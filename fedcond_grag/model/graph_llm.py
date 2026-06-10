import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch_scatter import scatter
from fedcond_grag.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Legacy Llama-2 chat markers — used only when the tokenizer is Llama-family.
# Other models get their native template via resolve_prompt_template().
BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


def resolve_prompt_template(tokenizer) -> tuple[str, str, str]:
    """Pick prompt markers native to the tokenizer's model family.

    Returns (bos_text, eos_user_text, eos_text). Training labels end with
    eos_text, so it must be the tokenizer's real EOS: with a frozen LLM the
    model can only stop generation through tokens it already knows. Hardcoded
    Llama markers fed to a Qwen tokenizer become literal text — the model then
    emits the string "</s>" instead of stopping.
    """
    if "<|im_start|>" in tokenizer.get_vocab():  # Qwen family
        if tokenizer.eos_token == "<|im_end|>":  # instruct-tuned → ChatML-aligned
            return (
                "<|im_start|>user\n",
                "<|im_end|>\n<|im_start|>assistant\n",
                "<|im_end|>",
            )
        # Base Qwen (eos <|endoftext|>): not ChatML-aligned — ChatML markers make
        # it echo the input instead of answering. Plain QA format works:
        # verified "desc\nquestion\nAnswer:" → correct answer on Qwen2.5-1.5B.
        return ("", "\nAnswer:", tokenizer.eos_token or "<|endoftext|>")
    if tokenizer.bos_token == "<s>":  # Llama family — original G-Retriever format
        return (BOS, EOS_USER, EOS)
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""
    return (bos, "\nAnswer:", eos)


class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.eval_max_new_tokens = int(getattr(args, "eval_max_new_tokens", self.max_new_tokens))

        print('Loading LLAMA')
        import torch
        n_gpus = torch.cuda.device_count()
        max_memory = {i: '80GiB' for i in range(n_gpus)} if n_gpus > 0 else None
        kwargs = {
            "device_map": "auto",
            "revision": "main",
        }
        if max_memory:
            kwargs["max_memory"] = max_memory

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        self.bos_text, self.eos_user_text, self.eos_text = resolve_prompt_template(self.tokenizer)
        print(f"Prompt template: bos={self.bos_text!r} eos_user={self.eos_user_text!r} eos={self.eos_text!r}")

        load_in_8bit = bool(getattr(args, "llm_load_in_8bit", False))
        load_in_4bit = bool(getattr(args, "llm_load_in_4bit", False))

        # Use PyTorch's built-in SDPA — automatically dispatches to the fastest
        # available kernel (Flash Attention on Ampere/Ada, memory-efficient on
        # older GPUs). No extra package needed; torch 2.x enables this by default.
        _attn_impl = "sdpa"
        print("Using SDPA attention (Flash Attention kernel auto-selected)")

        if load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_path,
                quantization_config=bnb_cfg,
                low_cpu_mem_usage=True,
                attn_implementation=_attn_impl,
                **kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_path,
                torch_dtype=None if load_in_8bit else torch.float16,
                low_cpu_mem_usage=True,
                load_in_8bit=load_in_8bit,
                attn_implementation=_attn_impl,
                **kwargs
            )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model

        # Propagate hf_device_map from the inner LLM to the wrapper so that
        # FedTrainer's "if not hasattr(model, 'hf_device_map'): model.to(cpu)"
        # guard also applies here — otherwise it would move graph_encoder to CPU
        # while word_embedding (a reference into the quantized LLM) stays on CUDA.
        if hasattr(model, "hf_device_map"):
            self.hf_device_map = model.hf_device_map

        if getattr(args, "llm_gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("Gradient checkpointing enabled — activation memory reduced ~4×")

        print('Finish loading LLAMA!')

        llm_hidden_size = self.model.config.hidden_size

        # word_embedding (nn.Embedding) is never quantized by bitsandbytes, so its
        # weight always lives on the true device — use it as the authoritative
        # device probe instead of self.model.device / next(model.parameters()),
        # both of which return 'cpu' for Params4bit objects in some bnb versions.
        self.word_embedding = self.model.model.get_input_embeddings()
        _true_device = self.word_embedding.weight.device

        _gnn_dtype = torch.bfloat16
        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(dtype=_gnn_dtype, device=_true_device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, llm_hidden_size),
        ).to(dtype=_gnn_dtype, device=_true_device)

        # Cache device + special-token embeddings so we don't iterate all
        # parameters or rebuild bos/pad embeddings on every forward pass.
        self._device_cache = _true_device
        with torch.no_grad():
            bos_ids = self.tokenizer(self.bos_text, add_special_tokens=False, return_tensors='pt').input_ids[0]
            self._bos_embeds_cached = self.word_embedding(bos_ids.to(self._device_cache)).detach()
            pad_id = torch.tensor(self.tokenizer.pad_token_id, device=self._device_cache)
            self._pad_embed_cached = self.word_embedding(pad_id).detach().unsqueeze(0)

    @property
    def device(self):
        return self._device_cache

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples['graph']
        graphs = graphs.to(next(self.graph_encoder.parameters()).device)
        _dtype = next(self.graph_encoder.parameters()).dtype
        x = graphs.x.to(_dtype)
        edge_attr = graphs.edge_attr.to(_dtype) if graphs.edge_attr is not None else None
        n_embeds, _ = self.graph_encoder(x, graphs.edge_index.long(), edge_attr)

        # mean pooling
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')

        return g_embeds

    def forward(self, samples):

        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens — eos/eos_user tokens are still cheap; bos/pad
        # embeds are cached buffers built once in __init__.
        eos_tokens = self.tokenizer(self.eos_text, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(self.eos_user_text, add_special_tokens=False)
        bos_embeds = self._bos_embeds_cached
        pad_embeds = self._pad_embed_cached

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        dev = self.model.device

        # Collect all token-id sequences; embed the whole batch in one GPU call.
        all_text_ids: list[list[int]] = []
        all_label_ids: list[list[int]] = []
        for i in range(batch_size):
            label_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            text_ids = (descriptions.input_ids[i][:self.max_txt_len]
                        + questions.input_ids[i]
                        + eos_user_tokens.input_ids
                        + label_ids)
            all_text_ids.append(text_ids)
            all_label_ids.append(label_ids)

        max_text_len = max(len(ids) for ids in all_text_ids)
        pad_id = self.tokenizer.pad_token_id
        text_id_tensor = torch.full((batch_size, max_text_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(all_text_ids):
            text_id_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        all_text_embeds = self.word_embedding(text_id_tensor.to(dev))  # [B, T, H]

        batch_inputs_embeds: list[torch.Tensor] = []
        seq_lens: list[int] = []
        batch_label_input_ids: list[list[int]] = []
        for i in range(batch_size):
            text_len = len(all_text_ids[i])
            label_len = len(all_label_ids[i])
            text_embeds = all_text_embeds[i, :text_len]
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), text_embeds], dim=0)
            seq_len = inputs_embeds.shape[0]
            batch_inputs_embeds.append(inputs_embeds)
            seq_lens.append(seq_len)
            batch_label_input_ids.append(
                [IGNORE_INDEX] * (seq_len - label_len) + all_label_ids[i]
            )

        # Left-pad; build attention_mask directly on GPU.
        max_length = max(seq_lens)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long, device=dev)
        padded_embeds: list[torch.Tensor] = []
        padded_labels: list[list[int]] = []
        for i in range(batch_size):
            pad_len = max_length - seq_lens[i]
            if pad_len > 0:
                padded_embeds.append(
                    torch.cat([pad_embeds.expand(pad_len, -1), batch_inputs_embeds[i]])
                )
            else:
                padded_embeds.append(batch_inputs_embeds[i])
            attention_mask[i, pad_len:] = 1
            padded_labels.append([IGNORE_INDEX] * pad_len + batch_label_input_ids[i])

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        label_input_ids = torch.tensor(padded_labels, dtype=torch.long, device=dev)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens — use cached bos/pad embeds.
        eos_user_tokens = self.tokenizer(self.eos_user_text, add_special_tokens=False)
        bos_embeds = self._bos_embeds_cached
        pad_embeds = self._pad_embed_cached

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        dev = self.model.device

        all_text_ids: list[list[int]] = []
        for i in range(batch_size):
            text_ids = (descriptions.input_ids[i][:self.max_txt_len]
                        + questions.input_ids[i]
                        + eos_user_tokens.input_ids)
            all_text_ids.append(text_ids)

        max_text_len = max(len(ids) for ids in all_text_ids)
        pad_id = self.tokenizer.pad_token_id
        text_id_tensor = torch.full((batch_size, max_text_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(all_text_ids):
            text_id_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        all_text_embeds = self.word_embedding(text_id_tensor.to(dev))  # [B, T, H]

        batch_inputs_embeds: list[torch.Tensor] = []
        seq_lens: list[int] = []
        for i in range(batch_size):
            text_embeds = all_text_embeds[i, :len(all_text_ids[i])]
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), text_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            seq_lens.append(inputs_embeds.shape[0])

        max_length = max(seq_lens)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long, device=dev)
        padded_embeds: list[torch.Tensor] = []
        for i in range(batch_size):
            pad_len = max_length - seq_lens[i]
            if pad_len > 0:
                padded_embeds.append(
                    torch.cat([pad_embeds.expand(pad_len, -1), batch_inputs_embeds[i]])
                )
            else:
                padded_embeds.append(batch_inputs_embeds[i])
            attention_mask[i, pad_len:] = 1

        inputs_embeds = torch.stack(padded_embeds, dim=0)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.eval_max_new_tokens,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
