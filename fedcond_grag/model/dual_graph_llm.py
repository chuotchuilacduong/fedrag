import torch
import torch.nn as nn
from torch_scatter import scatter

from fedcond_grag.model.gnn import load_gnn_model
from fedcond_grag.model.graph_llm import IGNORE_INDEX, GraphLLM


class DualGraphLLM(GraphLLM):
    """G-Retriever with evidence and condensed graph soft-prompt tokens."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.dual_graph_mode = getattr(args, "dual_graph_mode", "shared")

        if self.dual_graph_mode in ("shared", "no_synthetic"):
            # shared: one GNN encoder for both graphs.
            # no_synthetic: separate encoders exist but condensed slot always gets
            #   evidence graph (server synthetic graph is never used).
            # In both cases condensed_encoder / projector_c are not created.
            self.condensed_encoder = None
            self.projector_c = None
            if self.dual_graph_mode == "shared":
                print("[DualGraphLLM] shared-encoder ablation: condensed graph uses same GNN + projector as evidence graph")
            else:
                print("[DualGraphLLM] no-synthetic ablation: both slots use evidence graph (server synthetic graph ignored)")
        else:
            gnn_model_name_c = getattr(args, "gnn_model_name_c", "gat")
            gnn_num_layers_c = getattr(args, "gnn_num_layers_c", None) or args.gnn_num_layers
            gnn_num_heads_c = getattr(args, "gnn_num_heads_c", None) or args.gnn_num_heads
            gnn_hidden_dim_c = getattr(args, "gnn_hidden_dim_c", None) or args.gnn_hidden_dim
            gnn_in_dim_c = getattr(args, "gnn_in_dim_c", None) or args.gnn_in_dim
            prompt_dim = int(getattr(self.projector[-1], "out_features", getattr(self.model.config, "hidden_size", 4096)))

            _gnn_dtype = torch.bfloat16
            self.condensed_encoder = load_gnn_model[gnn_model_name_c](
                in_channels=gnn_in_dim_c,
                out_channels=gnn_hidden_dim_c,
                hidden_channels=gnn_hidden_dim_c,
                num_layers=gnn_num_layers_c,
                dropout=args.gnn_dropout,
                num_heads=gnn_num_heads_c,
            ).to(dtype=_gnn_dtype, device=self._device_cache)

            self.projector_c = nn.Sequential(
                nn.Linear(gnn_hidden_dim_c, 2048),
                nn.GELU(),
                nn.Linear(2048, prompt_dim),
            ).to(dtype=_gnn_dtype, device=self._device_cache)

    def _graph_edge_attr(self, graph):
        return getattr(graph, "edge_attr", None)

    def _graph_batch(self, graph):
        if hasattr(graph, "batch") and graph.batch is not None:
            return graph.batch
        return torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)

    def _encode_one_graph(self, graph, encoder, projector):
        _param = next(encoder.parameters(), None)
        enc_device = _param.device if _param is not None else graph.x.device
        graph = graph.to(enc_device)
        dtype = _param.dtype if _param is not None else graph.x.dtype
        x = graph.x.to(dtype)
        edge_attr = self._graph_edge_attr(graph)
        if edge_attr is not None:
            edge_attr = edge_attr.to(dtype)
        n_embeds, _ = encoder(x, graph.edge_index.long(), edge_attr)
        g_embeds = scatter(n_embeds, self._graph_batch(graph), dim=0, reduce="mean")
        return projector(g_embeds)

    def encode_graphs(self, samples):
        evidence_graph = samples.get("graph", samples.get("evidence_graph"))
        if evidence_graph is None:
            raise KeyError("DualGraphLLM requires samples['graph'] or samples['evidence_graph']")

        z_e = self._encode_one_graph(evidence_graph, self.graph_encoder, self.projector)

        condensed_graph = samples.get("condensed_graph")
        if self.dual_graph_mode == "no_synthetic":
            # No-synthetic ablation: ignore server graph entirely; both slots get
            # the evidence graph encoded by the same evidence encoder + projector.
            z_c = self._encode_one_graph(evidence_graph, self.graph_encoder, self.projector)
        elif self.dual_graph_mode == "shared":
            # Shared-encoder ablation: encode condensed graph with same GNN + projector
            if condensed_graph is None:
                z_c = z_e * 0.0
            else:
                z_c = self._encode_one_graph(condensed_graph, self.graph_encoder, self.projector)
        else:
            if condensed_graph is None:
                z_c = torch.zeros_like(z_e)
            else:
                z_c = self._encode_one_graph(condensed_graph, self.condensed_encoder, self.projector_c)

        return self._apply_dual_graph_mode(z_e, z_c)

    def _apply_dual_graph_mode(self, z_e, z_c):
        # Multiply-by-zero preserves grad_fn so frozen-LLM training does not
        # raise "element 0 ... does not require grad" when a branch is masked.
        mode = str(self.dual_graph_mode).lower()
        if mode in {"both", "dual", "shared", "no_synthetic"}:
            return z_e, z_c
        if mode in {"evidence_only", "ze_only", "z_e_only"}:
            return z_e, z_c * 0.0
        if mode in {"condensed_only", "zc_only", "z_c_only"}:
            return z_e * 0.0, z_c
        if mode in {"random_condensed", "random_zc", "random_z_c"}:
            return z_e, z_c * 0.0 + torch.randn_like(z_c).detach()
        if mode in {"none", "text_only"}:
            return z_e * 0.0, z_c * 0.0
        raise ValueError(f"Unsupported dual_graph_mode: {self.dual_graph_mode}")

    def _special_embeds(self):
        # bos/pad embeds were cached in GraphLLM.__init__ — reuse them.
        return self._bos_embeds_cached, self._pad_embed_cached

    def forward(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        eos_tokens = self.tokenizer(self.eos_text, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(self.eos_user_text, add_special_tokens=False)
        bos_embeds, pad_embeds = self._special_embeds()

        z_e, z_c = self.encode_graphs(samples)

        batch_size = len(samples["id"])
        dev = self._device_cache

        # Collect all token-id sequences before touching the GPU so we can
        # embed the entire batch in one call instead of batch_size separate calls.
        all_text_ids: list[list[int]] = []
        all_label_ids: list[list[int]] = []
        for i in range(batch_size):
            label_ids = labels.input_ids[i][: self.max_new_tokens] + eos_tokens.input_ids
            text_ids = (descriptions.input_ids[i][: self.max_txt_len]
                        + questions.input_ids[i]
                        + eos_user_tokens.input_ids
                        + label_ids)
            all_text_ids.append(text_ids)
            all_label_ids.append(label_ids)

        # One batched word_embedding call (1 GPU kernel) instead of batch_size calls.
        max_text_len = max(len(ids) for ids in all_text_ids)
        pad_id = self.tokenizer.pad_token_id
        text_id_tensor = torch.full((batch_size, max_text_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(all_text_ids):
            text_id_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        all_text_embeds = self.word_embedding(text_id_tensor.to(dev))  # [B, T, H]

        # Assemble per-sample sequences (variable length due to graph token prepend).
        batch_inputs_embeds = []
        seq_lens: list[int] = []
        batch_label_input_ids: list[list[int]] = []
        for i in range(batch_size):
            text_len = len(all_text_ids[i])
            label_len = len(all_label_ids[i])
            text_embeds = all_text_embeds[i, :text_len]                # [T_i, H]
            graph_tokens = torch.stack([z_e[i], z_c[i]], dim=0)        # [2, H]
            inputs_embeds = torch.cat([bos_embeds, graph_tokens, text_embeds], dim=0)
            seq_len = inputs_embeds.shape[0]
            batch_inputs_embeds.append(inputs_embeds)
            seq_lens.append(seq_len)
            batch_label_input_ids.append(
                [IGNORE_INDEX] * (seq_len - label_len) + all_label_ids[i]
            )

        # Left-pad to max_length; build attention_mask directly on GPU.
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
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        eos_user_tokens = self.tokenizer(self.eos_user_text, add_special_tokens=False)
        bos_embeds, pad_embeds = self._special_embeds()

        z_e, z_c = self.encode_graphs(samples)

        batch_size = len(samples["id"])
        dev = self._device_cache

        # Collect token ids; embed in one batched call.
        all_text_ids: list[list[int]] = []
        for i in range(batch_size):
            text_ids = (descriptions.input_ids[i][: self.max_txt_len]
                        + questions.input_ids[i]
                        + eos_user_tokens.input_ids)
            all_text_ids.append(text_ids)

        max_text_len = max(len(ids) for ids in all_text_ids)
        pad_id = self.tokenizer.pad_token_id
        text_id_tensor = torch.full((batch_size, max_text_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(all_text_ids):
            text_id_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        all_text_embeds = self.word_embedding(text_id_tensor.to(dev))  # [B, T, H]

        batch_inputs_embeds: list[torch.Tensor] = []
        seq_lens: list[int] = []
        for i in range(batch_size):
            text_embeds = all_text_embeds[i, : len(all_text_ids[i])]
            graph_tokens = torch.stack([z_e[i], z_c[i]], dim=0)
            inputs_embeds = torch.cat([bos_embeds, graph_tokens, text_embeds], dim=0)
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
        pred = [p.strip() for p in pred]

        return {
            "id": samples["id"],
            "pred": pred,
            "label": samples["label"],
            "question": samples["question"],
            "desc": samples["desc"],
        }
