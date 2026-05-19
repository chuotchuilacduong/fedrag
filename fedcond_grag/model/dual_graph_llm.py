import torch
import torch.nn as nn
from torch_scatter import scatter

from fedcond_grag.model.gnn import load_gnn_model
from fedcond_grag.model.graph_llm import BOS, EOS, EOS_USER, IGNORE_INDEX, GraphLLM


class DualGraphLLM(GraphLLM):
    """G-Retriever with evidence and condensed graph soft-prompt tokens."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        gnn_model_name_c = getattr(args, "gnn_model_name_c", "gat")
        gnn_num_layers_c = getattr(args, "gnn_num_layers_c", None) or args.gnn_num_layers
        gnn_num_heads_c = getattr(args, "gnn_num_heads_c", None) or args.gnn_num_heads
        gnn_hidden_dim_c = getattr(args, "gnn_hidden_dim_c", None) or args.gnn_hidden_dim
        gnn_in_dim_c = getattr(args, "gnn_in_dim_c", None) or args.gnn_in_dim
        prompt_dim = int(getattr(self.projector[-1], "out_features", getattr(self.model.config, "hidden_size", 4096)))

        self.condensed_encoder = load_gnn_model[gnn_model_name_c](
            in_channels=gnn_in_dim_c,
            out_channels=gnn_hidden_dim_c,
            hidden_channels=gnn_hidden_dim_c,
            num_layers=gnn_num_layers_c,
            dropout=args.gnn_dropout,
            num_heads=gnn_num_heads_c,
        ).to(self.model.device)

        self.projector_c = nn.Sequential(
            nn.Linear(gnn_hidden_dim_c, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, prompt_dim),
        ).to(self.model.device)

        self.dual_graph_mode = getattr(args, "dual_graph_mode", "both")

    def _graph_edge_attr(self, graph):
        return getattr(graph, "edge_attr", None)

    def _graph_batch(self, graph):
        if hasattr(graph, "batch") and graph.batch is not None:
            return graph.batch
        return torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)

    def _encode_one_graph(self, graph, encoder, projector):
        graph = graph.to(self.model.device)
        n_embeds, _ = encoder(graph.x, graph.edge_index.long(), self._graph_edge_attr(graph))
        g_embeds = scatter(n_embeds, self._graph_batch(graph), dim=0, reduce="mean")
        return projector(g_embeds)

    def encode_graphs(self, samples):
        evidence_graph = samples.get("graph", samples.get("evidence_graph"))
        if evidence_graph is None:
            raise KeyError("DualGraphLLM requires samples['graph'] or samples['evidence_graph']")

        z_e = self._encode_one_graph(evidence_graph, self.graph_encoder, self.projector)

        condensed_graph = samples.get("condensed_graph")
        if condensed_graph is None:
            z_c = torch.zeros_like(z_e)
        else:
            z_c = self._encode_one_graph(condensed_graph, self.condensed_encoder, self.projector_c)

        return self._apply_dual_graph_mode(z_e, z_c)

    def _apply_dual_graph_mode(self, z_e, z_c):
        mode = str(self.dual_graph_mode).lower()
        if mode in {"both", "dual"}:
            return z_e, z_c
        if mode in {"evidence_only", "ze_only", "z_e_only"}:
            return z_e, torch.zeros_like(z_c)
        if mode in {"condensed_only", "zc_only", "z_c_only"}:
            return torch.zeros_like(z_e), z_c
        if mode in {"random_condensed", "random_zc", "random_z_c"}:
            return z_e, torch.randn_like(z_c)
        if mode in {"none", "text_only"}:
            return torch.zeros_like(z_e), torch.zeros_like(z_c)
        raise ValueError(f"Unsupported dual_graph_mode: {self.dual_graph_mode}")

    def _special_embeds(self):
        bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.model.device)
        bos_embeds = self.word_embedding(bos_ids)
        pad_id = torch.tensor(self.tokenizer.pad_token_id, device=self.model.device)
        pad_embeds = self.word_embedding(pad_id).unsqueeze(0)
        return bos_embeds, pad_embeds

    def forward(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds, pad_embeds = self._special_embeds()

        z_e, z_c = self.encode_graphs(samples)

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][: self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][: self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=self.model.device))
            graph_tokens = torch.stack([z_e[i], z_c[i]], dim=0)
            inputs_embeds = torch.cat([bos_embeds, graph_tokens, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask, device=self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids, device=self.model.device)

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

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds, pad_embeds = self._special_embeds()

        z_e, z_c = self.encode_graphs(samples)

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = descriptions.input_ids[i][: self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=self.model.device))
            graph_tokens = torch.stack([z_e[i], z_c[i]], dim=0)
            inputs_embeds = torch.cat([bos_embeds, graph_tokens, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask, device=self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            "id": samples["id"],
            "pred": pred,
            "label": samples["label"],
            "question": samples["question"],
            "desc": samples["desc"],
        }
