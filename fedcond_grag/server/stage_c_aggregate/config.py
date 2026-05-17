"""Default config for the fedcond_qa Stage C algorithm."""

config = {
    "method": "GCond",
    "op_epoche": 5,
    "server_condense_iters": 50,
    "condense_iters": 50,
    "local_epochs": 0,
    "num_global_syn_nodes": 1024,
    "lr_feat": 1e-2,
    "lr_adj": 1e-2,
    "pge_hidden": 256,
    "pge_topk": 8,
    "type_emb_dim": 16,
    "surrogate_type_weight": 1.0,
    "surrogate_link_weight": 0.5,
    "surrogate_align_weight": 0.1,
    "match_norm_weight": 0.0,
    "condense_refresh_every": 10,
    "preserve_sep_topology": True,
}
