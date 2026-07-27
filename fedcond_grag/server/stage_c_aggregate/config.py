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
    # Stage C mode: "gradient_match" | "repr_align" | "both" | "fedrag"
    # "fedrag" = paper Algorithm 1: Phase 0 repr-align init, then client-side
    # query-conditioned synthetic-memory adaptation with server delta
    # aggregation (Eq. 18-19). Older modes refine the synthetic graph
    # server-side against client anchors every round.
    "server_stage_c_mode": "gradient_match",
    # FedRAG (mode "fedrag") knobs — paper §3.5/§3.6
    "syn_mem_steps": 5,        # K_mem local adaptation steps per round
    "syn_mem_lr": 1e-3,        # η_syn client-side synthetic-memory LR
    "syn_mem_batch_size": 0,   # 0 = reuse local_batch_size
    "syn_soft_tau": 0.1,       # τ for differentiable soft synthetic retrieval
    "lambda_gm": 0.1,          # λ_gm  gradient-matching weight
    "lambda_align_mem": 0.1,   # λ_align client-side alignment weight
    "lambda_reg_mem": 0.01,    # λ_reg  client-side regularization weight
    "eta_agg": 1.0,            # η_agg  server delta aggregation step size
    "eta_reg": 1e-2,           # η_reg  server L_reg gradient step size
    "server_reg_steps": 1,     # L_reg steps after delta aggregation
    "repr_align_weight": 1.0,
    "grad_match_weight": 1.0,
    "lambda_div": 0.1,
    "lambda_deg": 0.05,
    "repr_proj_out_dim": 4096,
    "gnn_model_name": "gcn",
    "gnn_model_name_c": "gcn",
}
