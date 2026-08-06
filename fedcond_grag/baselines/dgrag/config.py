"""DGRAG configuration.

Every knob from implement_dg_rag.md §6, tagged [PAPER] or [FREE].
All [FREE] values are exposed here so they appear in the run log — never
silently invented inside the algorithm modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Ablation / variant flags
# ---------------------------------------------------------------------------
AblationFlag = Literal["none", "w/o_CD", "w/o_SE", "w/o_BQ", "w/o_KG"]
VariantFlag = Literal["dgrag", "naive_rag", "local_rag", "cloud_rag", "cent_rag"]


@dataclass
class DGRAGConfig:
    # --- models [PAPER] ---
    # In the §9 no-cloud adaptation both roles use the same endpoint/model.
    slm_name: str = "qwen2.5:7b-instruct"         # edge SLM
    llm_name: str = "qwen2.5:7b-instruct"          # cloud LLM (same model, §9.3)
    llm_base_url: str = "http://localhost:11434/v1"
    embedding_model_name: str = "all-MiniLM-L6-v2"  # shared everywhere

    # --- chunking [FREE] ---
    chunk_size: int = 1200       # tokens (words proxy)
    chunk_overlap: int = 100

    # --- offline extraction [FREE] ---
    glean_rounds: int = 1        # gleaning passes to improve recall
    desc_max_tokens: int = 512   # max description length before LLM compression

    # --- graph partitioning [PAPER + DERIVED] ---
    target_entities_per_subgraph: int = 40   # ablation optimum §IV-E1
    subgraph_min_size: int = 10              # merge smaller communities
    leiden_n_iterations: int = 10
    leiden_seed: int = 42
    # Resolution sweep for hierarchical Leiden (derived, not in paper)
    leiden_resolutions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)

    # --- subgraph summaries [FREE] ---
    summary_max_tokens: int = 200

    # --- local retrieval [FREE] ---
    top_ent: int = 60
    top_rel: int = 60
    top_ent_expanded: int = 100
    top_rel_expanded: int = 100
    top_chunk: int = 5
    ctx_max_tokens: int = 8000

    # --- batch generation [FREE] ---
    batch_b: int = 3                   # candidates per query
    batch_temperature: float = 0.8
    batch_top_p: float = 0.95

    # --- gate mechanism [FREE] ---
    gate_threshold: float = 0.75       # calibrate to 91%/29.2% local rates

    # --- cross-edge retrieval [PAPER + FREE] ---
    thr_summary: float = 0.4           # [PAPER §IV-E2]
    thr_global: float = 0.4            # [PAPER §IV-E2]
    top_m: int = 1                     # summary matches: 1 domain-specific, 5 mixed [PAPER]
    top_k_edges: int = 1               # distinct edges contacted [FREE]
    cloud_ctx_max_tokens: int = 24000  # [FREE]

    # --- federation ---
    num_clients: int = 4

    # --- LLM call accounting (see §9.5) ---
    # Logged but not controlled here; the pipeline computes it at runtime.

    # --- ablation / variant ---
    ablation: AblationFlag = "none"
    variant: VariantFlag = "dgrag"

    # --- runtime ---
    device: str = "cpu"
    seed: int = 42
