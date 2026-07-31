"""Load a fedrag `--save-best` LoRA checkpoint into a baseline's own LLM.

Only meaningful for grag/gretriever (the two baselines with a local,
trainable LLM instance) -- hipporag/comorag/flare/linearrag call a frozen
LLM over an API and have no model object to load weights into.

The checkpoint (see fedcond_grag/trainer.py::FedTrainer._save_best) stores
only the LoRA adapter's state_dict (`lora_state_dict()` from
fedcond_grag/server/lora_aggregate/keys.py -- plain `model.state_dict()`
filtered to `.lora_A.`/`.lora_B.` keys, loaded back with plain
`load_state_dict()`, not PEFT's own get/set_peft_model_state_dict helpers),
keyed the same way regardless of which wrapper class built the model
(DualGraphLLM, or grag/gretriever's own GraphLLM) -- as long as the LoRA
config (rank, alpha, target_modules) matches the run that produced it.
grag/gretriever's own LoRA defaults (r=8, alpha=16, target_modules=[q_proj,
v_proj]) already match main.py's own defaults, so a checkpoint trained
without overriding those (e.g. via --dual-graph-mode none, a text-only LoRA
run) loads directly.
"""

from __future__ import annotations

import torch

from fedcond_grag.server.lora_aggregate import has_lora


def load_lora_checkpoint(model, path: str) -> None:
    """`model` is the underlying (PEFT-wrapped) causal LM, e.g. GraphLLM.model."""
    if not has_lora(model):
        raise RuntimeError(
            f"--load_checkpoint {path} given but this model has no LoRA adapter -- "
            "pass --llm_frozen False (matching the rank/alpha/target_modules used to "
            "produce this checkpoint) so there's a LoRA adapter to load into."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, _ = model.load_state_dict(payload["lora"], strict=False)
    lora_missing = [k for k in missing if ".lora_A." in k or ".lora_B." in k]
    if lora_missing:
        raise RuntimeError(
            f"--load_checkpoint {path}: {len(lora_missing)} LoRA keys not found in the "
            "current model (rank/alpha/target_modules mismatch with the run that "
            f"produced this checkpoint?) -- first few: {lora_missing[:5]}"
        )
    val_hit = (payload.get("val_metrics") or {}).get("hit")
    print(f"    [checkpoint] loaded LoRA from {path} "
          f"(round {payload.get('round')}, val hit {val_hit})", flush=True)
