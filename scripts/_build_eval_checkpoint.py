"""Splice one ablation arm's final synthetic memory into the reused frozen
backbone checkpoint, producing a --load-checkpoint-compatible payload for a
held-out --eval-only pass. See scripts/analyze_regularization_diversity.py
and the "Effect of Regularization on Node Diversity" experiment.
"""
import sys
from pathlib import Path

import torch

base_ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)  # checkpoints_5r3e/.../best.pt
last_snapshot = torch.load(sys.argv[2], map_location="cpu", weights_only=False)  # experiments/.../checkpoints/ckpt_00XX.pt
out_path = Path(sys.argv[3])

base_ckpt["synthetic_memory"] = last_snapshot["synthetic_memory_state"]
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(base_ckpt, out_path)
print(f"Wrote {out_path} (synthetic_memory from checkpoint_index={last_snapshot['checkpoint_index']}, "
      f"phase={last_snapshot['phase']}, round={last_snapshot['round']})")
