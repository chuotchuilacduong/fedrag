"""Optional WandB logging for baseline runs (grag/gretriever/hipporag/comorag/
flare/linearrag), landing in the SAME project as fedrag's own training
(`fedcond_grag/trainer.py`'s `wandb.init(project=... "fedcond-graphrag")`) so
baseline numbers show up on the same dashboard for comparison.

Best-effort: never raises. Matches trainer.py's own "always attempt to log,
don't gate on WANDB_API_KEY" behavior -- set WANDB_MODE=disabled/offline to
opt out, or just don't `wandb login`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _load_dotenv() -> None:
    """Load .env from project root into os.environ without overwriting existing vars.

    Same logic as fedcond_grag/trainer.py::FedTrainer._load_dotenv.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and val and key not in os.environ:
            os.environ[key] = val


def log_baseline_result(method: str, dataset: str, result: dict[str, Any]) -> None:
    """Log one baseline run's result to WandB.

    `result` is either a single-client dict (from run_client_baseline) or a
    run_all_clients summary (`{"per_client": [...], "mean": {...}, ...}`).
    """
    try:
        _load_dotenv()
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "fedcond-graphrag"),
            name=f"baseline-{method}-{dataset}",
            tags=["baseline", method, dataset],
            config={"method": method, "dataset": dataset,
                    **{k: v for k, v in result.items() if k != "per_client"}},
            reinit="finish_previous",
        )
        mean = result.get("mean", result)
        wandb.log({f"baseline/{k}": v for k, v in mean.items() if isinstance(v, (int, float))})
        for client_result in result.get("per_client", []):
            cid = client_result.get("client_id", "?")
            wandb.log({
                f"baseline/client_{cid}/{k}": v
                for k, v in client_result.items() if isinstance(v, (int, float))
            })
        run.finish()
        print(f"[wandb] logged: {run.url}", flush=True)
    except Exception as exc:
        print(f"[wandb] logging failed, continuing without: {exc}", flush=True)
