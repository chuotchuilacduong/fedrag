"""Per-client local FLARE baseline (https://github.com/jzbjyb/FLARE).

Training-free active-retrieval baseline: unlike `hipporag`/`gretriever`/
`grag`, there's no graph and no local training -- just iterative
retrieve-then-generate per question. Only FLARE's actual algorithmic
contribution (`ApiReturn`, the confidence-based query-masking logic) is
vendored under `_vendor/`; see `_vendor/VENDORED.md` for why the
orchestration loop itself is reimplemented in `client_runner.py` rather than
vendored (FLARE's own loop is tightly coupled to a legacy OpenAI API + ES
Wikipedia retrieval + dataset templates this project doesn't have/use).
"""

from .client_runner import run_client_baseline, run_all_clients

__all__ = ["run_client_baseline", "run_all_clients"]
