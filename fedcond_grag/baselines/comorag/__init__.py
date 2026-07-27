"""Per-client local ComoRAG baseline (https://github.com/EternityJune25/ComoRAG).

ComoRAG is itself a fork of HippoRAG with a "veridical/semantic/episodic"
memory-pool retrieval loop layered on top; vendored under `_vendor/` (not
pip-installable upstream -- no setup.py). See `_vendor/VENDORED.md` for the
lineage and packaging fixes. `client_runner.py` drives it per federated
client, mirroring `baselines/hipporag`'s methodology.
"""

from .client_runner import run_client_baseline, run_all_clients

__all__ = ["run_client_baseline", "run_all_clients"]
