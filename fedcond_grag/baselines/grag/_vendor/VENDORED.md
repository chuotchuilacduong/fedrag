# Vendored from HuieL/GRAG

The files under `src/` are copied from https://github.com/HuieL/GRAG (MIT
licensed, see `LICENSE` in this directory), fetched from `main` on
2026-07-26. GRAG has no `setup.py`/`pyproject.toml` (not pip-installable),
and unlike `baselines/gretriever`, its GNN encoder is genuinely different
from this project's own (`fedcond_grag/model/gnn.py`) -- GRAG conditions
node/edge features on the query embedding before each conv layer
(`GAT.forward(x, edge_index, question_node, edge_attr, question_edge)`),
so there's nothing to drop-in reuse; the model has to be vendored.

Only the files actually needed for the `augment="none"` retrieval path (the
default, and the one this baseline uses) are vendored:
- `src/model/gnn.py`, `src/model/graph_llm.py` -- the query-conditioned GNN
  + GraphLLM wrapper.
- `src/utils/graph_retrieval.py`, `src/utils/text_graph.py` -- GRAG's actual
  contribution: k-hop ego-subgraph retrieval + cosine-similarity top-k
  merging (`retrive_on_graphs`, `find_topk_subgraph`, `merge_graphs`).
- `src/utils/lm_modeling.py`, `src/utils/collate.py` -- small support
  modules those two above import.

`model/llm.py` is vendored too even though `client_runner.py` never imports
it directly -- upstream's own `model/__init__.py` (also vendored, unmodified)
does `from src.model.llm import LLM` unconditionally, so it has to be
present for the package to import at all.

Not vendored (redundant with what's already in this repo, same lineage --
see `fedcond_grag/baselines/gretriever/`'s own note about sharing code with
G-Retriever, which GRAG is itself built on): `utils/config.py`,
`utils/evaluate.py`, `utils/ckpt.py`, `utils/sampler.py`, `utils/seed.py`,
`utils/lr_schedule.py`, `model/__init__.py`'s `llama_model_path` (this
project's own `fedcond_grag.model.llama_model_path` is used instead,
matching `baselines/gretriever`).

Package layout preserves GRAG's own `src.model.*` / `src.utils.*` import
paths verbatim (added `_vendor/` to `sys.path` at import time in
`client_runner.py`) specifically so nothing needed hand-patched import
paths, unlike `baselines/hipporag/_vendor`'s deviations 2/3.

**Deviation 1:** `utils/graph_retrieval.py` loaded the sbert model
(~1.3GB) at *module import time* unconditionally, even though it's only
used when `augment != "none"`. Since this baseline always calls with
`augment="none"` (see `client_runner.py`), that load was 100% wasted every
run. Patched to load lazily on first actual use; behavior for `augment !=
"none"` callers is unchanged.

**Deviation 2:** `utils/graph_retrieval.py::get_triplets()` hardcoded
`.view(-1, 1024)`, assuming GRAG's paper-default 1024-dim embeddings
(all-roberta-large-v1). This project uses 384-dim (all-MiniLM-L6-v2)
throughout -- see `client_runner.py`'s module docstring. Patched to use
`subgraph.x.size(-1)` instead of the hardcoded constant.

**Deviation 3:** `model/graph_llm.py::__init__` hardcoded two things to
GRAG's own Llama-2-7B setup: a 4-GPU `max_memory` map (`{0,1,2,3: '20GiB'}`,
errors on some accelerate versions when those devices don't exist -- this
project's dev machine has one GPU) and the projector's output dim (`4096`,
Llama-2-7B's hidden size; their own comment even says "replace with
nn.Linear(2048, 5120) if using llama2-13b"). Patched both to be derived at
runtime (`torch.cuda.device_count()`, `self.model.config.hidden_size`) so
this works with any local LLM/GPU count, e.g. this project's Qwen2.5
checkpoints (hidden_size=1536) on a single GPU.

See `fedcond_grag/baselines/grag/client_runner.py` for how this package is
driven: per-client, reusing this project's own Tri-Graph (or HippoRAG's
cached OpenIE triples when available -- see that file's module docstring),
not GRAG's own WebQSP/ExplaGraphs KG datasets.
