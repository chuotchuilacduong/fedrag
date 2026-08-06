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

**Deviation 4:** `utils/graph_retrieval.py::find_topk_subgraph()` rebuilt
`graph.edge_index.T.tolist()` and did a Python `list.index()`/`in` scan over
it *per edge, per candidate node* to map an (src, dst) pair back to its
position in `graph.edge_attr` -- O(num_edges) per lookup. Harmless for
GRAG's own small task KGs but pathological against this project's Tri-Graph
fallback (tens of thousands of edges): with `client_runner.py::_retrieve_sample`
also not passing `sims` (see below), a single retrieval call iterated
*every node in the graph* doing this, taking upwards of an hour for one
query. Patched to build the (src, dst) -> edge-index dict once per call
(same pattern already used by `get_trunk_triplets`/`get_augmented_triplets`/
`get_augmented_path` elsewhere in this file) instead of re-deriving it with
a linear scan each time; output is identical, just no longer quadratic.
`client_runner.py::_retrieve_sample` was also fixed to pass a cheap
node-embedding cosine-similarity `sims` into `retrive_on_graphs`, which was
always called with `sims=None` -- that put every query down the
O(graph.num_nodes) "score every node's own subgraph" path instead of the
intended O(retrieval_topk) `find_topk_subgraph` fast path.

**Deviation 5:** `model/graph_llm.py::__init__` always loaded the LLM as
`torch_dtype=torch.float16` with no quantization option at all, unlike this
project's own `fedcond_grag/model/graph_llm.py` -- qwen2.5-7b in fp16 (~15GB)
doesn't fit an 8GB GPU without accelerate's automatic (slow, and here
outright broken -- see below) CPU offload. Patched to add the same
`llm_load_in_4bit` bitsandbytes path as this project's own model, and the
same `torch.backends.cuda.enable_cudnn_sdp(False)` workaround (newer torch
versions' cuDNN SDPA backend throws "No execution plans support the graph"
on some GPU/cuDNN combinations -- hit exactly this running the fp16+CPU-offload
path on this project's dev GPU before the 4-bit patch was added).

**Deviation 6:** `model/graph_llm.py::forward()`/`inference()` concatenate
`graph_embeds` (float32 -- `self.graph_encoder`/`self.projector` are plain
`nn.Module`s, only ever moved with `.to(device)`, never cast to the LLM's
dtype) with `bos_embeds`/`inputs_embeds` (whatever dtype the LLM was
loaded in) into one `inputs_embeds` tensor. On GPU this went unnoticed
(`maybe_autocast()` wraps the *following* `self.model(...)` call in
`torch.cuda.amp.autocast`, which is lenient enough about mixed float32/fp16
inputs to whitelisted ops). `maybe_autocast()` explicitly disables autocast
on CPU (`enable_autocast = self.device != torch.device("cpu")`), so a CPU
run hits a hard `RuntimeError: mat1 and mat2 must have the same dtype, but
got Float and BFloat16` the moment the mismatched `inputs_embeds` reaches
the first attention projection. Patched both call sites to
`.to(self.word_embedding.weight.dtype)` right after `self.projector(...)`,
so `graph_embeds` always matches the LLM's embedding dtype regardless of
device/dtype config.

See `fedcond_grag/baselines/grag/client_runner.py` for how this package is
driven: per-client, reusing this project's own Tri-Graph (or HippoRAG's
cached OpenIE triples when available -- see that file's module docstring),
not GRAG's own WebQSP/ExplaGraphs KG datasets.
