# Vendored from EternityJune25/ComoRAG

`src/comorag/` is a copy of https://github.com/EternityJune25/ComoRAG's
`src/comorag/` (MIT licensed, see `LICENSE` in this directory), fetched from
`main` on 2026-07-27.

**ComoRAG is itself a fork of HippoRAG.** Many files here are byte-identical
to what `baselines/hipporag` uses (`information_extraction/__init__.py`,
`prompts/__init__.py`, `prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json`,
`prompts/templates/ner.py`, `ner_query.py`, `triple_extraction.py`,
`utils/typing_utils.py`, `utils/logging_utils.py` -- same SHA hashes).
ComoRAG's actual contribution sits on top: `ComoRAG.py`'s "veridical /
semantic / episodic" memory-pool retrieval loop, plus new
`utils/cluster_utils.py` (soft clustering for the semantic/episodic layers),
`utils/memory_utils.py`, `utils/timeline_utils.py`,
`utils/summarization_utils.py`, `utils/agents.py`, and new prompt templates
(`agent_probe.py`, `memory_fusion.py`, `node_fusion.py`,
`rag_qa_mc*.py`, `rag_qa_narrativeqa.py`).

Vendored (not pip-installed, same reasoning as `baselines/hipporag`'s
original vendoring before it moved to a pinned git install): no
`setup.py`/`pyproject.toml` upstream. Kept as a self-contained copy here
rather than trying to share code with `baselines/hipporag`'s pip-installed
`hipporag` package, to avoid coupling two independently-evolving baselines'
versions together.

Package layout preserves the upstream `src.comorag.*` import path (this
repo's `ComoRAG.py` itself does `from src.comorag.utils.timeline_utils
import TimelineSummarizer` -- a hardcoded absolute import, not just a
fallback like `hipporag`'s template loader had) by vendoring under
`_vendor/src/comorag/` and adding `_vendor/` to `sys.path` in
`../client_runner.py`. This incidentally also fixes
`prompts/prompt_template_manager.py`'s dynamic template loader, which has
the exact same `importlib.import_module(module_name, 'comorag')`
hardcoded-package-name issue documented in `baselines/hipporag`'s
`VENDORED.md` deviation 2 -- its *first* fallback attempt is
`f"src.comorag.prompts.templates.{script_name}"`, which now resolves
correctly given the `src/` layout, so that file needed no direct patch here.

**Deviations** (all packaging/platform fixes, no algorithm changes):
1. `ComoRAG.py` imported `from click import prompt` and `from cv2 import
   log`, neither ever referenced anywhere in the file (verified by grep) --
   dead imports that would otherwise require installing `opencv-python` for
   nothing. Dropped both.
2. Same Windows-path colon issue as `baselines/hipporag` (Ollama tags like
   `qwen2.5:1.5b-instruct` break `.replace("/", "_")`-only sanitization):
   fixed in `ComoRAG.py` (working dir + OpenIE results filename),
   `llm/openai_gpt.py` and `llm/vllm_offline.py` (cache filenames).
3. `embedding_model/__init__.py`'s `_get_embedding_model_class()` default
   branch was `return` with no value -- the log message right above it says
   "using BGEEmbeddingModel as default" but it actually returned `None`,
   which would crash on any embedding_model_name that isn't `"bge-*"` or
   `"text-embedding-3-small"` (e.g. this project's usual
   `all-MiniLM-L6-v2`). `BGEEmbeddingModel` itself is a generic
   AutoModel/AutoTokenizer wrapper, not actually BGE-specific, so returning
   it here matches upstream's own stated intent. Fixed.
4. `ComoRAG.py` eagerly imports
   `.information_extraction.openie_vllm_offline.VLLMOfflineOpenIE`, which
   imports `vllm` at module level -- same situation as
   `baselines/hipporag`'s vllm-offline OpenIE backend. `client_runner.py`
   installs the same kind of `sys.modules` stub before importing `ComoRAG`
   so this succeeds without installing real `vllm` (which pins a
   torch/CUDA build this project's own pin conflicts with); it's never
   actually called since this baseline always uses the online
   OpenAI-compatible backend (`llm_base_url`).
5. `embedding_store.py`'s `EmbeddingStore._load_data()` only initializes
   `hash_id_to_text`/`text_to_hash_id` in the "loaded existing parquet file"
   branch, not the "no file yet, empty store" branch (which sets the other
   four attributes but not these two) -- a genuine upstream bug, reproduced
   and confirmed via smoke test: `ComoRAG.tri_retrieve()` reads
   `self.level_store.text_to_hash_id`, and `level_store` is the timeline
   summarizer's level-0 store, which starts out empty and can *stay* empty
   (e.g. if timeline summarization fails for all chunks -- observed here as
   a `RuntimeError: The size of tensor a (630) must match the size of
   tensor b (512)` from a too-long summary hitting BGE's tokenizer, on an 8
   passage/3 question corpus). Result was `AttributeError: 'EmbeddingStore'
   object has no attribute 'text_to_hash_id'` on every query. Fixed by
   initializing both dicts to `{}` in the empty-store branch, matching what
   the loaded-branch does for the equivalent empty case.

Note: going through `BGEEmbeddingModel` with a non-BGE model name
(`all-MiniLM-L6-v2`, chosen for consistency with the rest of this project)
means every input gets BGE's hardcoded instruction prefix ("Generate a
representation for this sentence to retrieve relevant articles:") prepended
before encoding, query and passage alike. Harmless for a plain
sentence-transformers model (no error, just a few extra unused tokens) but
worth knowing about if embedding quality looks off.

See `fedcond_grag/baselines/comorag/client_runner.py` for how this package
is driven (per-client indexing + evaluation, not part of the upstream repo).
