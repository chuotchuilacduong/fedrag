# Vendored from jzbjyb/FLARE

`api_return.py` is the `ApiReturn` class (+ its `Sentence` helper) extracted
verbatim from https://github.com/jzbjyb/FLARE's `src/templates.py` (MIT
licensed, see `LICENSE` in this directory), fetched from `main` on
2026-07-27. This is FLARE's actual algorithmic contribution: given a
generated span with per-token probabilities, truncate it at a sentence
boundary and mask out low-confidence tokens to form the retrieval query
(`use_as_query`, `mask_method='simple'` -- the setting FLARE's own published
`configs/2wikihop_flare_config.json` uses).

**Why only this one class, unlike `baselines/hipporag`/`grag` which vendor
the whole method:** FLARE's own orchestration (`src/openai_api.py`'s
`QueryAgent.complete()`/`ret_prompt()`, ~500 lines) is tightly coupled to
things this project doesn't use and can't reuse as-is:
- The legacy OpenAI `Completion.create(..., logprobs=0, echo=...)` API is
  the *only* path in their code that populates per-token
  tokens/probs/offsets -- required for the confidence-based masking above.
  Their `is_chat_model` branch (the one that would talk to a local
  OpenAI-compatible endpoint like Ollama) never populates them at all.
  Empirically: Ollama's `/v1/completions` does not return `logprobs`, but
  `/v1/chat/completions` does (`logprobs: true, top_logprobs: N`) -- so this
  baseline gets per-token probabilities through the chat endpoint instead,
  which upstream's code has no path for.
- A multiprocessing multi-API-key rotation manager (`KeyManager`,
  `CustomManager`) irrelevant to a single local endpoint.
- Retrieval via Elasticsearch over a full Wikipedia dump
  (`src/retriever.py`, `prep.py`), not a per-client passage shard.
- Dataset-specific few-shot prompt templates (`src/templates.py`'s
  `CtxPrompt`, `src/datasets.py`) for `StrategyQA`/`WikiMultiHopQA`/
  `WikiAsp`/`ASQA` -- none of which are HotpotQA or MuSiQue, and 2WikiMultiHop
  support is built for long-form generation eval (rouge/sacrebleu), not this
  project's Hit/EM/F1 convention.

Patching all of that to work here would mean rewriting most of it anyway.
Instead, the one piece that *is* FLARE's real contribution (`ApiReturn`) is
vendored verbatim, and `../client_runner.py` re-implements the orchestration
loop it documented in the paper (generate a look-ahead sentence -> mask
low-confidence tokens -> retrieve if the resulting query is non-empty ->
regenerate the sentence with retrieved context) against this project's own
per-client corpus and local Ollama chat-completions endpoint. Same
algorithm, honest about what's copied vs. reimplemented.

**Deviation:** `use_sentencizer` defaults to `'spacy'` here instead of
upstream's `'nltk'` (`nltk.tokenize.punkt.PunktSentenceTokenizer`) -- spacy
is already a dependency and `en_core_web_sm` already downloaded (Stage A's
NER), so this avoids adding `nltk` + a separate punkt corpus download for
the exact same job. The `'nltk'` branch is left as `raise NotImplementedError`
since nothing in this baseline selects it.
