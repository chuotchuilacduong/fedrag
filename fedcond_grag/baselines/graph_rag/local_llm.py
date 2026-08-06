"""Local Qwen LLM shim for the HippoRAG baseline (no API key, no vllm).

Upstream HippoRAG's own local paths are unusable here: ``Transformers-offline``
imports vllm/outlines, and ``TransformersLLM.infer`` returns a 2-tuple where
the online OpenIE / rerank code unpacks 3 values. This module provides:

- ``QwenLocalLLM``: 4-bit (bnb NF4) HF causal LM exposing the 3-tuple
  ``infer(messages=...) -> (response, metadata, cache_hit)`` contract that
  ``DSPyFilter`` (fact rerank) and ``OpenIE`` expect, plus a padded
  ``batch_infer`` for throughput. Responses are cached in the same sqlite
  ``LLM_Cache`` used upstream, so killed runs resume for free.
- ``BatchedLocalOpenIE``: an ``OpenIE`` subclass whose ``batch_openie`` uses
  ``batch_infer`` instead of one threaded request per chunk (threads would
  serialize on a single local model anyway). Prompts and parsers are reused
  from upstream unchanged.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_SRC = ROOT / "third_party" / "HippoRAG" / "src"
import sys  # noqa: E402

if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))

from hipporag.information_extraction.openie_openai import (  # noqa: E402
    OpenIE,
    _extract_ner_from_response,
)
from hipporag.llm.transformers_llm import LLM_Cache  # noqa: E402
from hipporag.utils.llm_utils import fix_broken_generated_json, filter_invalid_triples  # noqa: E402
from hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput  # noqa: E402

_TRIPLE_PATTERN = re.compile(r'\{[^{}]*"triples"\s*:\s*\[.*?\][^{}]*\}', re.DOTALL)


def _extract_triples_from_response(real_response: str) -> list:
    match = _TRIPLE_PATTERN.search(real_response)
    if match is None:
        return []
    return eval(match.group())["triples"]


class QwenLocalLLM:
    """Minimal BaseLLM-compatible local model for OpenIE + fact rerank."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        cache_dir: str | Path | None = None,
        batch_size: int = 4,
        load_in_4bit: bool = True,
        max_input_len: int = 4096,
        max_gpu_mem: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.llm_name = model_name
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self._lock = threading.Lock()

        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        # bitsandbytes 4-bit cannot CPU-offload (meta-tensor state_dict), so
        # the model must fit on the GPU. On this shared GPU free VRAM
        # fluctuates with other users' jobs: interpret max_gpu_mem as the
        # required free headroom and wait for it before loading, then retry
        # the load itself on transient OOM.
        if max_gpu_mem is not None and torch.cuda.is_available():
            need_gib = float(str(max_gpu_mem).lower().replace("gib", "").replace("gb", ""))
            self._wait_for_free_vram(need_gib)
        last_exc: Exception | None = None
        for attempt in range(20):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quant,
                    device_map={"": 0} if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                break
            except torch.OutOfMemoryError as exc:
                last_exc = exc
                torch.cuda.empty_cache()
                print(f"[QwenLocalLLM] OOM while loading (attempt {attempt + 1}/20), "
                      f"waiting 90s for shared GPU to free up...", flush=True)
                time.sleep(90)
        else:
            raise RuntimeError(f"Could not load {model_name}: GPU stayed full") from last_exc
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.cache = None
        if cache_dir is not None:
            self.cache = LLM_Cache(str(cache_dir), model_name.replace("/", "_"))

    @staticmethod
    def _wait_for_free_vram(need_gib: float, poll_s: int = 60, max_wait_s: int = 3600) -> None:
        waited = 0
        while waited <= max_wait_s:
            free_b, _total = torch.cuda.mem_get_info()
            if free_b / 1024**3 >= need_gib:
                return
            print(f"[QwenLocalLLM] {free_b / 1024**3:.1f}GiB free < {need_gib}GiB needed; "
                  f"waiting {poll_s}s for shared GPU...", flush=True)
            time.sleep(poll_s)
            waited += poll_s
        print("[QwenLocalLLM] proceeding despite low free VRAM (max wait reached)", flush=True)

    # -- generation ---------------------------------------------------------

    def _generate(self, messages_batch: List[list], max_new_tokens: int) -> List[Tuple[str, dict]]:
        for attempt in range(6):
            try:
                return self._generate_once(messages_batch, max_new_tokens)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                wait = min(60 * (attempt + 1), 300)
                print(f"[QwenLocalLLM] OOM during generate (attempt {attempt + 1}/6); "
                      f"retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("generate kept OOMing — shared GPU has no headroom")

    def _generate_once(self, messages_batch: List[list], max_new_tokens: int) -> List[Tuple[str, dict]]:
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_len,
        ).to(self.model.device)
        with self._lock, torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        results = []
        for row, text in zip(new_tokens, texts):
            n_new = int((row != self.tokenizer.pad_token_id).sum())
            metadata = {
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "completion_tokens": n_new,
                "finish_reason": "length" if n_new >= max_new_tokens else "stop",
            }
            results.append((text, metadata))
        return results

    def _cache_params(self, messages: list) -> dict:
        return {"model": self.llm_name, "temperature": 0.0, "messages": messages}

    # -- upstream contracts --------------------------------------------------

    def infer(self, messages: list, **kwargs: Any) -> Tuple[str, dict, bool]:
        """3-tuple contract used by OpenIE.ner/triple_extraction and DSPyFilter."""
        params = self._cache_params(messages)
        if self.cache is not None:
            hit = self.cache.read(params)
            if hit is not None:
                return hit[0], hit[1], True
        max_new = int(kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 512)
        text, metadata = self._generate([messages], max_new)[0]
        if self.cache is not None:
            self.cache.write(params, text, metadata)
        return text, metadata, False

    def batch_infer(
        self, messages_list: List[list], max_new_tokens: int = 512, desc: str = "local LLM"
    ) -> List[Tuple[str, dict]]:
        results: List[Tuple[str, dict] | None] = [None] * len(messages_list)
        pending: List[int] = []
        for i, messages in enumerate(messages_list):
            if self.cache is not None:
                hit = self.cache.read(self._cache_params(messages))
                if hit is not None:
                    results[i] = (hit[0], hit[1])
                    continue
            pending.append(i)

        for start in tqdm(range(0, len(pending), self.batch_size), desc=desc):
            idxs = pending[start : start + self.batch_size]
            outs = self._generate([messages_list[i] for i in idxs], max_new_tokens)
            for i, (text, metadata) in zip(idxs, outs):
                results[i] = (text, metadata)
                if self.cache is not None:
                    self.cache.write(self._cache_params(messages_list[i]), text, metadata)
        return results  # type: ignore[return-value]


class BatchedLocalOpenIE(OpenIE):
    """Upstream OpenIE with batched local inference instead of threaded API calls."""

    def __init__(self, llm_model: QwenLocalLLM) -> None:
        super().__init__(llm_model=llm_model)

    def batch_openie(self, chunks):
        chunk_passages = {k: c["content"] for k, c in chunks.items()}
        keys = list(chunk_passages)

        ner_messages = [
            self.prompt_template_manager.render(name="ner", passage=chunk_passages[k])
            for k in keys
        ]
        ner_out = self.llm_model.batch_infer(ner_messages, max_new_tokens=512, desc="NER (local)")
        ner_results = {}
        for k, (response, metadata) in zip(keys, ner_out):
            real = fix_broken_generated_json(response) if metadata.get("finish_reason") == "length" else response
            try:
                entities = list(dict.fromkeys(_extract_ner_from_response(real)))
            except Exception:
                entities = []
            ner_results[k] = NerRawOutput(k, response, entities, dict(metadata))

        triple_messages = [
            self.prompt_template_manager.render(
                name="triple_extraction",
                passage=chunk_passages[k],
                named_entity_json=json.dumps({"named_entities": ner_results[k].unique_entities}),
            )
            for k in keys
        ]
        triple_out = self.llm_model.batch_infer(
            triple_messages, max_new_tokens=1024, desc="Triples (local)"
        )
        triple_results = {}
        for k, (response, metadata) in zip(keys, triple_out):
            real = fix_broken_generated_json(response) if metadata.get("finish_reason") == "length" else response
            try:
                triples = filter_invalid_triples(triples=_extract_triples_from_response(real))
            except Exception:
                triples = []
            triple_results[k] = TripleRawOutput(
                chunk_id=k, response=response, metadata=dict(metadata), triples=triples
            )

        return ner_results, triple_results
