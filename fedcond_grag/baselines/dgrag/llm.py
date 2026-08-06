"""LLM/SLM wrapper for DGRAG.

Extends the repo's existing LLM_Model pattern (baselines/linearrag/utils.py)
with:
- generate_batch(prompt, n, temperature): B sequential calls with different
  per-call seeds (plan decision: works over any OpenAI-compatible endpoint
  including ollama, more reliable than native n=B which many endpoints ignore).
- infer_json(prompt): parse JSON from response, retry once on parse failure.
- infer(prompt, temperature): single call, temperature-configurable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import httpx
from openai import OpenAI

_log = logging.getLogger("dgrag.llm")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


class DGRAGModel:
    """Unified LLM/SLM wrapper for both edge (SLM) and cloud (LLM) roles.

    In the §9 no-cloud adaptation both roles use the same endpoint and model
    — just instantiate two DGRAGModel objects pointing at the same server.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "sk-",
        timeout: float = 120.0,
    ):
        self.model_name = model_name
        http_client = httpx.Client(timeout=timeout, trust_env=False)
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-"),
            base_url=base_url,
            http_client=http_client,
        )

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        """Single-shot completion."""
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            _log.warning("LLM infer failed: %s", exc)
            return ""

    def infer_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        default: Any = None,
    ) -> Any:
        """Parse JSON from response; retry once on failure with a correction prompt."""
        raw = self.infer(prompt, temperature=temperature, max_tokens=max_tokens)
        result = _extract_json(raw)
        if result is not None:
            return result
        # One retry: ask the model to fix its output
        fix_prompt = (
            f"The following output could not be parsed as JSON. "
            f"Return ONLY valid JSON, no explanation:\n{raw}"
        )
        raw2 = self.infer(fix_prompt, temperature=0.0, max_tokens=max_tokens)
        result2 = _extract_json(raw2)
        if result2 is not None:
            return result2
        _log.warning("JSON parse failed after retry. Raw: %s", raw[:200])
        return default if default is not None else {}

    def generate_batch(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> list[str]:
        """Generate n candidates by making n sequential calls with different seeds.

        Uses temperature > 0 so candidates genuinely differ (spec §4.1 note:
        identical candidates degenerate the gate similarity signal).
        """
        answers: list[str] = []
        for i in range(n):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=i,       # different seed per call → different samples
                )
                answers.append(resp.choices[0].message.content or "")
            except Exception as exc:
                _log.warning("Batch call %d/%d failed: %s", i + 1, n, exc)
                answers.append("")
        return answers
