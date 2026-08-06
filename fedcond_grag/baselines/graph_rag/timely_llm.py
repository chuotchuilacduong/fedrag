"""TimelyGPT LLM shim for the HippoRAG baseline.

TimelyGPT (https://hello.timelygpt.co.kr) is not OpenAI-compatible: it uses a
key->bearer-token auth handshake and a custom /llm-completion payload, so it
cannot be reached through HippoRAG's CacheOpenAI. This shim implements the
same 3-tuple ``infer(messages=...) -> (response, metadata, cache_hit)``
contract as ``QwenLocalLLM`` (see local_llm.py) and is safe under the
ThreadPoolExecutor concurrency of upstream ``OpenIE.batch_openie``.

- API key comes from TIMELY_GPT_KEY (or TIMELY_API_KEY) in the environment /
  project .env.
- A fresh session_id per request keeps calls stateless (DYNAMIC_CHAT sessions
  are server-side conversations otherwise).
- Responses are cached in the upstream sqlite LLM_Cache, so killed runs
  resume without repeating paid calls.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Tuple

import requests

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_SRC = ROOT / "third_party" / "HippoRAG" / "src"
import sys  # noqa: E402

if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))

from hipporag.llm.transformers_llm import LLM_Cache  # noqa: E402

DEFAULT_BASE_URL = "https://hello.timelygpt.co.kr/api/v2/chat"


def _load_dotenv_key() -> str | None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        for name in ("TIMELY_GPT_KEY", "TIMELY_API_KEY"):
            if line.startswith(f"{name}=") and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip()
    return None


class TimelyLLM:
    """Minimal BaseLLM-compatible client for TimelyGPT (OpenIE + fact rerank)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        cache_dir: str | Path | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_concurrency: int = 8,
        max_retries: int = 5,
        timeout: int = 120,
    ) -> None:
        self.llm_name = model_name
        self.base_url = (base_url or os.getenv("TIMELY_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("TIMELY_GPT_KEY")
            or os.getenv("TIMELY_API_KEY")
            or _load_dotenv_key()
        )
        if not self.api_key:
            raise ValueError("TIMELY_GPT_KEY is not set (env or .env)")
        self.max_retries = max_retries
        self.timeout = timeout

        self._token: str | None = None
        self._token_expires_at = 0.0
        self._auth_lock = threading.Lock()
        self._sem = threading.Semaphore(max_concurrency)

        self.cache = None
        if cache_dir is not None:
            self.cache = LLM_Cache(str(cache_dir), f"timely_{model_name.replace('/', '_')}")

    # -- auth -----------------------------------------------------------------

    def _ensure_auth(self) -> str:
        with self._auth_lock:
            if self._token and time.time() < self._token_expires_at:
                return self._token
            resp = requests.get(
                f"{self.base_url}/sdk-auth/authenticate",
                headers={"Content-Type": "application/json", "X-Timely-API": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            token = data.get("data", {}).get("access_token")
            if not data.get("success") or not token:
                raise RuntimeError(f"TimelyGPT auth failed: {data}")
            self._token = token
            self._token_expires_at = time.time() + 55 * 60
            return token

    def _invalidate_token(self) -> None:
        with self._auth_lock:
            self._token = None
            self._token_expires_at = 0.0

    # -- inference ------------------------------------------------------------

    def _call(self, messages: list) -> str:
        payload = {
            "session_id": f"fedrag_bench_{uuid.uuid4()}",
            "messages": messages,
            "chat_model_node": {"model": self.llm_name},
            "chat_type": "DYNAMIC_CHAT",
            "stream": False,
            "locale": "en",
        }
        last_err: str = "no attempts made"
        for attempt in range(self.max_retries):
            token = self._ensure_auth()
            try:
                with self._sem:
                    resp = requests.post(
                        f"{self.base_url}/llm-completion",
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {token}",
                        },
                        timeout=self.timeout,
                    )
            except requests.RequestException as exc:
                last_err = f"request error: {exc}"
                time.sleep(min(2 ** attempt, 30))
                continue

            if resp.status_code == 401:
                self._invalidate_token()
                last_err = "401 unauthorized"
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"{resp.status_code}: {resp.text[:200]}"
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"TimelyGPT error {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            msg_type = data.get("type")
            if msg_type == "final_response":
                return data.get("message", "") or ""
            if msg_type == "error":
                last_err = f"logic error: {data.get('error')}"
                time.sleep(min(2 ** attempt, 30))
                continue
            # Unknown but successful payload — return best-effort text.
            return str(data.get("message", data))
        raise RuntimeError(f"TimelyGPT failed after {self.max_retries} retries ({last_err})")

    def _cache_params(self, messages: list) -> dict:
        return {"model": self.llm_name, "temperature": 0.0, "messages": messages}

    def infer(self, messages: list, **kwargs: Any) -> Tuple[str, dict, bool]:
        """3-tuple contract used by OpenIE.ner/triple_extraction and DSPyFilter."""
        params = self._cache_params(messages)
        if self.cache is not None:
            hit = self.cache.read(params)
            if hit is not None:
                return hit[0], hit[1], True
        text = self._call(messages)
        metadata = {"finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0}
        if self.cache is not None:
            self.cache.write(params, text, metadata)
        return text, metadata, False
