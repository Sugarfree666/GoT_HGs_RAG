
from __future__ import annotations

import json
import logging
import random
import ssl
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib import error, request

import numpy as np


logger = logging.getLogger(__name__)


def _is_retryable_http_status(status: int) -> bool:
    return status in {408, 409, 429} or 500 <= status < 600


def _is_retryable_transport_error(exc: OSError) -> bool:
    reason = exc.reason if isinstance(exc, error.URLError) else exc
    if isinstance(reason, (ssl.CertificateError, ssl.SSLCertVerificationError)):
        return False
    return isinstance(reason, OSError)


def _retry_after_seconds(exc: error.HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


class OpenAIClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        embedding_model: str,
        timeout_seconds: int,
        temperature: float,
        base_url: str | None = None,
        max_attempts: int = 5,
        retry_base_delay_seconds: float = 1.0,
        retry_max_delay_seconds: float = 30.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_base_delay_seconds < 0 or retry_max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")

        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.max_attempts = max_attempts
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        #创建嵌入缓存

    #调用 embedding 接口，把文本转换成向量；
    def embed_text(self, text: str) -> np.ndarray:
        response = self._post(
            "/embeddings",
            {"model": self.embedding_model, "input": text},
        )
        #转成folat32浮点数向量
        return np.asarray(response["data"][0]["embedding"], dtype=np.float32)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send one JSON-mode chat request."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        response = self._post("/chat/completions", payload)
        return json.loads(response["choices"][0]["message"]["content"])

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        for attempt in range(1, self.max_attempts + 1):
            req = request.Request(
                f"{self.base_url}{endpoint}",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                if not _is_retryable_http_status(exc.code) or attempt == self.max_attempts:
                    raise
                retry_after = _retry_after_seconds(exc)
                failure = exc
            except OSError as exc:
                if not _is_retryable_transport_error(exc) or attempt == self.max_attempts:
                    raise
                retry_after = None
                failure = exc

            backoff = min(
                self.retry_max_delay_seconds,
                self.retry_base_delay_seconds * 2 ** (attempt - 1),
            )
            delay = (
                min(retry_after, self.retry_max_delay_seconds)
                if retry_after is not None
                else random.uniform(backoff / 2, backoff)
            )
            logger.warning(
                "Transient request failure for %s (attempt %d/%d): %s; retrying in %.2fs",
                endpoint,
                attempt,
                self.max_attempts,
                failure,
                delay,
            )
            time.sleep(delay)

        raise RuntimeError("unreachable")
