"""Small OpenAI-compatible client shared by analysis, retrieval, and answering."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import request

import numpy as np

from ..config import LLMConfig


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = os.getenv(config.api_key_env, "").strip()
        self.base_url = os.getenv(config.base_url_env, "https://api.openai.com/v1").rstrip("/")
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.response_cache: dict[str, str] = {}

    def chat_json(
        self,
        stage: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int = 1400,
    ) -> dict[str, Any]:
        response = json.loads(
            self.chat_text(stage, system_prompt, user_payload, max_tokens=max_tokens)
        )
        if not isinstance(response, dict):
            raise ValueError(f"Expected a JSON object from {stage}.")
        return response

    def chat_text(
        self,
        stage: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int = 1400,
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        cache_key = self._cache_key("/chat/completions", payload)
        cached = cache_key in self.response_cache
        if cached:
            content = self.response_cache[cache_key]
        else:
            response = self._post_json("/chat/completions", payload)
            content = str(response["choices"][0]["message"]["content"])
            self.response_cache[cache_key] = content
        return content

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        missing = [text for text in texts if text not in self.embedding_cache]
        if missing:
            response = self._post_json(
                "/embeddings",
                {"model": self.config.embedding_model, "input": missing},
            )
            vectors = sorted(response["data"], key=lambda item: item["index"])
            for text, item in zip(missing, vectors, strict=True):
                self.embedding_cache[text] = np.asarray(item["embedding"], dtype=np.float32)
        return [self.embedding_cache[text] for text in texts]

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{endpoint}",
            data=request_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _cache_key(endpoint: str, payload: dict[str, Any]) -> str:
        return json.dumps(
            {"endpoint": endpoint, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
