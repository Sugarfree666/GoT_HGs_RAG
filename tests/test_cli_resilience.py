from __future__ import annotations

import unittest
from unittest.mock import patch

from hyper_branch.client import OpenAIClient


class FakeHTTPResponse:
    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return b'{"ok": true}'


class LLMClientTest(unittest.TestCase):
    def test_post_json_uses_one_request(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="test-model",
            embedding_model="test-embedding-model",
            timeout_seconds=120,
            temperature=0.2,
        )
        with patch("hyper_branch.client.request.urlopen", return_value=FakeHTTPResponse()) as urlopen:
            self.assertEqual(client._post("/embeddings", {"input": ["x"]}), {"ok": True})
        self.assertEqual(urlopen.call_count, 1)

    def test_explicit_credentials_do_not_read_or_change_the_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "environment-key", "OPENAI_BASE_URL": "https://environment.example/v1"},
            clear=True,
        ):
            client = OpenAIClient(
                api_key="argument-key",
                model="test-model",
                embedding_model="test-embedding-model",
                timeout_seconds=120,
                temperature=0.2,
                base_url="https://argument.example/v1",
            )

        self.assertEqual(client.api_key, "argument-key")
        self.assertEqual(client.base_url, "https://argument.example/v1")
