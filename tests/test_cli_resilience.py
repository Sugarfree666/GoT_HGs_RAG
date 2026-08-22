from __future__ import annotations

import unittest
from unittest.mock import patch

from hyper_branch.config import LLMConfig
from hyper_branch.llm.client import OpenAICompatibleClient


class FakeHTTPResponse:
    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return b'{"ok": true}'


class LLMClientTest(unittest.TestCase):
    def test_post_json_uses_one_request(self) -> None:
        client = OpenAICompatibleClient(LLMConfig())
        with patch("hyper_branch.llm.client.request.urlopen", return_value=FakeHTTPResponse()) as urlopen:
            self.assertEqual(client._post_json("/embeddings", {"input": ["x"]}), {"ok": True})
        self.assertEqual(urlopen.call_count, 1)
