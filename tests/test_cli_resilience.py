from __future__ import annotations

import unittest
from unittest.mock import patch
from urllib import error

from hyper_branch.client import OpenAIClient


class FakeHTTPResponse:
    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return b'{"ok": true}'


class OpenAIClientTest(unittest.TestCase):
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

    def test_connection_reset_is_retried(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="test-model",
            embedding_model="test-embedding-model",
            timeout_seconds=120,
            temperature=0.2,
        )
        reset = error.URLError(ConnectionResetError(10054, "connection reset"))
        with (
            patch(
                "hyper_branch.client.request.urlopen",
                side_effect=[reset, FakeHTTPResponse()],
            ) as urlopen,
            patch("hyper_branch.client.random.uniform", return_value=0.75),
            patch("hyper_branch.client.time.sleep") as sleep,
        ):
            self.assertEqual(client._post("/embeddings", {"input": ["x"]}), {"ok": True})

        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.75)

    def test_retryable_http_errors_honor_retry_after(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="test-model",
            embedding_model="test-embedding-model",
            timeout_seconds=120,
            temperature=0.2,
        )
        rate_limit = error.HTTPError(
            "https://example.test/v1/embeddings",
            429,
            "Too Many Requests",
            {"Retry-After": "3"},
            None,
        )
        unavailable = error.HTTPError(
            "https://example.test/v1/embeddings",
            503,
            "Service Unavailable",
            {},
            None,
        )
        with (
            patch(
                "hyper_branch.client.request.urlopen",
                side_effect=[rate_limit, unavailable, FakeHTTPResponse()],
            ) as urlopen,
            patch("hyper_branch.client.random.uniform", return_value=1.5),
            patch("hyper_branch.client.time.sleep") as sleep,
        ):
            self.assertEqual(client._post("/embeddings", {"input": ["x"]}), {"ok": True})

        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [3.0, 1.5])

    def test_non_retryable_http_error_fails_immediately(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="test-model",
            embedding_model="test-embedding-model",
            timeout_seconds=120,
            temperature=0.2,
        )
        unauthorized = error.HTTPError(
            "https://example.test/v1/embeddings",
            401,
            "Unauthorized",
            {},
            None,
        )
        with (
            patch("hyper_branch.client.request.urlopen", side_effect=unauthorized) as urlopen,
            patch("hyper_branch.client.time.sleep") as sleep,
            self.assertRaises(error.HTTPError),
        ):
            client._post("/embeddings", {"input": ["x"]})

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_last_transport_error_is_reraised_after_attempts_are_exhausted(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="test-model",
            embedding_model="test-embedding-model",
            timeout_seconds=120,
            temperature=0.2,
            max_attempts=3,
        )
        reset = error.URLError(ConnectionResetError(10054, "connection reset"))
        with (
            patch("hyper_branch.client.request.urlopen", side_effect=reset) as urlopen,
            patch("hyper_branch.client.random.uniform", side_effect=[0.75, 1.5]),
            patch("hyper_branch.client.time.sleep") as sleep,
            self.assertRaises(error.URLError) as raised,
        ):
            client._post("/embeddings", {"input": ["x"]})

        self.assertIs(raised.exception, reset)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

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

    def test_chat_json_requests_json_mode(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="test-model",
            embedding_model="test-embedding-model",
            timeout_seconds=120,
            temperature=0.2,
        )
        response = {"choices": [{"message": {"content": '{"answer": "ok"}'}}]}
        with patch.object(client, "_post", return_value=response) as post:
            self.assertEqual(client.chat_json("system", "user", max_tokens=12), {"answer": "ok"})

        post.assert_called_once_with(
            "/chat/completions",
            {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "max_tokens": 12,
            },
        )
