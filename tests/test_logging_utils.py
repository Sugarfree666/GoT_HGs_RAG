from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from hyper_branch.logging_utils import TraceStore, configure_logging


class LoggingUtilsTest(unittest.TestCase):
    def test_default_console_logging_keeps_only_summary_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                logger = configure_logging(run_dir, "INFO")
                logger.info("Loading dataset from D:/dataset")
                logger.info("Executing atomic DAG with 3 node(s)")
                logger.warning("warning message")
                self._close_logger(logger)

            console_output = stderr.getvalue()
            self.assertIn("Executing atomic DAG with 3 node(s)", console_output)
            self.assertIn("warning message", console_output)
            self.assertNotIn("Loading dataset from D:/dataset", console_output)

            file_output = (run_dir / "run.log").read_text(encoding="utf-8")
            self.assertIn("Loading dataset from D:/dataset", file_output)
            self.assertIn("Executing atomic DAG with 3 node(s)", file_output)

    def test_verbose_console_logging_shows_detailed_info_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                logger = configure_logging(run_dir, "INFO", verbose_console=True)
                logger.info("Loading dataset from D:/dataset")
                self._close_logger(logger)

            self.assertIn("Loading dataset from D:/dataset", stderr.getvalue())

    def test_llm_trace_stores_sizes_without_prompt_or_response_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            trace_store = TraceStore(run_dir)
            trace_store.log_llm_call(
                "atomic_answer",
                {
                    "model": "test-model",
                    "messages": [
                        {"role": "system", "content": "secret prompt"},
                        {"role": "user", "content": "large evidence payload"},
                    ],
                    "max_tokens": 900,
                },
                {"content": "model answer", "cached": False},
            )

            record = json.loads((run_dir / "llm_calls.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(record["request"]["model"], "test-model")
        self.assertEqual(record["request"]["message_count"], 2)
        self.assertEqual(record["request"]["input_chars"], 35)
        self.assertEqual(record["response"]["output_chars"], 12)
        self.assertNotIn("messages", record["request"])
        self.assertNotIn("content", record["response"])

    def _close_logger(self, logger) -> None:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
