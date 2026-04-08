from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from hyper_branch.logging_utils import configure_logging


class LoggingUtilsTest(unittest.TestCase):
    def test_default_console_logging_keeps_only_summary_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                logger = configure_logging(run_dir, "INFO")
                logger.info("Loading dataset from D:/dataset")
                logger.info("Iterative reasoning step 1/3")
                logger.warning("warning message")
                self._close_logger(logger)

            console_output = stderr.getvalue()
            self.assertIn("Iterative reasoning step 1/3", console_output)
            self.assertIn("warning message", console_output)
            self.assertNotIn("Loading dataset from D:/dataset", console_output)

            file_output = (run_dir / "run.log").read_text(encoding="utf-8")
            self.assertIn("Loading dataset from D:/dataset", file_output)
            self.assertIn("Iterative reasoning step 1/3", file_output)

    def test_verbose_console_logging_shows_detailed_info_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                logger = configure_logging(run_dir, "INFO", verbose_console=True)
                logger.info("Loading dataset from D:/dataset")
                self._close_logger(logger)

            self.assertIn("Loading dataset from D:/dataset", stderr.getvalue())

    def _close_logger(self, logger) -> None:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
