from __future__ import annotations

import unittest

from hyper_branch.database import _display_text


class DatabaseTextTest(unittest.TestCase):
    def test_normalize_label(self) -> None:
        self.assertEqual(
            _display_text('<hyperedge>"Urban farms build trust through transparency."'),
            "Urban farms build trust through transparency.",
        )


if __name__ == "__main__":
    unittest.main()
