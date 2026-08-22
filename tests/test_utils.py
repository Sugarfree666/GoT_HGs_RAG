from __future__ import annotations

import unittest

from hyper_branch.utils import normalize_label


class UtilsTest(unittest.TestCase):
    def test_normalize_label(self) -> None:
        self.assertEqual(
            normalize_label('<hyperedge>"Urban farms build trust through transparency."'),
            "Urban farms build trust through transparency.",
        )


if __name__ == "__main__":
    unittest.main()
