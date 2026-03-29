from __future__ import annotations

import unittest

from goth_hyper.utils import extract_json_payload, normalize_label


class UtilsTest(unittest.TestCase):
    def test_extract_json_payload_from_fenced_block(self) -> None:
        payload = extract_json_payload('```json\n{"a": 1, "b": [2]}\n```')
        self.assertEqual(payload["a"], 1)
        self.assertEqual(payload["b"], [2])

    def test_normalize_label(self) -> None:
        self.assertEqual(
            normalize_label('<hyperedge>"Urban farms build trust through transparency."'),
            "Urban farms build trust through transparency.",
        )


if __name__ == "__main__":
    unittest.main()
