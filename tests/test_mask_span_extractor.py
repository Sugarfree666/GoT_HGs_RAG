from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from mask_span_extractor import ExplicitEntityExtractor  # noqa: E402


class EmptyEntityLLM:
    def chat_json(self, _system_prompt: str, _prompt: str) -> dict[str, Any]:
        return {
            "explicit_entities": [],
            "normalized_question": "When was the region around Blue Valley created?",
        }


class ExplicitEntityExtractorTest(unittest.TestCase):
    def test_empty_llm_result_keeps_the_question_unmasked(self) -> None:
        result = ExplicitEntityExtractor(EmptyEntityLLM()).extract(
            "When was the region around Blue Valley created?"
        )

        self.assertEqual(result.entities, [])


if __name__ == "__main__":
    unittest.main()
