from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from mask_span_extractor import MaskSpanExtractor, _heuristic_mask_spans  # noqa: E402
from placeholder import selective_entity_masking  # noqa: E402


class EmptyMaskLLM:
    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        del system_prompt, prompt
        return {"mask_spans": []}


class MaskSpanExtractorTest(unittest.TestCase):
    def test_multi_token_capitalized_region_name_is_detected_generically(self) -> None:
        question = "When was the region around Blue Valley created?"

        spans = _heuristic_mask_spans(question)

        self.assertEqual([span.text for span in spans], ["Blue Valley"])
        self.assertEqual(spans[0].semantic_type_hint, "Region")

    def test_llm_empty_result_is_augmented_with_deterministic_multi_token_entities(self) -> None:
        question = (
            "When was the region immediately north of the region that prevailed with the disgrace of "
            "Near East and the terrain feature on which shamal is located created?"
        )

        result = MaskSpanExtractor(EmptyMaskLLM()).extract(question)

        self.assertEqual([span.text for span in result.mask_spans], ["Near East"])
        replacement = selective_entity_masking(question=question, mask_spans=result)
        self.assertIn("RegionA", replacement.masked_question)
        self.assertNotIn("Near East", replacement.masked_question)


if __name__ == "__main__":
    unittest.main()
