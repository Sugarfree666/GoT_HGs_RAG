from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from mask_span_extractor import MaskSpanExtractor  # noqa: E402
from entity_masking_preprocessor import EntityMaskingPreprocessor  # noqa: E402


class EmptyEntityLLM:
    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        del system_prompt, prompt
        return {"explicit_entities": []}


class MaskSpanExtractorTest(unittest.TestCase):
    def test_no_llm_returns_no_guessed_capitalized_entity(self) -> None:
        question = "When was the region around Blue Valley created?"

        result = MaskSpanExtractor().extract(question)

        self.assertEqual(result.mask_spans, [])
        self.assertTrue(any("no LLM client" in warning for warning in result.warnings))

    def test_llm_empty_result_does_not_fall_back_to_deterministic_entities(self) -> None:
        question = (
            "When was the region immediately north of the region that prevailed with the disgrace of "
            "Near East and the terrain feature on which shamal is located created?"
        )

        result = MaskSpanExtractor(EmptyEntityLLM()).extract(question)

        self.assertEqual(result.mask_spans, [])
        preprocess_result = EntityMaskingPreprocessor(EmptyEntityLLM()).preprocess(question)
        self.assertEqual(preprocess_result.mask_mappings, [])
        self.assertEqual(preprocess_result.masked_question, question)


if __name__ == "__main__":
    unittest.main()
