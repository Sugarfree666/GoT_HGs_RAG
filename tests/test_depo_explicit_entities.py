from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from entity_masking_preprocessor import preprocess_question  # noqa: E402
from prompts import EXPLICIT_ENTITY_EXTRACTION_SYSTEM, build_explicit_entity_extraction_prompt  # noqa: E402


class ExplicitEntityExtractionTest(unittest.TestCase):
    def test_preprocessing_uses_llm_entities_and_masks_repeated_surfaces(self) -> None:
        question = "Did Shrek 2 influence Shrek 2's sequel?"
        result = preprocess_question(
            question,
            DirectEntityLLM(_payload(["Shrek 2"], question)),
        )

        self.assertEqual(result.entities, ["Shrek 2"])
        self.assertEqual(result.masked_question, "Did ENTITYA influence ENTITYA's sequel?")
        self.assertEqual(result.mask_mapping, {"ENTITYA": "Shrek 2"})

    def test_independent_entities_receive_distinct_placeholders(self) -> None:
        question = "Are Marufabad and Nasamkhrali both located in the same country?"
        result = preprocess_question(
            question,
            DirectEntityLLM(_payload(["Marufabad", "Nasamkhrali"], question)),
        )

        self.assertEqual(
            result.masked_question,
            "Are ENTITYA and ENTITYB both located in the same country?",
        )

    def test_empty_entities_keep_the_question_unmasked(self) -> None:
        question = "When was the region around Blue Valley created?"
        result = preprocess_question(question, DirectEntityLLM(_payload([], question)))

        self.assertEqual(result.entities, [])
        self.assertEqual(result.masked_question, question)


class ExplicitEntityPromptTest(unittest.TestCase):
    def test_prompt_uses_a_small_json_input_and_fixed_output_contract(self) -> None:
        question = "Who directed Shrek 2?"

        self.assertEqual(
            json.loads(build_explicit_entity_extraction_prompt(question)),
            {"question": question},
        )
        self.assertIn('"explicit_entities"', EXPLICIT_ENTITY_EXTRACTION_SYSTEM)
        self.assertIn('"normalized_question"', EXPLICIT_ENTITY_EXTRACTION_SYSTEM)
        self.assertIn("exact contiguous substring", EXPLICIT_ENTITY_EXTRACTION_SYSTEM)
        self.assertIn("never return\noverlapping surfaces", EXPLICIT_ENTITY_EXTRACTION_SYSTEM)
        self.assertIn("bare dates, years, or numeric values", EXPLICIT_ENTITY_EXTRACTION_SYSTEM)


class DirectEntityLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def chat_json(self, _system_prompt: str, _prompt: str) -> dict[str, object]:
        return self.payload


def _payload(entities: list[str], normalized_question: str) -> dict[str, object]:
    return {
        "explicit_entities": [{"surface": entity} for entity in entities],
        "normalized_question": normalized_question,
    }


if __name__ == "__main__":
    unittest.main()
