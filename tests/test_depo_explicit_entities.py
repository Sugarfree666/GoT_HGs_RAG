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


class ExplicitEntityExtractionTest(unittest.TestCase):
    def test_preprocessing_uses_llm_entities_and_masks_repeated_surfaces(self) -> None:
        question = "Did Shrek 2 influence Shrek 2's sequel?"
        result = preprocess_question(
            question,
            DirectEntityLLM(_payload(["Shrek 2"])),
        )

        self.assertEqual(result.entities, ["Shrek 2"])
        self.assertEqual(result.masked_question, "Did ENTITYA influence ENTITYA's sequel?")
        self.assertEqual(result.mask_mapping, {"ENTITYA": "Shrek 2"})

    def test_independent_entities_receive_distinct_placeholders(self) -> None:
        question = "Are Marufabad and Nasamkhrali both located in the same country?"
        result = preprocess_question(
            question,
            DirectEntityLLM(_payload(["Marufabad", "Nasamkhrali"])),
        )

        self.assertEqual(
            result.masked_question,
            "Are ENTITYA and ENTITYB both located in the same country?",
        )

    def test_empty_entities_keep_the_question_unmasked(self) -> None:
        question = "When was the region around Blue Valley created?"
        result = preprocess_question(question, DirectEntityLLM(_payload([])))

        self.assertEqual(result.entities, [])
        self.assertEqual(result.masked_question, question)


class ExplicitEntityPromptTest(unittest.TestCase):
    def test_prompt_uses_a_small_json_input_and_fixed_output_contract(self) -> None:
        question = "Who directed Shrek 2?"
        llm = DirectEntityLLM(_payload([]))
        preprocess_question(question, llm)
        prompt = (PROJECT_ROOT / "prompts" / "topic_entity_recognition.md").read_text(
            encoding="utf-8"
        )

        self.assertEqual(
            json.loads(llm.user_prompt),
            {"question": question},
        )
        self.assertEqual(llm.system_prompt, prompt.strip())
        self.assertIn('"entities"', prompt)
        self.assertIn("topic anchors", prompt)
        self.assertIn("Copy every entity exactly from the question", prompt)


class DirectEntityLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.system_prompt = ""
        self.user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.payload


def _payload(entities: list[str]) -> dict[str, object]:
    return {"entities": entities}


if __name__ == "__main__":
    unittest.main()
