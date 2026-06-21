from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from mask_span_extractor import ExplicitEntityExtractor, MaskSpanExtractor  # noqa: E402
from placeholder import selective_entity_masking  # noqa: E402


class ExplicitEntityExtractionTest(unittest.TestCase):
    def test_explicit_entity_extraction_person_possessive(self) -> None:
        question = "Why did John Middleton Murry's wife die?"
        result = ExplicitEntityExtractor(
            StaticEntityLLM([_entity(question, "John Middleton Murry", "Person")])
        ).extract(question)
        replacement = selective_entity_masking(
            question=question,
            mask_spans=MaskSpanExtractor(StaticEntityLLM([_entity(question, "John Middleton Murry", "Person")])).extract(question),
        )

        self.assertEqual([entity.text for entity in result.entities], ["John Middleton Murry"])
        self.assertNotIn("wife", [entity.text for entity in result.entities])
        self.assertNotIn("die", [entity.text for entity in result.entities])
        self.assertEqual(replacement.masked_question, "Why did PersonA's wife die?")
        self.assertEqual(replacement.mapping["PersonA"], "John Middleton Murry")

    def test_mask_all_single_token_entities(self) -> None:
        question = "Are Marufabad and Nasamkhrali both located in the same country?"
        replacement = selective_entity_masking(
            question=question,
            mask_spans=MaskSpanExtractor(
                StaticEntityLLM(
                    [
                        _entity(question, "Marufabad", "Location"),
                        _entity(question, "Nasamkhrali", "Location"),
                    ]
                )
            ).extract(question),
        )

        self.assertEqual(replacement.masked_question, "Are LocationA and LocationB both located in the same country?")
        self.assertEqual(replacement.mapping["LocationA"], "Marufabad")
        self.assertEqual(replacement.mapping["LocationB"], "Nasamkhrali")

    def test_do_not_extract_roles_slots_or_type_variables(self) -> None:
        question = "Which university did the CEO of the company that developed the AI game AlphaGo graduate from?"
        result = ExplicitEntityExtractor(
            StaticEntityLLM([_entity(question, "AlphaGo", "Game")])
        ).extract(question)
        replacement = selective_entity_masking(
            question=question,
            mask_spans=MaskSpanExtractor(StaticEntityLLM([_entity(question, "AlphaGo", "Game")])).extract(question),
        )

        self.assertEqual([entity.text for entity in result.entities], ["AlphaGo"])
        self.assertNotIn("CEO", [entity.text for entity in result.entities])
        self.assertNotIn("company", [entity.text for entity in result.entities])
        self.assertNotIn("university", [entity.text for entity in result.entities])
        self.assertEqual(
            replacement.masked_question,
            "Which university did the CEO of the company that developed the AI game GameA graduate from?",
        )

    def test_internal_and_inside_named_event_is_not_split(self) -> None:
        question = (
            "When was the region immediately north of the region where Israel is located and "
            "the location of the Battle of Qurah and Umm al Maradim created?"
        )
        result = ExplicitEntityExtractor(
            StaticEntityLLM(
                [
                    _entity(question, "Israel", "Country"),
                    _entity(question, "Battle of Qurah", "Event"),
                    _entity(question, "Umm al Maradim", "Location"),
                ]
            )
        ).extract(question)
        replacement = selective_entity_masking(
            question=question,
            mask_spans=MaskSpanExtractor(
                StaticEntityLLM(
                    [
                        _entity(question, "Israel", "Country"),
                        _entity(question, "Battle of Qurah", "Event"),
                        _entity(question, "Umm al Maradim", "Location"),
                    ]
                )
            ).extract(question),
        )

        self.assertEqual(
            [entity.text for entity in result.entities],
            ["Israel", "Battle of Qurah and Umm al Maradim"],
        )
        self.assertNotIn("Battle of Qurah", replacement.mapping.values())
        self.assertNotIn("Umm al Maradim", replacement.mapping.values())
        self.assertIn("Battle of Qurah and Umm al Maradim", replacement.mapping.values())
        self.assertEqual(
            replacement.masked_question,
            "When was the region immediately north of the region where CountryA is located and "
            "the location of the EventA created?",
        )

    def test_candidate_verification_handles_titles_apostrophe_and_non_ascii(self) -> None:
        question = "Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Br氓k?"
        llm = CandidateSelectingLLM(
            {
                "God'S Gift To Women": "Film",
                "Aldri Annet Enn Br氓k": "Film",
            }
        )
        result = ExplicitEntityExtractor(llm).extract(question)

        self.assertIn("Candidate spans", llm.prompt)
        self.assertIn("verified_entities", llm.prompt)
        self.assertEqual(
            [(entity.text, entity.semantic_type_hint) for entity in result.entities],
            [("God'S Gift To Women", "Film"), ("Aldri Annet Enn Br氓k", "Film")],
        )
        self.assertNotIn("God", [entity.text for entity in result.entities])
        self.assertNotIn("Br氓k", [entity.text for entity in result.entities])

    def test_colon_subtitle_title_is_kept_as_one_work_entity(self) -> None:
        question = "What music school did the singer of The Search for Everything: Wave One attend?"
        title = "The Search for Everything: Wave One"
        llm = CandidateSelectingLLM({title: "Work"})

        result = ExplicitEntityExtractor(llm).extract(question)

        self.assertIn("colon/subtitle", llm.prompt)
        self.assertEqual([(entity.text, entity.semantic_type_hint) for entity in result.entities], [(title, "Work")])
        self.assertNotIn("The Search for Everything", [entity.text for entity in result.entities])
        self.assertNotIn("Wave One", [entity.text for entity in result.entities])

    def test_title_after_type_head_can_begin_with_wh_word(self) -> None:
        question = "What nationality is the performer of song When The Stars Go Blue?"
        title = "When The Stars Go Blue"
        llm = CandidateSelectingLLM({title: "Song"})

        result = ExplicitEntityExtractor(llm).extract(question)

        self.assertIn(title, llm.prompt)
        self.assertIn("wh-looking word", llm.prompt)
        self.assertEqual([entity.text for entity in result.entities], [title])
        self.assertNotIn("The Stars Go Blue", [entity.text for entity in result.entities])

    def test_truncated_title_after_type_head_is_structurally_completed(self) -> None:
        question = "What nationality is the performer of song When The Stars Go Blue?"

        result = ExplicitEntityExtractor(
            StaticEntityLLM([_entity(question, "The Stars Go Blue", "Song")])
        ).extract(question)

        self.assertEqual([entity.text for entity in result.entities], ["When The Stars Go Blue"])
        self.assertTrue(any("typed-context title" in warning for warning in result.warnings))

    def test_free_span_possessive_boundary_is_repaired(self) -> None:
        question = "When did Lothair II's mother die?"
        result = ExplicitEntityExtractor(
            StaticEntityLLM(
                [
                    {
                        "text": "Lothair II's",
                        "start_char": question.index("Lothair"),
                        "end_char": question.index("Lothair") + len("Lothair II's"),
                        "semantic_type_hint": "Person",
                        "confidence": 0.9,
                    }
                ]
            )
        ).extract(question)

        self.assertEqual([entity.text for entity in result.entities], ["Lothair II"])

    def test_no_type_variable_masks_from_step2(self) -> None:
        question = "Which university did the CEO of the artificial intelligence company graduate from?"
        result = MaskSpanExtractor(
            StaticEntityLLM(
                [
                    {
                        "text": "artificial intelligence company",
                        "start_char": question.index("artificial intelligence company"),
                        "end_char": question.index("artificial intelligence company") + len("artificial intelligence company"),
                        "kind_hint": "type_variable",
                        "semantic_type_hint": "Company",
                    }
                ],
                field_name="mask_spans",
            )
        ).extract(question)
        replacement = selective_entity_masking(question=question, mask_spans=result)

        self.assertEqual(result.mask_spans, [])
        self.assertEqual(replacement.mask_mappings, [])


class StaticEntityLLM:
    def __init__(self, entities: list[dict[str, Any]], field_name: str = "entities") -> None:
        self.entities = entities
        self.field_name = field_name

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        del system_prompt, prompt
        return {self.field_name: self.entities}


class CandidateSelectingLLM:
    def __init__(self, selected_text_to_type: dict[str, str]) -> None:
        self.selected_text_to_type = selected_text_to_type
        self.prompt = ""

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        del system_prompt
        self.prompt = prompt
        marker = "Candidate spans:\n"
        candidates_json = prompt.split(marker, 1)[1].split("\n\nTask:", 1)[0]
        candidates = json.loads(candidates_json)
        verified = []
        for candidate in candidates:
            text = candidate["text"]
            verified.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "is_entity": text in self.selected_text_to_type,
                    "confidence": 0.98 if text in self.selected_text_to_type else 0.1,
                    "reason": "selected candidate" if text in self.selected_text_to_type else "not selected by test",
                }
            )
        return {"verified_entities": verified}


def _entity(question: str, text: str, semantic_type: str) -> dict[str, Any]:
    start = question.index(text)
    return {
        "text": text,
        "start_char": start,
        "end_char": start + len(text),
        "semantic_type_hint": semantic_type,
        "confidence": 0.95,
        "reason": "test entity",
    }


if __name__ == "__main__":
    unittest.main()
