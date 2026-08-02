from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from entity_masking_preprocessor import EntityMaskingPreprocessor  # noqa: E402
from mask_span_extractor import ExplicitEntityExtractor  # noqa: E402
from prompts import EXPLICIT_ENTITY_EXTRACTION_SYSTEM, build_explicit_entity_extraction_prompt  # noqa: E402


class ExplicitEntityExtractionTest(unittest.TestCase):
    def test_llm_directly_identifies_entities_without_candidates(self) -> None:
        question = "Which film featured Shrek 2?"
        llm = DirectEntityLLM(_payload([_entity("Shrek 2", "Work")]))

        result = ExplicitEntityExtractor(llm).extract(question)

        self.assertEqual([entity.text for entity in result.entities], ["Shrek 2"])
        self.assertEqual(result.entities[0].start_char, question.index("Shrek 2"))
        self.assertNotIn("Candidate spans", llm.user_prompt)
        self.assertNotIn("candidate_id", llm.user_prompt)
        self.assertNotIn("verified_entities", llm.user_prompt)

    def test_title_surfaces_keep_numbers_and_terminal_punctuation(self) -> None:
        cases = [
            ("What country produced Shrek 2?", "Shrek 2"),
            ("Who performed Back In The U.S.A.?", "Back In The U.S.A."),
            ("Who wrote Love, Honor And Oh-Baby!?", "Love, Honor And Oh-Baby!"),
        ]

        for question, title in cases:
            with self.subTest(title=title):
                result = ExplicitEntityExtractor(
                    DirectEntityLLM(_payload([_entity(title, "Work")]))
                ).extract(question)

                self.assertEqual([entity.text for entity in result.entities], [title])
                self.assertEqual(
                    (result.entities[0].start_char, result.entities[0].end_char),
                    (question.index(title), question.index(title) + len(title)),
                )

    def test_independent_coordinated_entities_are_masked_separately(self) -> None:
        question = "Are Marufabad and Nasamkhrali both located in the same country?"
        result = EntityMaskingPreprocessor(
            DirectEntityLLM(
                _payload(
                    [
                        _entity("Marufabad", "Location"),
                        _entity("Nasamkhrali", "Location"),
                    ]
                )
            )
        ).preprocess(question)

        self.assertEqual(
            result.masked_question,
            "Are ENTITYA and ENTITYB both located in the same country?",
        )
        self.assertEqual(
            [(mapping.placeholder, mapping.original_text) for mapping in result.mask_mappings],
            [("ENTITYA", "Marufabad"), ("ENTITYB", "Nasamkhrali")],
        )

    def test_llm_entity_order_controls_placeholder_order(self) -> None:
        question = "Was Beta released before Alpha?"
        result = EntityMaskingPreprocessor(
            DirectEntityLLM(
                _payload([_entity("Alpha", "Work"), _entity("Beta", "Work")])
            )
        ).preprocess(question)

        self.assertEqual(result.masked_question, "Was ENTITYB released before ENTITYA?")
        self.assertEqual(
            [mapping.original_text for mapping in result.mask_mappings],
            ["Alpha", "Beta"],
        )

    def test_llm_boundaries_are_not_merged_split_or_sorted(self) -> None:
        question = "What nationality is Beatrice I, Countess Of Burgundy's husband?"
        result = EntityMaskingPreprocessor(
            DirectEntityLLM(
                _payload(
                    [
                        _entity("Countess Of Burgundy", "Other"),
                        _entity("Beatrice I", "Person"),
                    ]
                )
            )
        ).preprocess(question)

        self.assertEqual(
            [entity.text for entity in result.explicit_entities.entities],
            ["Countess Of Burgundy", "Beatrice I"],
        )
        self.assertEqual(
            result.masked_question,
            "What nationality is ENTITYB, ENTITYA's husband?",
        )

    def test_duplicate_llm_surface_is_invalid_instead_of_deduplicated(self) -> None:
        result = ExplicitEntityExtractor(
            DirectEntityLLM(
                _payload([_entity("Shrek 2", "Work"), _entity("Shrek 2", "Work")])
            )
        ).extract("Who produced Shrek 2?")

        self.assertEqual(result.entities, [])
        self.assertTrue(any("duplicate explicit entity surface" in warning for warning in result.warnings))

    def test_case_insensitive_duplicate_surface_is_invalid(self) -> None:
        result = ExplicitEntityExtractor(
            DirectEntityLLM(
                _payload([_entity("Shrek 2", "Work"), _entity("shrek 2", "Work")])
            )
        ).extract("Who produced Shrek 2?")

        self.assertEqual(result.entities, [])
        self.assertTrue(any("duplicate explicit entity surface='Shrek 2'" in warning for warning in result.warnings))

    def test_case_insensitive_surface_match_uses_original_question_casing(self) -> None:
        question = (
            "An Indy car race was held in the capital of the state where the performer of "
            "Mingus Plays Piano was born. Who won the race?"
        )
        result = EntityMaskingPreprocessor(
            DirectEntityLLM(
                _payload(
                    [
                        _entity("Indy Car Race", "Event"),
                        _entity("Mingus Plays Piano", "Work"),
                    ]
                )
            )
        ).preprocess(question)

        self.assertEqual(
            [entity.text for entity in result.explicit_entities.entities],
            ["Indy car race", "Mingus Plays Piano"],
        )
        self.assertEqual(
            result.masked_question,
            "An ENTITYA was held in the capital of the state where the performer of ENTITYB was born. Who won the race?",
        )
        self.assertEqual(result.warnings, [])

    def test_missing_surface_with_spacing_or_punctuation_difference_is_invalid(self) -> None:
        result = ExplicitEntityExtractor(
            DirectEntityLLM(_payload([_entity("Shrek-2", "Work")]))
        ).extract("Who produced Shrek 2?")

        self.assertEqual(result.entities, [])
        self.assertTrue(
            any("was not found in the original question" in warning for warning in result.warnings)
        )

    def test_overlapping_llm_surfaces_are_invalid_instead_of_resolved(self) -> None:
        question = "What nationality is Beatrice I, Countess Of Burgundy's husband?"
        result = ExplicitEntityExtractor(
            DirectEntityLLM(
                _payload(
                    [
                        _entity("Beatrice I", "Person"),
                        _entity("Beatrice I, Countess Of Burgundy", "Person"),
                    ]
                )
            )
        ).extract(question)

        self.assertEqual(result.entities, [])
        self.assertTrue(any("overlapping explicit entity span" in warning for warning in result.warnings))

    def test_repeated_surface_reuses_one_placeholder_and_mapping(self) -> None:
        question = "Did Shrek 2 influence Shrek 2's sequel?"
        result = EntityMaskingPreprocessor(
            DirectEntityLLM(_payload([_entity("Shrek 2", "Work")]))
        ).preprocess(question)

        self.assertEqual(result.masked_question, "Did ENTITYA influence ENTITYA's sequel?")
        self.assertEqual(len(result.mask_mappings), 1)
        self.assertEqual(result.mask_mappings[0].placeholder, "ENTITYA")
        self.assertEqual(result.mask_mappings[0].original_text, "Shrek 2")

    def test_masked_question_and_mappings_use_the_same_entities(self) -> None:
        question = "Did Alpha meet Beta?"
        result = EntityMaskingPreprocessor(
            DirectEntityLLM(
                _payload([_entity("Beta", "Person"), _entity("Alpha", "Person")])
            )
        ).preprocess(question)

        self.assertEqual(
            [entity.text for entity in result.explicit_entities.entities],
            [mapping.original_text for mapping in result.mask_mappings],
        )
        for mapping in result.mask_mappings:
            self.assertIn(mapping.placeholder, result.masked_question)
            self.assertIn(mapping.original_text, question)

    def test_normalized_question_payload_is_preserved(self) -> None:
        question = "Who is the child of the director of film An Event?"
        normalized = "Who is the child of the person who directed the film An Event?"
        result = ExplicitEntityExtractor(
            DirectEntityLLM(
                _payload(
                    [_entity("An Event", "Work")],
                    normalized_question=normalized,
                    normalization_changed=True,
                    normalization_note="Expanded the nested relation.",
                )
            )
        ).extract(question)

        self.assertEqual(result.normalized_question, normalized)
        self.assertTrue(result.normalization_changed)
        self.assertEqual(result.normalization_note, "Expanded the nested relation.")

    def test_implicit_attribute_ownership_normalization_payload_is_preserved(self) -> None:
        question = "What nationality is Lamprocles's father?"
        normalized = "What is the nationality of Lamprocles's father?"
        result = ExplicitEntityExtractor(
            DirectEntityLLM(
                _payload(
                    [_entity("Lamprocles", "Person")],
                    normalized_question=normalized,
                    normalization_changed=True,
                    normalization_note="Made the attribute ownership explicit.",
                )
            )
        ).extract(question)

        self.assertEqual(result.normalized_question, normalized)
        self.assertTrue(result.normalization_changed)
        self.assertIn(result.entities[0].text, result.normalized_question)

    def test_explicit_attribute_relation_normalization_payload_remains_unchanged(self) -> None:
        question = "What country is Lamprocles located in?"
        result = ExplicitEntityExtractor(
            DirectEntityLLM(
                _payload([_entity("Lamprocles", "Person")], normalized_question=question)
            )
        ).extract(question)

        self.assertEqual(result.normalized_question, question)
        self.assertFalse(result.normalization_changed)


class ExplicitEntityPromptTest(unittest.TestCase):
    def test_direct_prompt_keeps_normalization_policy_without_candidate_rules(self) -> None:
        question = "What is the place of birth of the performer of song Changed It?"
        prompt = build_explicit_entity_extraction_prompt(question)
        system_prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM

        self.assertIn("topic entity extraction and narrow structural question normalization", system_prompt)
        self.assertIn("the place of birth of the performer of song X", system_prompt)
        self.assertIn("Where was the person who performed the song Changed It born?", system_prompt)
        self.assertIn("What is the capital of France?  (single-layer \"of\")", system_prompt)
        self.assertIn('"born later" is not "younger"', system_prompt)
        self.assertIn("Directly identify explicit named topic entities", prompt)
        self.assertNotIn("Candidate spans", prompt)
        self.assertNotIn("Judge only the supplied candidates", prompt)
        self.assertNotIn("candidate_id", prompt)
        self.assertNotIn("verified_entities", prompt)

    def test_prompt_has_entity_boundary_rules_for_observed_failures(self) -> None:
        build_explicit_entity_extraction_prompt(
            "What nationality is Beatrice I, Countess Of Burgundy's husband?"
        )
        system_prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM

        self.assertIn('return "B Boy (Song)" rather than "song B Boy (Song)"', system_prompt)
        self.assertIn('Never return "the director", "the performer"', system_prompt)
        self.assertIn('return "Lamprocles" only', system_prompt)
        self.assertIn('"Maurice, Prince Of Orange"', system_prompt)
        self.assertIn('"Beatrice I, Countess Of Burgundy"', system_prompt)

    def test_prompt_scans_full_question_and_rejects_generic_capitalized_labels(self) -> None:
        prompt = build_explicit_entity_extraction_prompt(
            "Which prime minister met WikiLeaks at the Gujarat Legislative Assembly?"
        )

        self.assertIn("Directly identify explicit named topic entities", prompt)
        self.assertIn("Each surface is an exact case-preserving contiguous substring", prompt)
        self.assertIn("Keep normalized_question equal to the original question", prompt)

        system_prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM
        self.assertIn("Scan the whole original question from left to right", system_prompt)
        self.assertIn('"prime minister", "Jamaican cricketer", "British racing driver"', system_prompt)
        self.assertIn('"Gujarat Legislative Assembly"', system_prompt)
        self.assertIn('"Wexner Graduate Fellowships"', system_prompt)
        self.assertIn("FINAL ENTITY OUTPUT GATE", system_prompt)
        self.assertIn("NORMALIZATION FIREWALL", system_prompt)

    def test_prompt_prefers_recall_for_identifier_like_anchors(self) -> None:
        prompt = build_explicit_entity_extraction_prompt(
            "Where is the country with ISO code ISO 3166-2:CV located?"
        )
        system_prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM

        self.assertIn("Prefer recall for exact named or identifier-like surfaces", system_prompt)
        self.assertIn('"ISO 3166-2:CV"', system_prompt)
        self.assertIn('"MLB MVP"', system_prompt)
        self.assertIn('"FA Cup"', system_prompt)
        self.assertIn('"Auctor"', system_prompt)
        self.assertIn('"KZAR"', system_prompt)
        self.assertIn('"Darling Mills Creek"', system_prompt)
        self.assertIn("codes, awards, competitions, named events", system_prompt)
        self.assertIn("identifier-like", prompt)

    def test_prompt_calibrates_relaxed_step1_musique_failures(self) -> None:
        prompt = build_explicit_entity_extraction_prompt(
            "What year did the Governor of the city where the basilica named after the same saint as the one that Mantua Cathedral is dedicated to die?"
        )
        system_prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM

        self.assertIn('"Mantua Cathedral"', system_prompt)
        self.assertIn('"Birmingham"', system_prompt)
        self.assertIn('"Near East"', system_prompt)
        self.assertIn('"Susie"', system_prompt)
        self.assertIn('"Indy Car Race"', system_prompt)
        self.assertIn('"Cabinet"', system_prompt)
        self.assertIn('"NATO"', system_prompt)
        self.assertIn("named buildings/facilities", system_prompt)
        self.assertIn('"city"', system_prompt)
        self.assertIn('"film company"', system_prompt)
        self.assertIn('"league"', system_prompt)
        self.assertIn('"body of water"', system_prompt)
        self.assertIn('"1999"', system_prompt)
        self.assertIn('"American"', system_prompt)
        self.assertIn('"Italian"', system_prompt)
        self.assertIn("generic path nodes", prompt)

    def test_prompt_excludes_possessive_relations_and_requires_global_non_overlap(self) -> None:
        build_explicit_entity_extraction_prompt(
            "Who is the father of Empress Wang's husband?"
        )

        system_prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM
        self.assertIn('return "marty mcfly", not "marty mcfly\'s daughter"', system_prompt)
        self.assertIn('"Empress Wang\'s husband"', system_prompt)
        self.assertIn("All exact occurrences of all selected surfaces", system_prompt)
        self.assertIn("omit the short candidate", system_prompt)

    def test_prompt_keeps_internal_coordination_and_parenthetical_disambiguators(self) -> None:
        prompt = build_explicit_entity_extraction_prompt(
            "When did Christopher Newton (Criminal) visit the Battle of X and Y?"
        )
        system_prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM

        self.assertIn('"Battle of X and Y"', system_prompt)
        self.assertIn('"Christopher Newton (Criminal)"', system_prompt)
        self.assertIn("mutually non-overlapping", system_prompt)
        self.assertIn("Directly identify explicit named topic entities", prompt)

    def test_system_prompt_preserves_existing_normalization_constraints(self) -> None:
        prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM

        self.assertIn("normalized_question must usually equal the original question", prompt)
        self.assertIn("two or more mutually nested nominal \"of\" relations", prompt)
        self.assertIn("Do not normalize for style, fluency, articles, general parser-friendliness", prompt)
        self.assertIn("Preserve the exact answer set", prompt)
        self.assertIn("ENTITYA/ENTITYB placeholders", prompt)
        self.assertIn("Stop an entity before a possessive relation", prompt)
        self.assertIn("comma-separated personal name and capitalized rank/designation", prompt)

    def test_prompt_allows_only_strict_implicit_attribute_ownership_normalization(self) -> None:
        prompt = EXPLICIT_ENTITY_EXTRACTION_SYSTEM

        self.assertIn("Implicit attribute ownership", prompt)
        self.assertIn("What nationality is Lamprocles's father?", prompt)
        self.assertIn("What is the nationality of Lamprocles's father?", prompt)
        self.assertIn("What profession is ENTITYA's husband?", prompt)
        self.assertIn("What religion was ENTITYA's mother?", prompt)
        self.assertIn("not an attribute-word whitelist", prompt)
        self.assertIn("What country is ENTITYA located in?", prompt)
        self.assertIn("Which film has the director born later, A or B?", prompt)
        self.assertIn("Preserve and/or, both/either/neither, same, all, superlatives", prompt)


class DirectEntityLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.system_prompt = ""
        self.user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.payload


def _entity(surface: str, entity_type: str) -> dict[str, str]:
    return {"surface": surface, "type": entity_type}


def _payload(
    entities: list[dict[str, str]],
    *,
    normalized_question: str | None = None,
    normalization_changed: bool = False,
    normalization_note: str = "",
) -> dict[str, object]:
    return {
        "explicit_entities": entities,
        "normalized_question": normalized_question,
        "normalization_changed": normalization_changed,
        "normalization_note": normalization_note,
        "warnings": [],
    }


if __name__ == "__main__":
    unittest.main()
