from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from entity_path_projector import build_entity_start_nodes_from_explicit_entities  # noqa: E402
from graph_builder import GraphBuilder  # noqa: E402
from mask_span_extractor import ExplicitEntityExtractor, MaskSpanExtractor  # noqa: E402
from models import CoreNLPToken, DependencyEdge, DependencyParse, MaskMapping, MaskReplacement, RestoredGraphNodeCandidate  # noqa: E402
from path_projector import build_undirected_dependency_graph  # noqa: E402
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

    def test_explicit_entity_extraction_lothair_possessive(self) -> None:
        question = "When did Lothair II's mother die?"
        replacement = selective_entity_masking(
            question=question,
            mask_spans=MaskSpanExtractor(StaticEntityLLM([_entity(question, "Lothair II", "Person")])).extract(question),
        )
        starts = build_entity_start_nodes_from_explicit_entities(
            dependency_graph=_graph_with_placeholders(["PersonA", "mother", "die"]),
            restored_graph_node_candidates=[_candidate("1", "PersonA", "Lothair II", "Person")],
            replacement=replacement,
        )

        self.assertEqual(replacement.masked_question, "When did PersonA's mother die?")
        self.assertEqual([entity.text for entity in starts], ["Lothair II"])

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
        starts = build_entity_start_nodes_from_explicit_entities(
            dependency_graph=_graph_with_placeholders(["LocationA", "and", "LocationB"]),
            restored_graph_node_candidates=[
                _candidate("1", "LocationA", "Marufabad", "Location"),
                _candidate("3", "LocationB", "Nasamkhrali", "Location"),
            ],
            replacement=replacement,
        )

        self.assertEqual(replacement.masked_question, "Are LocationA and LocationB both located in the same country?")
        self.assertEqual([entity.text for entity in starts], ["Marufabad", "Nasamkhrali"])

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

    def test_coordinated_films_split_entities(self) -> None:
        question = "Which film was released first, Aas Ka Panchhi or Phoolwari?"
        replacement = selective_entity_masking(
            question=question,
            mask_spans=MaskSpanExtractor(
                StaticEntityLLM(
                    [
                        _entity(question, "Aas Ka Panchhi", "Film"),
                        _entity(question, "Phoolwari", "Film"),
                    ]
                )
            ).extract(question),
        )
        starts = build_entity_start_nodes_from_explicit_entities(
            dependency_graph=_graph_with_placeholders(["FilmA", "or", "FilmB"]),
            restored_graph_node_candidates=[
                _candidate("1", "FilmA", "Aas Ka Panchhi", "Film"),
                _candidate("3", "FilmB", "Phoolwari", "Film"),
            ],
            replacement=replacement,
        )

        self.assertEqual(replacement.masked_question, "Which film was released first, FilmA or FilmB?")
        self.assertNotIn("was released first, Aas Ka Panchhi or Phoolwari", replacement.mapping.values())
        self.assertEqual([entity.text for entity in starts], ["Aas Ka Panchhi", "Phoolwari"])

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
        question = "Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?"
        llm = CandidateSelectingLLM(
            {
                "God'S Gift To Women": "Film",
                "Aldri Annet Enn Bråk": "Film",
            }
        )
        result = ExplicitEntityExtractor(llm).extract(question)

        self.assertIn("Deterministic entity candidates", llm.prompt)
        self.assertIn("verified_entities", llm.prompt)
        self.assertEqual(
            [(entity.text, entity.semantic_type_hint) for entity in result.entities],
            [("God'S Gift To Women", "Film"), ("Aldri Annet Enn Bråk", "Film")],
        )
        self.assertNotIn("God", [entity.text for entity in result.entities])
        self.assertNotIn("Bråk", [entity.text for entity in result.entities])

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

    def test_step6_does_not_redetect_extra_entities(self) -> None:
        replacement = _replacement_for_one_entity()
        graph = _graph_with_placeholders(["FilmA", "OtherNNP", "director"])
        starts = build_entity_start_nodes_from_explicit_entities(
            dependency_graph=graph,
            restored_graph_node_candidates=[
                _candidate("1", "FilmA", "Known Film", "Film"),
                _candidate("2", None, "OtherNNP", "Entity", placeholder_text="OtherNNP"),
                _candidate("3", None, "director", "Person", placeholder_text="director"),
            ],
            replacement=replacement,
        )

        self.assertEqual([entity.text for entity in starts], ["Known Film"])

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

    def test_placeholder_alignment_uses_masked_char_span_overlap(self) -> None:
        replacement = MaskReplacement(
            question="Who is PersonA's paternal grandfather?",
            mapping={"PersonA": "Raghnall Mac Ruaidhrí"},
            original_question="Who is Raghnall Mac Ruaidhrí's paternal grandfather?",
            mask_mapping={
                "PersonA": {
                    "text": "Raghnall Mac Ruaidhrí",
                    "kind": "entity",
                    "semantic_type": "Person",
                    "span": {"start": 7, "end": 27},
                    "masked_span": {"start": 7, "end": 14},
                }
            },
            mask_mappings=[
                MaskMapping(
                    placeholder="PersonA",
                    original_text="Raghnall Mac Ruaidhrí",
                    kind_hint="entity",
                    semantic_type_hint="Person",
                    original_char_span=[7, 27],
                    masked_char_span=[7, 14],
                )
            ],
        )
        dependency_parse = DependencyParse(
            tokens=[
                CoreNLPToken(index=1, word="Who", character_offset_begin=0, character_offset_end=3),
                CoreNLPToken(index=2, word="is", character_offset_begin=4, character_offset_end=6),
                CoreNLPToken(index=3, word="PersonA's", character_offset_begin=7, character_offset_end=16, pos="NNP"),
                CoreNLPToken(index=4, word="paternal", character_offset_begin=17, character_offset_end=25),
                CoreNLPToken(index=5, word="grandfather", character_offset_begin=26, character_offset_end=37),
                CoreNLPToken(index=6, word="?", character_offset_begin=37, character_offset_end=38),
            ],
            edges=[
                DependencyEdge("grandfather", "nmod:poss", "PersonA's", 5, 3),
                DependencyEdge("grandfather", "amod", "paternal", 5, 4),
                DependencyEdge("grandfather", "punct", "?", 5, 6),
            ],
        )
        graph_builder = GraphBuilder()
        graph_candidates = graph_builder.build_graph_node_candidates(dependency_parse, replacement)
        restored_candidates = graph_builder.restore_graph_node_candidates(graph_candidates, replacement)
        graph = build_undirected_dependency_graph(dependency_parse, restored_candidates)
        starts = build_entity_start_nodes_from_explicit_entities(
            dependency_graph=graph,
            restored_graph_node_candidates=restored_candidates,
            replacement=replacement,
        )

        self.assertEqual(graph_candidates[2].placeholder, "PersonA")
        self.assertEqual(graph_candidates[2].display_text, "Raghnall Mac Ruaidhrí")
        self.assertEqual([entity.text for entity in starts], ["Raghnall Mac Ruaidhrí"])
        self.assertEqual(starts[0].graph_node_ids, ["3"])


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
        marker = "Deterministic entity candidates:\n"
        candidates_json = prompt.split(marker, 1)[1].split("\n\nCandidate-driven extraction mode:", 1)[0]
        candidates = json.loads(candidates_json)
        verified = []
        for candidate in candidates:
            text = candidate["text"]
            if text not in self.selected_text_to_type:
                verified.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "is_entity": False,
                        "reason": "not selected by test",
                    }
                )
                continue
            verified.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "is_entity": True,
                    "semantic_type_hint": self.selected_text_to_type[text],
                    "confidence": 0.98,
                    "reason": "selected candidate",
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


def _graph_with_placeholders(words: list[str]) -> nx.Graph:
    graph = nx.Graph()
    for index, word in enumerate(words, start=1):
        graph.add_node(str(index), word=word, text=word, order=index)
    return graph


def _candidate(
    node_id: str,
    placeholder: str | None,
    text: str,
    semantic_type: str,
    *,
    placeholder_text: str | None = None,
) -> RestoredGraphNodeCandidate:
    return RestoredGraphNodeCandidate(
        node_id=node_id,
        token_index=int(node_id),
        graph_text=placeholder or placeholder_text or text,
        placeholder=placeholder,
        restored_text=text,
        display_text=text,
        is_mask_placeholder=placeholder is not None,
        kind_hint="entity_candidate" if placeholder is not None else "context",
        semantic_type_hint=semantic_type,
        source_token_indices=[int(node_id)],
        text=text,
    )


def _replacement_for_one_entity() -> MaskReplacement:
    return MaskReplacement(
        question="FilmA",
        mapping={"FilmA": "Known Film"},
        original_question="Known Film",
        mask_mapping={
            "FilmA": {
                "text": "Known Film",
                "kind": "entity",
                "semantic_type": "Film",
                "span": {"start": 0, "end": 10},
                "masked_span": {"start": 0, "end": 5},
            }
        },
        mask_mappings=[
            MaskMapping(
                placeholder="FilmA",
                original_text="Known Film",
                kind_hint="entity",
                semantic_type_hint="Film",
                original_char_span=[0, 10],
                masked_char_span=[0, 5],
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
