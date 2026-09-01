from __future__ import annotations

import json
import sys
import unittest
from math import inf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from hanlp_sdp_parser import HanLPSDPParser  # noqa: E402
from entity_masking_preprocessor import preprocess_question  # noqa: E402
from pipeline import extract_question_structure, run_depo  # noqa: E402
from models import HanLPSDPEdge, HanLPSDPResult  # noqa: E402
from tri_sdp_reasoning_compiler import (  # noqa: E402
    TokenReasoningEdge,
    _edge_cost,
    add_pas_preposition_contraction_edges,
    build_evidence_graph,
    classify_node,
    compile_token_reasoning_structure,
)


class _LLM:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def chat_json(self, _system: str, _prompt: str) -> dict[str, object]:
        self.calls += 1
        self.prompts.append(_prompt)
        if self.calls == 1:
            return {
                "entities": ["Ryan Tubridy", "Mauro Massironi"],
            }
        return {
            "atomic_questions": [
                {"id": "q1", "question": "Who is older?", "depends_on": []}
            ]
        }


class _Parser:
    def parse(self, text: str) -> HanLPSDPResult:
        self.text = text
        return HanLPSDPResult(
            tokens=["Who", "is", "older", "ENTITYA", "or", "ENTITYB"],
            edges=[
                HanLPSDPEdge(3, "adj_ARG1", 1),
                HanLPSDPEdge(3, "adj_ARG1", 4),
                HanLPSDPEdge(3, "adj_ARG1", 6),
            ],
        )


class _EntityLLM:
    def __init__(self, entities: list[str]) -> None:
        self.entities = entities

    def chat_json(self, _system: str, _prompt: str) -> dict[str, object]:
        return {"entities": self.entities}


class HanLPSDPPipelineTest(unittest.TestCase):
    def test_masking_uses_case_insensitive_word_boundaries(self) -> None:
        result = preprocess_question(
            "Who played the girlfriend of marty mcfly in Back to the Future 2?",
            _EntityLLM(["Marty McFly", "Back to the Future 2"]),
        )

        self.assertEqual(
            result.masked_question,
            "Who played the girlfriend of ENTITYA in ENTITYB?",
        )
        self.assertEqual(
            result.mask_mapping,
            {"ENTITYA": "marty mcfly", "ENTITYB": "Back to the Future 2"},
        )

    def test_masking_does_not_replace_an_entity_inside_a_larger_word(self) -> None:
        result = preprocess_question(
            "Sand Mountainis part of a mountain chain.",
            _EntityLLM(["Sand Mountain"]),
        )

        self.assertEqual(
            result.masked_question,
            "Sand Mountainis part of a mountain chain.",
        )
        self.assertEqual(result.mask_mapping, {})

    def test_masking_drops_unmatched_entities(self) -> None:
        result = preprocess_question(
            "Where is the library?",
            _EntityLLM(["Missing Entity"]),
        )

        self.assertEqual(result.entities, [])
        self.assertEqual(result.mask_mapping, {})
        self.assertEqual(result.masked_question, "Where is the library?")

    def test_parser_keeps_pas_edges_and_syntax_heads(self) -> None:
        parser = HanLPSDPParser()
        parser._pipeline = lambda _text: {  # type: ignore[assignment]
            "tok": ["ENTITYA", "target"],
            "sdp/pas": [[(2, "ARG1")], [(0, "root")]],
            "dep": [(2, "nsubj"), (0, "root")],
        }

        result = parser.parse("ENTITYA target")

        self.assertEqual(result.tokens, ["ENTITYA", "target"])
        self.assertEqual(
            [(edge.head_idx, edge.relation, edge.dep_idx) for edge in result.edges],
            [(2, "ARG1", 1), (0, "root", 2)],
        )
        self.assertEqual(result.syntax_heads, {"1": 2, "2": 0})

    def test_compiler_returns_entity_paths_only(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "older", "Who"],
            edges=[
                HanLPSDPEdge(2, "adj_ARG1", 1),
                HanLPSDPEdge(2, "adj_ARG1", 3),
            ],
        )

        self.assertEqual(
            compile_token_reasoning_structure(result, ["ENTITYA"]),
            [["ENTITYA", "older", "Who"]],
        )

    def test_edge_costs_match_the_method(self) -> None:
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"act_ARG2"})), 1)
        self.assertEqual(
            _edge_cost(TokenReasoningEdge(rules={"pas_possessive_contraction"})),
            1,
        )
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"relative_ARG1"})), 2)
        self.assertEqual(
            _edge_cost(
                TokenReasoningEdge(rules={"pas_coordination_candidate_attachment"})
            ),
            2,
        )
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"coord_ARG1"})), 3)
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"aux_ARG1"})), 3)
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"unknown_ARG"})), 3)
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"punct"})), inf)

    def test_question_words_are_content_nodes(self) -> None:
        self.assertEqual(classify_node("why"), "content")
        self.assertEqual(classify_node("how"), "content")

    def test_function_words_remain_searchable_path_bridges(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "from", "country", "same"],
            edges=[
                HanLPSDPEdge(1, "verb_ARG1", 2),
                HanLPSDPEdge(2, "prep_ARG2", 3),
                HanLPSDPEdge(3, "adj_ARG1", 4),
            ],
        )

        self.assertEqual(
            compile_token_reasoning_structure(result, ["ENTITYA"]),
            [["ENTITYA", "from", "country", "same"]],
        )
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"prep_ARG2"})), 3)

    def test_sp_prefers_the_complete_path_across_a_function_word(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "film", "has", "director", "born", "later"],
            edges=[
                HanLPSDPEdge(2, "noun_ARG1", 1),
                HanLPSDPEdge(3, "verb_ARG1", 2),
                HanLPSDPEdge(3, "verb_ARG2", 4),
                HanLPSDPEdge(5, "verb_ARG2", 4),
                HanLPSDPEdge(6, "adj_ARG1", 5),
            ],
        )

        self.assertEqual(
            compile_token_reasoning_structure(result, ["ENTITYA"]),
            [["ENTITYA", "film", "has", "director", "born", "later"]],
        )

    def test_root_word_is_not_treated_as_the_virtual_root_node(self) -> None:
        self.assertEqual(classify_node("root"), "content")

    def test_preposition_contraction_keeps_the_entity_path(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "of", "capital", "Who"],
            edges=[
                HanLPSDPEdge(2, "prep_ARG1", 1),
                HanLPSDPEdge(2, "prep_ARG2", 3),
                HanLPSDPEdge(3, "adj_ARG1", 4),
            ],
        )

        self.assertEqual(
            compile_token_reasoning_structure(result, ["ENTITYA"]),
            [["ENTITYA", "capital", "Who"]],
        )

    def test_preposition_contraction_updates_the_single_graph(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "of", "capital"],
            edges=[
                HanLPSDPEdge(2, "prep_ARG1", 1),
                HanLPSDPEdge(2, "prep_ARG2", 3),
            ],
        )

        state = build_evidence_graph(result)
        add_pas_preposition_contraction_edges(state)

        self.assertNotIn("2", state.graph)
        self.assertIn("3", state.graph["1"])
        self.assertIs(state.graph["1"]["3"], state.graph["3"]["1"])
        self.assertEqual(
            state.graph["1"]["3"].rules,
            {"pas_preposition_contraction"},
        )

    def test_possessive_contraction_keeps_the_entity_path(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "'", "s", "capital", "Who"],
            edges=[
                HanLPSDPEdge(2, "poss_ARG2", 1),
                HanLPSDPEdge(3, "poss_ARG2", 1),
                HanLPSDPEdge(2, "poss_ARG1", 4),
                HanLPSDPEdge(3, "poss_ARG1", 4),
                HanLPSDPEdge(4, "adj_ARG1", 5),
            ],
        )

        self.assertEqual(
            compile_token_reasoning_structure(result, ["ENTITYA"]),
            [["ENTITYA", "capital", "Who"]],
        )

    def test_possessive_contraction_allows_an_isolated_suffix_marker(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "'", "s", "capital", "Who"],
            edges=[
                HanLPSDPEdge(2, "poss_ARG2", 1),
                HanLPSDPEdge(2, "poss_ARG1", 4),
                HanLPSDPEdge(4, "adj_ARG1", 5),
            ],
        )

        self.assertEqual(
            compile_token_reasoning_structure(result, ["ENTITYA"]),
            [["ENTITYA", "capital", "Who"]],
        )

    def test_possessive_contraction_supports_combined_and_curly_markers(self) -> None:
        for marker_tokens, marker_edges, possessed_id in [
            (["'s"], [HanLPSDPEdge(2, "poss_ARG2", 1)], 2),
            (["’", "s"], [HanLPSDPEdge(2, "poss_ARG2", 1)], 3),
        ]:
            with self.subTest(marker_tokens=marker_tokens):
                capital_id = 2 + len(marker_tokens)
                result = HanLPSDPResult(
                    tokens=["ENTITYA", *marker_tokens, "capital", "Who"],
                    edges=[
                        *marker_edges,
                        HanLPSDPEdge(possessed_id, "poss_ARG1", capital_id),
                        HanLPSDPEdge(capital_id, "adj_ARG1", capital_id + 1),
                    ],
                )

                self.assertEqual(
                    compile_token_reasoning_structure(result, ["ENTITYA"]),
                    [["ENTITYA", "capital", "Who"]],
                )

    def test_coordination_uses_the_shared_syntactic_attachment(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "and", "ENTITYB", "won", "Who"],
            edges=[
                HanLPSDPEdge(2, "coord", 1),
                HanLPSDPEdge(2, "coord", 3),
                HanLPSDPEdge(4, "verb_ARG1", 5),
            ],
            syntax_heads={"1": 4, "2": 4, "3": 4, "4": 0, "5": 4},
        )

        self.assertEqual(
            compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"]),
            [["ENTITYA", "won", "Who"], ["ENTITYB", "won", "Who"]],
        )

    def test_coordination_does_not_attach_entities_to_root(self) -> None:
        result = HanLPSDPResult(
            tokens=["ENTITYA", "or", "ENTITYB"],
            edges=[
                HanLPSDPEdge(2, "coord_ARG1", 1),
                HanLPSDPEdge(2, "coord_ARG2", 3),
            ],
            syntax_heads={"1": 0, "2": 3, "3": 1},
        )

        self.assertEqual(
            compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"]),
            [],
        )

    def test_pipeline_uses_masked_question_and_generates_dag(self) -> None:
        llm = _LLM()
        parser = _Parser()
        result = run_depo(
            "Who is older, Ryan Tubridy or Mauro Massironi?",
            parser,
            llm,
        )

        self.assertEqual(parser.text, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(result["atomic_question_dag"]["nodes"][0]["question"], "Who is older?")

    def test_pipeline_can_extract_question_structure_without_generating_dag(self) -> None:
        llm = _LLM()
        parser = _Parser()

        result = extract_question_structure(
            "Who is older, Ryan Tubridy or Mauro Massironi?",
            parser,
            llm,
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(parser.text, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(
            result["question_structure"],
            [
                ["Ryan Tubridy", "older", "Who"],
                ["Mauro Massironi", "older", "Who"],
            ],
        )

    def test_pipeline_can_override_question_structure(self) -> None:
        llm = _LLM()
        parser = _Parser()

        run_depo(
            "Who is older, Ryan Tubridy or Mauro Massironi?",
            parser,
            llm,
            question_structure_override=[],
        )

        dag_request = json.loads(llm.prompts[1])
        self.assertEqual(dag_request["question_structure"], [])


if __name__ == "__main__":
    unittest.main()
