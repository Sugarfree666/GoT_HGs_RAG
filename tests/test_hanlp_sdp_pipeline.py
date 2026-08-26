from __future__ import annotations

import sys
import unittest
from math import inf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from hanlp_sdp_parser import HanLPSDPParser  # noqa: E402
from pipeline import run_depo  # noqa: E402
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

    def chat_json(self, _system: str, _prompt: str) -> dict[str, object]:
        self.calls += 1
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


class HanLPSDPPipelineTest(unittest.TestCase):
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
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"unknown_ARG"})), 3)
        self.assertEqual(_edge_cost(TokenReasoningEdge(relations={"punct"})), inf)

    def test_question_words_are_content_nodes(self) -> None:
        self.assertEqual(classify_node("why", 1), "content")
        self.assertEqual(classify_node("how", 1), "content")

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


if __name__ == "__main__":
    unittest.main()
