from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from atomic_question_dag import (  # noqa: E402
    ATOMIC_QUESTION_DAG_SYSTEM,
    PathAlignedAtomicDAGGenerator,
    RestoredTokenPath,
    restore_entity_paths,
)
from entity_masking_preprocessor import EntityMaskingPreprocessor  # noqa: E402
from main import run_hanlp_sdp_pipeline  # noqa: E402
from models import HanLPSDPEdge, HanLPSDPResult, MaskMapping, QuestionRecord  # noqa: E402


class AtomicQuestionDAGTest(unittest.TestCase):
    def test_llm_input_contract_contains_only_original_question_and_paths(self) -> None:
        paths = [
            RestoredTokenPath(
                "P1",
                ("Ten9Eight: Shoot For The Moon", "director", "share", "nationality"),
            )
        ]
        llm = RecordingStep5LLM(_johnny_payload())

        PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Do director of film Ten9Eight: Shoot For The Moon share the same nationality?",
            paths=paths,
        )

        payload = json.loads(llm.user_prompts[0])
        self.assertEqual(set(payload), {"original_question", "paths"})
        serialized = json.dumps(payload, ensure_ascii=False)
        forbidden = [
            "answer_anchor",
            "entity_anchors",
            "constraints",
            "candidate_sets",
            "path_type",
            "masked_question",
            "entity_map",
            "ENTITYA",
            "ENTITYB",
        ]
        for item in forbidden:
            self.assertNotIn(item, serialized)
        self.assertEqual(
            payload["paths"][0]["nodes"],
            [
                {"index": 0, "text": "Ten9Eight: Shoot For The Moon"},
                {"index": 1, "text": "director"},
                {"index": 2, "text": "share"},
                {"index": 3, "text": "nationality"},
            ],
        )

    def test_restore_entity_paths_replaces_complete_placeholder_tokens_only(self) -> None:
        paths = [
            SimpleNamespace(path_id="P1", nodes=["ENTITYA", "director", "ENTITYBish"]),
            SimpleNamespace(path_id="P2", nodes=["ENTITYB", "born"]),
        ]
        mappings = [
            MaskMapping("ENTITYA", "Ten9Eight: Shoot For The Moon", "entity"),
            MaskMapping("ENTITYB", "Sabotage (1936 Film)", "entity"),
        ]

        restored = restore_entity_paths(paths, mappings)

        self.assertEqual(restored[0].nodes, ("Ten9Eight: Shoot For The Moon", "director", "ENTITYBish"))
        self.assertEqual(restored[1].nodes, ("Sabotage (1936 Film)", "born"))

    def test_restore_entity_paths_rejects_unmapped_placeholder(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unresolved entity placeholder"):
            restore_entity_paths([SimpleNamespace(path_id="P1", nodes=["ENTITYA", "born"])], [])

    def test_single_path_predicate_chain_generates_dependent_atomic_questions(self) -> None:
        paths = [
            RestoredTokenPath("P1", ("Johnny Majors", "defeated", "player", "born", "year"))
        ]
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_johnny_payload())).generate(
            original_question="The player who defeated Johnny Majors for the Heisman Trophy in 1956 was born in what year?",
            paths=paths,
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([node.id for node in result.nodes], ["q1", "q2"])
        self.assertEqual(result.nodes[1].depends_on, ("q1",))
        self.assertEqual(result.edges[0].to_dict(), {"source": "q1", "target": "q2"})
        self.assertEqual(result.leaf_node_ids, ["q2"])
        self.assertEqual(result.nodes[0].support.nodes, ("Johnny Majors", "defeated", "player"))
        self.assertEqual(result.nodes[1].support.nodes, ("player", "born", "year"))

    def test_parallel_nationality_paths_generate_independent_evidence_branches(self) -> None:
        paths = [
            RestoredTokenPath("P1", ("Ten9Eight: Shoot For The Moon", "director", "share", "nationality")),
            RestoredTokenPath("P2", ("Sabotage (1936 Film)", "director", "share", "nationality")),
        ]
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_parallel_nationality_payload())).generate(
            original_question=(
                "Do director of film Ten9Eight: Shoot For The Moon and director of film "
                "Sabotage (1936 Film) share the same nationality?"
            ),
            paths=paths,
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}, {"source": "q3", "target": "q4"}])
        self.assertEqual(result.leaf_node_ids, ["q2", "q4"])
        self.assertFalse(any("same" in node.question.lower() for node in result.nodes))
        self.assertFalse(any(len(node.depends_on) > 1 for node in result.nodes))

    def test_born_later_paths_generate_date_evidence_not_final_selection(self) -> None:
        paths = [
            RestoredTokenPath("P1", ("Gideon Johnson Pillow", "born", "later")),
            RestoredTokenPath("P2", ("Holm Jølsen", "born", "later")),
        ]
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_born_later_payload())).generate(
            original_question="Who was born later, Gideon Johnson Pillow or Holm Jølsen?",
            paths=paths,
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([node.question for node in result.nodes], ["When was Gideon Johnson Pillow born?", "When was Holm Jølsen born?"])
        self.assertFalse(any("who was born later" in node.question.lower() for node in result.nodes))
        self.assertEqual(result.edges, [])

    def test_younger_director_paths_generate_evidence_not_candidate_selection(self) -> None:
        paths = [
            RestoredTokenPath("P1", ("Dangerously They Live", "director", "younger")),
            RestoredTokenPath("P2", ("Salad By The Roots", "director", "younger")),
        ]
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_younger_director_payload())).generate(
            original_question="Which film whose director is younger, Dangerously They Live or Salad By The Roots?",
            paths=paths,
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}, {"source": "q3", "target": "q4"}])
        self.assertFalse(any("which film" in node.question.lower() for node in result.nodes))
        self.assertEqual(result.nodes[1].support.nodes, ("director", "younger"))

    def test_dell_long_path_requires_contiguous_support_cover(self) -> None:
        paths = [
            RestoredTokenPath("P1", ("FireWire", "replacing", "interface", "letting", "feature", "call", "What"))
        ]
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_dell_payload())).generate(
            original_question=(
                "What does dell call the feature letting the interface replacing FireWire "
                "to remain powered when the computer is off?"
            ),
            paths=paths,
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([node.support.nodes for node in result.nodes], [
            ("FireWire", "replacing", "interface"),
            ("interface", "letting", "feature"),
            ("feature", "call", "What"),
        ])

    def test_invalid_path_id_fails_validation(self) -> None:
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_payload_with_support("P9", 0, 1))).generate(
            original_question="Question?",
            paths=[RestoredTokenPath("P1", ("A", "B"))],
        )

        self.assertFalse(result.valid)
        self.assertIn("support.path_id does not exist", "; ".join(result.validation_errors))

    def test_invalid_index_or_reversed_span_fails_validation(self) -> None:
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_payload_with_support("P1", 1, 0))).generate(
            original_question="Question?",
            paths=[RestoredTokenPath("P1", ("A", "B"))],
        )

        self.assertFalse(result.valid)
        self.assertIn("start_index must be <= end_index", "; ".join(result.validation_errors))

    def test_depends_on_must_match_question_reference(self) -> None:
        payload = {
            "nodes": [
                {"id": "q1", "question": "Who directed A?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 1}},
                {"id": "q2", "question": "What is the nationality?", "depends_on": ["q1"], "support": {"path_id": "P1", "start_index": 1, "end_index": 2}},
            ]
        }
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(payload)).generate(
            original_question="Question?",
            paths=[RestoredTokenPath("P1", ("A", "director", "nationality"))],
        )

        self.assertFalse(result.valid)
        self.assertIn("question references [] but depends_on is ['q1']", "; ".join(result.validation_errors))

    def test_braced_question_reference_is_rejected(self) -> None:
        payload = {
            "nodes": [
                {"id": "q1", "question": "Who directed A?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 1}},
                {"id": "q2", "question": "What is the nationality of {{q1}}?", "depends_on": ["q1"], "support": {"path_id": "P1", "start_index": 1, "end_index": 2}},
            ]
        }
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(payload)).generate(
            original_question="Question?",
            paths=[RestoredTokenPath("P1", ("A", "director", "nationality"))],
        )

        self.assertFalse(result.valid)
        self.assertIn("question references [] but depends_on is ['q1']", "; ".join(result.validation_errors))

    def test_cross_path_dependency_fails_validation(self) -> None:
        payload = {
            "nodes": [
                {"id": "q1", "question": "Who directed A?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 1}},
                {"id": "q2", "question": "What is the nationality of q1's answer?", "depends_on": ["q1"], "support": {"path_id": "P2", "start_index": 0, "end_index": 1}},
            ]
        }
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(payload)).generate(
            original_question="Question?",
            paths=[RestoredTokenPath("P1", ("A", "director")), RestoredTokenPath("P2", ("B", "nationality"))],
        )

        self.assertFalse(result.valid)
        self.assertIn("crosses paths", "; ".join(result.validation_errors))

    def test_node_depends_on_two_branches_fails_validation(self) -> None:
        payload = {
            "nodes": [
                {"id": "q1", "question": "When was A born?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 1}},
                {"id": "q2", "question": "When was B born?", "depends_on": [], "support": {"path_id": "P2", "start_index": 0, "end_index": 1}},
                {"id": "q3", "question": "Compare q1's answer and q2's answer?", "depends_on": ["q1", "q2"], "support": {"path_id": "P1", "start_index": 0, "end_index": 1}},
            ]
        }
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(payload)).generate(
            original_question="Question?",
            paths=[RestoredTokenPath("P1", ("A", "born")), RestoredTokenPath("P2", ("B", "born"))],
        )

        self.assertFalse(result.valid)
        self.assertIn("at most one previous node", "; ".join(result.validation_errors))

    def test_path_edge_not_covered_fails_validation(self) -> None:
        result = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_payload_with_support("P1", 0, 1))).generate(
            original_question="Question?",
            paths=[RestoredTokenPath("P1", ("A", "B", "C"))],
        )

        self.assertFalse(result.valid)
        self.assertIn("uncovered adjacent edge", "; ".join(result.validation_errors))

    def test_deterministic_output_for_same_payload(self) -> None:
        paths = [RestoredTokenPath("P1", ("Johnny Majors", "defeated", "player", "born", "year"))]
        first = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_johnny_payload())).generate(
            original_question="Question?",
            paths=paths,
        )
        second = PathAlignedAtomicDAGGenerator(RecordingStep5LLM(_johnny_payload())).generate(
            original_question="Question?",
            paths=paths,
        )

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_pipeline_integration_returns_atomic_question_dag(self) -> None:
        llm = FullPipelineLLM()
        result = run_hanlp_sdp_pipeline(
            record=QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?"),
            index=1,
            preprocessor=EntityMaskingPreprocessor(llm),
            parser=FakeOlderParser(),
        )

        dag = result["atomic_question_dag"]
        self.assertTrue(dag.valid, dag.validation_errors)
        self.assertEqual([node.question for node in dag.nodes], ["When was Ryan Tubridy born?", "When was Mauro Massironi born?"])
        payload = json.loads(llm.step5_user_prompt)
        self.assertEqual(set(payload), {"original_question", "paths"})
        self.assertNotIn("ENTITYA", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("ENTITYB", json.dumps(payload, ensure_ascii=False))


class RecordingStep5LLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        self.assert_step5_prompt(system_prompt, user_prompt)
        return self.payload

    @staticmethod
    def assert_step5_prompt(system_prompt: str, user_prompt: str) -> None:
        if system_prompt != ATOMIC_QUESTION_DAG_SYSTEM:
            raise AssertionError("Unexpected system prompt")
        payload = json.loads(user_prompt)
        if set(payload) != {"original_question", "paths"}:
            raise AssertionError(f"Unexpected Step5 prompt keys: {set(payload)}")


class FullPipelineLLM:
    def __init__(self) -> None:
        self.step5_user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "DEPO Step 2" in system_prompt:
            question = "Who is older, Ryan Tubridy or Mauro Massironi?"
            return {
                "entities": [
                    _entity(question, "Ryan Tubridy"),
                    _entity(question, "Mauro Massironi"),
                ],
                "warnings": [],
            }
        if system_prompt == ATOMIC_QUESTION_DAG_SYSTEM:
            self.step5_user_prompt = user_prompt
            return {
                "nodes": [
                    {"id": "q1", "question": "When was Ryan Tubridy born?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 1}},
                    {"id": "q2", "question": "When was Mauro Massironi born?", "depends_on": [], "support": {"path_id": "P2", "start_index": 0, "end_index": 1}},
                ]
            }
        raise AssertionError(f"Unexpected system prompt: {system_prompt}")


class FakeOlderParser:
    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        del placeholders
        tokens = ["Who", "is", "older", ",", "ENTITYA", "or", "ENTITYB", "?"]
        return HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=["tok", "sdp/dm", "sdp/pas", "sdp/psd"],
            sdp_graphs={"sdp/dm": [], "sdp/pas": [], "sdp/psd": []},
            edges=[
                HanLPSDPEdge("sdp/dm", 0, "ROOT", "root", 3, "older"),
                HanLPSDPEdge("sdp/dm", 3, "older", "ARG1", 1, "Who"),
                HanLPSDPEdge("sdp/dm", 3, "older", "ARG2", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 6, "or", "coord_ARG1", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 6, "or", "coord_ARG2", 7, "ENTITYB"),
                HanLPSDPEdge("sdp/psd", 6, "or", "DISJ.member", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/psd", 6, "or", "DISJ.member", 7, "ENTITYB"),
                HanLPSDPEdge("sdp/psd", 3, "older", "ACT-arg", 7, "ENTITYB"),
            ],
            raw={"tok": tokens},
            model="fake",
        )


def _johnny_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": "q1",
                "question": "Who defeated Johnny Majors for the Heisman Trophy in 1956?",
                "depends_on": [],
                "support": {"path_id": "P1", "start_index": 0, "end_index": 2},
            },
            {
                "id": "q2",
                "question": "What year was q1's answer born?",
                "depends_on": ["q1"],
                "support": {"path_id": "P1", "start_index": 2, "end_index": 4},
            },
        ]
    }


def _parallel_nationality_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "q1", "question": "Who directed Ten9Eight: Shoot For The Moon?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 1}},
            {"id": "q2", "question": "What is the nationality of q1's answer?", "depends_on": ["q1"], "support": {"path_id": "P1", "start_index": 1, "end_index": 3}},
            {"id": "q3", "question": "Who directed Sabotage (1936 Film)?", "depends_on": [], "support": {"path_id": "P2", "start_index": 0, "end_index": 1}},
            {"id": "q4", "question": "What is the nationality of q3's answer?", "depends_on": ["q3"], "support": {"path_id": "P2", "start_index": 1, "end_index": 3}},
        ]
    }


def _born_later_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "q1", "question": "When was Gideon Johnson Pillow born?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 2}},
            {"id": "q2", "question": "When was Holm Jølsen born?", "depends_on": [], "support": {"path_id": "P2", "start_index": 0, "end_index": 2}},
        ]
    }


def _younger_director_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "q1", "question": "Who directed Dangerously They Live?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 1}},
            {"id": "q2", "question": "When was q1's answer born?", "depends_on": ["q1"], "support": {"path_id": "P1", "start_index": 1, "end_index": 2}},
            {"id": "q3", "question": "Who directed Salad By The Roots?", "depends_on": [], "support": {"path_id": "P2", "start_index": 0, "end_index": 1}},
            {"id": "q4", "question": "When was q3's answer born?", "depends_on": ["q3"], "support": {"path_id": "P2", "start_index": 1, "end_index": 2}},
        ]
    }


def _dell_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "q1", "question": "What interface replaced FireWire?", "depends_on": [], "support": {"path_id": "P1", "start_index": 0, "end_index": 2}},
            {"id": "q2", "question": "What feature lets q1's answer remain powered?", "depends_on": ["q1"], "support": {"path_id": "P1", "start_index": 2, "end_index": 4}},
            {"id": "q3", "question": "What does Dell call q2's answer?", "depends_on": ["q2"], "support": {"path_id": "P1", "start_index": 4, "end_index": 6}},
        ]
    }


def _payload_with_support(path_id: str, start: int, end: int) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": "q1",
                "question": "What is the supported fact?",
                "depends_on": [],
                "support": {"path_id": path_id, "start_index": start, "end_index": end},
            }
        ]
    }


def _entity(question: str, text: str) -> dict[str, Any]:
    start = question.index(text)
    return {
        "text": text,
        "start_char": start,
        "end_char": start + len(text),
        "confidence": 1.0,
        "reason": "test entity",
    }


if __name__ == "__main__":
    unittest.main()
