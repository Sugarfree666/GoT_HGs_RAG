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
    ATOMIC_QUESTION_DAG_NO_PATH_SYSTEM,
    ATOMIC_QUESTION_DAG_SYSTEM,
    NoPathAtomicDAGGenerator,
    PathAlignedAtomicDAGGenerator,
    prompt_input_payload,
    restore_entity_paths,
    restore_global_best_path,
    restore_global_best_paths,
    validate_atomic_question_dag,
)
from entity_masking_preprocessor import EntityMaskingPreprocessor  # noqa: E402
from main import run_hanlp_sdp_pipeline  # noqa: E402
from models import HanLPSDPEdge, HanLPSDPResult, MaskMapping, QuestionRecord  # noqa: E402


class AtomicQuestionDAGTest(unittest.TestCase):
    def test_prompt_input_contract_contains_only_original_entities_and_global_best_paths(self) -> None:
        payload = prompt_input_payload(
            original_question="Question?",
            explicit_entities=["Ten9Eight: Shoot For The Moon"],
            global_best_paths=[["Ten9Eight: Shoot For The Moon", "director", "nationality"]],
        )

        self.assertEqual(set(payload), {"original_question", "explicit_entities", "global_best_paths"})
        self.assertEqual(payload["explicit_entities"], ["Ten9Eight: Shoot For The Moon"])
        self.assertEqual(payload["global_best_paths"], [["Ten9Eight: Shoot For The Moon", "director", "nationality"]])
        serialized = json.dumps(payload, ensure_ascii=False)
        for forbidden in ("masked_question", "normalized_question", "sdp", "candidate_sets", "ENTITYA"):
            self.assertNotIn(forbidden, serialized)

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

    def test_restore_global_best_path_replaces_placeholders(self) -> None:
        restored = restore_global_best_path(
            {"nodes": ["ENTITYA", "signed", "person"]},
            [MaskMapping("ENTITYA", "Barcelona", "entity")],
        )

        self.assertEqual(restored, ["Barcelona", "signed", "person"])

    def test_restore_global_best_paths_replaces_each_selected_path(self) -> None:
        paths = [
            SimpleNamespace(nodes=["ENTITYA", "director", "born"]),
            SimpleNamespace(nodes=["ENTITYB", "director", "born"]),
        ]
        restored = restore_global_best_paths(
            paths,
            [
                MaskMapping("ENTITYA", "Illusions (1982 Film)", "entity"),
                MaskMapping("ENTITYB", "It'S A Wonderful Afterlife", "entity"),
            ],
        )

        self.assertEqual(
            restored,
            [
                ["Illusions (1982 Film)", "director", "born"],
                ["It'S A Wonderful Afterlife", "director", "born"],
            ],
        )

    def test_restore_global_best_path_rejects_unmapped_placeholder(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unresolved entity placeholder"):
            restore_global_best_path({"nodes": ["ENTITYA", "born"]}, [])

    def test_valid_new_payload_builds_dag_from_atomic_questions(self) -> None:
        result = validate_atomic_question_dag(_johnny_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([node.id for node in result.nodes], ["q1", "q2"])
        self.assertEqual(result.nodes[1].depends_on, ("q1",))
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])
        self.assertEqual(result.leaf_node_ids, ["q2"])
        self.assertIn("semantic_reasoning_paths", result.raw_payload)
        self.assertEqual(result.nodes[0].operation, "lookup")
        self.assertEqual(result.nodes[0].semantic_edge_ids, ("p1_e1",))

    def test_path_generator_keeps_step5_input_contract(self) -> None:
        llm = RecordingStep5LLM(_johnny_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="The player who defeated Johnny Majors was born in what year?",
            explicit_entities=["Johnny Majors"],
            global_best_paths=[["Johnny Majors", "defeated", "player", "born", "year"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        payload = json.loads(llm.user_prompts[0])
        self.assertEqual(set(payload), {"original_question", "explicit_entities", "global_best_paths"})
        self.assertEqual(payload["global_best_paths"], [["Johnny Majors", "defeated", "player", "born", "year"]])

    def test_empty_global_best_path_fails_before_llm_call(self) -> None:
        llm = RecordingStep5LLM(_johnny_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Question?",
            explicit_entities=[],
            global_best_paths=[],
        )

        self.assertFalse(result.valid)
        self.assertEqual(llm.user_prompts, [])
        self.assertIn("Step5 requires at least one non-empty global_best_paths entry.", result.validation_errors)

    def test_old_actions_payload_is_rejected_for_path_aligned_step5(self) -> None:
        result = validate_atomic_question_dag(
            {
                "actions": [
                    {
                        "id": "q1",
                        "consume": ["A"],
                        "produce": "q1_answer",
                        "question": "What is A?",
                    }
                ]
            }
        )

        self.assertFalse(result.valid)
        self.assertIn("semantic_reasoning_paths must be a non-empty list.", result.validation_errors)
        self.assertIn("atomic_questions must be a non-empty list.", result.validation_errors)

    def test_invalid_dependencies_are_rejected(self) -> None:
        cases = [
            ("future", ["q3"], "depends_on references non-previous node 'q3'"),
            ("self", ["q1"], "depends_on references non-previous node 'q1'"),
            ("unknown", ["qx"], "depends_on contains invalid dependency id 'qx'"),
        ]
        for _, dependencies, expected in cases:
            with self.subTest(expected=expected):
                payload = _johnny_payload()
                payload["atomic_questions"][0]["depends_on"] = dependencies
                result = validate_atomic_question_dag(payload)
                self.assertFalse(result.valid)
                self.assertIn(expected, "; ".join(result.validation_errors))

    def test_unresolved_entity_placeholder_in_question_is_rejected(self) -> None:
        payload = _johnny_payload()
        payload["atomic_questions"][0]["question"] = "Who defeated ENTITYA?"

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("question contains unresolved ENTITY placeholder", "; ".join(result.validation_errors))

    def test_unresolved_entity_placeholder_in_semantic_path_is_rejected(self) -> None:
        payload = _johnny_payload()
        payload["semantic_reasoning_paths"][0]["semantic_nodes"][0]["label"] = "ENTITYA"

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("label contains unresolved ENTITY placeholder", "; ".join(result.validation_errors))

    def test_unresolved_entity_placeholder_in_support_tokens_is_rejected(self) -> None:
        payload = _johnny_payload()
        payload["semantic_reasoning_paths"][0]["semantic_edges"][0]["support_tokens"] = ["ENTITYA"]

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("support_tokens contains unresolved ENTITY placeholder", "; ".join(result.validation_errors))

    def test_support_tokens_must_be_copied_from_source_token_path(self) -> None:
        payload = _johnny_payload()
        payload["semantic_reasoning_paths"][0]["semantic_edges"][0]["support_tokens"] = ["not-in-path"]

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("token not copied from source_token_path", "; ".join(result.validation_errors))

    def test_braced_question_reference_is_rejected(self) -> None:
        payload = _johnny_payload()
        payload["atomic_questions"][1]["question"] = "What year was {{q1}} born?"

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("must use qN's answer references", "; ".join(result.validation_errors))

    def test_comparison_case_generates_expected_edges(self) -> None:
        result = validate_atomic_question_dag(_parallel_nationality_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(
            [node.question for node in result.nodes],
            [
                "Who directed Ten9Eight: Shoot For The Moon?",
                "What is the nationality of q1's answer?",
                "Who directed Sabotage (1936 Film)?",
                "What is the nationality of q3's answer?",
                "Do q2's answer and q4's answer indicate that the two directors share the same nationality?",
            ],
        )
        self.assertEqual(
            [edge.to_dict() for edge in result.edges],
            [
                {"source": "q1", "target": "q2"},
                {"source": "q3", "target": "q4"},
                {"source": "q2", "target": "q5"},
                {"source": "q4", "target": "q5"},
            ],
        )
        self.assertEqual(result.leaf_node_ids, ["q5"])

    def test_no_path_generator_still_accepts_isolated_old_action_trace(self) -> None:
        llm = RecordingNoPathLLM(
            {
                "actions": [
                    {
                        "id": "q1",
                        "consume": ["A", "relation"],
                        "produce": "q1_answer",
                        "question": "Who is the performer of Song A?",
                    },
                    {
                        "id": "q2",
                        "consume": ["q1_answer"],
                        "produce": "q2_answer",
                        "question": "Where was q1's answer born?",
                    },
                ]
            }
        )
        result = NoPathAtomicDAGGenerator(llm).generate(original_question="Where was the performer of Song A born?")

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.raw_payload["actions"][0]["consume"], [])
        self.assertEqual(result.nodes[1].depends_on, ("q1",))
        self.assertIn("no-path mode ignored non-empty consume", "; ".join(result.warnings))

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
        self.assertIn("semantic_reasoning_paths", dag.raw_payload)
        payload = json.loads(llm.step5_user_prompt)
        self.assertEqual(set(payload), {"original_question", "explicit_entities", "global_best_paths"})
        self.assertEqual(payload["explicit_entities"], ["Ryan Tubridy", "Mauro Massironi"])
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
        if system_prompt != ATOMIC_QUESTION_DAG_SYSTEM:
            raise AssertionError("Unexpected system prompt")
        payload = json.loads(user_prompt)
        if set(payload) != {"original_question", "explicit_entities", "global_best_paths"}:
            raise AssertionError(f"Unexpected Step5 prompt keys: {set(payload)}")
        return self.payload


class RecordingNoPathLLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if system_prompt != ATOMIC_QUESTION_DAG_NO_PATH_SYSTEM:
            raise AssertionError("Unexpected no-path system prompt")
        payload = json.loads(user_prompt)
        if set(payload) != {"original_question"}:
            raise AssertionError(f"Unexpected no-path Step5 prompt keys: {set(payload)}")
        return self.payload


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
            payload = json.loads(user_prompt)
            if set(payload) != {"original_question", "explicit_entities", "global_best_paths"}:
                raise AssertionError(f"Unexpected Step5 prompt keys: {set(payload)}")
            return _older_payload()
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


def _semantic_path(
    branch_id: str,
    source_token_path: list[str],
    edge_ids: list[str],
) -> dict[str, Any]:
    nodes = [
        {"id": f"{branch_id}_n1", "label": source_token_path[0], "kind": "entity"},
        {"id": f"{branch_id}_n2", "label": source_token_path[-1], "kind": "value_slot"},
    ]
    edges = [
        {
            "id": edge_id,
            "source": f"{branch_id}_n1",
            "target": f"{branch_id}_n2",
            "relation": " ".join(source_token_path[1:]) or "lookup",
            "support_tokens": source_token_path[:2] if len(source_token_path) > 1 else source_token_path,
        }
        for edge_id in edge_ids
    ]
    return {
        "branch_id": branch_id,
        "source_token_path": source_token_path,
        "semantic_nodes": nodes,
        "semantic_edges": edges,
        "terminal_node_id": f"{branch_id}_n2",
    }


def _johnny_payload() -> dict[str, Any]:
    return {
        "semantic_reasoning_paths": [
            _semantic_path("p1", ["Johnny Majors", "defeated", "player", "born", "year"], ["p1_e1", "p1_e2"])
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who defeated Johnny Majors for the Heisman Trophy in 1956?",
                "depends_on": [],
                "operation": "lookup",
                "semantic_edge_ids": ["p1_e1"],
            },
            {
                "id": "q2",
                "question": "What year was q1's answer born?",
                "depends_on": ["q1"],
                "operation": "lookup",
                "semantic_edge_ids": ["p1_e2"],
            },
        ],
    }


def _parallel_nationality_payload() -> dict[str, Any]:
    return {
        "semantic_reasoning_paths": [
            _semantic_path("p1", ["Ten9Eight: Shoot For The Moon", "director", "nationality"], ["p1_e1", "p1_e2"]),
            _semantic_path("p2", ["Sabotage (1936 Film)", "director", "nationality"], ["p2_e1", "p2_e2"]),
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who directed Ten9Eight: Shoot For The Moon?",
                "depends_on": [],
                "operation": "lookup",
                "semantic_edge_ids": ["p1_e1"],
            },
            {
                "id": "q2",
                "question": "What is the nationality of q1's answer?",
                "depends_on": ["q1"],
                "operation": "lookup",
                "semantic_edge_ids": ["p1_e2"],
            },
            {
                "id": "q3",
                "question": "Who directed Sabotage (1936 Film)?",
                "depends_on": [],
                "operation": "lookup",
                "semantic_edge_ids": ["p2_e1"],
            },
            {
                "id": "q4",
                "question": "What is the nationality of q3's answer?",
                "depends_on": ["q3"],
                "operation": "lookup",
                "semantic_edge_ids": ["p2_e2"],
            },
            {
                "id": "q5",
                "question": "Do q2's answer and q4's answer indicate that the two directors share the same nationality?",
                "depends_on": ["q2", "q4"],
                "operation": "verify",
                "semantic_edge_ids": [],
            },
        ],
    }


def _older_payload() -> dict[str, Any]:
    return {
        "semantic_reasoning_paths": [
            _semantic_path("p1", ["Ryan Tubridy", "older"], ["p1_e1"]),
            _semantic_path("p2", ["Mauro Massironi", "older"], ["p2_e1"]),
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "When was Ryan Tubridy born?",
                "depends_on": [],
                "operation": "lookup",
                "semantic_edge_ids": ["p1_e1"],
            },
            {
                "id": "q2",
                "question": "When was Mauro Massironi born?",
                "depends_on": [],
                "operation": "lookup",
                "semantic_edge_ids": ["p2_e1"],
            },
        ],
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
