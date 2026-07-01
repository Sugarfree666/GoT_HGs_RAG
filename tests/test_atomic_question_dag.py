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
        self.assertIn("reasoning_steps", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("support_step_ids", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_nodes"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_edges"', ATOMIC_QUESTION_DAG_SYSTEM)

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
        self.assertIn("question_plan", result.raw_payload)
        self.assertIn("semantic_reasoning_paths", result.raw_payload)
        self.assertEqual(result.nodes[0].operation, "lookup")
        self.assertEqual(result.nodes[0].support_step_ids, ("p1_s1",))
        self.assertEqual(result.nodes[0].to_dict()["support_step_ids"], ["p1_s1"])
        self.assertEqual(result.nodes[0].to_dict()["semantic_edge_ids"], [])

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

    def test_v1_semantic_graph_schema_is_rejected_for_path_aligned_step5(self) -> None:
        payload = {
            "question_plan": {
                "final_answer_intent": "find the birth year",
                "final_answer_type": "date",
                "must_preserve_constraints": [],
            },
            "semantic_reasoning_paths": [
                {
                    "branch_id": "p1",
                    "source_token_path": ["Johnny Majors", "defeated", "player", "born", "year"],
                    "semantic_nodes": [{"id": "p1_n1", "label": "Johnny Majors", "kind": "entity"}],
                    "semantic_edges": [],
                    "terminal_node_id": "p1_n1",
                }
            ],
            "atomic_questions": [
                {
                    "id": "q1",
                    "question": "Who defeated Johnny Majors?",
                    "depends_on": [],
                    "operation": "lookup",
                    "support_step_ids": ["p1_s1"],
                    "output_type": "person",
                }
            ],
        }

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        joined = "; ".join(result.validation_errors)
        self.assertIn("Step5 V2 uses reasoning_steps, not semantic_nodes/semantic_edges", joined)
        self.assertIn("reasoning_steps must be a non-empty list", joined)

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
        payload["semantic_reasoning_paths"][0]["reasoning_steps"][0]["output"] = "ENTITYA"

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("output contains unresolved ENTITY placeholder", "; ".join(result.validation_errors))

    def test_unresolved_entity_placeholder_in_path_evidence_is_rejected(self) -> None:
        payload = _johnny_payload()
        payload["semantic_reasoning_paths"][0]["reasoning_steps"][0]["path_evidence"] = ["ENTITYA"]

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("path_evidence contains unresolved ENTITY placeholder", "; ".join(result.validation_errors))

    def test_path_evidence_must_be_copied_from_source_token_path(self) -> None:
        payload = _johnny_payload()
        payload["semantic_reasoning_paths"][0]["reasoning_steps"][0]["path_evidence"] = ["not-in-path"]

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("token not copied from source_token_path", "; ".join(result.validation_errors))

    def test_braced_question_reference_is_rejected(self) -> None:
        payload = _johnny_payload()
        payload["atomic_questions"][1]["question"] = "What year was {{q1}} born?"

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("must use qN's answer references", "; ".join(result.validation_errors))

    def test_unknown_support_step_id_is_rejected(self) -> None:
        payload = _johnny_payload()
        payload["atomic_questions"][0]["support_step_ids"] = ["p9_s1"]

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("unknown support_step_id 'p9_s1'", "; ".join(result.validation_errors))

    def test_lookup_question_requires_support_step_id(self) -> None:
        payload = _johnny_payload()
        payload["atomic_questions"][0]["support_step_ids"] = []

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("lookup questions must include at least one support_step_id", "; ".join(result.validation_errors))

    def test_path_grounded_step_requires_path_evidence(self) -> None:
        payload = _johnny_payload()
        payload["semantic_reasoning_paths"][0]["reasoning_steps"][0]["path_evidence"] = []

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("path_evidence may be empty only for question_only_required or operator", "; ".join(result.validation_errors))

    def test_final_answer_type_mismatch_is_rejected(self) -> None:
        payload = _johnny_payload()
        payload["question_plan"]["final_answer_type"] = "person"

        result = validate_atomic_question_dag(payload)

        self.assertFalse(result.valid)
        self.assertIn("final leaf output_type 'date' does not match question_plan.final_answer_type 'person'", "; ".join(result.validation_errors))

    def test_bad_baby_i_token_path_relabeling_is_rejected(self) -> None:
        result = validate_atomic_question_dag(_bad_baby_i_payload())

        self.assertFalse(result.valid)
        self.assertIn("likely token-path relabeling", "; ".join(result.validation_errors))

    def test_correct_baby_i_case_is_compressed_and_builds_expected_edge(self) -> None:
        result = validate_atomic_question_dag(
            _baby_i_payload(),
            global_best_paths=[["Baby I", "performer", "One Last Time", "video", "stars", "Who"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        steps = result.raw_payload["semantic_reasoning_paths"][0]["reasoning_steps"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["path_evidence"], ["Baby I", "performer"])
        self.assertEqual(steps[1]["known_inputs"], ["One Last Time", "p1_s1 output"])
        self.assertEqual([node.question for node in result.nodes], ["Who is the performer of Baby I?", "Who stars in the video One Last Time by q1's answer?"])
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])

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


def _reasoning_path(
    branch_id: str,
    source_token_path: list[str],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "branch_id": branch_id,
        "source_token_path": source_token_path,
        "reasoning_steps": steps,
    }


def _johnny_payload() -> dict[str, Any]:
    return {
        "question_plan": {
            "final_answer_intent": "find the birth year of the player who defeated Johnny Majors",
            "final_answer_type": "date",
            "must_preserve_constraints": ["defeated Johnny Majors", "birth year"],
        },
        "semantic_reasoning_paths": [
            _reasoning_path(
                "p1",
                ["Johnny Majors", "defeated", "player", "born", "year"],
                [
                    {
                        "id": "p1_s1",
                        "path_evidence": ["Johnny Majors", "defeated", "player"],
                        "question_evidence": ["player who defeated Johnny Majors"],
                        "known_inputs": ["Johnny Majors"],
                        "operation": "find the player who defeated Johnny Majors",
                        "output": "player who defeated Johnny Majors",
                        "output_type": "person",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    },
                    {
                        "id": "p1_s2",
                        "path_evidence": ["player", "born", "year"],
                        "question_evidence": ["born in what year"],
                        "known_inputs": ["p1_s1 output"],
                        "operation": "find the birth year of the player found in p1_s1",
                        "output": "birth year of player",
                        "output_type": "date",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    },
                ],
            )
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who defeated Johnny Majors for the Heisman Trophy in 1956?",
                "depends_on": [],
                "operation": "lookup",
                "support_step_ids": ["p1_s1"],
                "output_type": "person",
            },
            {
                "id": "q2",
                "question": "What year was q1's answer born?",
                "depends_on": ["q1"],
                "operation": "lookup",
                "support_step_ids": ["p1_s2"],
                "output_type": "date",
            },
        ],
    }


def _parallel_nationality_payload() -> dict[str, Any]:
    return {
        "question_plan": {
            "final_answer_intent": "verify whether the two film directors share the same nationality",
            "final_answer_type": "boolean",
            "must_preserve_constraints": ["director of each film", "same nationality"],
        },
        "semantic_reasoning_paths": [
            _reasoning_path(
                "p1",
                ["Ten9Eight: Shoot For The Moon", "director", "nationality"],
                [
                    {
                        "id": "p1_s1",
                        "path_evidence": ["Ten9Eight: Shoot For The Moon", "director"],
                        "question_evidence": ["director of film Ten9Eight: Shoot For The Moon"],
                        "known_inputs": ["Ten9Eight: Shoot For The Moon"],
                        "operation": "find the director of the film Ten9Eight: Shoot For The Moon",
                        "output": "director of Ten9Eight: Shoot For The Moon",
                        "output_type": "person",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    },
                    {
                        "id": "p1_s2",
                        "path_evidence": ["director", "nationality"],
                        "question_evidence": ["same nationality"],
                        "known_inputs": ["p1_s1 output"],
                        "operation": "find the nationality of the director found in p1_s1",
                        "output": "nationality of first director",
                        "output_type": "value",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    },
                ],
            ),
            _reasoning_path(
                "p2",
                ["Sabotage (1936 Film)", "director", "nationality"],
                [
                    {
                        "id": "p2_s1",
                        "path_evidence": ["Sabotage (1936 Film)", "director"],
                        "question_evidence": ["director of film Sabotage (1936 Film)"],
                        "known_inputs": ["Sabotage (1936 Film)"],
                        "operation": "find the director of the film Sabotage (1936 Film)",
                        "output": "director of Sabotage (1936 Film)",
                        "output_type": "person",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    },
                    {
                        "id": "p2_s2",
                        "path_evidence": ["director", "nationality"],
                        "question_evidence": ["same nationality"],
                        "known_inputs": ["p2_s1 output"],
                        "operation": "find the nationality of the director found in p2_s1",
                        "output": "nationality of second director",
                        "output_type": "value",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    },
                ],
            ),
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who directed Ten9Eight: Shoot For The Moon?",
                "depends_on": [],
                "operation": "lookup",
                "support_step_ids": ["p1_s1"],
                "output_type": "person",
            },
            {
                "id": "q2",
                "question": "What is the nationality of q1's answer?",
                "depends_on": ["q1"],
                "operation": "lookup",
                "support_step_ids": ["p1_s2"],
                "output_type": "value",
            },
            {
                "id": "q3",
                "question": "Who directed Sabotage (1936 Film)?",
                "depends_on": [],
                "operation": "lookup",
                "support_step_ids": ["p2_s1"],
                "output_type": "person",
            },
            {
                "id": "q4",
                "question": "What is the nationality of q3's answer?",
                "depends_on": ["q3"],
                "operation": "lookup",
                "support_step_ids": ["p2_s2"],
                "output_type": "value",
            },
            {
                "id": "q5",
                "question": "Do q2's answer and q4's answer indicate that the two directors share the same nationality?",
                "depends_on": ["q2", "q4"],
                "operation": "verify",
                "support_step_ids": [],
                "output_type": "boolean",
            },
        ],
    }


def _older_payload() -> dict[str, Any]:
    return {
        "question_plan": {
            "final_answer_intent": "select which person is older",
            "final_answer_type": "person",
            "must_preserve_constraints": ["older comparison between Ryan Tubridy and Mauro Massironi"],
        },
        "semantic_reasoning_paths": [
            _reasoning_path(
                "p1",
                ["Mauro Massironi", "older", "Ryan Tubridy"],
                [
                    {
                        "id": "p1_s1",
                        "path_evidence": ["Ryan Tubridy", "older"],
                        "question_evidence": ["Ryan Tubridy", "older"],
                        "known_inputs": ["Ryan Tubridy"],
                        "operation": "find the birth date needed for an older-person comparison",
                        "output": "birth date of Ryan Tubridy",
                        "output_type": "date",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    }
                ],
            ),
            _reasoning_path(
                "p2",
                ["Mauro Massironi", "older", "Ryan Tubridy"],
                [
                    {
                        "id": "p2_s1",
                        "path_evidence": ["Mauro Massironi", "older"],
                        "question_evidence": ["Mauro Massironi", "older"],
                        "known_inputs": ["Mauro Massironi"],
                        "operation": "find the birth date needed for an older-person comparison",
                        "output": "birth date of Mauro Massironi",
                        "output_type": "date",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    }
                ],
            ),
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "When was Ryan Tubridy born?",
                "depends_on": [],
                "operation": "lookup",
                "support_step_ids": ["p1_s1"],
                "output_type": "date",
            },
            {
                "id": "q2",
                "question": "When was Mauro Massironi born?",
                "depends_on": [],
                "operation": "lookup",
                "support_step_ids": ["p2_s1"],
                "output_type": "date",
            },
        ],
    }


def _bad_baby_i_payload() -> dict[str, Any]:
    source_path = ["Baby I", "performer", "One Last Time", "video", "stars", "Who"]
    outputs = ["performer", "One Last Time", "video", "stars", "Who"]
    return {
        "question_plan": {
            "final_answer_intent": "find who stars in the video One Last Time by the performer of Baby I",
            "final_answer_type": "person",
            "must_preserve_constraints": ["One Last Time", "performer of Baby I"],
        },
        "semantic_reasoning_paths": [
            _reasoning_path(
                "p1",
                source_path,
                [
                    {
                        "id": f"p1_s{index}",
                        "path_evidence": [output],
                        "question_evidence": [output],
                        "known_inputs": ["Baby I"] if index == 1 else [f"p1_s{index - 1} output"],
                        "operation": f"find {output}",
                        "output": output,
                        "output_type": "person" if output in {"performer", "Who"} else "value",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    }
                    for index, output in enumerate(outputs, start=1)
                ],
            )
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who is the performer of Baby I?",
                "depends_on": [],
                "operation": "lookup",
                "support_step_ids": ["p1_s1"],
                "output_type": "person",
            }
        ],
    }


def _baby_i_payload() -> dict[str, Any]:
    return {
        "question_plan": {
            "final_answer_intent": "find who stars in the video One Last Time by the performer of Baby I",
            "final_answer_type": "person",
            "must_preserve_constraints": ["One Last Time", "performer of Baby I"],
        },
        "semantic_reasoning_paths": [
            _reasoning_path(
                "p1",
                ["Baby I", "performer", "One Last Time", "video", "stars", "Who"],
                [
                    {
                        "id": "p1_s1",
                        "path_evidence": ["Baby I", "performer"],
                        "question_evidence": ["performer of Baby I"],
                        "known_inputs": ["Baby I"],
                        "operation": "find the performer of the song Baby I",
                        "output": "performer of Baby I",
                        "output_type": "person",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    },
                    {
                        "id": "p1_s2",
                        "path_evidence": ["One Last Time", "video", "stars"],
                        "question_evidence": ["stars in the video One Last Time by the performer of Baby I"],
                        "known_inputs": ["One Last Time", "p1_s1 output"],
                        "operation": "find who stars in the video One Last Time by the performer found in p1_s1",
                        "output": "person who stars in the video One Last Time",
                        "output_type": "person",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    },
                ],
            )
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who is the performer of Baby I?",
                "depends_on": [],
                "operation": "lookup",
                "support_step_ids": ["p1_s1"],
                "output_type": "person",
            },
            {
                "id": "q2",
                "question": "Who stars in the video One Last Time by q1's answer?",
                "depends_on": ["q1"],
                "operation": "lookup",
                "support_step_ids": ["p1_s2"],
                "output_type": "person",
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
