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
        for forbidden in (
            "masked_question",
            "normalized_question",
            "sdp",
            "candidate_paths",
            "candidate_sets",
            "debug",
            "raw",
            "ENTITYA",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertIn('"semantic_nodes"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn('"semantic_edges"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn('"semantic_edge_ids"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"reasoning_steps"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"support_step_ids"', ATOMIC_QUESTION_DAG_SYSTEM)

    def test_prompt_requires_dependency_mentions_in_question_text(self) -> None:
        self.assertIn("question text must explicitly mention every dependency as q1's answer", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("do not write a dependent question that can be read without the dependency", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("Do not add a follow-up question that merely restates or generalizes", ATOMIC_QUESTION_DAG_SYSTEM)

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

    def test_valid_simple_semantic_path_builds_dag(self) -> None:
        result = validate_atomic_question_dag(_a_nest_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([node.id for node in result.nodes], ["q1", "q2"])
        self.assertEqual([node.semantic_edge_ids for node in result.nodes], [("p1_e1",), ("p1_e2",)])
        self.assertEqual(result.nodes[1].depends_on, ("q1",))
        self.assertEqual(result.nodes[1].output_node_id, "p1_n3")
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])
        self.assertEqual(result.leaf_node_ids, ["q2"])
        self.assertIn("question_plan", result.raw_payload)
        self.assertIn("semantic_reasoning_paths", result.raw_payload)
        self.assertEqual(result.nodes[0].to_dict()["support_step_ids"], [])

    def test_path_generator_keeps_step5_input_contract(self) -> None:
        llm = RecordingStep5LLM(_a_nest_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Where does the director of film A Nest Of Noblemen work at?",
            explicit_entities=["A Nest Of Noblemen"],
            global_best_paths=[["A Nest Of Noblemen", "director", "work"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        payload = json.loads(llm.user_prompts[0])
        self.assertEqual(set(payload), {"original_question", "explicit_entities", "global_best_paths"})
        self.assertEqual(payload["global_best_paths"], [["A Nest Of Noblemen", "director", "work"]])

    def test_empty_global_best_path_fails_before_llm_call(self) -> None:
        llm = RecordingStep5LLM(_a_nest_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Question?",
            explicit_entities=[],
            global_best_paths=[],
        )

        self.assertFalse(result.valid)
        self.assertEqual(llm.user_prompts, [])
        self.assertIn("Step5 requires at least one non-empty global_best_paths entry.", result.validation_errors)

    def test_old_actions_payload_does_not_block_path_aligned_output(self) -> None:
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

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes, [])
        self.assertEqual(result.validation_errors, [])

    def test_v2_reasoning_steps_schema_is_preserved_without_step6_validation(self) -> None:
        result = validate_atomic_question_dag(_v2_reasoning_steps_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([node.id for node in result.nodes], ["q1"])
        self.assertEqual(result.nodes[0].support_step_ids, ("p1_s1",))
        self.assertEqual(result.validation_errors, [])

    def test_dependencies_are_not_validated_and_edges_are_still_derived(self) -> None:
        cases = [
            ("future", "q2", ["q3"], {"source": "q3", "target": "q2"}),
            ("self", "q1", ["q1"], {"source": "q1", "target": "q1"}),
            ("unknown", "q1", ["qx"], {"source": "qx", "target": "q1"}),
        ]
        for _, question_id, dependencies, expected_edge in cases:
            with self.subTest(expected_edge=expected_edge):
                payload = _a_nest_payload()
                index = int(question_id[1:]) - 1
                payload["atomic_questions"][index]["depends_on"] = dependencies
                result = validate_atomic_question_dag(payload)
                self.assertTrue(result.valid, result.validation_errors)
                self.assertIn(expected_edge, [edge.to_dict() for edge in result.edges])
                self.assertEqual(result.validation_errors, [])

    def test_self_reference_and_missing_dependency_reference_are_preserved(self) -> None:
        payload = _baby_i_payload()
        payload["atomic_questions"][0]["question"] = "Who stars in the video 'One Last Time' by q1's answer?"

        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[0].question, "Who stars in the video 'One Last Time' by q1's answer?")

        payload = _baby_i_payload()
        payload["atomic_questions"][1]["depends_on"] = []
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[1].depends_on, ())

    def test_unresolved_entity_placeholders_are_preserved_without_step6_validation(self) -> None:
        payload = _a_nest_payload()
        payload["atomic_questions"][0]["question"] = "Who directed ENTITYA?"
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[0].question, "Who directed ENTITYA?")

        payload = _a_nest_payload()
        payload["semantic_reasoning_paths"][0]["semantic_nodes"][0]["label"] = "ENTITYA"
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.raw_payload["semantic_reasoning_paths"][0]["semantic_nodes"][0]["label"], "ENTITYA")

        payload = _a_nest_payload()
        payload["semantic_reasoning_paths"][0]["semantic_edges"][0]["support_tokens"] = ["ENTITYA"]
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.raw_payload["semantic_reasoning_paths"][0]["semantic_edges"][0]["support_tokens"], ["ENTITYA"])

    def test_evidence_tokens_outside_source_token_path_are_preserved(self) -> None:
        payload = _a_nest_payload()
        payload["semantic_reasoning_paths"][0]["semantic_nodes"][0]["path_evidence"] = ["not-in-path"]
        payload["semantic_reasoning_paths"][0]["semantic_edges"][0]["support_tokens"] = ["director", "film title from question"]

        result = validate_atomic_question_dag(payload)

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.raw_payload["semantic_reasoning_paths"][0]["semantic_nodes"][0]["path_evidence"], ["not-in-path"])

    def test_braced_question_reference_is_preserved(self) -> None:
        payload = _a_nest_payload()
        payload["atomic_questions"][1]["question"] = "Where does {{q1}} work?"

        result = validate_atomic_question_dag(payload)

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[1].question, "Where does {{q1}} work?")

    def test_semantic_edge_fields_are_not_validated(self) -> None:
        payload = _a_nest_payload()
        payload["atomic_questions"][0]["semantic_edge_ids"] = []
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[0].semantic_edge_ids, ())

        payload = _a_nest_payload()
        payload["atomic_questions"][0]["semantic_edge_ids"] = ["p9_e1"]
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[0].semantic_edge_ids, ("p9_e1",))

        payload = _a_nest_payload()
        payload["atomic_questions"][0]["output_node_id"] = "p9_n1"
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[0].output_node_id, "p9_n1")

        payload = _a_nest_payload()
        payload["semantic_reasoning_paths"][0]["semantic_edges"][0]["source"] = "p9_n1"
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.raw_payload["semantic_reasoning_paths"][0]["semantic_edges"][0]["source"], "p9_n1")

        payload = _a_nest_payload()
        payload["semantic_reasoning_paths"][0]["semantic_edges"][1]["condition_node_ids"] = ["p9_n1"]
        result = validate_atomic_question_dag(payload)
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.raw_payload["semantic_reasoning_paths"][0]["semantic_edges"][1]["condition_node_ids"], ["p9_n1"])

    def test_final_answer_type_mismatch_is_not_validated(self) -> None:
        payload = _a_nest_payload()
        payload["question_plan"]["final_answer_type"] = "person"

        result = validate_atomic_question_dag(payload)

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.raw_payload["question_plan"]["final_answer_type"], "person")

    def test_fort_deposit_case_uses_county_answer_directly(self) -> None:
        result = validate_atomic_question_dag(_fort_deposit_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(
            [node.question for node in result.nodes],
            ["What county is Fort Deposit located in?", "What is the capital of q1's answer?"],
        )
        self.assertNotIn("county of q1's answer", result.nodes[1].question)
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])

    def test_baby_i_missing_path_semantics_case(self) -> None:
        result = validate_atomic_question_dag(
            _baby_i_payload(),
            global_best_paths=[["One Last Time", "video", "stars", "Who"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        path = result.raw_payload["semantic_reasoning_paths"][0]
        self.assertEqual(path["semantic_edges"][0]["evidence_status"], "question_required")
        self.assertEqual(path["semantic_edges"][1]["condition_node_ids"], ["p1_n2"])
        self.assertEqual(
            [node.question for node in result.nodes],
            ["Who is the performer of Baby I?", "Who stars in the video 'One Last Time' by q1's answer?"],
        )
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])

    def test_possessive_wh_case_preserves_possessor_semantics(self) -> None:
        result = validate_atomic_question_dag(_possessive_wh_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(
            [node.question for node in result.nodes],
            ["Who played Susie in miracle on 34th street?", "Whose sister is q1's answer?"],
        )
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])

        bad = _possessive_wh_payload()
        bad["atomic_questions"][0]["question"] = "Who is the sister of the person who played Susie?"
        self.assertTrue(validate_atomic_question_dag(bad).valid)

    def test_bad_baby_i_token_path_relabeling_is_preserved_without_step6_validation(self) -> None:
        result = validate_atomic_question_dag(_bad_baby_i_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.validation_errors, [])

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


def _plan(final_answer_intent: str, final_answer_type: str, constraints: list[str] | None = None, variables: list[str] | None = None) -> dict[str, Any]:
    return {
        "final_answer_intent": final_answer_intent,
        "final_answer_type": final_answer_type,
        "required_constraints": constraints or [],
        "required_intermediate_variables": variables or [],
    }


def _node(
    node_id: str,
    label: str,
    kind: str,
    output_type: str,
    origin: str,
    path_evidence: list[str],
    question_evidence: list[str],
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "kind": kind,
        "output_type": output_type,
        "origin": origin,
        "path_evidence": path_evidence,
        "question_evidence": question_evidence,
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    support_tokens: list[str],
    question_evidence: list[str],
    atomic_question_hint: str,
    *,
    condition_node_ids: list[str] | None = None,
    edge_type: str = "lookup",
    evidence_status: str = "path_grounded",
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "condition_node_ids": condition_node_ids or [],
        "relation": relation,
        "edge_type": edge_type,
        "evidence_status": evidence_status,
        "support_tokens": support_tokens,
        "question_evidence": question_evidence,
        "atomic_question_hint": atomic_question_hint,
    }


def _path(branch_id: str, source_token_path: list[str], nodes: list[dict[str, Any]], edges: list[dict[str, Any]], terminal_node_id: str) -> dict[str, Any]:
    return {
        "branch_id": branch_id,
        "source_token_path": source_token_path,
        "semantic_nodes": nodes,
        "semantic_edges": edges,
        "terminal_node_id": terminal_node_id,
        "folded_or_discarded_tokens": [],
    }


def _question(
    question_id: str,
    text: str,
    depends_on: list[str],
    semantic_edge_ids: list[str],
    output_node_id: str,
    output_type: str,
    operation: str = "lookup",
) -> dict[str, Any]:
    return {
        "id": question_id,
        "question": text,
        "depends_on": depends_on,
        "operation": operation,
        "semantic_edge_ids": semantic_edge_ids,
        "output_node_id": output_node_id,
        "output_type": output_type,
    }


def _a_nest_payload() -> dict[str, Any]:
    return {
        "question_plan": _plan(
            "find where the director of A Nest Of Noblemen works",
            "place",
            ["director of film A Nest Of Noblemen"],
            ["director of A Nest Of Noblemen", "workplace of director"],
        ),
        "semantic_reasoning_paths": [
            _path(
                "p1",
                ["A Nest Of Noblemen", "director", "work"],
                [
                    _node("p1_n1", "A Nest Of Noblemen", "entity", "work", "explicit_entity", ["A Nest Of Noblemen"], ["A Nest Of Noblemen"]),
                    _node("p1_n2", "director of A Nest Of Noblemen", "intermediate_variable", "person", "derived_variable", ["director"], ["director of film A Nest Of Noblemen"]),
                    _node("p1_n3", "workplace of director", "value_slot", "place", "derived_variable", ["work"], ["Where does the director work at"]),
                ],
                [
                    _edge("p1_e1", "p1_n1", "p1_n2", "director of film", ["A Nest Of Noblemen", "director"], ["director of film A Nest Of Noblemen"], "Who directed A Nest Of Noblemen?"),
                    _edge("p1_e2", "p1_n2", "p1_n3", "works at", ["director", "work"], ["Where does the director work at"], "Where does the director work?"),
                ],
                "p1_n3",
            )
        ],
        "atomic_questions": [
            _question("q1", "Who directed A Nest Of Noblemen?", [], ["p1_e1"], "p1_n2", "person"),
            _question("q2", "Where does q1's answer work?", ["q1"], ["p1_e2"], "p1_n3", "place"),
        ],
    }


def _fort_deposit_payload() -> dict[str, Any]:
    return {
        "question_plan": _plan("find the capital of the county containing Fort Deposit", "place", ["Fort Deposit is located in the county"], ["county containing Fort Deposit"]),
        "semantic_reasoning_paths": [
            _path(
                "p1",
                ["Fort Deposit", "located", "county", "capital"],
                [
                    _node("p1_n1", "Fort Deposit", "entity", "place", "explicit_entity", ["Fort Deposit"], ["Fort Deposit"]),
                    _node("p1_n2", "county containing Fort Deposit", "intermediate_variable", "place", "derived_variable", ["located", "county"], ["county where Fort Deposit is located"]),
                    _node("p1_n3", "capital of county", "value_slot", "place", "derived_variable", ["capital"], ["capital of the county"]),
                ],
                [
                    _edge("p1_e1", "p1_n1", "p1_n2", "located in county", ["Fort Deposit", "located", "county"], ["county where Fort Deposit is located"], "What county is Fort Deposit located in?"),
                    _edge("p1_e2", "p1_n2", "p1_n3", "capital of county", ["county", "capital"], ["capital of the county"], "What is the capital of the county?"),
                ],
                "p1_n3",
            )
        ],
        "atomic_questions": [
            _question("q1", "What county is Fort Deposit located in?", [], ["p1_e1"], "p1_n2", "place"),
            _question("q2", "What is the capital of q1's answer?", ["q1"], ["p1_e2"], "p1_n3", "place"),
        ],
    }


def _baby_i_payload() -> dict[str, Any]:
    return {
        "question_plan": _plan("find who stars in the video 'One Last Time' by the performer of Baby I", "person", ["video 'One Last Time'", "performer of Baby I"], ["performer of Baby I"]),
        "semantic_reasoning_paths": [
            _path(
                "p1",
                ["One Last Time", "video", "stars", "Who"],
                [
                    _node("p1_n1", "Baby I", "entity", "work", "question_required", [], ["Baby I"]),
                    _node("p1_n2", "performer of Baby I", "intermediate_variable", "person", "question_required", [], ["performer of Baby I"]),
                    _node("p1_n3", "video 'One Last Time'", "entity", "work", "explicit_entity", ["One Last Time", "video"], ["video 'One Last Time'"]),
                    _node("p1_n4", "person who stars in the video", "answer_slot", "person", "derived_variable", ["stars", "Who"], ["Who stars in the video"]),
                ],
                [
                    _edge("p1_e1", "p1_n1", "p1_n2", "performer of song", [], ["performer of Baby I"], "Who is the performer of Baby I?", evidence_status="question_required"),
                    _edge(
                        "p1_e2",
                        "p1_n3",
                        "p1_n4",
                        "stars in video constrained by performer",
                        ["One Last Time", "video", "stars"],
                        ["Who stars in the video 'One Last Time' by the performer of Baby I"],
                        "Who stars in the video 'One Last Time' by the performer?",
                        condition_node_ids=["p1_n2"],
                        evidence_status="mixed",
                    ),
                ],
                "p1_n4",
            )
        ],
        "atomic_questions": [
            _question("q1", "Who is the performer of Baby I?", [], ["p1_e1"], "p1_n2", "person"),
            _question("q2", "Who stars in the video 'One Last Time' by q1's answer?", ["q1"], ["p1_e2"], "p1_n4", "person"),
        ],
    }


def _possessive_wh_payload() -> dict[str, Any]:
    return {
        "question_plan": _plan("find whose sister played Susie in miracle on 34th street", "person", ["played Susie", "miracle on 34th street"], ["actor who played Susie"]),
        "semantic_reasoning_paths": [
            _path(
                "p1",
                ["Whose", "sister", "played", "Susie"],
                [
                    _node("p1_n1", "Susie in miracle on 34th street", "entity", "value", "question_required", ["Susie"], ["Susie in miracle on 34th street"]),
                    _node("p1_n2", "actor who played Susie", "intermediate_variable", "person", "derived_variable", ["played", "Susie"], ["played Susie in miracle on 34th street"]),
                    _node("p1_n3", "person whose sister is the actor", "answer_slot", "person", "derived_variable", ["Whose", "sister"], ["Whose sister"]),
                ],
                [
                    _edge("p1_e1", "p1_n1", "p1_n2", "played by in work", ["played", "Susie"], ["played Susie in miracle on 34th street"], "Who played Susie in miracle on 34th street?", evidence_status="mixed"),
                    _edge("p1_e2", "p1_n2", "p1_n3", "is sister of whose person", ["Whose", "sister"], ["Whose sister"], "Whose sister is the actor?"),
                ],
                "p1_n3",
            )
        ],
        "atomic_questions": [
            _question("q1", "Who played Susie in miracle on 34th street?", [], ["p1_e1"], "p1_n2", "person"),
            _question("q2", "Whose sister is q1's answer?", ["q1"], ["p1_e2"], "p1_n3", "person"),
        ],
    }


def _bad_baby_i_payload() -> dict[str, Any]:
    source_path = ["Baby I", "performer", "One Last Time", "video", "stars", "Who"]
    nodes = [
        _node(f"p1_n{index}", token, "intermediate_variable", "value", "path_evidence", [token], [token])
        for index, token in enumerate(source_path, start=1)
    ]
    edges = [
        _edge(f"p1_e{index}", f"p1_n{index}", f"p1_n{index + 1}", "related to", [source_path[index - 1], source_path[index]], [source_path[index]], "What is related?")
        for index in range(1, len(source_path))
    ]
    return {
        "question_plan": _plan("find who stars in the video One Last Time by the performer of Baby I", "person", ["One Last Time", "performer of Baby I"], []),
        "semantic_reasoning_paths": [_path("p1", source_path, nodes, edges, "p1_n6")],
        "atomic_questions": [_question("q1", "Who is the performer of Baby I?", [], ["p1_e1"], "p1_n2", "person")],
    }


def _parallel_nationality_payload() -> dict[str, Any]:
    return {
        "question_plan": _plan("verify whether the two film directors share the same nationality", "boolean", ["same nationality", "director of each film"], ["director of each film", "nationality of each director"]),
        "semantic_reasoning_paths": [
            _path(
                "p1",
                ["Ten9Eight: Shoot For The Moon", "director", "nationality"],
                [
                    _node("p1_n1", "Ten9Eight: Shoot For The Moon", "entity", "work", "explicit_entity", ["Ten9Eight: Shoot For The Moon"], ["Ten9Eight: Shoot For The Moon"]),
                    _node("p1_n2", "director of Ten9Eight: Shoot For The Moon", "intermediate_variable", "person", "derived_variable", ["director"], ["director of film Ten9Eight: Shoot For The Moon"]),
                    _node("p1_n3", "nationality of first director", "value_slot", "value", "derived_variable", ["nationality"], ["same nationality"]),
                ],
                [
                    _edge("p1_e1", "p1_n1", "p1_n2", "director of film", ["Ten9Eight: Shoot For The Moon", "director"], ["director of film Ten9Eight: Shoot For The Moon"], "Who directed Ten9Eight: Shoot For The Moon?"),
                    _edge("p1_e2", "p1_n2", "p1_n3", "nationality of director", ["director", "nationality"], ["same nationality"], "What is the nationality of the director?"),
                ],
                "p1_n3",
            ),
            _path(
                "p2",
                ["Sabotage (1936 Film)", "director", "nationality"],
                [
                    _node("p2_n1", "Sabotage (1936 Film)", "entity", "work", "explicit_entity", ["Sabotage (1936 Film)"], ["Sabotage (1936 Film)"]),
                    _node("p2_n2", "director of Sabotage (1936 Film)", "intermediate_variable", "person", "derived_variable", ["director"], ["director of film Sabotage (1936 Film)"]),
                    _node("p2_n3", "nationality of second director", "value_slot", "value", "derived_variable", ["nationality"], ["same nationality"]),
                ],
                [
                    _edge("p2_e1", "p2_n1", "p2_n2", "director of film", ["Sabotage (1936 Film)", "director"], ["director of film Sabotage (1936 Film)"], "Who directed Sabotage (1936 Film)?"),
                    _edge("p2_e2", "p2_n2", "p2_n3", "nationality of director", ["director", "nationality"], ["same nationality"], "What is the nationality of the director?"),
                ],
                "p2_n3",
            ),
        ],
        "atomic_questions": [
            _question("q1", "Who directed Ten9Eight: Shoot For The Moon?", [], ["p1_e1"], "p1_n2", "person"),
            _question("q2", "What is the nationality of q1's answer?", ["q1"], ["p1_e2"], "p1_n3", "value"),
            _question("q3", "Who directed Sabotage (1936 Film)?", [], ["p2_e1"], "p2_n2", "person"),
            _question("q4", "What is the nationality of q3's answer?", ["q3"], ["p2_e2"], "p2_n3", "value"),
            _question("q5", "Do q2's answer and q4's answer indicate that the two directors share the same nationality?", ["q2", "q4"], [], "", "boolean", "verify"),
        ],
    }


def _older_payload() -> dict[str, Any]:
    return {
        "question_plan": _plan("select which person is older", "person", ["older comparison between Ryan Tubridy and Mauro Massironi"], ["birth date of Ryan Tubridy", "birth date of Mauro Massironi"]),
        "semantic_reasoning_paths": [
            _path(
                "p1",
                ["Mauro Massironi", "older", "Ryan Tubridy"],
                [
                    _node("p1_n1", "Ryan Tubridy", "entity", "person", "explicit_entity", ["Ryan Tubridy"], ["Ryan Tubridy"]),
                    _node("p1_n2", "birth date of Ryan Tubridy", "value_slot", "date", "derived_variable", ["older"], ["older"]),
                ],
                [_edge("p1_e1", "p1_n1", "p1_n2", "birth date for older comparison", ["Ryan Tubridy", "older"], ["Ryan Tubridy", "older"], "When was Ryan Tubridy born?")],
                "p1_n2",
            ),
            _path(
                "p2",
                ["Mauro Massironi", "older", "Ryan Tubridy"],
                [
                    _node("p2_n1", "Mauro Massironi", "entity", "person", "explicit_entity", ["Mauro Massironi"], ["Mauro Massironi"]),
                    _node("p2_n2", "birth date of Mauro Massironi", "value_slot", "date", "derived_variable", ["older"], ["older"]),
                ],
                [_edge("p2_e1", "p2_n1", "p2_n2", "birth date for older comparison", ["Mauro Massironi", "older"], ["Mauro Massironi", "older"], "When was Mauro Massironi born?")],
                "p2_n2",
            ),
        ],
        "atomic_questions": [
            _question("q1", "When was Ryan Tubridy born?", [], ["p1_e1"], "p1_n2", "date"),
            _question("q2", "When was Mauro Massironi born?", [], ["p2_e1"], "p2_n2", "date"),
        ],
    }


def _v2_reasoning_steps_payload() -> dict[str, Any]:
    return {
        "question_plan": _plan("find where the director works", "place", [], []),
        "semantic_reasoning_paths": [
            {
                "branch_id": "p1",
                "source_token_path": ["A Nest Of Noblemen", "director", "work"],
                "reasoning_steps": [
                    {
                        "id": "p1_s1",
                        "path_evidence": ["A Nest Of Noblemen", "director"],
                        "question_evidence": ["director of film A Nest Of Noblemen"],
                        "known_inputs": ["A Nest Of Noblemen"],
                        "operation": "find director",
                        "output": "director",
                        "output_type": "person",
                        "step_type": "lookup",
                        "evidence_status": "path_grounded",
                    }
                ],
            }
        ],
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who directed A Nest Of Noblemen?",
                "depends_on": [],
                "operation": "lookup",
                "support_step_ids": ["p1_s1"],
                "output_type": "person",
            }
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
