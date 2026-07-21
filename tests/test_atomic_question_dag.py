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
    prompt_input_payload,
    restore_entity_paths,
    restore_global_best_path,
    restore_global_best_paths,
    validate_atomic_question_dag,
)
from models import MaskMapping  # noqa: E402


class AtomicQuestionDAGTest(unittest.TestCase):
    def test_prompt_contract_uses_step4_paths_as_hints_only(self) -> None:
        payload = prompt_input_payload(
            original_question="Question?",
            explicit_entities=["When The Stars Go Blue"],
            global_best_paths=[["When The Stars Go Blue", "song", "performer", "nationality"]],
        )

        self.assertEqual(set(payload), {"original_question", "topic_entities", "step4_paths"})
        self.assertEqual(payload["topic_entities"], ["When The Stars Go Blue"])
        self.assertEqual(payload["step4_paths"], [["When The Stars Go Blue", "song", "performer", "nationality"]])
        self.assertNotIn("semantic_reasoning_paths", json.dumps(payload, ensure_ascii=False))

        self.assertIn("complex-question decomposition", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("step4_paths are noisy structural hints", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("DAG nodes do not need path support", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("latent-bridge test", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("nationality is not country of birth", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("Direct one-hop", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("Parallel comparison", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("Whose sister played Susie", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("FINAL-ANSWER CONTRACT", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("CONSTRAINT BRANCH COMPLETENESS", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("BINDING INTEGRITY", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("unique leaf", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("not a boolean verification", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("Which country artist recorded Heartbreak Hurricane", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("Whose sister is q1's answer", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("series with The Bag or the Bat", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("Joint location constraints", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("birth city of the composer of X", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("silently audit", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("qN's answer", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_reasoning_paths"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_nodes"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_edges"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_edge_ids"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"output_node_id"', ATOMIC_QUESTION_DAG_SYSTEM)

    def test_direct_atomic_questions_build_dag(self) -> None:
        result = validate_atomic_question_dag(_bridge_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([node.id for node in result.nodes], ["q1", "q2"])
        self.assertEqual([node.question for node in result.nodes], [
            "Who performed When The Stars Go Blue?",
            "What is the nationality of q1's answer?",
        ])
        self.assertEqual(result.nodes[1].depends_on, ("q1",))
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])
        self.assertEqual(result.leaf_node_ids, ["q2"])
        self.assertEqual(result.raw_payload, _bridge_payload())

    def test_missing_node_id_or_question_is_invalid(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {"question": "Who performed When The Stars Go Blue?"},
                    {"id": "q2", "depends_on": []},
                ]
            }
        )

        self.assertFalse(result.valid)
        self.assertTrue(any(".id must be a non-empty string" in error for error in result.validation_errors))
        self.assertTrue(any(".question must be a non-empty string" in error for error in result.validation_errors))

    def test_only_atomic_questions_are_required(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "Which city shares a county with Helvetia?",
                        "depends_on": [],
                        "operation": "lookup",
                        "output_type": "place",
                    }
                ]
            }
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.leaf_node_ids, ["q1"])

    def test_empty_or_missing_atomic_questions_are_invalid(self) -> None:
        for payload in ({"atomic_questions": []}, {"actions": [{"id": "q1", "question": "What is A?"}]}):
            with self.subTest(payload=payload):
                result = validate_atomic_question_dag(payload)
                self.assertFalse(result.valid)
                self.assertIn("atomic_questions must be a non-empty list.", result.validation_errors)

    def test_nested_atomic_question_dag_envelope_is_invalid(self) -> None:
        result = validate_atomic_question_dag({"atomic_question_dag": {"atomic_questions": _bridge_payload()["atomic_questions"]}})

        self.assertFalse(result.valid)
        self.assertIn("atomic_questions must be a non-empty list.", result.validation_errors)

    def test_path_hints_are_not_used_as_support_requirements(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "Who is the performer of Baby I?",
                        "depends_on": [],
                        "operation": "lookup",
                        "output_type": "person",
                    },
                    {
                        "id": "q2",
                        "question": "Who stars in the video 'One Last Time' by q1's answer?",
                        "depends_on": ["q1"],
                        "operation": "lookup",
                        "output_type": "person",
                    },
                ]
            },
            original_question="Who stars in the video 'One Last Time' by the performer of Baby I?",
            explicit_entities=["One Last Time", "Baby I"],
            global_best_paths=[["One Last Time", "video", "stars", "Who"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])

    def test_candidate_comparison_dag_edges_are_derived_from_dependencies(self) -> None:
        result = validate_atomic_question_dag(_younger_director_payload())

        self.assertTrue(result.valid, result.validation_errors)
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
        self.assertEqual(result.nodes[-1].operation, "select")
        self.assertEqual(result.nodes[-1].output_type, "work")

    def test_path_generator_sends_topic_entities_and_step4_paths(self) -> None:
        llm = RecordingStep5LLM(_bridge_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="What nationality is the performer of the song When The Stars Go Blue?",
            explicit_entities=["When The Stars Go Blue"],
            global_best_paths=[["When The Stars Go Blue", "song", "performer", "nationality"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(llm.system_prompts, [ATOMIC_QUESTION_DAG_SYSTEM])
        payload = json.loads(llm.user_prompts[0])
        self.assertEqual(set(payload), {"original_question", "topic_entities", "step4_paths"})
        self.assertEqual(payload["topic_entities"], ["When The Stars Go Blue"])
        self.assertEqual(payload["step4_paths"], [["When The Stars Go Blue", "song", "performer", "nationality"]])

    def test_dependency_binding_mismatch_is_invalid(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "Who is the singer of Come Away with Me?",
                        "depends_on": [],
                        "operation": "lookup",
                        "output_type": "person",
                    },
                    {
                        "id": "q2",
                        "question": "Who wrote Turn Me On?",
                        "depends_on": ["q1"],
                        "operation": "lookup",
                        "output_type": "person",
                    },
                ]
            },
            original_question="Who wrote Turn Me On by the singer of Come Away with Me?",
            explicit_entities=["Turn Me On", "Come Away with Me"],
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("does not reference that answer" in error for error in result.validation_errors))

    def test_validator_does_not_apply_possessive_semantic_heuristics(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "Who played Susie in Miracle on 34th Street?",
                        "depends_on": [],
                        "operation": "lookup",
                        "output_type": "person",
                    },
                    {
                        "id": "q2",
                        "question": "Who is the sister of q1's answer?",
                        "depends_on": ["q1"],
                        "operation": "lookup",
                        "output_type": "person",
                    },
                ]
            },
            original_question="Whose sister played Susie in Miracle on 34th Street?",
            explicit_entities=["Susie", "Miracle on 34th Street"],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.warnings, [])

    def test_validator_does_not_apply_lifespan_semantic_heuristics(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "When was Ludwig Elsbett born?",
                        "depends_on": [],
                        "operation": "lookup",
                        "output_type": "date",
                    },
                    {
                        "id": "q2",
                        "question": "When was Pamela Ann Rymer born?",
                        "depends_on": [],
                        "operation": "lookup",
                        "output_type": "date",
                    },
                    {
                        "id": "q3",
                        "question": "Based on q1's answer and q2's answer, who lived longer: Ludwig Elsbett or Pamela Ann Rymer?",
                        "depends_on": ["q1", "q2"],
                        "operation": "compare",
                        "output_type": "person",
                    },
                ]
            },
            original_question="Who lived longer, Ludwig Elsbett or Pamela Ann Rymer?",
            explicit_entities=["Ludwig Elsbett", "Pamela Ann Rymer"],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.warnings, [])

    def test_validator_does_not_apply_appositive_semantic_heuristics(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "Who is John Ernest's father-in-law?",
                        "depends_on": [],
                        "operation": "lookup",
                        "output_type": "person",
                    }
                ]
            },
            original_question="Who is the father-in-law of John Ernest, Duke Of Saxe-Eisenach?",
            explicit_entities=["John Ernest", "Duke Of Saxe-Eisenach"],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.warnings, [])

    def test_path_generator_returns_structural_invalidity_from_its_single_call(self) -> None:
        bad_payload = {
            "atomic_questions": [
                {
                    "id": "q1",
                    "question": "Who is the singer of Come Away with Me?",
                    "depends_on": [],
                    "operation": "lookup",
                    "output_type": "person",
                },
                {
                    "id": "q2",
                    "question": "Who wrote Turn Me On?",
                    "depends_on": ["q1"],
                    "operation": "lookup",
                    "output_type": "person",
                },
            ]
        }
        llm = RecordingStep5LLM(bad_payload)

        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Who wrote Turn Me On by the singer of Come Away with Me?",
            explicit_entities=["Turn Me On", "Come Away with Me"],
            global_best_paths=[["Come Away with Me", "singer", "Turn Me On", "wrote", "Who"]],
        )

        self.assertFalse(result.valid)
        self.assertEqual(len(llm.user_prompts), 1)
        self.assertTrue(any("does not reference that answer" in error for error in result.validation_errors))

    def test_validator_rejects_deterministic_dag_structure_errors(self) -> None:
        cases = [
            (
                "duplicate ids",
                {
                    "atomic_questions": [
                        {"id": "q1", "question": "Who is A?", "depends_on": []},
                        {"id": "q1", "question": "Who is B?", "depends_on": []},
                    ]
                },
                "duplicates node id",
            ),
            (
                "unknown dependency",
                {
                    "atomic_questions": [
                        {"id": "q1", "question": "Who is A?", "depends_on": ["q9"]},
                    ]
                },
                "references unknown node",
            ),
            (
                "self dependency",
                {
                    "atomic_questions": [
                        {"id": "q1", "question": "Who is q1's answer?", "depends_on": ["q1"]},
                    ]
                },
                "references itself",
            ),
            (
                "later dependency",
                {
                    "atomic_questions": [
                        {"id": "q1", "question": "Who is q2's answer?", "depends_on": ["q2"]},
                        {"id": "q2", "question": "Who is B?", "depends_on": []},
                    ]
                },
                "references a later node",
            ),
            (
                "unknown answer reference",
                {
                    "atomic_questions": [
                        {"id": "q1", "question": "Who is q9's answer?", "depends_on": ["q9"]},
                    ]
                },
                "references unknown answer",
            ),
            (
                "answer reference missing dependency",
                {
                    "atomic_questions": [
                        {"id": "q1", "question": "Who is A?", "depends_on": []},
                        {"id": "q2", "question": "Where was q1's answer born?", "depends_on": []},
                    ]
                },
                "depends_on does not include it",
            ),
            (
                "unresolved placeholder",
                {
                    "atomic_questions": [
                        {"id": "q1", "question": "Who directed ENTITYA?", "depends_on": []},
                    ]
                },
                "unresolved ENTITY placeholder",
            ),
            (
                "cycle",
                {
                    "atomic_questions": [
                        {"id": "q1", "question": "Who is q2's answer?", "depends_on": ["q2"]},
                        {"id": "q2", "question": "Who is q1's answer?", "depends_on": ["q1"]},
                    ]
                },
                "dependency cycle",
            ),
        ]

        for name, payload, expected_error in cases:
            with self.subTest(name=name):
                result = validate_atomic_question_dag(payload)
                self.assertFalse(result.valid)
                self.assertTrue(any(expected_error in error for error in result.validation_errors))

    def test_empty_global_best_path_fails_before_llm_call(self) -> None:
        llm = RecordingStep5LLM(_bridge_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Question?",
            explicit_entities=[],
            global_best_paths=[],
        )

        self.assertFalse(result.valid)
        self.assertEqual(llm.user_prompts, [])
        self.assertIn("Step5 requires at least one non-empty step4_paths/global_best_paths entry.", result.validation_errors)

    def test_restore_entity_paths_replaces_placeholders_inside_punctuated_tokens(self) -> None:
        paths = [
            SimpleNamespace(path_id="P1", nodes=["ENTITYA!", "director", "ENTITYBish"]),
            SimpleNamespace(path_id="P2", nodes=["ENTITYB.", "born", "ENTITYC/ENTITYD"]),
        ]
        mappings = [
            MaskMapping("ENTITYA", "Ten9Eight: Shoot For The Moon", "entity"),
            MaskMapping("ENTITYB", "Sabotage (1936 Film)", "entity"),
            MaskMapping("ENTITYC", "Fort Nelson", "entity"),
            MaskMapping("ENTITYD", "Gordon Field Airport", "entity"),
        ]

        restored = restore_entity_paths(paths, mappings)

        self.assertEqual(restored[0].nodes, ("Ten9Eight: Shoot For The Moon!", "director", "ENTITYBish"))
        self.assertEqual(restored[1].nodes, ("Sabotage (1936 Film).", "born", "Fort Nelson/Gordon Field Airport"))

    def test_restore_global_best_paths_replace_placeholders(self) -> None:
        self.assertEqual(
            restore_global_best_path(
                {"nodes": ["ENTITYA", "signed", "person"]},
                [MaskMapping("ENTITYA", "Barcelona", "entity")],
            ),
            ["Barcelona", "signed", "person"],
        )
        self.assertEqual(
            restore_global_best_paths(
                [SimpleNamespace(nodes=["ENTITYA", "director", "born"])],
                [MaskMapping("ENTITYA", "Illusions (1982 Film)", "entity")],
            ),
            [["Illusions (1982 Film)", "director", "born"]],
        )
        with self.assertRaisesRegex(ValueError, "Unresolved entity placeholder"):
            restore_global_best_path({"nodes": ["ENTITYA", "born"]}, [])

class RecordingStep5LLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return self.payload


def _bridge_payload() -> dict[str, Any]:
    return {
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who performed When The Stars Go Blue?",
                "depends_on": [],
                "operation": "lookup",
                "output_type": "person",
            },
            {
                "id": "q2",
                "question": "What is the nationality of q1's answer?",
                "depends_on": ["q1"],
                "operation": "lookup",
                "output_type": "value",
            },
        ]
    }


def _younger_director_payload() -> dict[str, Any]:
    return {
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who directed Dangerously They Live?",
                "depends_on": [],
                "operation": "lookup",
                "output_type": "person",
            },
            {
                "id": "q2",
                "question": "When was q1's answer born?",
                "depends_on": ["q1"],
                "operation": "lookup",
                "output_type": "date",
            },
            {
                "id": "q3",
                "question": "Who directed Salad By The Roots?",
                "depends_on": [],
                "operation": "lookup",
                "output_type": "person",
            },
            {
                "id": "q4",
                "question": "When was q3's answer born?",
                "depends_on": ["q3"],
                "operation": "lookup",
                "output_type": "date",
            },
            {
                "id": "q5",
                "question": "Based on q2's answer and q4's answer, which film has the younger director: Dangerously They Live or Salad By The Roots?",
                "depends_on": ["q2", "q4"],
                "operation": "select",
                "output_type": "work",
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
