from __future__ import annotations

import json
import re
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
    QuestionStructureAtomicDAGGenerator,
    prompt_input_text,
    restore_entity_paths,
    restore_global_best_path,
    restore_global_best_paths,
    validate_atomic_question_dag,
)
from models import MaskMapping  # noqa: E402


class AtomicQuestionDAGTest(unittest.TestCase):
    def test_prompt_contract_renders_question_structure_text(self) -> None:
        prompt = prompt_input_text(
            original_question="Question?",
            question_entities=["When The Stars Go Blue"],
            question_structure=[["When The Stars Go Blue", "song", "performer", "nationality"]],
        )
        payload = json.loads(prompt)

        self.assertEqual(payload["original_question"], "Question?")
        self.assertEqual(payload["question_entities"], ["When The Stars Go Blue"])
        self.assertEqual(
            payload["question_structure"],
            ["When The Stars Go Blue -- song -- performer -- nationality"],
        )
        self.assertNotIn("step4_paths", prompt)
        self.assertNotIn("global_best_paths", prompt)
        self.assertNotIn("topic_entities", prompt)
        self.assertNotIn("semantic_reasoning_paths", prompt)

        self.assertIn("question_structure", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("question_entities", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("exact unknown span as `ANSWER`", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("candidate carriers", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("all_ids - referenced_ids == {last_id}", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("original question is the only source of meaning", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_reasoning_paths"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_nodes"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_edges"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"semantic_edge_ids"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn('"output_node_id"', ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn("output_type", ATOMIC_QUESTION_DAG_SYSTEM)

    def test_prompt_renders_multiple_branches_and_ignores_empty_nodes(self) -> None:
        prompt = prompt_input_text(
            original_question="Which film has the younger director, Dangerously They Live or Salad By The Roots?",
            question_entities=[
                "Dangerously They Live",
                "Dangerously They Live",
                "Salad By The Roots",
                "",
            ],
            question_structure=[
                ["Dangerously They Live", "", "director", "born"],
                [],
                ["Salad By The Roots", "director", "born"],
            ],
        )
        payload = json.loads(prompt)

        self.assertEqual(
            payload,
            {
                "original_question": "Which film has the younger director, Dangerously They Live or Salad By The Roots?",
                "question_entities": ["Dangerously They Live", "Salad By The Roots"],
                "question_structure": [
                    "Dangerously They Live -- director -- born",
                    "Salad By The Roots -- director -- born",
                ],
            },
        )

    def test_all_step5_prompt_examples_use_json_input_and_one_final_leaf(self) -> None:
        input_payloads = [
            json.loads(raw_input)
            for raw_input in re.findall(
                r"Input:\n\n(\{.*?\})\n\nOutput:",
                ATOMIC_QUESTION_DAG_SYSTEM,
                flags=re.DOTALL,
            )
        ]
        output_payloads = [
            json.loads(line)
            for line in ATOMIC_QUESTION_DAG_SYSTEM.splitlines()
            if line.startswith('{"atomic_questions":')
        ]

        self.assertEqual(len(input_payloads), 8)
        self.assertEqual(len(output_payloads), 8)
        for payload in input_payloads:
            self.assertEqual(
                set(payload),
                {"original_question", "question_entities", "question_structure"},
            )
            self.assertIsInstance(payload["question_entities"], list)
            self.assertTrue(
                all(isinstance(branch, str) for branch in payload["question_structure"])
            )

        for payload in output_payloads:
            nodes = payload["atomic_questions"]
            all_ids = {node["id"] for node in nodes}
            referenced_ids = {
                dependency
                for node in nodes
                for dependency in node["depends_on"]
            }
            self.assertEqual(all_ids - referenced_ids, {nodes[-1]["id"]})
            for node in nodes:
                self.assertEqual(
                    set(node),
                    {"id", "question", "depends_on", "operation"},
                )
                literal_references = set(
                    re.findall(r"\b(q\d+)'s answer\b", node["question"])
                )
                self.assertEqual(literal_references, set(node["depends_on"]))

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

    def test_question_structure_hints_are_not_used_as_support_requirements(self) -> None:
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
            question_structure=[["One Last Time", "video", "stars", "Who"]],
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
        self.assertEqual(result.nodes[-1].output_type, "unknown")

    def test_legacy_output_type_is_accepted_for_saved_payloads(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "Who is A?",
                        "depends_on": [],
                        "operation": "lookup",
                        "output_type": "person",
                    }
                ]
            }
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[0].output_type, "person")

    def test_question_structure_generator_sends_text_prompt_once(self) -> None:
        llm = RecordingStep5LLM(_bridge_payload())
        result = QuestionStructureAtomicDAGGenerator(llm).generate(
            original_question="What nationality is the performer of the song When The Stars Go Blue?",
            question_entities=["When The Stars Go Blue"],
            question_structure=[["When The Stars Go Blue", "song", "performer", "nationality"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(llm.system_prompts, [ATOMIC_QUESTION_DAG_SYSTEM])
        self.assertEqual(len(llm.user_prompts), 1)
        prompt_payload = json.loads(llm.user_prompts[0])
        self.assertEqual(
            prompt_payload,
            {
                "original_question": "What nationality is the performer of the song When The Stars Go Blue?",
                "question_entities": ["When The Stars Go Blue"],
                "question_structure": [
                    "When The Stars Go Blue -- song -- performer -- nationality"
                ],
            },
        )
        self.assertNotIn("step4_paths", llm.user_prompts[0])
        self.assertNotIn("global_best_paths", llm.user_prompts[0])
        self.assertNotIn("topic_entities", llm.user_prompts[0])

    def test_dependency_binding_mismatch_is_warning_only(self) -> None:
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
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.validation_errors, [])
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])
        self.assertTrue(any("does not reference that answer" in warning for warning in result.warnings))

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
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.warnings, [])

    def test_question_structure_generator_keeps_binding_mismatch_as_warning(self) -> None:
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

        result = QuestionStructureAtomicDAGGenerator(llm).generate(
            original_question="Who wrote Turn Me On by the singer of Come Away with Me?",
            question_entities=["Turn Me On", "Come Away with Me"],
            question_structure=[["Come Away with Me", "singer", "Turn Me On", "wrote", "Who"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(len(llm.user_prompts), 1)
        self.assertTrue(any("does not reference that answer" in warning for warning in result.warnings))

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

    def test_empty_question_structure_still_calls_llm(self) -> None:
        llm = RecordingStep5LLM(_bridge_payload())
        result = QuestionStructureAtomicDAGGenerator(llm).generate(
            original_question="Question?",
            question_entities=[],
            question_structure=[],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(len(llm.user_prompts), 1)
        self.assertEqual(
            json.loads(llm.user_prompts[0]),
            {
                "original_question": "Question?",
                "question_entities": [],
                "question_structure": [],
            },
        )

    def test_missing_depends_on_reference_is_warning_only(self) -> None:
        result = validate_atomic_question_dag(
            {
                "atomic_questions": [
                    {"id": "q1", "question": "Who is A?", "depends_on": []},
                    {"id": "q2", "question": "Where was q1's answer born?", "depends_on": []},
                ]
            }
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.edges, [])
        self.assertTrue(any("depends_on does not include it" in warning for warning in result.warnings))

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
            },
            {
                "id": "q2",
                "question": "What is the nationality of q1's answer?",
                "depends_on": ["q1"],
                "operation": "lookup",
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
            },
            {
                "id": "q2",
                "question": "When was q1's answer born?",
                "depends_on": ["q1"],
                "operation": "lookup",
            },
            {
                "id": "q3",
                "question": "Who directed Salad By The Roots?",
                "depends_on": [],
                "operation": "lookup",
            },
            {
                "id": "q4",
                "question": "When was q3's answer born?",
                "depends_on": ["q3"],
                "operation": "lookup",
            },
            {
                "id": "q5",
                "question": "Based on q2's answer and q4's answer, which film has the younger director: Dangerously They Live or Salad By The Roots?",
                "depends_on": ["q2", "q4"],
                "operation": "select",
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
