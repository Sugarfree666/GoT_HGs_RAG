from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
DEPO_ROOT = PROJECT_ROOT / "depo"
for path in (SCRIPTS_ROOT, DEPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_direct_llm_atomic_dag_batch import (  # noqa: E402
    DIRECT_LLM_ATOMIC_DAG_SYSTEM,
    build_direct_atomic_dag_prompt,
    build_result_payload,
    decompose_question_direct,
)
from prompts import ATOMIC_QUESTION_DAG_SYSTEM  # noqa: E402


class DirectLLMAtomicDAGBatchTest(unittest.TestCase):
    def test_prompt_contains_only_original_question_input(self) -> None:
        prompt = build_direct_atomic_dag_prompt("What is the capital of France?")

        self.assertIn("Original question:", prompt)
        self.assertIn("What is the capital of France?", prompt)
        self.assertNotIn("question_structure", prompt)
        self.assertNotIn("step4_paths", prompt)
        self.assertNotIn("global_best_paths", prompt)

        self.assertIn('"atomic_questions"', DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertNotIn('"nodes"', DIRECT_LLM_ATOMIC_DAG_SYSTEM)

    def test_prompt_preserves_candidate_target_for_comparisons(self) -> None:
        self.assertIn("candidate carriers", DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertIn("Evidence values are", DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertIn("Alternative-choice wording", DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertIn("Use `select` when the final answer is one of the candidate entities", DIRECT_LLM_ATOMIC_DAG_SYSTEM)

    def test_prompt_requires_original_answer_slot_and_exactly_one_leaf(self) -> None:
        self.assertIn("exact unknown span as `ANSWER`", DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertIn("answer must fill that same slot", DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertIn(
            "`all_ids - referenced_ids == {last_id}`",
            DIRECT_LLM_ATOMIC_DAG_SYSTEM,
        )
        self.assertIn("sole final answer node", DIRECT_LLM_ATOMIC_DAG_SYSTEM)

    def test_prompt_keeps_supplied_constraints_in_the_target_lookup(self) -> None:
        self.assertIn("Distinguish **given constraints** from unknowns", DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertIn("it is not a separate answer to", DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertIn("supplied facts are joint filters", DIRECT_LLM_ATOMIC_DAG_SYSTEM)
        self.assertIn("founded by Jordan Lee", DIRECT_LLM_ATOMIC_DAG_SYSTEM)

    def test_prompt_requires_dependencies_to_equal_literal_references(self) -> None:
        self.assertIn(
            "must equal its `depends_on` set exactly",
            DIRECT_LLM_ATOMIC_DAG_SYSTEM,
        )

    def test_step5_prompt_retains_the_direct_prompt_behavior_contract(self) -> None:
        shared_contract_markers = (
            "exact unknown span as `ANSWER`",
            "candidate carriers",
            "given constraints",
            "faithful span substitution",
            "Alternative-choice wording",
            "trailing interrogative",
            "all_ids - referenced_ids == {last_id}",
            "ANSWER`-slot test",
        )
        for marker in shared_contract_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, DIRECT_LLM_ATOMIC_DAG_SYSTEM)
                self.assertIn(marker, ATOMIC_QUESTION_DAG_SYSTEM)

        self.assertIn("question_entities", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("question_structure", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertNotIn("output_type", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn(
            "First derive the answer contract and candidate DAG from the original question",
            ATOMIC_QUESTION_DAG_SYSTEM,
        )

    def test_all_prompt_examples_have_one_final_leaf_and_exact_references(self) -> None:
        example_payloads = [
            json.loads(line)
            for line in DIRECT_LLM_ATOMIC_DAG_SYSTEM.splitlines()
            if line.startswith('{"atomic_questions":')
        ]

        self.assertEqual(len(example_payloads), 9)
        for payload in example_payloads:
            nodes = payload["atomic_questions"]
            all_ids = {node["id"] for node in nodes}
            referenced_ids = {
                dependency
                for node in nodes
                for dependency in node["depends_on"]
            }
            self.assertEqual(all_ids - referenced_ids, {nodes[-1]["id"]})

            for node in nodes:
                literal_references = set(
                    re.findall(r"\b(q\d+)'s answer\b", node["question"])
                )
                self.assertEqual(literal_references, set(node["depends_on"]))

    def test_direct_decomposition_calls_llm_once_and_validates_schema(self) -> None:
        llm = FakeLLM(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "Who performed the song Changed It?",
                        "depends_on": [],
                        "operation": "lookup",
                    },
                    {
                        "id": "q2",
                        "question": "Where was q1's answer born?",
                        "depends_on": ["q1"],
                        "operation": "lookup",
                    },
                ]
            }
        )

        result = decompose_question_direct(
            llm,
            "What is the place of birth of the performer of song Changed It?",
        )

        self.assertTrue(result.valid)
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("Original question:", llm.calls[0]["user_prompt"])

    def test_invalid_dag_is_reported_in_payload(self) -> None:
        llm = FakeLLM(
            {
                "atomic_questions": [
                    {
                        "id": "q1",
                        "question": "What is q2's answer?",
                        "depends_on": ["q2"],
                        "operation": "lookup",
                    }
                ]
            }
        )
        result = decompose_question_direct(llm, "What is the answer?")
        payload = build_result_payload(
            dataset="sample",
            questions_file=Path("questions/sample/questions.json"),
            item={"index": 1, "qid": None, "question": "What is the answer?", "answer": None, "raw": {}},
            dag_result=result,
        )

        self.assertEqual(payload["status"], "invalid")
        self.assertFalse(payload["atomic_question_dag"]["valid"])
        self.assertTrue(payload["atomic_question_dag"]["validation_errors"])


class FakeLLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_retries": max_retries,
            }
        )
        return self.payload


if __name__ == "__main__":
    unittest.main()
