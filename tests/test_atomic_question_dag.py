from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from atomic_question_dag import generate_atomic_question_dag, restore_paths  # noqa: E402
from prompts import ATOMIC_QUESTION_DAG_SYSTEM, build_atomic_question_dag_prompt  # noqa: E402


class AtomicQuestionDAGTest(unittest.TestCase):
    def test_prompt_keeps_the_dag_execution_contract(self) -> None:
        self.assertIn("exactly one final leaf", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("last node", ATOMIC_QUESTION_DAG_SYSTEM)
        self.assertIn("qN's answer", ATOMIC_QUESTION_DAG_SYSTEM)

    def test_prompt_contains_only_step5_inputs(self) -> None:
        payload = json.loads(
            build_atomic_question_dag_prompt(
                original_question="Question?",
                question_entities=["An Event"],
                question_structure=[["An Event", "director"]],
            )
        )

        self.assertEqual(
            payload,
            {
                "original_question": "Question?",
                "question_entities": ["An Event"],
                "question_structure": ["An Event -- director"],
            },
        )

    def test_generator_builds_dag_edges(self) -> None:
        llm = RecordingLLM(_payload())
        result = generate_atomic_question_dag(
            llm,
            "Who directed An Event?",
            ["An Event"],
            [["An Event", "director"]],
        )

        self.assertEqual(len(llm.prompts), 1)
        self.assertEqual(result["nodes"][0]["id"], "q1")
        self.assertEqual(result["edges"], [{"source": "q1", "target": "q2"}])
        self.assertEqual(result["leaf_node_ids"], ["q2"])

    def test_restores_selected_paths(self) -> None:
        self.assertEqual(
            restore_paths([["ENTITYA!", "director"]], {"ENTITYA": "An Event"}),
            [["An Event!", "director"]],
        )


class RecordingLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def chat_json(self, _system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.prompts.append(user_prompt)
        return self.payload


def _payload() -> dict[str, object]:
    return {
        "atomic_questions": [
            {
                "id": "q1",
                "question": "Who directed An Event?",
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


if __name__ == "__main__":
    unittest.main()
