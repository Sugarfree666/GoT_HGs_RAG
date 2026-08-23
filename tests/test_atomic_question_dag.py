from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from atomic_question_dag import (  # noqa: E402
    QuestionStructureAtomicDAGGenerator,
    restore_global_best_paths,
    validate_atomic_question_dag,
)
from models import MaskMapping  # noqa: E402
from prompts import build_atomic_question_dag_prompt  # noqa: E402


class AtomicQuestionDAGTest(unittest.TestCase):
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

    def test_valid_payload_builds_edges(self) -> None:
        result = validate_atomic_question_dag(_payload())

        self.assertEqual([node.id for node in result.nodes], ["q1", "q2"])
        self.assertEqual(
            [edge.to_dict() for edge in result.edges],
            [{"source": "q1", "target": "q2"}],
        )
        self.assertEqual(result.leaf_node_ids, ["q2"])

    def test_restores_selected_paths(self) -> None:
        paths = [SimpleNamespace(nodes=["ENTITYA!", "director"])]
        mappings = [MaskMapping("ENTITYA", "An Event")]

        self.assertEqual(
            restore_global_best_paths(paths, mappings),
            [["An Event!", "director"]],
        )

    def test_generator_calls_llm_once(self) -> None:
        llm = RecordingLLM(_payload())
        result = QuestionStructureAtomicDAGGenerator(llm).generate(
            original_question="Who directed An Event?",
            question_entities=["An Event"],
            question_structure=[["An Event", "director"]],
        )

        self.assertEqual(len(llm.prompts), 1)


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
