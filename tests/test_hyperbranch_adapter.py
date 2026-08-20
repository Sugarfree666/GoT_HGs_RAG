from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from hyperbranch_adapter import build_hyperbranch_dag_payload, explicit_entity_texts  # noqa: E402


class HyperBranchAdapterTest(unittest.TestCase):
    def test_adapts_valid_depo_dag_and_deduplicates_entity_texts(self) -> None:
        payload = {
            "question": "Who is Alice?",
            "stages": {
                "1_explicit_entities": {"entities": [{"text": "Alice"}, {"text": "alice"}, {"text": "Bob"}]},
                "6_atomic_question_dag": {
                    "valid": True,
                    "nodes": [{"id": "q1", "question": "Who is Alice?"}],
                    "edges": [],
                    "leaf_node_ids": ["q1"],
                },
            },
        }

        result = build_hyperbranch_dag_payload(payload)

        self.assertEqual(explicit_entity_texts(payload), ["Alice", "Bob"])
        self.assertEqual(result["question"], "Who is Alice?")
        self.assertEqual(result["topic_entities"], ["Alice", "Bob"])
        self.assertEqual(result["nodes"], payload["stages"]["6_atomic_question_dag"]["nodes"])

    def test_rejects_invalid_depo_dag(self) -> None:
        payload = {"stages": {"6_atomic_question_dag": {"valid": False, "validation_errors": ["bad DAG"]}}}

        with self.assertRaisesRegex(ValueError, "DEPO atomic DAG is invalid"):
            build_hyperbranch_dag_payload(payload)


if __name__ == "__main__":
    unittest.main()
