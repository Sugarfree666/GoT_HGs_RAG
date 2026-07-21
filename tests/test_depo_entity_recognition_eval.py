from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_depo_entity_recognition_eval import (  # noqa: E402
    _slice_items,
    build_error_payload,
    build_markdown_result,
    build_result_payload,
)


class EntityRecognitionEvalScriptTest(unittest.TestCase):
    def test_range_selection_is_one_based_inclusive_and_honors_limit(self) -> None:
        items = [{"index": index} for index in range(1, 7)]

        selected = _slice_items(items, start=2, end=5, limit=2)

        self.assertEqual([item["index"] for item in selected], [2, 3])

    def test_result_payload_includes_entities_spans_and_normalization(self) -> None:
        question = "Did Shrek 2 influence Shrek 2's sequel?"
        result = SimpleNamespace(
            entities=[
                SimpleNamespace(
                    text="Shrek 2",
                    semantic_type_hint="Work",
                    start_char=4,
                    end_char=11,
                    reason="LLM direct explicit entity",
                )
            ],
            normalized_question="Did Shrek 2 influence Shrek 2's sequel?",
            normalization_changed=False,
            normalization_note="",
            warnings=["sample warning"],
            raw_payload={"explicit_entities": [{"surface": "Shrek 2", "type": "Work"}]},
        )
        item = {"index": 7, "qid": "q7", "question": question, "answer": "yes"}

        payload = build_result_payload(
            dataset="2wikimultihopqa",
            questions_file=Path("questions/2wikimultihopqa/questions.json"),
            item=item,
            result=result,
        )

        self.assertEqual(payload["method"], "depo_step1_entity_recognition")
        self.assertEqual(payload["explicit_entities"][0]["surface"], "Shrek 2")
        self.assertEqual(payload["explicit_entities"][0]["type"], "Work")
        self.assertEqual(
            payload["explicit_entities"][0]["matched_spans"],
            [{"start_char": 4, "end_char": 11}, {"start_char": 22, "end_char": 29}],
        )
        self.assertEqual(payload["normalized_question"], question)
        self.assertFalse(payload["normalization_changed"])
        self.assertEqual(payload["warnings"], ["sample warning"])
        self.assertEqual(payload["raw_llm_payload"], result.raw_payload)

    def test_markdown_report_shows_normalization_and_warning(self) -> None:
        payload = {
            "status": "ok",
            "index": 3,
            "qid": None,
            "question": "Who directed An Event?",
            "explicit_entities": [
                {
                    "surface": "An Event",
                    "type": "Work",
                    "matched_spans": [{"start_char": 13, "end_char": 21}],
                }
            ],
            "normalized_question": "Who directed An Event?",
            "normalization_changed": False,
            "normalization_note": "",
            "warnings": ["sample warning"],
        }

        report = "\n".join(build_markdown_result(payload))

        self.assertIn("### Explicit Entities", report)
        self.assertIn("`An Event` (Work; [13:21])", report)
        self.assertIn("### Normalized Question", report)
        self.assertIn("- sample warning", report)

    def test_error_payload_keeps_the_original_question_for_inspection(self) -> None:
        item = {"index": 4, "qid": None, "question": "Who directed An Event?", "answer": None}

        payload = build_error_payload(
            dataset="musique",
            questions_file=Path("questions/musique/questions.json"),
            item=item,
            exc=RuntimeError("request failed"),
        )

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["normalized_question"], item["question"])
        self.assertIn("RuntimeError: request failed", payload["warnings"][0])


if __name__ == "__main__":
    unittest.main()
