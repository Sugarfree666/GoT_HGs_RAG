from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_depo_decomposition_batch import (  # noqa: E402
    _manifest_key,
    _processed_manifest_keys,
    build_markdown_report,
)


class DEPOBatchReportTest(unittest.TestCase):
    def test_markdown_report_includes_repaired_graph_with_cost_only(self) -> None:
        payload = {
            "status": "ok",
            "method": "depo_hanlp_sdp_atomic_dag",
            "dataset": "2wikimultihopqa",
            "questions_file": "questions/2wikimultihopqa/questions.json",
            "index": 7,
            "qid": "q7",
            "question": "Which film has the director born later, ENTITYA or ENTITYB?",
            "gold_answer": None,
            "stages": {
                "1_explicit_entities": {
                    "entities": [
                        {"text": "Romance On The Run", "start_char": 40, "end_char": 58},
                        {"text": "The Palace Of Angels", "start_char": 62, "end_char": 82},
                    ]
                },
                "2_entity_masking": {
                    "masked_question": "Which film has the director born later, ENTITYA or ENTITYB?",
                    "mask_mappings": [
                        {
                            "placeholder": "ENTITYA",
                            "original_text": "Romance On The Run",
                        },
                        {
                            "placeholder": "ENTITYB",
                            "original_text": "The Palace Of Angels",
                        },
                    ],
                },
                "3_hanlp_sdp_parsing": {
                    "model": "fake",
                    "tokens": ["Which", "film", "has", "director", "later"],
                    "mask_token_checks": {},
                    "edges": [],
                },
                "4_token_reasoning_structure": {
                    "repaired_evidence_edges": [
                        {
                            "source": "2",
                            "source_text": "film",
                            "target": "3",
                            "target_text": "has",
                            "edge_cost": 1,
                            "rule": "raw_evidence",
                            "provenance": [{"relation": "verb_ARG1"}],
                        },
                        {
                            "source": "9",
                            "source_text": "ENTITYA",
                            "target": "10",
                            "target_text": "or",
                            "edge_cost": None,
                            "rule": "raw_evidence",
                            "provenance": [{"relation": "coord_ARG1"}],
                        },
                    ],
                    "paths": [],
                },
                "5_step5_action_trace": {
                    "input": {
                        "step4_paths": [
                            [
                                "Romance On The Run",
                                "film",
                                "has",
                                "director",
                                "born",
                                "later",
                            ],
                            [
                                "The Palace Of Angels",
                                "film",
                                "has",
                                "director",
                                "born",
                                "later",
                            ],
                        ]
                    },
                    "atomic_questions": [],
                },
                "6_atomic_question_dag": {"valid": True, "nodes": []},
            },
        }

        markdown = build_markdown_report(payload, heading_level=2)

        self.assertIn("## DEPO Decomposition #7", markdown)
        self.assertIn("### 4. Token Reasoning Structure", markdown)
        self.assertIn("#### Repaired Evidence Graph", markdown)
        self.assertIn("film[2] -- has[3] (cost=1)", markdown)
        self.assertIn("ENTITYA[9] -- or[10] (cost=blocked)", markdown)
        self.assertIn("#### Entity Branch Best Paths", markdown)
        self.assertIn(
            "P1: Romance On The Run ---- film ---- has ---- director ---- born ---- later",
            markdown,
        )
        self.assertNotIn("## 4. Global Best Path", markdown)
        self.assertNotIn("; rule=", markdown)
        self.assertNotIn("; rel=", markdown)

    def test_processed_manifest_keys_use_completed_items_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        '{"dataset":"2wikimultihopqa","index":1,"qid":"a","status":"ok"}',
                        '{"dataset":"2wikimultihopqa","index":2,"qid":"b","status":"sample"}',
                        '{"dataset":"2wikimultihopqa","index":3,"qid":"c","status":"skipped"}',
                    ]
                ),
                encoding="utf-8",
            )

            keys = _processed_manifest_keys(manifest_path)

        self.assertIn(_manifest_key("2wikimultihopqa", 1, "a"), keys)
        self.assertIn(_manifest_key("2wikimultihopqa", 2, "b"), keys)
        self.assertNotIn(_manifest_key("2wikimultihopqa", 3, "c"), keys)


if __name__ == "__main__":
    unittest.main()
