from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_depo_decomposition_batch import (  # noqa: E402
    _jsonable,
    _manifest_key,
    _processed_manifest_keys,
    _question_dir_name,
    _write_json,
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
                        "question_entities": [
                            "Romance On The Run",
                            "The Palace Of Angels",
                        ],
                        "question_structure": [
                            "Romance On The Run -- film -- has -- director -- born -- later",
                            "The Palace Of Angels -- film -- has -- director -- born -- later",
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
        self.assertIn("#### Question Structure", markdown)
        self.assertIn(
            "Branch 1: Romance On The Run -- film -- has -- director -- born -- later",
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

    def test_question_dir_name_is_available_for_hyperbranch_batch(self) -> None:
        name = _question_dir_name(
            12,
            "Q/12",
            "Which film has the director born later, Illusions or Afterlife?",
        )

        self.assertEqual(
            name,
            "00012_q-12_which-film-has-the-director-born-later-illusions-or-afterlife",
        )

    def test_json_writer_handles_to_dict_payloads_for_hyperbranch_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "payload.json"
            payload = {"node": SimpleNamespace(to_dict=lambda: {"id": "q1"}), "deps": ("q1",)}

            _write_json(output_path, payload)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{\n  "node": {\n    "id": "q1"\n  },\n  "deps": [\n    "q1"\n  ]\n}',
            )
            self.assertEqual(_jsonable(payload), {"node": {"id": "q1"}, "deps": ["q1"]})


if __name__ == "__main__":
    unittest.main()
