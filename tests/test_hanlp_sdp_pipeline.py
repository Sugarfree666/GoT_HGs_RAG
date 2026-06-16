from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from main import print_hanlp_sdp_result, run_hanlp_sdp_pipeline  # noqa: E402
from mask_span_extractor import ExplicitEntityExtractor  # noqa: E402
from models import HanLPSDPEdge, HanLPSDPResult, QuestionRecord  # noqa: E402


class HanLPSDPMainlineTest(unittest.TestCase):
    def test_hanlp_sdp_pipeline_masks_entities_and_prints_only_hanlp_stages(self) -> None:
        record = QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?")
        parser = FakeHanLPSDPParser()

        result = run_hanlp_sdp_pipeline(
            record=record,
            index=1,
            mask_span_extractor=ExplicitEntityExtractor(None),
            parser=parser,
        )

        replacement = result["replacement"]
        self.assertEqual(replacement.masked_question, "Who is older, PERSONA or PERSONB?")
        self.assertEqual([mapping.placeholder for mapping in replacement.mask_mappings], ["PERSONA", "PERSONB"])
        self.assertEqual(parser.placeholders, ["PERSONA", "PERSONB"])

        stream = io.StringIO()
        with redirect_stdout(stream):
            print_hanlp_sdp_result(1, record, result)
        output = stream.getvalue()

        self.assertIn("[Original Question]", output)
        self.assertIn("[1. Explicit Entities]", output)
        self.assertIn(" - Ryan Tubridy [Person]", output)
        self.assertIn("[2. Entity Masking]", output)
        self.assertIn(" - PERSONA -> Ryan Tubridy", output)
        self.assertIn("[3. HanLP SDP Parsing]", output)
        self.assertIn("[Readable SDP Edges]", output)
        self.assertIn("[SDP: sdp/pas]", output)
        self.assertIn("older[3] --ARG1--> Who[1]", output)
        self.assertNotIn("Relation-Carrier Declarative Views", output)
        self.assertNotIn("CoreNLP + OpenIE View Annotations", output)
        self.assertNotIn("Semantic Reasoning Path Induction", output)
        self.assertNotIn("Semantic-Path-Guided Atomic DAG", output)


class FakeHanLPSDPParser:
    def __init__(self) -> None:
        self.placeholders: list[str] = []

    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        self.placeholders = list(placeholders or [])
        tokens = ["Who", "is", "older", ",", "PERSONA", "or", "PERSONB", "?"]
        return HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=["tok", "sdp/pas"],
            sdp_graphs={"sdp/pas": [[(3, "ARG1")], [], [(0, "root")], [], [(3, "ARG2")], [], [(3, "ARG2")], []]},
            edges=[
                HanLPSDPEdge("sdp/pas", 3, "older", "ARG1", 1, "Who"),
                HanLPSDPEdge("sdp/pas", 0, "ROOT", "root", 3, "older"),
                HanLPSDPEdge("sdp/pas", 3, "older", "ARG2", 5, "PERSONA"),
                HanLPSDPEdge("sdp/pas", 3, "older", "ARG2", 7, "PERSONB"),
            ],
            raw={"tok": tokens, "sdp/pas": []},
            warnings=[],
            model="fake.hanlp.model",
            mask_token_checks={placeholder: "OK" for placeholder in self.placeholders},
        )


if __name__ == "__main__":
    unittest.main()
