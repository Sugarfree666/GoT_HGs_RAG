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

from hanlp_sdp_preprocessor import HanLPSDPPreprocessor  # noqa: E402
from main import print_hanlp_sdp_result, run_hanlp_sdp_pipeline  # noqa: E402
from models import HanLPSDPEdge, HanLPSDPResult, QuestionRecord  # noqa: E402


class HanLPSDPMainlineTest(unittest.TestCase):
    def test_hanlp_sdp_pipeline_preprocesses_once_and_parses_declarative_sentence(self) -> None:
        record = QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?")
        parser = FakeHanLPSDPParser()
        llm = FakePreprocessLLM()
        preprocessor = HanLPSDPPreprocessor(llm)

        result = run_hanlp_sdp_pipeline(
            record=record,
            index=1,
            preprocessor=preprocessor,
            parser=parser,
        )

        preprocess_result = result["preprocess_result"]
        self.assertEqual(preprocess_result.masked_question, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(preprocess_result.sdp_input_sentence, "ANSWER is older, ENTITYA or ENTITYB.")
        self.assertEqual([mapping.placeholder for mapping in preprocess_result.mask_mappings], ["ENTITYA", "ENTITYB"])
        self.assertEqual(parser.placeholders, ["ENTITYA", "ENTITYB"])
        self.assertEqual(parser.text, "ANSWER is older, ENTITYA or ENTITYB.")
        self.assertEqual(llm.calls, 1)

        stream = io.StringIO()
        with redirect_stdout(stream):
            print_hanlp_sdp_result(1, record, result)
        output = stream.getvalue()

        self.assertIn("[Original Question]", output)
        self.assertIn("[1. Explicit Entities]", output)
        self.assertIn(" - Ryan Tubridy [Person]", output)
        self.assertIn("[2. SDP-Oriented Rewrite]", output)
        self.assertIn(" - ENTITYA -> Ryan Tubridy", output)
        self.assertIn("Masked question: Who is older, ENTITYA or ENTITYB?", output)
        self.assertIn("SDP input sentence: ANSWER is older, ENTITYA or ENTITYB.", output)
        self.assertIn("[3. HanLP SDP Parsing]", output)
        self.assertIn("[Readable SDP Edges]", output)
        self.assertIn("[SDP: sdp/pas]", output)
        self.assertIn("older[3] --ARG1--> ANSWER[1]", output)
        self.assertNotIn("Relation-Carrier Declarative Views", output)
        self.assertNotIn("CoreNLP + OpenIE View Annotations", output)
        self.assertNotIn("Semantic Reasoning Path Induction", output)
        self.assertNotIn("Semantic-Path-Guided Atomic DAG", output)

    def test_preprocessor_smoke_examples(self) -> None:
        cases = [
            (
                "Who is the spouse of Young Man Luther's author?",
                {
                    "explicit_entities": [
                        _entity("Who is the spouse of Young Man Luther's author?", "Young Man Luther"),
                    ],
                    "mask_mappings": [{"placeholder": "ENTITYA", "original_text": "Young Man Luther"}],
                    "masked_question": "Who is the spouse of ENTITYA's author?",
                    "sdp_input_sentence": "ANSWER is the spouse of the author of ENTITYA.",
                    "warnings": [],
                },
                "Who is the spouse of ENTITYA's author?",
                "ANSWER is the spouse of the author of ENTITYA.",
            ),
            (
                "What is the date of death of the director of film FilmA?",
                {
                    "explicit_entities": [
                        _entity("What is the date of death of the director of film FilmA?", "FilmA"),
                    ],
                    "mask_mappings": [{"placeholder": "ENTITYA", "original_text": "FilmA"}],
                    "masked_question": "What is the date of death of the director of film ENTITYA?",
                    "sdp_input_sentence": "ANSWER is the date of death of the director of ENTITYA.",
                    "warnings": [],
                },
                "What is the date of death of the director of film ENTITYA?",
                "ANSWER is the date of death of the director of ENTITYA.",
            ),
            (
                "Where was the person who wrote about the rioting being a dividing factor in Birmingham educated?",
                {
                    "explicit_entities": [
                        _entity(
                            "Where was the person who wrote about the rioting being a dividing factor in Birmingham educated?",
                            "Birmingham",
                        ),
                    ],
                    "mask_mappings": [{"placeholder": "ENTITYA", "original_text": "Birmingham"}],
                    "masked_question": (
                        "Where was the person who wrote about the rioting being a dividing factor in ENTITYA educated?"
                    ),
                    "sdp_input_sentence": (
                        "The person who wrote about the rioting being a dividing factor in ENTITYA was educated at ANSWER."
                    ),
                    "warnings": [],
                },
                "Where was the person who wrote about the rioting being a dividing factor in ENTITYA educated?",
                "The person who wrote about the rioting being a dividing factor in ENTITYA was educated at ANSWER.",
            ),
            (
                "Who is older, Ryan Tubridy or Mauro Massironi?",
                {
                    "explicit_entities": [
                        _entity("Who is older, Ryan Tubridy or Mauro Massironi?", "Ryan Tubridy", "Person"),
                        _entity("Who is older, Ryan Tubridy or Mauro Massironi?", "Mauro Massironi", "Person"),
                    ],
                    "mask_mappings": [
                        {"placeholder": "ENTITYA", "original_text": "Ryan Tubridy"},
                        {"placeholder": "ENTITYB", "original_text": "Mauro Massironi"},
                    ],
                    "masked_question": "Who is older, ENTITYA or ENTITYB?",
                    "sdp_input_sentence": "ANSWER is older, ENTITYA or ENTITYB.",
                    "warnings": [],
                },
                "Who is older, ENTITYA or ENTITYB?",
                "ANSWER is older, ENTITYA or ENTITYB.",
            ),
        ]
        for question, payload, expected_masked, expected_sdp in cases:
            with self.subTest(question=question):
                llm = StaticPreprocessLLM(payload)
                result = HanLPSDPPreprocessor(llm).preprocess(question)

                self.assertEqual(llm.calls, 1)
                self.assertEqual(result.masked_question, expected_masked)
                self.assertEqual(result.sdp_input_sentence, expected_sdp)
                self.assertIn("ANSWER", result.sdp_input_sentence)
                for mapping in result.mask_mappings:
                    self.assertTrue(mapping.placeholder.startswith("ENTITY"))
                    self.assertIn(mapping.placeholder, result.masked_question)
                    self.assertIn(mapping.placeholder, result.sdp_input_sentence)


class FakePreprocessLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        self.calls += 1
        assert max_retries == 1
        assert "not decomposing" in system_prompt
        assert "Who is older, Ryan Tubridy or Mauro Massironi?" in user_prompt
        return {
            "explicit_entities": [
                {
                    "text": "Ryan Tubridy",
                    "semantic_type_hint": "Person",
                    "start_char": 14,
                    "end_char": 26,
                    "confidence": 1.0,
                    "reason": "explicit person name",
                },
                {
                    "text": "Mauro Massironi",
                    "semantic_type_hint": "Person",
                    "start_char": 30,
                    "end_char": 45,
                    "confidence": 1.0,
                    "reason": "explicit person name",
                },
            ],
            "mask_mappings": [
                {"placeholder": "ENTITYA", "original_text": "Ryan Tubridy"},
                {"placeholder": "ENTITYB", "original_text": "Mauro Massironi"},
            ],
            "masked_question": "Who is older, ENTITYA or ENTITYB?",
            "sdp_input_sentence": "ANSWER is older, ENTITYA or ENTITYB.",
            "warnings": [],
        }


class StaticPreprocessLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, object]:
        del system_prompt, user_prompt
        self.calls += 1
        assert max_retries == 1
        return self.payload


class FakeHanLPSDPParser:
    def __init__(self) -> None:
        self.placeholders: list[str] = []
        self.text = ""

    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        self.placeholders = list(placeholders or [])
        self.text = text
        tokens = ["ANSWER", "is", "older", ",", "ENTITYA", "or", "ENTITYB", "."]
        return HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=["tok", "sdp/pas"],
            sdp_graphs={"sdp/pas": [[(3, "ARG1")], [], [(0, "root")], [], [(3, "ARG2")], [], [(3, "ARG2")], []]},
            edges=[
                HanLPSDPEdge("sdp/pas", 3, "older", "ARG1", 1, "ANSWER"),
                HanLPSDPEdge("sdp/pas", 0, "ROOT", "root", 3, "older"),
                HanLPSDPEdge("sdp/pas", 3, "older", "ARG2", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 3, "older", "ARG2", 7, "ENTITYB"),
            ],
            raw={"tok": tokens, "sdp/pas": []},
            warnings=[],
            model="fake.hanlp.model",
            mask_token_checks={placeholder: "OK" for placeholder in self.placeholders},
        )


def _entity(question: str, text: str, semantic_type: str = "Entity") -> dict[str, object]:
    start = question.index(text)
    return {
        "text": text,
        "semantic_type_hint": semantic_type,
        "start_char": start,
        "end_char": start + len(text),
        "confidence": 1.0,
        "reason": "test entity",
    }


if __name__ == "__main__":
    unittest.main()
