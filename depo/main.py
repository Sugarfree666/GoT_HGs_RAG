from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import QuestionRecord


if TYPE_CHECKING:
    from entity_masking_preprocessor import EntityMaskingPreprocessor
    from hanlp_sdp_parser import HanLPSDPParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an atomic-question DAG from a question with DEPO."
    )
    parser.add_argument("--question", help="Process one question.")
    parser.add_argument("--questions-file", default="questions.json")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--hanlp-model")
    parser.add_argument("--debug", action="store_true", help="Save Step4 debug JSON.")
    parser.add_argument("--debug-dir", default="debug/hanlp_sdp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [QuestionRecord(question=args.question)] if args.question else read_questions(args.questions_file)
    return _run_hanlp_sdp_cli(args, records)


def _run_hanlp_sdp_cli(args: argparse.Namespace, records: list[QuestionRecord]) -> int:
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    from entity_masking_preprocessor import EntityMaskingPreprocessor
    from hanlp_sdp_parser import HanLPSDPParser
    from llm_client import LLMClient

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
    preprocessor = EntityMaskingPreprocessor(llm_client)
    parser = HanLPSDPParser(args.hanlp_model)

    for index, record in enumerate(records, start=1):
        print(f"[run {index}/{len(records)}] {record.question}")
        run_hanlp_sdp_pipeline(
            record=record,
            index=index,
            preprocessor=preprocessor,
            parser=parser,
            llm_client=llm_client,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )
        print(f"[ok]  #{index}")
    return 0


def run_hanlp_sdp_pipeline(
    record: QuestionRecord,
    index: int,
    preprocessor: "EntityMaskingPreprocessor",
    parser: "HanLPSDPParser",
    llm_client: Any,
    debug: bool = False,
    debug_dir: str | None = None,
) -> dict[str, Any]:
    from atomic_question_dag import QuestionStructureAtomicDAGGenerator, restore_global_best_paths
    from tri_sdp_reasoning_compiler import compile_token_reasoning_structure

    preprocess_result = preprocessor.preprocess(record.question)
    hanlp_input_sentence = preprocess_result.sdp_input_sentence
    explicit_entities = [mapping.placeholder for mapping in preprocess_result.mask_mappings]
    hanlp_sdp_result = parser.parse(hanlp_input_sentence, placeholders=explicit_entities)
    token_reasoning_structure = compile_token_reasoning_structure(
        hanlp_sdp_result,
        explicit_entities=explicit_entities,
        masked_question=preprocess_result.masked_question,
        original_question=preprocess_result.original_question,
        normalized_question=preprocess_result.normalized_question or preprocess_result.original_question,
        normalization_changed=preprocess_result.normalization_changed,
        normalization_note=preprocess_result.normalization_note,
        question_id=record.qid or f"q{index}",
        debug=debug,
        debug_dir=debug_dir,
    )
    question_structure = restore_global_best_paths(
        token_reasoning_structure.paths,
        preprocess_result.mask_mappings,
    )
    atomic_question_dag = QuestionStructureAtomicDAGGenerator(llm_client).generate(
        original_question=record.question,
        question_entities=[entity.text for entity in preprocess_result.explicit_entities.entities],
        question_structure=question_structure,
    )
    return {
        "preprocess_result": preprocess_result,
        "explicit_entities": preprocess_result.explicit_entities,
        "explicit_entity_payload": preprocess_result.explicit_entities.raw_payload,
        "original_question": preprocess_result.original_question,
        "normalized_question": preprocess_result.normalized_question,
        "normalization_changed": preprocess_result.normalization_changed,
        "normalization_note": preprocess_result.normalization_note,
        "masked_question": preprocess_result.masked_question,
        "sdp_input_sentence": preprocess_result.sdp_input_sentence,
        "hanlp_input_sentence": hanlp_input_sentence,
        "entity_mask_mappings": preprocess_result.mask_mappings,
        "hanlp_sdp_result": hanlp_sdp_result,
        "token_reasoning_structure": token_reasoning_structure,
        "atomic_question_dag": atomic_question_dag,
    }


if __name__ == "__main__":
    raise SystemExit(main())
