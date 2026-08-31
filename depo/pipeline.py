"""DEPO algorithm entry points."""

from __future__ import annotations

from typing import Any

from atomic_question_dag import generate_atomic_question_dag, restore_paths
from entity_masking_preprocessor import preprocess_question
from hanlp_sdp_parser import HanLPSDPParser
from hyper_branch.client import OpenAIClient
from tri_sdp_reasoning_compiler import compile_token_reasoning_structure


def extract_question_structure(
    question: str,
    parser: HanLPSDPParser,
    llm_client: OpenAIClient,
) -> dict[str, Any]:
    """Run DEPO through Step4 and return the restored question structure."""
    preprocessed = preprocess_question(question, llm_client)
    pas_result = parser.parse(preprocessed.masked_question)
    masked_paths = compile_token_reasoning_structure(
        pas_result,
        list(preprocessed.mask_mapping),
    )
    question_structure = restore_paths(masked_paths, preprocessed.mask_mapping)
    return {
        "entities": preprocessed.entities,
        "masked_question": preprocessed.masked_question,
        "mask_mapping": preprocessed.mask_mapping,
        "masked_question_structure": masked_paths,
        "question_structure": question_structure,
    }


def run_depo(
    question: str,
    parser: HanLPSDPParser,
    llm_client: OpenAIClient,
    *,
    question_structure_override: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Generate an atomic-question DAG from a natural-language question."""
    structure_result = extract_question_structure(question, parser, llm_client)
    question_structure = structure_result["question_structure"]
    if question_structure_override is not None:
        question_structure = question_structure_override

    dag = generate_atomic_question_dag(
        llm_client,
        question,
        structure_result["entities"],
        question_structure,
    )
    return {"entities": structure_result["entities"], "atomic_question_dag": dag}
