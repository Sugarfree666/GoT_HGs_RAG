from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mask_span_extractor import ExplicitEntityExtractor
from models import ExplicitEntity, ExplicitEntityResult, HanLPSDPPreprocessResult, MaskMapping

if TYPE_CHECKING:
    from llm_client import LLMClient


class EntityMaskingPreprocessor:
    """Generic Step 2 entity masking for DEPO parser pipelines.

    LLM usage is limited to explicit entity extraction via
    EXPLICIT_ENTITY_EXTRACTION_SYSTEM/build_explicit_entity_extraction_prompt.
    Placeholder assignment and masked question construction are deterministic.
    """

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.entity_extractor = ExplicitEntityExtractor(llm_client)

    def preprocess(self, question: str) -> HanLPSDPPreprocessResult:
        explicit_entities = self.entity_extractor.extract(question)
        masked_question = _masked_question_from_entities(question, explicit_entities.entities)
        mask_mappings = _mask_mappings_from_entities(masked_question, explicit_entities.entities)
        _validate_preprocess_result(
            question=question,
            masked_question=masked_question,
            mask_mappings=mask_mappings,
            warnings=explicit_entities.warnings,
        )
        return HanLPSDPPreprocessResult(
            original_question=question,
            explicit_entities=ExplicitEntityResult(
                entities=list(explicit_entities.entities),
                warnings=list(explicit_entities.warnings),
                raw_payload=explicit_entities.raw_payload,
            ),
            masked_question=masked_question,
            sdp_input_sentence=masked_question,
            mask_mappings=mask_mappings,
            warnings=list(explicit_entities.warnings),
            raw_payload=explicit_entities.raw_payload,
        )


def _masked_question_from_entities(question: str, entities: list[ExplicitEntity]) -> str:
    masked = question
    replacements = [
        (entity.start_char, entity.end_char, f"ENTITY{_letter_suffix(index)}")
        for index, entity in enumerate(sorted(entities, key=lambda item: (item.start_char, item.end_char)))
    ]
    for start, end, placeholder in sorted(replacements, key=lambda item: item[0], reverse=True):
        masked = masked[:start] + placeholder + masked[end:]
    return masked


def _mask_mappings_from_entities(masked_question: str, entities: list[ExplicitEntity]) -> list[MaskMapping]:
    mappings: list[MaskMapping] = []
    for index, entity in enumerate(sorted(entities, key=lambda item: (item.start_char, item.end_char))):
        placeholder = f"ENTITY{_letter_suffix(index)}"
        masked_span = _find_placeholder_span(masked_question, placeholder)
        mappings.append(
            MaskMapping(
                placeholder=placeholder,
                original_text=entity.text,
                kind_hint="entity",
                semantic_type_hint=entity.semantic_type_hint or "Entity",
                original_char_span=[entity.start_char, entity.end_char],
                masked_char_span=list(masked_span) if masked_span else [],
            )
        )
    return mappings


def _validate_preprocess_result(
    question: str,
    masked_question: str,
    mask_mappings: list[MaskMapping],
    warnings: list[str],
) -> None:
    for mapping in mask_mappings:
        if mapping.placeholder not in masked_question:
            raise ValueError(f"masked_question is missing placeholder {mapping.placeholder}.")
        if len(mapping.original_text.split()) > 1 and mapping.original_text in masked_question:
            raise ValueError(f"masked_question still contains unmasked multi-word entity {mapping.original_text!r}.")
        if mapping.original_text not in question:
            warnings.append(f"Mask mapping original_text was not found in original question: {mapping.original_text!r}.")


def _find_placeholder_span(text: str, placeholder: str) -> tuple[int, int] | None:
    match = re.search(rf"\b{re.escape(placeholder)}\b", text)
    if not match:
        return None
    return match.start(), match.end()


def _letter_suffix(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    current = index
    while True:
        label = alphabet[current % len(alphabet)] + label
        current = current // len(alphabet) - 1
        if current < 0:
            return label
