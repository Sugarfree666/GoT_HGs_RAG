from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from models import ExplicitEntity, ExplicitEntityResult, HanLPSDPPreprocessResult, MaskMapping
from prompts import HANLP_SDP_PREPROCESS_SYSTEM, build_hanlp_sdp_preprocess_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient


class HanLPSDPPreprocessor:
    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def preprocess(self, question: str) -> HanLPSDPPreprocessResult:
        raw_payload = self.llm_client.chat_json(
            HANLP_SDP_PREPROCESS_SYSTEM,
            build_hanlp_sdp_preprocess_prompt(question),
            max_retries=1,
        )
        warnings = _coerce_warnings(raw_payload.get("warnings"))
        explicit_entities = _explicit_entities_from_payload(question, raw_payload, warnings)
        placeholder_rewrites = _placeholder_rewrites(raw_payload, explicit_entities.entities, warnings)
        masked_question = _masked_question_from_entities(question, explicit_entities.entities, placeholder_rewrites)
        raw_masked_question = str(raw_payload.get("masked_question") or "").strip()
        if raw_masked_question and raw_masked_question != masked_question:
            warnings.append("Rebuilt masked_question from validated entity spans and canonical ENTITY* placeholders.")

        sdp_input_sentence = masked_question
        mask_mappings = _mask_mappings_from_entities(masked_question, explicit_entities.entities, placeholder_rewrites)
        _validate_preprocess_result(
            question=question,
            masked_question=masked_question,
            mask_mappings=mask_mappings,
            warnings=warnings,
        )
        return HanLPSDPPreprocessResult(
            original_question=question,
            explicit_entities=explicit_entities,
            masked_question=masked_question,
            sdp_input_sentence=sdp_input_sentence,
            mask_mappings=mask_mappings,
            warnings=warnings,
            raw_payload=raw_payload,
        )


def _explicit_entities_from_payload(
    question: str,
    payload: dict[str, Any],
    warnings: list[str],
) -> ExplicitEntityResult:
    raw_entities = payload.get("explicit_entities", [])
    if not isinstance(raw_entities, list):
        raise ValueError("HanLP SDP preprocess payload explicit_entities must be a list.")

    entities: list[ExplicitEntity] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = _coerce_int(item.get("start_char", item.get("start")))
        end = _coerce_int(item.get("end_char", item.get("end")))
        resolved = _resolve_entity_span(question, text, start, end)
        if resolved is None:
            warnings.append(f"Dropped explicit entity whose text was not found in the original question: {text!r}.")
            continue
        resolved_start, resolved_end = resolved
        if (start, end) != (resolved_start, resolved_end):
            warnings.append(
                f"Corrected explicit entity span for {text!r} from ({start}, {end}) "
                f"to ({resolved_start}, {resolved_end})."
            )
        entities.append(
            ExplicitEntity(
                text=question[resolved_start:resolved_end],
                start_char=resolved_start,
                end_char=resolved_end,
                semantic_type_hint=str(item.get("semantic_type_hint") or "Entity").strip() or "Entity",
                confidence=_coerce_float(item.get("confidence"), 1.0),
                reason=str(item.get("reason") or "").strip(),
            )
        )

    entities = _remove_overlapping_entities(entities, warnings)
    return ExplicitEntityResult(entities=entities, warnings=warnings, raw_payload=payload)


def _placeholder_rewrites(
    payload: dict[str, Any],
    entities: list[ExplicitEntity],
    warnings: list[str],
) -> dict[str, str]:
    raw_mappings = payload.get("mask_mappings", [])
    raw_placeholder_by_text: dict[str, str] = {}
    if isinstance(raw_mappings, list):
        for item in raw_mappings:
            if not isinstance(item, dict):
                continue
            original_text = str(item.get("original_text") or item.get("text") or "").strip()
            placeholder = str(item.get("placeholder") or "").strip()
            if original_text and placeholder:
                raw_placeholder_by_text.setdefault(_norm(original_text), placeholder)

    rewrites: dict[str, str] = {}
    for index, entity in enumerate(sorted(entities, key=lambda item: item.start_char)):
        canonical = f"ENTITY{_letter_suffix(index)}"
        raw_placeholder = raw_placeholder_by_text.get(_norm(entity.text), canonical)
        if raw_placeholder != canonical:
            warnings.append(f"Canonicalized placeholder {raw_placeholder!r} to {canonical!r}.")
        rewrites[raw_placeholder] = canonical
        rewrites[canonical] = canonical
    return rewrites


def _masked_question_from_entities(
    question: str,
    entities: list[ExplicitEntity],
    placeholder_rewrites: dict[str, str],
) -> str:
    replacements: list[tuple[int, int, str]] = []
    for index, entity in enumerate(sorted(entities, key=lambda item: item.start_char)):
        placeholder = placeholder_rewrites.get(f"ENTITY{_letter_suffix(index)}", f"ENTITY{_letter_suffix(index)}")
        replacements.append((entity.start_char, entity.end_char, placeholder))

    masked = question
    for start, end, placeholder in sorted(replacements, key=lambda item: item[0], reverse=True):
        masked = masked[:start] + placeholder + masked[end:]
    return masked


def _mask_mappings_from_entities(
    masked_question: str,
    entities: list[ExplicitEntity],
    placeholder_rewrites: dict[str, str],
) -> list[MaskMapping]:
    mappings: list[MaskMapping] = []
    for index, entity in enumerate(sorted(entities, key=lambda item: item.start_char)):
        placeholder = placeholder_rewrites.get(f"ENTITY{_letter_suffix(index)}", f"ENTITY{_letter_suffix(index)}")
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


def _resolve_entity_span(
    question: str,
    text: str,
    start: int | None,
    end: int | None,
) -> tuple[int, int] | None:
    if start is not None and end is not None and 0 <= start < end <= len(question):
        if question[start:end] == text:
            return start, end
    matches = list(re.finditer(re.escape(text), question))
    if not matches:
        matches = list(re.finditer(re.escape(text), question, flags=re.IGNORECASE))
    if not matches:
        return None
    if start is not None:
        best = min(matches, key=lambda match: abs(match.start() - start))
    else:
        best = matches[0]
    return best.start(), best.end()


def _remove_overlapping_entities(entities: list[ExplicitEntity], warnings: list[str]) -> list[ExplicitEntity]:
    ordered = sorted(entities, key=lambda item: (item.start_char, -(item.end_char - item.start_char)))
    kept: list[ExplicitEntity] = []
    occupied: list[tuple[int, int]] = []
    for entity in ordered:
        if any(not (entity.end_char <= start or entity.start_char >= end) for start, end in occupied):
            warnings.append(f"Dropped overlapping explicit entity span text={entity.text!r}.")
            continue
        kept.append(entity)
        occupied.append((entity.start_char, entity.end_char))
    return kept


def _find_placeholder_span(text: str, placeholder: str) -> tuple[int, int] | None:
    match = re.search(rf"\b{re.escape(placeholder)}\b", text)
    if not match:
        return None
    return match.start(), match.end()


def _coerce_warnings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _letter_suffix(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    current = index
    while True:
        label = alphabet[current % len(alphabet)] + label
        current = current // len(alphabet) - 1
        if current < 0:
            return label


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()
