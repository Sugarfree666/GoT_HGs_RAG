from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterable

from models import DeclarativeView, RelationCarrierViewResult, SemanticNormalizationResult
from prompts import (
    RELATION_CARRIER_DECLARATIVE_SYSTEM,
    SEMANTIC_QUESTION_NORMALIZATION_SYSTEM,
    build_relation_carrier_declarative_prompt,
    build_semantic_question_normalization_prompt,
)

if TYPE_CHECKING:
    from llm_client import LLMClient


PLACEHOLDER_BASES = (
    "Album",
    "Age",
    "Book",
    "City",
    "Company",
    "Country",
    "Date",
    "Entity",
    "Film",
    "Institution",
    "Location",
    "Movie",
    "Nationality",
    "Network",
    "Organization",
    "Person",
    "Population",
    "Region",
    "Series",
    "SomeEntity",
    "Song",
    "Space",
    "System",
    "Time",
    "University",
    "Variable",
    "Work",
)
PLACEHOLDER_RE = re.compile(
    r"\b(?:X\d+(?:_[A-Za-z0-9]+)?|(?:"
    + "|".join(sorted(PLACEHOLDER_BASES, key=len, reverse=True))
    + r")(?:[A-Z][A-Za-z0-9]*|\d+))\b"
)

QUESTION_START_WORDS = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "how",
    "is",
    "may",
    "might",
    "must",
    "should",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "will",
    "would",
}


class SemanticQuestionNormalizer:
    """LLM-backed parser-facing view generator.

    ``normalize`` is kept as a deprecated compatibility wrapper for older
    callers. The new DEPO pipeline calls ``generate_relation_carrier_views``
    after explicit entity masking.
    """

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.llm_client = llm_client

    def normalize(
        self,
        question: str,
        placeholders: Iterable[str] | None = None,
    ) -> SemanticNormalizationResult:
        original_question = _normalize_space(question)
        explicit_placeholders = list(dict.fromkeys(placeholders or _extract_placeholders(original_question)))
        warnings: list[str] = []

        if self.llm_client is None:
            return SemanticNormalizationResult(
                original_question=original_question,
                normalized_question=original_question,
                changed=False,
                warnings=["Semantic normalization LLM unavailable; using original question."],
            )

        payload: dict[str, Any] = {}
        try:
            payload = self.llm_client.chat_json(
                SEMANTIC_QUESTION_NORMALIZATION_SYSTEM,
                build_semantic_question_normalization_prompt(
                    question=original_question,
                    placeholders=explicit_placeholders,
                ),
            )
        except Exception as exc:
            warnings.append(f"Semantic normalization LLM failed; using original question: {exc}")
            return SemanticNormalizationResult(
                original_question=original_question,
                normalized_question=original_question,
                changed=False,
                warnings=warnings,
                raw_payload=payload or None,
            )

        candidate = _clean_candidate_question(_candidate_from_payload(payload))
        added_type_variables = _parse_added_type_variables(payload.get("added_type_variables", []))
        if not candidate:
            warnings.append("Semantic normalization returned an empty question; using original question.")
            return SemanticNormalizationResult(
                original_question=original_question,
                normalized_question=original_question,
                changed=False,
                added_type_variables=[],
                warnings=warnings,
                raw_payload=payload or None,
            )

        return SemanticNormalizationResult(
            original_question=original_question,
            normalized_question=candidate,
            changed=_normalize_for_compare(candidate) != _normalize_for_compare(original_question),
            added_type_variables=added_type_variables,
            warnings=warnings,
            raw_payload=payload or None,
        )

    def generate_relation_carrier_views(
        self,
        *,
        original_question: str,
        masked_question: str,
        placeholders: Iterable[str] | None = None,
    ) -> RelationCarrierViewResult:
        masked_question = _normalize_space(masked_question)
        original_question = _normalize_space(original_question)
        explicit_placeholders = list(dict.fromkeys(placeholders or _extract_placeholders(masked_question)))
        warnings: list[str] = []

        if self.llm_client is None:
            return _heuristic_relation_carrier_result(
                original_question=original_question,
                masked_question=masked_question,
                warnings=["Relation-carrier LLM unavailable; using heuristic declarative views."],
            )

        payload: dict[str, Any] = {}
        try:
            payload = self.llm_client.chat_json(
                RELATION_CARRIER_DECLARATIVE_SYSTEM,
                build_relation_carrier_declarative_prompt(
                    original_question=original_question,
                    masked_question=masked_question,
                    placeholders=explicit_placeholders,
                ),
            )
        except Exception as exc:
            warnings.append(f"Relation-carrier LLM failed; using heuristic declarative views: {exc}")
            result = _heuristic_relation_carrier_result(
                original_question=original_question,
                masked_question=masked_question,
                warnings=warnings,
            )
            result.raw_payload = payload or None
            return result

        result = _parse_relation_carrier_payload(
            payload,
            original_question=original_question,
            masked_question=masked_question,
            placeholders=explicit_placeholders,
        )
        if warnings:
            result.warnings[:0] = warnings
        return result


class RelationCarrierDeclarativeGenerator(SemanticQuestionNormalizer):
    """Named alias for the new parser-facing relation-carrier generator."""

    pass


def _candidate_from_payload(payload: dict[str, Any]) -> str:
    for key in ("normalized_question", "normalizedQuestion", "question"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _parse_added_type_variables(raw_items: Any) -> list[dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    result: list[dict[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        text = _normalize_space(str(raw.get("text", "")))
        trigger_text = _normalize_space(str(raw.get("trigger_text", raw.get("trigger", ""))))
        reason = _normalize_space(str(raw.get("reason", "")))
        if not text or not trigger_text:
            continue
        result.append({"text": text, "trigger_text": trigger_text, "reason": reason})
    return result


def _parse_relation_carrier_payload(
    payload: dict[str, Any],
    *,
    original_question: str,
    masked_question: str,
    placeholders: list[str],
) -> RelationCarrierViewResult:
    if not isinstance(payload, dict):
        return _heuristic_relation_carrier_result(
            original_question=original_question,
            masked_question=masked_question,
            warnings=["Relation-carrier LLM returned a non-object payload; using heuristic declarative views."],
        )

    warnings = _str_list(payload.get("warnings"))
    raw_views = payload.get("declarative_views")
    views: list[DeclarativeView] = []
    if isinstance(raw_views, list):
        for index, item in enumerate(raw_views, start=1):
            if not isinstance(item, dict):
                continue
            view_id = _normalize_space(str(item.get("id") or f"view_{index}"))
            sentence = _clean_declarative_sentence(str(item.get("sentence") or ""))
            purpose = _normalize_space(str(item.get("purpose") or "relation_carrier")) or "relation_carrier"
            if not sentence:
                continue
            missing = [placeholder for placeholder in placeholders if placeholder in masked_question and placeholder not in sentence]
            metadata = {"missing_placeholders": missing} if missing else {}
            views.append(DeclarativeView(id=view_id, sentence=sentence, purpose=purpose, metadata=metadata))

    if not views:
        fallback = _heuristic_relation_carrier_result(
            original_question=original_question,
            masked_question=masked_question,
            warnings=["Relation-carrier LLM produced no usable views; using heuristic declarative views."],
        )
        fallback.raw_payload = payload
        return fallback

    operator_intent = payload.get("operator_intent") if isinstance(payload.get("operator_intent"), dict) else {}
    return RelationCarrierViewResult(
        masked_question=masked_question,
        declarative_views=views,
        operator_intent=_normalize_operator_intent(operator_intent, masked_question),
        warnings=warnings,
        raw_payload=payload,
    )


def _heuristic_relation_carrier_result(
    *,
    original_question: str,
    masked_question: str,
    warnings: list[str] | None = None,
) -> RelationCarrierViewResult:
    del original_question
    views = _heuristic_declarative_views(masked_question)
    return RelationCarrierViewResult(
        masked_question=masked_question,
        declarative_views=views,
        operator_intent=_infer_operator_intent(masked_question),
        warnings=list(warnings or []),
    )


def _heuristic_declarative_views(masked_question: str) -> list[DeclarativeView]:
    text = _normalize_space(masked_question.rstrip("?"))
    views: list[DeclarativeView] = []

    placeholder_pattern = _placeholder_capture_pattern()
    nationality_song = re.search(
        r"\bwhat\s+(?:is\s+the\s+)?nationality\s+(?:is\s+)?(?:of\s+)?the\s+performer\s+of\s+(?:the\s+)?song\s+(" + placeholder_pattern + r")\b",
        text,
        flags=re.IGNORECASE,
    )
    if nationality_song:
        song = nationality_song.group(1)
        views.append(DeclarativeView(id="view_1", sentence=f"The song {song} has a performer."))
        views.append(DeclarativeView(id="view_2", sentence="The performer has a nationality."))
        return views

    performer_song = re.search(r"\bperformer\s+of\s+(?:the\s+)?song\s+(" + placeholder_pattern + r")\b", text, flags=re.IGNORECASE)
    if performer_song:
        song = performer_song.group(1)
        views.append(DeclarativeView(id="view_1", sentence=f"The song {song} has a performer."))

    for index, phrase in enumerate(_of_relation_phrases(text), start=len(views) + 1):
        views.append(DeclarativeView(id=f"view_{index}", sentence=phrase))

    if not views:
        views.append(DeclarativeView(id="view_1", sentence=_question_to_soft_declarative(text)))
    return views


def _of_relation_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    pattern = re.compile(r"\b(?P<head>[A-Za-z][A-Za-z0-9_-]*)\s+of\s+(?P<tail>[A-Z][A-Za-z0-9_]*|the\s+[A-Za-z][A-Za-z0-9_-]*)", re.IGNORECASE)
    for match in pattern.finditer(text):
        head = match.group("head")
        tail = match.group("tail")
        if head.lower() in QUESTION_START_WORDS or head.lower() in {"place", "which", "what"}:
            continue
        phrases.append(f"{tail.strip().capitalize()} has a {head}.")
    return phrases


def _placeholder_capture_pattern() -> str:
    pattern = PLACEHOLDER_RE.pattern
    if pattern.startswith(r"\b"):
        pattern = pattern[2:]
    if pattern.endswith(r"\b"):
        pattern = pattern[:-2]
    return pattern


def _question_to_soft_declarative(text: str) -> str:
    cleaned = _normalize_space(text)
    if not cleaned:
        return "The question has a relation."
    if cleaned.endswith("."):
        return cleaned
    return cleaned[0].upper() + cleaned[1:] + "."


def _clean_declarative_sentence(candidate: str) -> str:
    cleaned = _normalize_space(candidate.strip().strip("\"'"))
    if not cleaned:
        return ""
    if cleaned.endswith("?"):
        cleaned = cleaned[:-1].strip()
    if not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def _normalize_operator_intent(raw: dict[str, Any], text: str) -> dict[str, Any]:
    inferred = _infer_operator_intent(text)
    result = dict(inferred)
    for key, value in raw.items():
        if value not in (None, "", []):
            result[str(key)] = value
    result.setdefault("type", inferred["type"])
    result.setdefault("cues", inferred.get("cues", []))
    return result


def _infer_operator_intent(text: str) -> dict[str, Any]:
    lower = text.lower()
    cues: list[str] = []
    operator_type = "lookup"
    if "how many" in lower:
        operator_type = "count"
        cues.append("how many")
    elif any(cue in lower for cue in ("both", "same", "whether", "are ", "do ", "does ")):
        operator_type = "boolean"
        cues.extend([cue.strip() for cue in ("both", "same") if cue in lower])
    elif any(cue in lower for cue in ("older", "younger", "first", "earlier", "later", "largest", "smallest")):
        operator_type = "comparison"
        cues.extend([cue for cue in ("older", "younger", "first", "earlier", "later", "largest", "smallest") if cue in lower])
    elif " or " in lower and lower.startswith("which"):
        operator_type = "comparison"
        cues.append("which/or")
    elif "common" in lower or "both" in lower and "what" in lower:
        operator_type = "intersection"
        cues.append("both")

    target_hint = ""
    answer_type_hint = ""
    if "nationality" in lower:
        target_hint = "nationality"
        answer_type_hint = "Nationality"
    elif "where" in lower or "place" in lower:
        target_hint = "location"
        answer_type_hint = "Location"
    elif "when" in lower or "year" in lower or "date" in lower:
        target_hint = "date"
        answer_type_hint = "Date"
    elif "who" in lower:
        target_hint = "person"
        answer_type_hint = "Person"

    return {
        "type": operator_type,
        "target_hint": target_hint,
        "answer_type_hint": answer_type_hint,
        "cues": cues,
    }


def _str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        value = _normalize_space(raw)
        return [value] if value else []
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for item in raw:
        value = _normalize_space(str(item))
        if value:
            result.append(value)
    return result


def _clean_candidate_question(candidate: str) -> str:
    cleaned = _normalize_space(candidate.strip().strip("\"'"))
    if not cleaned:
        return ""
    if cleaned.endswith(".") and _looks_like_question(cleaned[:-1] + "?"):
        cleaned = cleaned[:-1] + "?"
    if not cleaned.endswith("?") and _looks_like_question(cleaned):
        cleaned += "?"
    return cleaned


def _looks_like_question(text: str) -> bool:
    first = _first_word(text)
    return first in QUESTION_START_WORDS or bool(
        re.match(r"^\s*(?:in|on|at|from|to|for)\s+(?:what|which)\b", text, flags=re.IGNORECASE)
    )


def _extract_placeholders(text: str) -> list[str]:
    return [match.group(0) for match in PLACEHOLDER_RE.finditer(text)]


def _first_word(text: str) -> str:
    match = re.search(r"[A-Za-z]+", text)
    return match.group(0).lower() if match else ""


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_compare(text: str) -> str:
    return _normalize_space(text).lower()
