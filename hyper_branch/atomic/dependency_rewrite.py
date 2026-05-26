from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..utils import normalize_label


_INSUFFICIENT_ANSWERS = {
    "",
    "insufficient_evidence",
    "insufficient evidence",
    "unknown",
    "none",
    "n/a",
    "yes",
    "no",
    "true",
    "false",
}
_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
_NON_ENTITY_SINGLE_WORDS = {
    "american",
    "athenian",
    "british",
    "canadian",
    "chinese",
    "dutch",
    "english",
    "french",
    "german",
    "greek",
    "indian",
    "irish",
    "italian",
    "japanese",
    "polish",
    "russian",
    "spanish",
}
_NON_ENTITY_ANSWER_TYPES = {
    "age",
    "boolean",
    "count",
    "date",
    "duration",
    "nationality",
    "number",
    "time",
    "year",
}


@dataclass(slots=True)
class DependencyReplacement:
    dependency_node_id: str
    replacement_span: str
    replacement_answer: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dependency_node_id": self.dependency_node_id,
            "replacement_span": self.replacement_span,
            "replacement_answer": self.replacement_answer,
        }


@dataclass(slots=True)
class DependencyQuestionRewrite:
    original_question: str
    retrieval_question: str
    whether_rewritten: bool
    replacement_span: str = ""
    replacement_answer: str = ""
    replacements: list[DependencyReplacement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "retrieval_question": self.retrieval_question,
            "whether_rewritten": self.whether_rewritten,
            "replacement_span": self.replacement_span,
            "replacement_answer": self.replacement_answer,
            "replacements": [item.to_dict() for item in self.replacements],
        }


def resolve_dependency_question(
    question: str,
    dependency_answers: list[dict[str, Any]],
    *,
    confidence_threshold: float = 0.7,
) -> DependencyQuestionRewrite:
    """Rewrite only resolved intermediate phrases using high-confidence entity answers."""

    retrieval_question = question
    replacements: list[DependencyReplacement] = []
    for dependency in dependency_answers:
        answer = str(dependency.get("answer", "") or "").strip()
        confidence = _safe_float(dependency.get("confidence", 0.0))
        if confidence <= confidence_threshold:
            continue
        if not is_entity_like_answer(answer, dependency.get("answer_type")):
            continue
        span = _resolved_intermediate_span(str(dependency.get("question", "") or ""))
        if not span:
            continue
        match = _find_span_match(retrieval_question, span)
        if match is None:
            continue
        retrieval_question = (
            retrieval_question[: match.start()]
            + answer
            + retrieval_question[match.end() :]
        )
        replacements.append(
            DependencyReplacement(
                dependency_node_id=str(dependency.get("node_id", "") or ""),
                replacement_span=match.group(0),
                replacement_answer=answer,
            )
        )

    first_replacement = replacements[0] if replacements else None
    return DependencyQuestionRewrite(
        original_question=question,
        retrieval_question=retrieval_question,
        whether_rewritten=bool(replacements),
        replacement_span=first_replacement.replacement_span if first_replacement else "",
        replacement_answer=first_replacement.replacement_answer if first_replacement else "",
        replacements=replacements,
    )


def is_entity_like_answer(answer: str, answer_type: Any = None) -> bool:
    text = normalize_label(answer).strip()
    lowered = text.lower()
    if lowered in _INSUFFICIENT_ANSWERS:
        return False
    if any(answer_type_name in normalize_label(str(answer_type)).lower() for answer_type_name in _NON_ENTITY_ANSWER_TYPES):
        return False
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", text)
    if not tokens or len(tokens) > 10 or len(text) > 100:
        return False
    token_lowers = {token.lower().strip(".") for token in tokens}
    if token_lowers & _MONTHS:
        return False
    if re.fullmatch(r"(?:c\.\s*)?\d{1,4}(?:[-/]\d{1,2})?(?:[-/]\d{1,4})?", lowered):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?%?", lowered):
        return False
    if len(tokens) == 1 and lowered in _NON_ENTITY_SINGLE_WORDS:
        return False
    if re.search(r"[.!?]\s+\w", text):
        return False
    if len(tokens) > 4 and re.search(r"\b(?:is|are|was|were|because|therefore)\b", lowered):
        return False
    return any(any(char.isupper() for char in token) for token in tokens)


def _resolved_intermediate_span(dependency_question: str) -> str:
    text = normalize_label(dependency_question).strip().rstrip("?")
    patterns = (
        r"^(?:who|what) (?:is|was|are|were) (?P<span>.+)$",
        r"^(?:who|what) (?:did|does|do) (?P<span>.+?) (?:become|be)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        span = match.group("span").strip()
        if _is_intermediate_span(span):
            return span
    return ""


def _is_intermediate_span(span: str) -> bool:
    lowered = span.lower()
    if " of " in lowered:
        return True
    if "'s " in lowered or "’s " in lowered:
        return True
    return False


def _find_span_match(question: str, span: str) -> re.Match[str] | None:
    for variant in _span_variants(span):
        pattern = r"(?<!\w)" + re.escape(variant) + r"(?!\w)"
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match
    return None


def _span_variants(span: str) -> list[str]:
    variants = [normalize_label(span).strip()]
    replacements = (
        (" of the song ", " of "),
        (" of song ", " of "),
        (" of the film ", " of "),
        (" of film ", " of "),
        (" of the movie ", " of "),
        (" of movie ", " of "),
    )
    for source, target in replacements:
        variant = re.sub(
            re.escape(source.strip()),
            target.strip(),
            variants[0],
            flags=re.IGNORECASE,
        )
        if variant != variants[0]:
            variants.append(normalize_label(variant))
    return [variant for index, variant in enumerate(variants) if variant and variant not in variants[:index]]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
