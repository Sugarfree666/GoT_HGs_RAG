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
    unresolved_dependencies: list[dict[str, Any]] = field(default_factory=list)
    primary_anchor_entities: list[str] = field(default_factory=list)
    dependency_answers_used: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "resolved_question": self.retrieval_question,
            "retrieval_question": self.retrieval_question,
            "whether_rewritten": self.whether_rewritten,
            "replacement_span": self.replacement_span,
            "replacement_answer": self.replacement_answer,
            "replacements": [item.to_dict() for item in self.replacements],
            "dependency_replacements": [item.to_dict() for item in self.replacements],
            "unresolved_dependencies": list(self.unresolved_dependencies),
            "primary_anchor_entities": list(self.primary_anchor_entities),
            "dependency_answers_used": list(self.dependency_answers_used),
        }


def resolve_dependency_question(
    question: str,
    dependency_answers: list[dict[str, Any]],
    *,
    confidence_threshold: float = 0.7,
) -> DependencyQuestionRewrite:
    """Resolve explicit dependency-answer variables before legacy phrase rewriting."""

    retrieval_question = question
    replacements: list[DependencyReplacement] = []
    unresolved_dependencies: list[dict[str, Any]] = []
    primary_anchor_entities: list[str] = []
    dependency_answers_used: list[dict[str, Any]] = []

    for dependency in dependency_answers:
        dependency_node_id = str(dependency.get("node_id", "") or "").strip()
        if not dependency_node_id:
            continue
        matches = _find_dependency_variable_matches(retrieval_question, dependency_node_id)
        if not matches:
            continue
        answer = str(dependency.get("answer", "") or "").strip()
        confidence = _safe_float(dependency.get("confidence", 1.0))
        has_confidence = "confidence" in dependency
        if (
            not answer
            or normalize_label(answer).lower() in _INSUFFICIENT_ANSWERS
            or (has_confidence and confidence <= 0.0)
        ):
            unresolved_dependencies.append(
                {
                    "node_id": dependency_node_id,
                    "reason": "missing_or_low_confidence_answer",
                    "answer": answer,
                    "confidence": confidence,
                }
            )
            continue
        for match in reversed(matches):
            retrieval_question = (
                retrieval_question[: match.start()]
                + answer
                + retrieval_question[match.end() :]
            )
            replacements.insert(
                0,
                DependencyReplacement(
                    dependency_node_id=dependency_node_id,
                    replacement_span=match.group(0),
                    replacement_answer=answer,
                ),
            )
        if answer not in primary_anchor_entities:
            primary_anchor_entities.append(answer)
        dependency_answers_used.append(_dependency_answer_summary(dependency))

    if replacements:
        first_replacement = replacements[0]
        return DependencyQuestionRewrite(
            original_question=question,
            retrieval_question=retrieval_question,
            whether_rewritten=True,
            replacement_span=first_replacement.replacement_span,
            replacement_answer=first_replacement.replacement_answer,
            replacements=replacements,
            unresolved_dependencies=unresolved_dependencies,
            primary_anchor_entities=primary_anchor_entities,
            dependency_answers_used=dependency_answers_used,
        )

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
        if answer not in primary_anchor_entities:
            primary_anchor_entities.append(answer)
        dependency_answers_used.append(_dependency_answer_summary(dependency))

    first_replacement = replacements[0] if replacements else None
    return DependencyQuestionRewrite(
        original_question=question,
        retrieval_question=retrieval_question,
        whether_rewritten=bool(replacements),
        replacement_span=first_replacement.replacement_span if first_replacement else "",
        replacement_answer=first_replacement.replacement_answer if first_replacement else "",
        replacements=replacements,
        unresolved_dependencies=unresolved_dependencies,
        primary_anchor_entities=primary_anchor_entities,
        dependency_answers_used=dependency_answers_used,
    )


def _find_dependency_variable_matches(question: str, dependency_node_id: str) -> list[re.Match[str]]:
    qid = re.escape(dependency_node_id)
    patterns = (
        rf"\{{\s*{qid}\.answer\s*\}}",
        rf"\b{qid}\s*['\u2019]s\s+answer\b",
        rf"\b{qid}\s+answer\b",
        rf"\banswer\s+(?:of|to)\s+{qid}\b",
    )
    matches: list[re.Match[str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            span = (match.start(), match.end())
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in occupied):
                continue
            matches.append(match)
            occupied.append(span)
    return sorted(matches, key=lambda item: item.start())


def _dependency_answer_summary(dependency: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": str(dependency.get("node_id", "") or ""),
        "question": str(dependency.get("question", "") or ""),
        "answer": str(dependency.get("answer", "") or ""),
        "confidence": _safe_float(dependency.get("confidence", 0.0)),
        "answer_type": str(dependency.get("answer_type", "") or ""),
    }


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
