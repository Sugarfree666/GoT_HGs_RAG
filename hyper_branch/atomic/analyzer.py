from __future__ import annotations

import re
from typing import Any

from ..llm.service import AtomicLLMService
from ..utils import ensure_list, normalize_label
from .models import AtomicQuestionAnalysis


CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9'&.-]+)(?:\s+(?:[A-Z][a-zA-Z0-9'&.-]+|of|the|and|&))*"
)
WH_WORDS = {"what", "which", "who", "where", "when", "why", "how"}
POSSESSIVE_ROLE_TERMS = {
    "actor",
    "actress",
    "artist",
    "author",
    "brother",
    "child",
    "composer",
    "creator",
    "daughter",
    "director",
    "father",
    "founder",
    "grandfather",
    "grandmother",
    "husband",
    "mother",
    "parent",
    "performer",
    "producer",
    "sister",
    "son",
    "spouse",
    "wife",
    "writer",
}


class AtomicQuestionAnalyzer:
    def __init__(self, llm_service: AtomicLLMService | None = None) -> None:
        self.llm_service = llm_service

    def analyze(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]] | None = None,
    ) -> AtomicQuestionAnalysis:
        dependency_answers = dependency_answers or []
        if self.llm_service is not None:
            payload = self.llm_service.analyze_atomic_question(
                atomic_question=atomic_question,
                dependency_answers=dependency_answers,
            )
        else:
            payload = self._heuristic_analysis(atomic_question)
        return self._coerce_payload(payload, atomic_question)

    def _coerce_payload(self, payload: Any, atomic_question: str) -> AtomicQuestionAnalysis:
        if not isinstance(payload, dict):
            payload = self._heuristic_analysis(atomic_question)
        entities = self._clean_entity_mentions(payload.get("entities", []))
        answer_type = _infer_answer_type(atomic_question)
        return AtomicQuestionAnalysis(
            entities=entities,
            answer_type=answer_type,
        )

    def _heuristic_analysis(self, atomic_question: str) -> dict[str, Any]:
        entities = _extract_capitalized_entities(atomic_question)
        return {
            "entities": entities,
        }

    def _clean_entity_mentions(self, value: Any) -> list[str]:
        cleaned: list[str] = []
        for item in ensure_list(value):
            text = normalize_label(str(item).strip())
            if not text:
                continue
            entity = _strip_possessive_role_tail(text)
            if entity and entity not in cleaned:
                cleaned.append(entity)
        return cleaned


def _extract_capitalized_entities(question: str) -> list[str]:
    entities: list[str] = []
    for match in CAPITALIZED_PHRASE_RE.finditer(question):
        text = normalize_label(match.group(0))
        if not text:
            continue
        if text.lower() in WH_WORDS:
            continue
        if text not in entities:
            entities.append(text)
    return entities


def _strip_possessive_role_tail(text: str) -> str:
    match = re.match(r"^(.+?)'s\s+([A-Za-z][A-Za-z -]*)$", text)
    if not match:
        return text
    owner = normalize_label(match.group(1))
    tail = normalize_label(match.group(2)).lower()
    if tail in POSSESSIVE_ROLE_TERMS:
        return owner
    return text


def _infer_answer_type(question: str) -> str:
    lowered = question.strip().lower()
    if lowered.startswith("which "):
        tokens = lowered.split()
        if len(tokens) > 1:
            return tokens[1].strip(" ?.,")
    if lowered.startswith("what "):
        tokens = lowered.split()
        if len(tokens) > 1 and tokens[1] not in {"is", "are", "was", "were", "did", "does", "do"}:
            return tokens[1].strip(" ?.,")
        return "entity, concept, or phrase"
    if lowered.startswith("who "):
        return "person or organization"
    if lowered.startswith("where "):
        return "location"
    if lowered.startswith("when "):
        return "time or date"
    return "grounded short answer"
