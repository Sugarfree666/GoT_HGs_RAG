from __future__ import annotations

import re
from typing import Any

from ..utils import content_tokens, ensure_list, normalize_label
from .models import AtomicQuestionAnalysis


CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9'&.-]+)(?:\s+(?:[A-Z][a-zA-Z0-9'&.-]+|of|the|and|&))*"
)
WH_WORDS = {"what", "which", "who", "where", "when", "why", "how"}
RELATION_STOPWORDS = WH_WORDS | {"did", "does", "do", "is", "are", "was", "were", "a", "an", "the"}


class AtomicQuestionAnalyzer:
    def __init__(self, llm_service: Any | None = None) -> None:
        self.llm_service = llm_service

    def analyze(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]] | None = None,
    ) -> AtomicQuestionAnalysis:
        dependency_answers = dependency_answers or []
        if self.llm_service is not None and hasattr(self.llm_service, "analyze_atomic_question"):
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
        entities = self._clean_text_list(payload.get("entities", []))
        relations = self._clean_text_list(payload.get("relations", []))
        relation_query = normalize_label(str(payload.get("relation_query", "") or "")).strip()
        answer_type = normalize_label(str(payload.get("answer_type", "") or "")).strip()
        if not relation_query:
            relation_query = self._mask_entities(atomic_question, entities)
        if not answer_type:
            answer_type = _infer_answer_type(atomic_question)
        return AtomicQuestionAnalysis(
            entities=entities,
            relations=relations,
            relation_query=relation_query,
            answer_type=answer_type,
        )

    def _heuristic_analysis(self, atomic_question: str) -> dict[str, Any]:
        entities = _extract_capitalized_entities(atomic_question)
        relations = _extract_relation_phrases(atomic_question, entities)
        answer_type = _infer_answer_type(atomic_question)
        relation_query = self._mask_entities(atomic_question, entities)
        return {
            "entities": entities,
            "relations": relations,
            "relation_query": relation_query,
            "answer_type": answer_type,
        }

    def _mask_entities(self, atomic_question: str, entities: list[str]) -> str:
        masked = atomic_question.strip().rstrip("?")
        for entity in entities:
            if not entity:
                continue
            masked = re.sub(re.escape(entity), "an entity", masked, flags=re.IGNORECASE)
        tokens = content_tokens(masked)
        if not tokens:
            return masked
        if entities:
            return " ".join(tokens)
        return masked

    def _clean_text_list(self, value: Any) -> list[str]:
        cleaned: list[str] = []
        for item in ensure_list(value):
            text = normalize_label(str(item).strip())
            if text and text not in cleaned:
                cleaned.append(text)
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


def _extract_relation_phrases(question: str, entities: list[str]) -> list[str]:
    lowered = question.lower().rstrip("?")
    for entity in entities:
        lowered = lowered.replace(entity.lower(), " ")
    tokens = [token for token in content_tokens(lowered) if token not in RELATION_STOPWORDS]
    if not tokens:
        return []
    if "from" in lowered and "graduate" in lowered:
        return ["graduate from"]
    if "known for" in lowered:
        return ["known for"]
    if "located in" in lowered:
        return ["located in"]
    return [" ".join(tokens[:5])]


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
