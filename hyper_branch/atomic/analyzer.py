"""Use the LLM's entity mentions as retrieval anchors."""

from __future__ import annotations

from typing import Any

from ..llm.service import AtomicLLMService
from ..utils import normalize_label
from .models import AtomicQuestionAnalysis


class AtomicQuestionAnalyzer:
    """Normalize the entity mentions returned for one atomic question."""

    def __init__(self, llm_service: AtomicLLMService) -> None:
        self.llm_service = llm_service

    def analyze(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
    ) -> AtomicQuestionAnalysis:
        payload = self.llm_service.analyze_atomic_question(
            atomic_question=atomic_question,
            dependency_answers=dependency_answers,
        )
        entities: list[str] = []
        for raw_entity in payload["entities"]:
            entity = normalize_label(str(raw_entity).strip())
            if entity and entity not in entities:
                entities.append(entity)
        return AtomicQuestionAnalysis(entities=entities, answer_type=_infer_answer_type(atomic_question))


def _infer_answer_type(question: str) -> str:
    lowered = question.strip().lower()
    if lowered.startswith(("which ", "what ")):
        return "short answer"
    if lowered.startswith("who "):
        return "person or organization"
    if lowered.startswith("where "):
        return "location"
    if lowered.startswith("when "):
        return "time or date"
    return "grounded short answer"
