"""Use the LLM's entity mentions as retrieval anchors."""

from __future__ import annotations

from typing import Any

from ..llm.service import OpenAIAtomicLLMService
from .models import AtomicQuestionAnalysis


class AtomicQuestionAnalyzer:
    """Normalize the entity mentions returned for one atomic question."""

    def __init__(self, llm_service: OpenAIAtomicLLMService) -> None:
        self.llm_service = llm_service
    #让 LLM 分析一个 atomic question，从问题中提取实体
    def analyze(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
    ) -> AtomicQuestionAnalysis:
        payload = self.llm_service.analyze_atomic_question(
            atomic_question=atomic_question,
            dependency_answers=dependency_answers,
        )
        return AtomicQuestionAnalysis(entities=payload["entities"])
