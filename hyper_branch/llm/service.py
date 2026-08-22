"""LLM boundary for atomic analysis and evidence-grounded answers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .client import OpenAICompatibleClient
from .prompts import PromptManager


class AtomicLLMService(ABC):
    @abstractmethod
    def analyze_atomic_question(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, Any],
        dependency_answers: list[dict[str, Any]],
        evidence: dict[str, list[dict[str, Any]]],
        original_question: str = "",
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIAtomicLLMService(AtomicLLMService):
    def __init__(self, client: OpenAICompatibleClient, prompts: PromptManager) -> None:
        self.client = client
        self.prompts = prompts

    def analyze_atomic_question(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "atomic_question_analysis",
            self.prompts.get("atomic_question_analysis"),
            {"atomic_question": atomic_question, "dependency_answers": dependency_answers},
            max_tokens=300,
        )
        return {"entities": response["entities"]}

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, Any],
        dependency_answers: list[dict[str, Any]],
        evidence: dict[str, list[dict[str, Any]]],
        original_question: str = "",
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "atomic_answer",
            self.prompts.get("atomic_answer"),
            {
                "original_question": original_question,
                "atomic_question": atomic_question,
                "answer_contract": answer_contract,
                "dependency_answers": dependency_answers,
                "evidence_blocks": _answer_evidence_blocks(evidence),
            },
            max_tokens=900,
        )
        return {"answer": str(response["answer"]).strip()}


def _answer_evidence_blocks(evidence: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(block) for block in evidence["evidence_blocks"]]
