"""LLM boundary for evidence-grounded atomic answers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .client import OpenAICompatibleClient

ATOMIC_ANSWER_PROMPT = (
    Path(__file__).resolve().parents[2] / "prompts" / "atomic_answer.md"
).read_text(encoding="utf-8").strip()


class OpenAIAtomicLLMService:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def answer_atomic_question(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
        evidence: dict[str, list[dict[str, Any]]],
        original_question: str = "",
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            ATOMIC_ANSWER_PROMPT,
            {
                "original_question": original_question,
                "atomic_question": atomic_question,
                "dependency_answers": dependency_answers,
                "evidence_blocks": _answer_evidence_blocks(evidence),
            },
            max_tokens=900,
        )
        return {"answer": str(response["answer"]).strip()}


def _answer_evidence_blocks(evidence: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(block) for block in evidence["evidence_blocks"]]
