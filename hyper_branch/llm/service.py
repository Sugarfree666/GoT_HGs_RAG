from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..utils import content_tokens, ensure_list, normalize_label, short_text
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
        evidence: Any,
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
            {
                "atomic_question": atomic_question,
                "dependency_answers": dependency_answers,
            },
            max_tokens=300,
        )
        response.setdefault("entities", [])
        return {"entities": response.get("entities", [])}

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, Any],
        dependency_answers: list[dict[str, Any]],
        evidence: Any,
        original_question: str = "",
    ) -> dict[str, Any]:
        evidence_blocks = _answer_evidence_blocks(evidence)
        response = self.client.chat_json(
            "atomic_answer",
            self.prompts.get("atomic_answer"),
            {
                "original_question": original_question,
                "atomic_question": atomic_question,
                "answer_contract": answer_contract,
                "dependency_answers": dependency_answers,
                "evidence_blocks": evidence_blocks,
            },
            max_tokens=900,
        )
        if not isinstance(response, dict):
            return {"answer": ""}
        payload = dict(response)
        payload["answer"] = str(response.get("answer", "") or "")
        return payload

class MockAtomicLLMService(AtomicLLMService):
    def __init__(
        self,
        answer_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.answer_responses = list(answer_responses or [])
        self.answer_calls: list[dict[str, Any]] = []

    def analyze_atomic_question(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del dependency_answers
        entities = [
            phrase
            for phrase in _extract_topic_phrases(atomic_question)
            if phrase.lower() not in {"what", "which", "who", "where", "when", "how"}
        ][:4]
        return {"entities": entities}

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, Any],
        dependency_answers: list[dict[str, Any]],
        evidence: Any,
        original_question: str = "",
    ) -> dict[str, Any]:
        evidence_blocks = _answer_evidence_blocks(evidence)
        self.answer_calls.append(
            {
                "original_question": original_question,
                "atomic_question": atomic_question,
                "answer_contract": answer_contract,
                "dependency_answers": dependency_answers,
                "evidence_blocks": evidence_blocks,
            }
        )
        if self.answer_responses:
            response = self.answer_responses.pop(0)
            return {"answer": str(response.get("answer", "") or "")} if isinstance(response, dict) else {"answer": ""}
        if not evidence_blocks:
            return {"answer": "INSUFFICIENT_EVIDENCE"}

        query_tokens = set(content_tokens(atomic_question))
        answer = ""
        first_hyperedge_text = ""
        for block in evidence_blocks:
            block_texts = [
                str(block.get("title", "") or ""),
                str(block.get("text", "") or ""),
            ]
            for hyperedge in ensure_list(block.get("hyperedges", [])):
                if not isinstance(hyperedge, dict):
                    continue
                hyperedge_text = str(hyperedge.get("hyperedge_text", "") or "")
                if not first_hyperedge_text:
                    first_hyperedge_text = hyperedge_text
                block_texts.append(str(hyperedge.get("first_hop_hyperedge_text", "") or ""))
                block_texts.append(hyperedge_text)
            for token in content_tokens(" ".join(block_texts)):
                if token not in query_tokens:
                    answer = normalize_label(token)
                    break
            if answer:
                break
        if not answer:
            answer = short_text(first_hyperedge_text, 160)

        return {"answer": answer}

def _extract_topic_phrases(question: str) -> list[str]:
    cleaned = question.replace("?", " ").replace(",", " ").replace(";", " ")
    tokens = [token.strip() for token in cleaned.split() if token.strip()]
    capitalized: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token[:1].isupper():
            current.append(token)
        elif current:
            capitalized.append(" ".join(current))
            current = []
    if current:
        capitalized.append(" ".join(current))
    if capitalized:
        return capitalized

    content = content_tokens(question)
    phrases: list[str] = []
    for index in range(len(content)):
        phrases.append(content[index])
        if index + 1 < len(content):
            phrases.append(f"{content[index]} {content[index + 1]}")
    return _dedupe_texts(phrases)[:6]


def _dedupe_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = normalize_label(str(value).strip())
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _answer_evidence_blocks(evidence: Any) -> list[dict[str, Any]]:
    if isinstance(evidence, dict):
        blocks = [dict(item) for item in ensure_list(evidence.get("evidence_blocks")) if isinstance(item, dict)]
        if blocks:
            return blocks
        return _legacy_evidence_sections_to_blocks(
            [dict(item) for item in ensure_list(evidence.get("evidence")) if isinstance(item, dict)],
            [dict(item) for item in ensure_list(evidence.get("contexts")) if isinstance(item, dict)],
        )
    blocks = [dict(item) for item in ensure_list(evidence) if isinstance(item, dict)]
    if blocks and any("hyperedges" in item for item in blocks):
        return blocks
    return []


def _legacy_evidence_sections_to_blocks(
    evidence_items: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    context_by_id = {str(item.get("chunk_id", "") or ""): item for item in contexts}
    blocks: list[dict[str, Any]] = []
    block_by_context_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(evidence_items, start=1):
        hyperedge = {
            "hyperedge_id": f"H{index}",
            "hyperedge_text": str(item.get("hyperedge_text", "") or "").strip(),
        }
        chunk_ids = [str(chunk_id or "").strip() for chunk_id in ensure_list(item.get("chunk_ids")) if str(chunk_id or "").strip()]
        if not chunk_ids:
            chunk_ids = ["__NO_LINKED_CHUNK__"]
        for chunk_id in chunk_ids:
            context = context_by_id.get(chunk_id, {})
            block = block_by_context_id.get(chunk_id)
            if block is None:
                block = {
                    "chunk_id": f"C{len(blocks) + 1}",
                    "title": str(context.get("title", "") or ""),
                    "text": str(context.get("text", "") or ""),
                    "hyperedges": [],
                }
                block_by_context_id[chunk_id] = block
                blocks.append(block)
            block["hyperedges"].append(dict(hyperedge))
    return blocks
