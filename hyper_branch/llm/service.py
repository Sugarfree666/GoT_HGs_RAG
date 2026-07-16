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
    def rewrite_atomic_fact_query(
        self,
        atomic_question: str,
        answer_type: str,
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

    @abstractmethod
    def select_anchor_entity(
        self,
        question: str,
        mention: str,
        analysis: Any,
        candidates: list[dict[str, Any]],
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

    def rewrite_atomic_fact_query(
        self,
        atomic_question: str,
        answer_type: str,
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "atomic_fact_query",
            self.prompts.get("atomic_fact_query"),
            {
                "atomic_question": atomic_question,
                "answer_type": answer_type,
            },
            max_tokens=200,
        )
        return _fact_query_payload(response)

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, Any],
        dependency_answers: list[dict[str, Any]],
        evidence: Any,
        original_question: str = "",
    ) -> dict[str, Any]:
        evidence_items, contexts = _answer_evidence_sections(evidence)
        response = self.client.chat_json(
            "atomic_answer",
            self.prompts.get("atomic_answer"),
            {
                "original_question": original_question,
                "atomic_question": atomic_question,
                "answer_contract": answer_contract,
                "dependency_answers": dependency_answers,
                "evidence": evidence_items,
                "contexts": contexts,
            },
            max_tokens=900,
        )
        if not isinstance(response, dict):
            return {"answer": ""}
        payload = dict(response)
        payload["answer"] = str(response.get("answer", "") or "")
        return payload

    def select_anchor_entity(
        self,
        question: str,
        mention: str,
        analysis: Any,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "anchor_entity_selection",
            self.prompts.get("anchor_entity_selection"),
            {
                "question": question,
                "mention": mention,
                "analysis": _analysis_payload(analysis),
                "candidate_entities": candidates,
            },
            max_tokens=300,
        )
        response.setdefault("selected_entity_id", "NONE")
        response.setdefault("confidence", 0.0)
        response.setdefault("reason", "")
        return response


class MockAtomicLLMService(AtomicLLMService):
    def __init__(
        self,
        answer_responses: list[dict[str, Any]] | None = None,
        fact_query_responses: list[dict[str, Any] | BaseException] | None = None,
    ) -> None:
        self.answer_responses = list(answer_responses or [])
        self.fact_query_responses = list(fact_query_responses or [])
        self.answer_calls: list[dict[str, Any]] = []
        self.fact_query_calls: list[dict[str, Any]] = []

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

    def rewrite_atomic_fact_query(
        self,
        atomic_question: str,
        answer_type: str,
    ) -> dict[str, Any]:
        self.fact_query_calls.append(
            {
                "atomic_question": atomic_question,
                "answer_type": answer_type,
            }
        )
        if self.fact_query_responses:
            response = self.fact_query_responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return _fact_query_payload(response)
        return {"fact_query": str(atomic_question or "").strip()}

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, Any],
        dependency_answers: list[dict[str, Any]],
        evidence: Any,
        original_question: str = "",
    ) -> dict[str, Any]:
        evidence_items, contexts = _answer_evidence_sections(evidence)
        self.answer_calls.append(
            {
                "original_question": original_question,
                "atomic_question": atomic_question,
                "answer_contract": answer_contract,
                "dependency_answers": dependency_answers,
                "evidence": evidence_items,
                "contexts": contexts,
            }
        )
        if self.answer_responses:
            response = self.answer_responses.pop(0)
            return {"answer": str(response.get("answer", "") or "")} if isinstance(response, dict) else {"answer": ""}
        if not evidence_items and not contexts:
            return {"answer": "INSUFFICIENT_EVIDENCE"}

        query_tokens = set(content_tokens(atomic_question))
        answer = ""
        first = evidence_items[0] if evidence_items else {}
        context_by_id = {str(item.get("chunk_id", "") or ""): item for item in contexts}
        for item in evidence_items:
            context_texts: list[str] = []
            for chunk_id in ensure_list(item.get("chunk_ids", [])):
                context = context_by_id.get(str(chunk_id))
                if not context:
                    continue
                context_texts.extend(
                    [
                        str(context.get("title", "") or ""),
                        str(context.get("text", "") or ""),
                    ]
                )
            text = " ".join(
                [
                    str(item.get("hyperedge_text", "") or ""),
                    *[str(chunk) for chunk in ensure_list(item.get("chunk_texts", []))],
                    *context_texts,
                ]
            )
            for token in content_tokens(text):
                if token not in query_tokens:
                    answer = normalize_label(token)
                    break
            if answer:
                break
        if not answer:
            answer = short_text(str(first.get("hyperedge_text", "")), 160)

        return {"answer": answer}

    def select_anchor_entity(
        self,
        question: str,
        mention: str,
        analysis: Any,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del question, mention, analysis
        if not candidates:
            return {
                "selected_entity_id": "NONE",
                "confidence": 0.0,
                "reason": "No candidates were provided.",
            }
        selected = str(candidates[0].get("entity_id", "")).strip()
        return {
            "selected_entity_id": selected or "NONE",
            "confidence": 1.0 if selected else 0.0,
            "reason": "Deterministic mock selects the first candidate.",
        }


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


def _analysis_payload(analysis: Any) -> dict[str, Any]:
    if hasattr(analysis, "to_dict") and callable(analysis.to_dict):
        payload = dict(analysis.to_dict())
    elif isinstance(analysis, dict):
        payload = dict(analysis)
    else:
        payload = {
            "entities": getattr(analysis, "entities", []),
        }
    return {
        "entities": [str(item) for item in ensure_list(payload.get("entities", []))],
    }


def _fact_query_payload(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"fact_query": ""}
    return {"fact_query": str(response.get("fact_query", "") or "").strip()}


def _answer_evidence_sections(evidence: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(evidence, dict):
        evidence_items = [dict(item) for item in ensure_list(evidence.get("evidence")) if isinstance(item, dict)]
        contexts = [dict(item) for item in ensure_list(evidence.get("contexts")) if isinstance(item, dict)]
        return evidence_items, contexts
    return [dict(item) for item in ensure_list(evidence) if isinstance(item, dict)], []
