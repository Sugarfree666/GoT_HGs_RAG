from __future__ import annotations

import re
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
        evidence: list[dict[str, Any]],
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
            max_tokens=700,
        )
        response.setdefault("entities", [])
        response.setdefault("relations", [])
        response.setdefault("relation_query", "")
        response.setdefault("answer_type", "")
        return response

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, Any],
        dependency_answers: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "atomic_answer",
            self.prompts.get("atomic_answer"),
            {
                "atomic_question": atomic_question,
                "answer_contract": answer_contract,
                "dependency_answers": dependency_answers,
                "evidence": evidence,
            },
            max_tokens=900,
        )
        response.setdefault("answer", "")
        response.setdefault("confidence", 0.0)
        response.setdefault("reasoning_summary", "")
        response.setdefault("used_evidence_ids", [])
        response.setdefault("insufficient", False)
        return response

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
        relation_intent = _infer_relation_intent(atomic_question)
        return {
            "entities": entities,
            "relations": [relation_intent] if relation_intent else [],
            "relation_query": _mask_entities(atomic_question, entities),
            "answer_type": _infer_answer_type(atomic_question),
        }

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, Any],
        dependency_answers: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.answer_calls.append(
            {
                "atomic_question": atomic_question,
                "answer_contract": answer_contract,
                "dependency_answers": dependency_answers,
                "evidence": evidence,
            }
        )
        if self.answer_responses:
            return self.answer_responses.pop(0)
        if not evidence:
            return {
                "answer": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "reasoning_summary": "No top evidence was provided.",
                "used_evidence_ids": [],
                "insufficient": True,
            }

        query_tokens = set(content_tokens(atomic_question))
        answer = ""
        first = evidence[0]
        for item in evidence:
            text = " ".join(
                [
                    str(item.get("hyperedge_text", "") or ""),
                    *[str(chunk) for chunk in ensure_list(item.get("chunk_texts", []))],
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

        confidence = min(0.95, 0.35 + (0.03 * len(evidence)))
        evidence_texts = ensure_list(first.get("chunk_texts", []))
        summary = short_text(
            " ".join(str(item) for item in evidence_texts) or str(first.get("hyperedge_text", "")),
            420,
        )
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_summary": summary,
            "used_evidence_ids": [str(first.get("evidence_id", ""))],
            "insufficient": False,
        }

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


def _mask_entities(question: str, entities: list[str]) -> str:
    masked = question.strip().rstrip("?")
    for entity in entities:
        text = str(entity).strip()
        if text:
            masked = re.sub(re.escape(text), "an entity", masked, flags=re.IGNORECASE)
    tokens = content_tokens(masked)
    return " ".join(tokens) if tokens else masked


def _infer_answer_type(question: str) -> str:
    lowered = question.lower().strip()
    if lowered.startswith("when"):
        return "time or date"
    if lowered.startswith("where"):
        return "location"
    if lowered.startswith("who"):
        return "person or organization"
    if lowered.startswith("which "):
        tokens = lowered.split()
        if len(tokens) > 1:
            return tokens[1].strip(" ?.,")
    if lowered.startswith("what "):
        return "entity, concept, or phrase"
    return "grounded short answer"


def _infer_relation_intent(question: str) -> str:
    lowered = question.lower()
    if "graduate" in lowered and "from" in lowered:
        return "graduate from"
    if "released" in lowered:
        return "release date"
    if "known for" in lowered:
        return "known for"
    if "located in" in lowered:
        return "located in"
    return "connect the topic entities to the missing answer"


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
            "relations": getattr(analysis, "relations", []),
            "relation_query": getattr(analysis, "relation_query", ""),
        }
    return {
        "entities": [str(item) for item in ensure_list(payload.get("entities", []))],
        "relations": [str(item) for item in ensure_list(payload.get("relations", []))],
        "relation_query": str(payload.get("relation_query", "") or ""),
    }
