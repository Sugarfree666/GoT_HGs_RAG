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
        dependency_answers: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compose_final_answer(
        self,
        original_question: str,
        atomic_results: list[dict[str, Any]],
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
        dependency_answers: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "atomic_answer",
            self.prompts.get("atomic_answer"),
            {
                "atomic_question": atomic_question,
                "dependency_answers": dependency_answers,
                "top_evidence": evidence,
            },
            max_tokens=900,
        )
        response.setdefault("answer", "")
        response.setdefault("confidence", 0.0)
        response.setdefault("reasoning_summary", "")
        response.setdefault("used_hyperedge_ids", [])
        response.setdefault("insufficient", False)
        return response

    def compose_final_answer(
        self,
        original_question: str,
        atomic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "final_answer_composer",
            self.prompts.get("final_answer_composer"),
            {
                "original_question": original_question,
                "atomic_results": atomic_results,
            },
            max_tokens=1100,
        )
        response.setdefault("answer", "")
        response.setdefault("reasoning_summary", "")
        response.setdefault("confidence", 0.0)
        response.setdefault("atomic_answer_trace", [])
        response.setdefault("remaining_gaps", [])
        return response


class MockAtomicLLMService(AtomicLLMService):
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
        dependency_answers: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del dependency_answers
        if not evidence:
            return {
                "answer": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "reasoning_summary": "No top evidence was provided.",
                "used_hyperedge_ids": [],
                "insufficient": True,
            }

        query_tokens = set(content_tokens(atomic_question))
        answer = ""
        first = evidence[0]
        for item in evidence:
            for entity_id in ensure_list(item.get("entity_ids", [])):
                label = normalize_label(str(entity_id))
                label_tokens = set(content_tokens(label))
                if label and not label_tokens.issubset(query_tokens):
                    answer = label
                    break
            if answer:
                break
        if not answer:
            answer = short_text(str(first.get("hyperedge_text", "")), 160)

        confidence = min(0.95, 0.35 + (0.1 * len(first.get("branch_support", []))) + (0.03 * len(evidence)))
        evidence_texts = ensure_list(first.get("evidence_texts", []))
        summary = short_text(
            " ".join(str(item) for item in evidence_texts) or str(first.get("hyperedge_text", "")),
            420,
        )
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_summary": summary,
            "used_hyperedge_ids": [str(first.get("hyperedge_id", ""))],
            "insufficient": False,
        }

    def compose_final_answer(
        self,
        original_question: str,
        atomic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del original_question
        answers = [
            str(item.get("answer", "")).strip()
            for item in atomic_results
            if str(item.get("answer", "")).strip() and str(item.get("answer", "")).strip() != "INSUFFICIENT_EVIDENCE"
        ]
        answer = answers[-1] if answers else "INSUFFICIENT_EVIDENCE"
        if len(answers) > 1:
            answer = "; ".join(answers)
        confidence_values = [float(item.get("confidence", 0.0) or 0.0) for item in atomic_results]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        return {
            "answer": answer,
            "reasoning_summary": short_text(
                " | ".join(str(item.get("reasoning_summary", "")) for item in atomic_results),
                600,
            ),
            "confidence": confidence,
            "atomic_answer_trace": [
                {
                    "node_id": item.get("node_id", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "used_hyperedge_ids": list(item.get("used_hyperedge_ids", [])),
                }
                for item in atomic_results
            ],
            "remaining_gaps": [
                item.get("node_id", "")
                for item in atomic_results
                if str(item.get("answer", "")).strip() in {"", "INSUFFICIENT_EVIDENCE"}
            ],
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
