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
    def route_reasoning_paths(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
        hop: int,
        candidate_paths: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def answer_atomic_question_from_paths(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        evidence_mode: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compose_final_answer(
        self,
        original_question: str,
        dag_nodes: list[dict[str, Any]],
        atomic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def finalize_answer_span(
        self,
        original_question: str,
        synthesis_candidate: dict[str, Any],
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

    def route_reasoning_paths(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
        hop: int,
        candidate_paths: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "atomic_path_router",
            self.prompts.get("atomic_path_router"),
            {
                "atomic_question": atomic_question,
                "dependency_answers": dependency_answers,
                "current_hop": hop,
                "maximum_hops": 2,
                "candidate_paths": candidate_paths,
            },
            max_tokens=1200,
        )
        response.setdefault("labels", [])
        return response

    def answer_atomic_question_from_paths(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        evidence_mode: str,
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "atomic_path_answer",
            self.prompts.get("atomic_path_answer"),
            {
                "atomic_question": atomic_question,
                "dependency_answers": dependency_answers,
                "evidence_mode": evidence_mode,
                "paths": paths,
            },
            max_tokens=900,
        )
        response.setdefault("answer", "")
        response.setdefault("confidence", 0.0)
        response.setdefault("reasoning_summary", "")
        response.setdefault("used_path_ids", [])
        response.setdefault("used_hyperedge_ids", [])
        response.setdefault("insufficient", False)
        return response

    def compose_final_answer(
        self,
        original_question: str,
        dag_nodes: list[dict[str, Any]],
        atomic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self.client.chat_json(
            "final_answer_composer",
            self.prompts.get("final_answer_composer"),
            {
                "original_question": original_question,
                "dag": dag_nodes,
                "atomic_results": atomic_results,
            },
            max_tokens=1400,
        )
        response.setdefault("answer", response.get("candidate_answer", ""))
        response.setdefault("candidate_answer", response.get("answer", ""))
        response.setdefault("semantic_answer", response.get("candidate_answer", response.get("answer", "")))
        response.setdefault("judgment", None)
        response.setdefault("reasoning_summary", "")
        response.setdefault("answer_span_reasoning", "")
        response.setdefault("confidence", 0.0)
        response.setdefault("atomic_answer_trace", [])
        response.setdefault("remaining_gaps", [])
        return response

    def finalize_answer_span(
        self,
        original_question: str,
        synthesis_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        # Legacy interface retained for compatibility. The default HyperBranch
        # pipeline resolves and canonicalizes the final answer in
        # compose_final_answer() and does not call this method.
        response = self.client.chat_json(
            "final_answer_span",
            self.prompts.get("final_answer_span"),
            {
                "original_question": original_question,
                "candidate_answer": synthesis_candidate.get("candidate_answer", synthesis_candidate.get("answer", "")),
                "reasoning_summary": synthesis_candidate.get("reasoning_summary", ""),
            },
            max_tokens=400,
        )
        response.setdefault("answer", "")
        response.setdefault("confidence", synthesis_candidate.get("confidence", 0.0))
        response.setdefault("answer_span_reasoning", "")
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
        route_responses: list[dict[str, Any]] | None = None,
        path_answer_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.route_responses = list(route_responses or [])
        self.path_answer_responses = list(path_answer_responses or [])
        self.route_calls: list[dict[str, Any]] = []
        self.path_answer_calls: list[dict[str, Any]] = []

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

    def route_reasoning_paths(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
        hop: int,
        candidate_paths: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.route_calls.append(
            {
                "atomic_question": atomic_question,
                "dependency_answers": dependency_answers,
                "hop": hop,
                "candidate_paths": candidate_paths,
            }
        )
        if self.route_responses:
            return self.route_responses.pop(0)

        labels: list[dict[str, Any]] = []
        question_tokens = set(content_tokens(atomic_question))
        for path in candidate_paths:
            path_id = str(path.get("path_id", ""))
            path_tokens = set(content_tokens(_path_text(path)))
            entity_ids = [str(item) for item in ensure_list(path.get("entity_ids", []))]
            final_entity = entity_ids[-1] if entity_ids else ""
            overlap = len(question_tokens & path_tokens)
            if overlap >= max(1, min(2, len(question_tokens))):
                label = "ANSWER"
                answer_entity_ids = [final_entity] if final_entity else []
            else:
                label = "EXPAND" if hop < 2 else "DROP"
                answer_entity_ids = []
            labels.append(
                {
                    "path_id": path_id,
                    "label": label,
                    "answer_entity_ids": answer_entity_ids,
                    "reason": "Deterministic mock path routing.",
                }
            )
        return {"labels": labels}

    def answer_atomic_question_from_paths(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        evidence_mode: str,
    ) -> dict[str, Any]:
        self.path_answer_calls.append(
            {
                "atomic_question": atomic_question,
                "dependency_answers": dependency_answers,
                "paths": paths,
                "evidence_mode": evidence_mode,
            }
        )
        if self.path_answer_responses:
            return self.path_answer_responses.pop(0)
        if not paths:
            return {
                "answer": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "reasoning_summary": "No reasoning paths were provided.",
                "used_path_ids": [],
                "used_hyperedge_ids": [],
                "insufficient": True,
            }

        first = paths[0]
        query_tokens = set(content_tokens(atomic_question))
        answer = ""
        for entity_id in ensure_list(first.get("answer_entity_ids", [])):
            label = normalize_label(str(entity_id))
            if label:
                answer = label
                break
        if not answer:
            for entity_id in ensure_list(first.get("entity_ids", [])):
                label = normalize_label(str(entity_id))
                if label and not set(content_tokens(label)).issubset(query_tokens):
                    answer = label
                    break
        if not answer:
            answer = short_text(_path_text(first), 160)

        used_hyperedge_ids = [str(item) for item in ensure_list(first.get("hyperedge_ids", [])) if str(item)]
        return {
            "answer": answer or "INSUFFICIENT_EVIDENCE",
            "confidence": 0.75 if answer else 0.0,
            "reasoning_summary": short_text(_path_text(first), 420),
            "used_path_ids": [str(first.get("path_id", ""))],
            "used_hyperedge_ids": used_hyperedge_ids,
            "insufficient": not bool(answer),
        }

    def compose_final_answer(
        self,
        original_question: str,
        dag_nodes: list[dict[str, Any]],
        atomic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del original_question, dag_nodes
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
        judgment = answer.lower() if answer.lower() in {"yes", "no"} else None
        return {
            "answer": answer,
            "candidate_answer": answer,
            "semantic_answer": answer,
            "judgment": judgment,
            "reasoning_summary": short_text(
                " | ".join(str(item.get("reasoning_summary", "")) for item in atomic_results),
                600,
            ),
            "answer_span_reasoning": "Mock single-stage final resolver mirrors the selected answer.",
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

    def finalize_answer_span(
        self,
        original_question: str,
        synthesis_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        # Legacy interface retained for compatibility. The main pipeline does
        # not call this method.
        del original_question
        return {
            "answer": str(synthesis_candidate.get("candidate_answer", synthesis_candidate.get("answer", ""))).strip(),
            "confidence": float(synthesis_candidate.get("confidence", 0.0) or 0.0),
            "answer_span_reasoning": "Mock final span mirrors the candidate answer.",
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


def _path_text(path: dict[str, Any]) -> str:
    texts: list[str] = []
    texts.extend(str(item) for item in ensure_list(path.get("entity_ids", [])))
    for step in ensure_list(path.get("steps", [])):
        if not isinstance(step, dict):
            continue
        texts.append(str(step.get("hyperedge_text", "")))
        texts.extend(str(item) for item in ensure_list(step.get("chunk_texts", [])))
    return " ".join(text for text in texts if text)
