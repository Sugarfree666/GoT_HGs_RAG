from __future__ import annotations

from typing import Any

from ..llm.service import AtomicLLMService
from ..utils import short_text
from .models import AtomicAnswerResult, AtomicQuestionNode


class FinalAnswerComposer:
    def __init__(self, llm_service: AtomicLLMService | None = None) -> None:
        self.llm_service = llm_service

    def compose(
        self,
        original_question: str,
        atomic_results: list[AtomicAnswerResult],
        dag_nodes: list[AtomicQuestionNode] | None = None,
    ) -> dict[str, Any]:
        payload_results = [self._result_payload(result) for result in atomic_results]
        dag_payload = [node.to_dict() for node in dag_nodes or []]
        if self.llm_service is not None:
            candidate_payload = self.llm_service.compose_final_answer(
                original_question=original_question,
                dag_nodes=dag_payload,
                atomic_results=payload_results,
            )
            answer_span_payload = self.llm_service.finalize_answer_span(
                original_question=original_question,
                synthesis_candidate=candidate_payload,
            )
        else:
            candidate_payload = self._fallback_compose(original_question, payload_results)
            answer_span_payload = {
                "answer": candidate_payload.get("candidate_answer", candidate_payload.get("answer", "")),
                "confidence": candidate_payload.get("confidence", 0.0),
                "answer_span_reasoning": "Fallback final span mirrors the candidate answer.",
            }
        return self._coerce_payload(candidate_payload, answer_span_payload, original_question, payload_results)

    def _result_payload(self, result: AtomicAnswerResult) -> dict[str, Any]:
        return {
            "node_id": result.node_id,
            "question": result.question,
            "answer": result.answer,
            "confidence": result.confidence,
            "reasoning_summary": result.reasoning_summary,
            "used_hyperedge_ids": list(result.used_hyperedge_ids),
            "top_evidence": [
                {
                    "hyperedge_id": evidence.hyperedge_id,
                    "hyperedge_text": evidence.hyperedge_text,
                    "branch_support": sorted(evidence.branch_support),
                    "score_breakdown": dict(evidence.score_breakdown),
                    "evidence_texts": list(evidence.evidence_texts),
                }
                for evidence in result.evidence
            ],
        }

    def _fallback_compose(self, original_question: str, atomic_results: list[dict[str, Any]]) -> dict[str, Any]:
        usable_answers = [
            str(item.get("answer", "")).strip()
            for item in atomic_results
            if str(item.get("answer", "")).strip() and str(item.get("answer", "")).strip() != "INSUFFICIENT_EVIDENCE"
        ]
        answer = usable_answers[-1] if usable_answers else "INSUFFICIENT_EVIDENCE"
        if len(usable_answers) > 1:
            answer = "; ".join(usable_answers)
        confidence_values = [float(item.get("confidence", 0.0) or 0.0) for item in atomic_results]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        remaining_gaps = [
            item.get("node_id", "")
            for item in atomic_results
            if not str(item.get("answer", "")).strip() or str(item.get("answer", "")).strip() == "INSUFFICIENT_EVIDENCE"
        ]
        return {
            "candidate_answer": answer,
            "reasoning_summary": short_text(" | ".join(str(item.get("reasoning_summary", "")) for item in atomic_results), 800),
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
            "remaining_gaps": remaining_gaps,
        }

    def _coerce_payload(
        self,
        candidate_payload: Any,
        answer_span_payload: Any,
        original_question: str,
        atomic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(candidate_payload, dict):
            candidate_payload = self._fallback_compose(original_question, atomic_results)
        if not isinstance(answer_span_payload, dict):
            answer_span_payload = {}

        candidate_payload.setdefault("candidate_answer", candidate_payload.get("answer", ""))
        candidate_payload.setdefault("reasoning_summary", "")
        candidate_payload.setdefault("confidence", 0.0)
        candidate_payload.setdefault(
            "atomic_answer_trace",
            [
                {
                    "node_id": item.get("node_id", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "used_hyperedge_ids": list(item.get("used_hyperedge_ids", [])),
                }
                for item in atomic_results
            ],
        )
        candidate_payload.setdefault("remaining_gaps", [])

        final_answer = str(answer_span_payload.get("answer", "") or "").strip()
        if not final_answer:
            final_answer = str(candidate_payload.get("candidate_answer", "") or "").strip()
        confidence = answer_span_payload.get("confidence", candidate_payload.get("confidence", 0.0))

        payload = {
            "answer": final_answer,
            "candidate_answer": str(candidate_payload.get("candidate_answer", "") or "").strip(),
            "reasoning_summary": str(candidate_payload.get("reasoning_summary", "") or "").strip(),
            "answer_span_reasoning": str(answer_span_payload.get("answer_span_reasoning", "") or "").strip(),
            "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
            "atomic_answer_trace": list(candidate_payload.get("atomic_answer_trace", [])),
            "remaining_gaps": list(candidate_payload.get("remaining_gaps", [])),
        }
        if not payload["answer"]:
            payload["answer"] = "INSUFFICIENT_EVIDENCE"
            payload["confidence"] = 0.0
        return payload
