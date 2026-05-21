from __future__ import annotations

from typing import Any

from ..llm.service import AtomicLLMService
from ..utils import short_text
from .models import AtomicAnswerResult


class FinalAnswerComposer:
    def __init__(self, llm_service: AtomicLLMService | None = None) -> None:
        self.llm_service = llm_service

    def compose(self, original_question: str, atomic_results: list[AtomicAnswerResult]) -> dict[str, Any]:
        payload_results = [self._result_payload(result) for result in atomic_results]
        if self.llm_service is not None:
            payload = self.llm_service.compose_final_answer(
                original_question=original_question,
                atomic_results=payload_results,
            )
        else:
            payload = self._fallback_compose(original_question, payload_results)
        return self._coerce_payload(payload, original_question, payload_results)

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
            "answer": answer,
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
        payload: Any,
        original_question: str,
        atomic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = self._fallback_compose(original_question, atomic_results)
        payload.setdefault("answer", "")
        payload.setdefault("reasoning_summary", "")
        payload.setdefault("confidence", 0.0)
        payload.setdefault(
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
        payload.setdefault("remaining_gaps", [])
        payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
        return payload
