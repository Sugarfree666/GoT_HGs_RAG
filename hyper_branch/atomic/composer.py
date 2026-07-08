from __future__ import annotations

import re
from typing import Any

from ..llm.service import AtomicLLMService
from ..utils import normalize_label, short_text
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
            final_payload = self.llm_service.compose_final_answer(
                original_question=original_question,
                dag_nodes=dag_payload,
                atomic_results=payload_results,
            )
        else:
            final_payload = self._fallback_compose(original_question, payload_results)
            final_payload.setdefault("answer", final_payload.get("candidate_answer", ""))
            final_payload.setdefault("answer_span_reasoning", "Fallback final answer mirrors the candidate answer.")
        payload = self._coerce_single_payload(final_payload, original_question, payload_results)
        return _postprocess_final_answer(
            payload=payload,
            original_question=original_question,
            atomic_results=payload_results,
            dag_nodes=dag_payload,
        )

    def _result_payload(self, result: AtomicAnswerResult) -> dict[str, Any]:
        return {
            "node_id": result.node_id,
            "question": result.question,
            "answer": result.answer,
            "confidence": result.confidence,
            "reasoning_summary": result.reasoning_summary,
            "used_dependencies": list(result.used_dependencies),
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
            "candidate_answer": answer,
            "semantic_answer": answer,
            "judgment": _yes_no_judgment(answer),
            "reasoning_summary": short_text(" | ".join(str(item.get("reasoning_summary", "")) for item in atomic_results), 800),
            "answer_span_reasoning": "Fallback final answer mirrors the candidate answer.",
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

    def _coerce_single_payload(
        self,
        final_payload: Any,
        original_question: str,
        atomic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del original_question
        if not isinstance(final_payload, dict):
            final_payload = self._fallback_compose("", atomic_results)

        answer = str(final_payload.get("answer", "") or "").strip()
        candidate_answer = str(final_payload.get("candidate_answer", "") or "").strip()
        semantic_answer = str(final_payload.get("semantic_answer", "") or "").strip()
        if not answer:
            answer = candidate_answer or semantic_answer
        if not candidate_answer:
            candidate_answer = answer
        if not semantic_answer:
            semantic_answer = candidate_answer or answer

        confidence = _bounded_confidence(final_payload.get("confidence", 0.0))
        if not answer:
            answer = "INSUFFICIENT_EVIDENCE"
            candidate_answer = "INSUFFICIENT_EVIDENCE"
            semantic_answer = "INSUFFICIENT_EVIDENCE"
            confidence = 0.0
        if answer.upper() == "INSUFFICIENT_EVIDENCE":
            confidence = 0.0

        trace = final_payload.get("atomic_answer_trace")
        if not isinstance(trace, list) or len(trace) != len(atomic_results):
            trace = _atomic_answer_trace(atomic_results)
        remaining_gaps = final_payload.get("remaining_gaps", [])
        if not isinstance(remaining_gaps, list):
            remaining_gaps = [str(remaining_gaps)] if str(remaining_gaps).strip() else []

        judgment = final_payload.get("judgment", None)
        if isinstance(judgment, str):
            normalized_judgment = judgment.strip().lower()
            judgment = normalized_judgment if normalized_judgment in {"yes", "no"} else None
        else:
            judgment = None

        payload = {
            "answer": answer,
            "candidate_answer": candidate_answer,
            "semantic_answer": semantic_answer,
            "judgment": judgment,
            "reasoning_summary": str(final_payload.get("reasoning_summary", "") or "").strip(),
            "answer_span_reasoning": str(final_payload.get("answer_span_reasoning", "") or "").strip(),
            "confidence": confidence,
            "atomic_answer_trace": list(trace),
            "remaining_gaps": [str(item).strip() for item in remaining_gaps if str(item).strip()],
        }
        return payload


def _atomic_answer_trace(atomic_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": item.get("node_id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "used_hyperedge_ids": list(item.get("used_hyperedge_ids", [])),
        }
        for item in atomic_results
    ]


def _bounded_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _yes_no_judgment(answer: str) -> str | None:
    normalized = str(answer or "").strip().lower()
    return normalized if normalized in {"yes", "no"} else None


_COUNTRY_TO_DEMONYM = {
    "austria": "Austrian",
    "canada": "Canadian",
    "china": "Chinese",
    "czech republic": "Czech",
    "england": "English",
    "france": "French",
    "germany": "German",
    "ireland": "Irish",
    "italy": "Italian",
    "japan": "Japanese",
    "poland": "Polish",
    "romania": "Romanian",
    "russia": "Russian",
    "spain": "Spanish",
    "sweden": "Swedish",
    "united kingdom": "British",
    "united states": "American",
    "united states of america": "American",
}

_DEMONYM_TO_COUNTRY = {
    "american": "United States",
    "british": "United Kingdom",
    "canadian": "Canada",
    "chinese": "China",
    "czech": "Czech Republic",
    "english": "England",
    "french": "France",
    "german": "Germany",
    "irish": "Ireland",
    "italian": "Italy",
    "japanese": "Japan",
    "polish": "Poland",
    "romanian": "Romania",
    "russian": "Russia",
    "spanish": "Spain",
    "swedish": "Sweden",
}

_GEOGRAPHIC_SUFFIXES = {
    "argentina",
    "australia",
    "austria",
    "belgium",
    "brazil",
    "canada",
    "california",
    "china",
    "england",
    "france",
    "germany",
    "great britain",
    "india",
    "ireland",
    "italy",
    "japan",
    "london",
    "netherlands",
    "new york",
    "poland",
    "russia",
    "scotland",
    "spain",
    "sweden",
    "united kingdom",
    "united states",
    "united states of america",
    "wales",
}

_LOCATION_CONTEXT_SUFFIXES = {
    "argentina",
    "australia",
    "austria",
    "belgium",
    "brazil",
    "canada",
    "china",
    "denmark",
    "england",
    "france",
    "germany",
    "great britain",
    "india",
    "ireland",
    "italy",
    "japan",
    "netherlands",
    "northern ireland",
    "poland",
    "russia",
    "russian federation",
    "scotland",
    "spain",
    "sweden",
    "united kingdom",
    "united states",
    "united states of america",
    "wales",
}


def _postprocess_final_answer(
    *,
    payload: dict[str, Any],
    original_question: str,
    atomic_results: list[dict[str, Any]],
    dag_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    corrected = dict(payload)
    deterministic = _deterministic_comparison_answer(original_question, atomic_results, dag_nodes)
    if deterministic:
        corrected["answer"] = deterministic["answer"]
        corrected["candidate_answer"] = deterministic["answer"]
        corrected["confidence"] = max(float(corrected.get("confidence", 0.0) or 0.0), deterministic["confidence"])
        corrected["answer_span_reasoning"] = deterministic["reason"]
        corrected["deterministic_final_correction"] = deterministic
        return corrected

    span = _minimal_candidate_span(
        str(corrected.get("answer", "") or corrected.get("candidate_answer", "") or ""),
        original_question,
    )
    if span and span != corrected.get("answer"):
        corrected["answer"] = span
        corrected["answer_span_reasoning"] = (
            str(corrected.get("answer_span_reasoning", "") or "").strip()
            or "Deterministic span normalizer selected the minimal candidate span."
        )
        corrected["deterministic_span_normalized"] = True
    date_span = _date_canonical_span(
        str(corrected.get("answer", "") or ""),
        original_question,
    )
    if date_span and date_span != corrected.get("answer"):
        corrected["answer"] = date_span
        corrected["answer_span_reasoning"] = "Deterministic span normalizer selected the first supported year."
        corrected["deterministic_date_normalized"] = True
    alias = _parenthetical_alias_span(
        str(corrected.get("answer", "") or ""),
        original_question,
        atomic_results,
        dag_nodes,
    )
    if alias and alias != corrected.get("answer"):
        corrected["answer"] = alias
        corrected["answer_span_reasoning"] = "Deterministic span normalizer selected a parenthetical alias."
        corrected["deterministic_parenthetical_alias"] = True
    location = _location_canonical_span(
        str(corrected.get("answer", "") or ""),
        original_question,
    )
    if location and location != corrected.get("answer"):
        corrected["answer"] = location
        corrected["answer_span_reasoning"] = "Deterministic span normalizer removed geographic context from a place answer."
        corrected["deterministic_location_normalized"] = True
    organization = _organization_alias_span(
        str(corrected.get("answer", "") or ""),
        original_question,
    )
    if organization and organization != corrected.get("answer"):
        corrected["answer"] = organization
        corrected["answer_span_reasoning"] = "Deterministic span normalizer selected a shorter organization alias."
        corrected["deterministic_organization_alias"] = True
    nationality = _nationality_canonical_span(
        str(corrected.get("answer", "") or ""),
        original_question,
    )
    if nationality and nationality != corrected.get("answer"):
        corrected["answer"] = nationality
        corrected["answer_span_reasoning"] = "Deterministic span normalizer canonicalized a nationality answer."
        corrected["deterministic_nationality_normalized"] = True
    return corrected


def _deterministic_comparison_answer(
    original_question: str,
    atomic_results: list[dict[str, Any]],
    dag_nodes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    mode = _comparison_mode(original_question)
    if mode is None:
        return _deterministic_yes_no_answer(original_question, atomic_results, dag_nodes)

    records = _comparable_records(original_question, atomic_results, dag_nodes)
    if len(records) < 2:
        return None
    usable = [record for record in records if record["value"] is not None and record["label"]]
    if len(usable) < 2:
        return None

    if mode == "max_duration":
        duration_records = [record for record in usable if record["kind"] == "duration"]
        if len(duration_records) >= 2:
            selected = max(duration_records, key=lambda item: item["value"])
            return _comparison_payload(selected, "Selected the branch with the longest parsed lifespan/duration.")
        return None

    if mode in {"older", "younger"}:
        if all(record["kind"] == "number" for record in usable):
            selected = max(usable, key=lambda item: item["value"]) if mode == "older" else min(usable, key=lambda item: item["value"])
            return _comparison_payload(selected, f"Selected by numeric age for {mode}.")
        date_records = [record for record in usable if record["kind"] == "date"]
        if len(date_records) >= 2:
            selected = min(date_records, key=lambda item: item["value"]) if mode == "older" else max(date_records, key=lambda item: item["value"])
            return _comparison_payload(selected, f"Selected by birth-date ordering for {mode}.")
        return None

    date_records = [record for record in usable if record["kind"] == "date"]
    if len(date_records) < 2:
        return None
    selected = min(date_records, key=lambda item: item["value"]) if mode == "min_date" else max(date_records, key=lambda item: item["value"])
    return _comparison_payload(selected, f"Selected by deterministic date ordering: {mode}.")


def _comparison_payload(record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "answer": str(record["label"]).strip(),
        "confidence": 0.95,
        "reason": reason,
        "selected_node_id": record.get("node_id", ""),
        "selected_value": record.get("raw_answer", ""),
    }


def _comparison_mode(question: str) -> str | None:
    lowered = _comparison_cue_text(question).lower()
    if "lived longer" in lowered or "live longer" in lowered:
        return "max_duration"
    if re.search(r"\b(?:born|birth)\b.*\blater\b|\blater\b.*\b(?:born|birth)\b", lowered):
        return "max_date"
    if re.search(r"\b(?:born|birth)\b.*\b(?:first|earlier)\b|\b(?:first|earlier)\b.*\b(?:born|birth)\b", lowered):
        return "min_date"
    if re.search(r"\b(?:released|release)\b.*\blater\b|\blater\b.*\b(?:released|release)\b", lowered):
        return "max_date"
    if re.search(r"\b(?:released|release|came out|come out)\b.*\b(?:first|earlier)\b|\b(?:first|earlier)\b.*\b(?:released|release|came out|come out)\b", lowered):
        return "min_date"
    if re.search(r"\b(?:died|death|die)\b.*\blater\b|\blater\b.*\b(?:died|death|die)\b", lowered):
        return "max_date"
    if re.search(r"\b(?:died|death|die)\b.*\b(?:first|earlier)\b|\b(?:first|earlier)\b.*\b(?:died|death|die)\b", lowered):
        return "min_date"
    if "older" in lowered:
        return "older"
    if "younger" in lowered:
        return "younger"
    return None


def _comparison_cue_text(question: str) -> str:
    """Return the question relation text before candidate alternatives when possible."""
    text = normalize_label(question).strip()
    if "," in text:
        return text.split(",", 1)[0]
    return text


def _deterministic_yes_no_answer(
    original_question: str,
    atomic_results: list[dict[str, Any]],
    dag_nodes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    lowered = original_question.lower()
    answers = _terminal_atomic_answers(atomic_results, dag_nodes)
    explicit_terminal = [
        answer
        for answer in answers
        if _normalize_comparison_answer(answer) in {"yes", "no"}
    ]
    if explicit_terminal:
        answer = _normalize_comparison_answer(explicit_terminal[-1])
        return {
            "answer": answer,
            "confidence": 0.95,
            "reason": "Used the explicit terminal yes/no atomic answer.",
            "compared_answers": answers,
        }
    if not ("same " in lowered or "share the same" in lowered):
        return None
    if len(answers) < 2:
        return None
    normalized = [_normalize_comparison_answer(answer) for answer in answers[-2:]]
    if not normalized[0] or not normalized[1]:
        return None
    return {
        "answer": "yes" if normalized[0] == normalized[1] else "no",
        "confidence": 0.9,
        "reason": "Deterministic yes/no comparison over the final branch atomic answers.",
        "compared_answers": answers[-2:],
    }


def _terminal_atomic_answers(
    atomic_results: list[dict[str, Any]],
    dag_nodes: list[dict[str, Any]],
) -> list[str]:
    nodes_by_id = {str(item.get("node_id", "")): item for item in dag_nodes}
    consumed_dependencies: set[str] = set()
    for result in atomic_results:
        node_id = str(result.get("node_id", "") or "")
        node = nodes_by_id.get(node_id, {})
        consumed_dependencies.update(_dependencies_for_node(result, node))
    answers = [
        str(item.get("answer", "") or "").strip()
        for item in atomic_results
        if str(item.get("node_id", "") or "") not in consumed_dependencies
        and str(item.get("answer", "") or "").strip()
        and str(item.get("answer", "") or "").strip().upper() != "INSUFFICIENT_EVIDENCE"
    ]
    if len(answers) >= 2:
        return answers
    return [
        str(item.get("answer", "") or "").strip()
        for item in atomic_results
        if str(item.get("answer", "") or "").strip()
        and str(item.get("answer", "") or "").strip().upper() != "INSUFFICIENT_EVIDENCE"
    ]


def _parenthetical_alias_span(
    answer: str,
    original_question: str,
    atomic_results: list[dict[str, Any]],
    dag_nodes: list[dict[str, Any]],
) -> str | None:
    del original_question
    normalized_answer = _norm_text(answer)
    if not normalized_answer:
        return None
    for terminal_answer in reversed(_terminal_atomic_answers(atomic_results, dag_nodes)):
        match = re.match(r"^\s*(?P<prefix>.+?)\s*\((?P<alias>[^()]+)\)\s*$", terminal_answer)
        if not match:
            continue
        prefix = normalize_label(match.group("prefix")).strip()
        alias = normalize_label(match.group("alias")).strip()
        if not alias or not _is_plausible_parenthetical_alias(alias):
            continue
        if normalized_answer in {_norm_text(prefix), _norm_text(terminal_answer)}:
            return alias
    return None


def _is_plausible_parenthetical_alias(alias: str) -> bool:
    lowered = alias.lower()
    if re.search(r"\b(?:film|song|album|book|born|died|b\.|d\.|disambiguation)\b", lowered):
        return False
    if re.search(r"\d", alias):
        return False
    return 1 <= len(alias.split()) <= 5


def _location_canonical_span(answer: str, original_question: str) -> str | None:
    answer = normalize_label(answer).strip()
    if not answer or ";" in answer:
        return None
    if _looks_like_date_span(answer):
        return None
    if _norm_text(answer) and _norm_text(answer) in _norm_text(original_question):
        return None
    question = original_question.lower()
    if re.search(r"\b(?:country|nationality|state|province|region|kingdom|empire|caliphate|commonwealth)\b", question):
        return None
    if not re.search(
        r"\b(?:place of birth|place of death|birthplace|deathplace|born|die|died|death|birth|location|located|where)\b",
        question,
    ):
        return None

    parts = [part.strip() for part in answer.split(",") if part.strip()]
    if len(parts) < 2:
        return None
    if len(parts) >= 3 and parts[1].lower().startswith("near "):
        return parts[0]
    if len(parts) == 2:
        first_norm = _norm_text(parts[0])
        second_norm = _norm_text(parts[1])
        if second_norm == first_norm:
            return parts[0]
        if first_norm and second_norm == f"{first_norm} prefecture":
            return parts[0]
        if second_norm in _LOCATION_CONTEXT_SUFFIXES:
            return parts[0]
    return None


def _looks_like_date_span(answer: str) -> bool:
    return bool(
        re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
            answer,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(r"\s*\d{1,2}\s*,\s*(?:1[0-9]{3}|20[0-9]{2}|[1-9][0-9]{2})\s*", answer)
    )


def _organization_alias_span(answer: str, original_question: str) -> str | None:
    answer = normalize_label(answer).strip()
    if not answer or " of " not in answer.lower():
        return None
    question = original_question.lower()
    if not re.search(r"\b(?:work|institution|organization|organisation|company|employer|where)\b", question):
        return None
    match = re.match(r"^(?P<prefix>.+?)\s+of\s+(?P<suffix>[^,;()]+)$", answer, flags=re.IGNORECASE)
    if not match:
        return None
    prefix = match.group("prefix").strip()
    suffix = match.group("suffix").strip()
    if len(prefix.split()) < 2:
        return None
    if not re.search(
        r"\b(?:institution|institute|society|association|academy|museum|library|gallery|college|university|foundation|school|hospital|company|corporation|council|center|centre|laboratory|observatory)\b",
        prefix,
        flags=re.IGNORECASE,
    ):
        return None
    if _norm_text(suffix) not in _GEOGRAPHIC_SUFFIXES:
        return None
    return prefix


def _date_canonical_span(answer: str, original_question: str) -> str | None:
    answer = normalize_label(answer).strip()
    if not answer:
        return None
    if not re.search(r"\b(?:when|what year|born|birth|died|death|released)\b", original_question.lower()):
        return None
    match = re.fullmatch(r"\s*((?:1[0-9]{3}|20[0-9]{2}|[1-9][0-9]{2}))\s*/\s*((?:1[0-9]{3}|20[0-9]{2}|[1-9][0-9]{2}))\s*", answer)
    if not match:
        return None
    return match.group(1)


def _nationality_canonical_span(answer: str, original_question: str) -> str | None:
    question = original_question.lower()
    answer = normalize_label(answer).strip()
    if not answer or "-" in answer or "/" in answer or "," in answer:
        return None
    # Compound labels stay with the single-stage resolver. This deterministic
    # fallback only handles one-label country/demonym aliases in explicitly
    # typed country/nationality questions.
    if "nationality" in question:
        return _COUNTRY_TO_DEMONYM.get(_norm_text(answer))
    if re.search(r"\bcountry\b", question):
        return _DEMONYM_TO_COUNTRY.get(_norm_text(answer))
    return None


def _comparable_records(
    original_question: str,
    atomic_results: list[dict[str, Any]],
    dag_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results_by_id = {str(item.get("node_id", "")): item for item in atomic_results}
    dag_by_id = {str(item.get("node_id", "")): item for item in dag_nodes}
    original_candidates = _candidate_phrases_from_question(original_question)
    records: list[dict[str, Any]] = []
    for result in atomic_results:
        node_id = str(result.get("node_id", "") or "")
        question = str(result.get("question", "") or "")
        answer = str(result.get("answer", "") or "").strip()
        if not answer or answer.upper() == "INSUFFICIENT_EVIDENCE":
            continue
        kind, value = _parse_comparable_value(question, answer)
        if value is None:
            continue
        label = _branch_label_for_result(
            result=result,
            results_by_id=results_by_id,
            dag_by_id=dag_by_id,
            original_candidates=original_candidates,
        )
        if not label:
            continue
        records.append(
            {
                "node_id": node_id,
                "label": label,
                "kind": kind,
                "value": value,
                "raw_answer": answer,
                "question": question,
            }
        )
    return records


def _parse_comparable_value(question: str, answer: str) -> tuple[str, float | None]:
    lowered_q = question.lower()
    duration = _parse_duration(answer)
    if duration is not None and ("live" in lowered_q or "lived" in lowered_q):
        return "duration", duration
    number = _parse_number(answer)
    if number is not None and ("how old" in lowered_q or " age" in lowered_q or "birth year" in lowered_q):
        if "birth" in lowered_q:
            return "date", number
        return "number", number
    year = _parse_year(answer)
    if year is not None:
        return "date", year
    return "unknown", None


def _parse_year(answer: str) -> float | None:
    years = [int(item) for item in re.findall(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2}|[1-9][0-9]{2})(?!\d)", answer)]
    if not years:
        return None
    return float(years[0])


def _parse_number(answer: str) -> float | None:
    match = re.search(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", answer)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_duration(answer: str) -> float | None:
    years = [int(item) for item in re.findall(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2}|[1-9][0-9]{2})(?!\d)", answer)]
    if len(years) >= 2:
        return float(abs(years[1] - years[0]))
    return None


def _branch_label_for_result(
    *,
    result: dict[str, Any],
    results_by_id: dict[str, dict[str, Any]],
    dag_by_id: dict[str, dict[str, Any]],
    original_candidates: list[str],
) -> str:
    node_id = str(result.get("node_id", "") or "")
    dependencies = _dependencies_for_node(result, dag_by_id.get(node_id, {}))
    if dependencies:
        return _branch_label_for_node(
            dependencies[0],
            results_by_id=results_by_id,
            dag_by_id=dag_by_id,
            original_candidates=original_candidates,
            seen={node_id},
        )
    return _extract_branch_label(str(result.get("question", "") or ""), original_candidates)


def _branch_label_for_node(
    node_id: str,
    *,
    results_by_id: dict[str, dict[str, Any]],
    dag_by_id: dict[str, dict[str, Any]],
    original_candidates: list[str],
    seen: set[str],
) -> str:
    if node_id in seen:
        return ""
    seen.add(node_id)
    node = dag_by_id.get(node_id, {})
    result = results_by_id.get(node_id, {})
    dependencies = _dependencies_for_node(result, node)
    if dependencies:
        return _branch_label_for_node(
            dependencies[0],
            results_by_id=results_by_id,
            dag_by_id=dag_by_id,
            original_candidates=original_candidates,
            seen=seen,
        )
    return _extract_branch_label(
        str(node.get("question") or result.get("question") or ""),
        original_candidates,
    )


def _dependencies_for_node(result: dict[str, Any], node: dict[str, Any]) -> list[str]:
    raw = result.get("used_dependencies")
    if raw is None:
        raw = node.get("dependencies")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _extract_branch_label(question: str, original_candidates: list[str]) -> str:
    normalized_question = _norm_text(question)
    for candidate in sorted(original_candidates, key=len, reverse=True):
        if _norm_text(candidate) in normalized_question:
            return candidate

    patterns = (
        r"\b(?:director|performer|composer|author|spouse|husband|wife|mother|father|child)\s+of\s+(?:the\s+)?(?:film|movie|song|book)?\s*(?P<label>.+?)\??$",
        r"\b(?:born|released|die|died|live|lived)\s+(?:in\s+|on\s+)?(?P<label>.+?)\??$",
        r"\b(?:of|for)\s+(?P<label>[A-Z][^?]+?)\??$",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return _clean_candidate_label(match.group("label"))
    return ""


def _candidate_phrases_from_question(question: str) -> list[str]:
    cleaned = question.strip().rstrip("? ")
    candidates: list[str] = []
    if "," in cleaned:
        tail = cleaned.split(",", 1)[-1]
        candidates.extend(_split_candidate_tail(tail))
    out_of = re.search(r"\bout of\s+(.+)$", cleaned, flags=re.IGNORECASE)
    if out_of:
        candidates.extend(_split_candidate_tail(out_of.group(1)))
    subject_choices = re.search(
        r"^(?:was|were|is|are|did|do|does)\s+(.+?)\s+\b(?:born|released|died|die|live|lived)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if subject_choices:
        candidates.extend(_split_candidate_tail(subject_choices.group(1)))
    both = re.search(r"\bboth\s+(.+?)\s+(?:located|from|born|of|share|in)\b", cleaned, flags=re.IGNORECASE)
    if both:
        candidates.extend(_split_candidate_tail(both.group(1)))
    candidates.extend(_capitalized_phrases(cleaned))
    result: list[str] = []
    for candidate in candidates:
        candidate = _clean_candidate_label(candidate)
        if candidate and candidate not in result and len(candidate) > 1:
            result.append(candidate)
    return result


def _split_candidate_tail(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"\s+\b(?:or|and)\b\s+", text)
        if part.strip()
    ]


def _capitalized_phrases(text: str) -> list[str]:
    return re.findall(
        r"\b[A-Z][A-Za-z0-9'’.\-]*(?:\s+(?:[A-Z][A-Za-z0-9'’.\-]*|of|the|and|de|le|la|von|van))*",
        text,
    )


def _clean_candidate_label(text: str) -> str:
    text = normalize_label(text).strip(" ?.,;:")
    text = re.sub(
        r"^(?:which|what|who|whom|whose|was|were|is|are|did|do|does)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:the\s+)?(?:film|movie|song|book|director|performer|composer)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" ?.,;:")


def _minimal_candidate_span(answer: str, original_question: str) -> str:
    answer = normalize_label(answer)
    if not answer or answer.upper() == "INSUFFICIENT_EVIDENCE":
        return answer
    lowered_q = original_question.lower().strip()
    if lowered_q.startswith(("are ", "do ", "does ", "did ", "is ", "was ", "were ")):
        yes_no = re.search(r"\b(yes|no)\b", answer, flags=re.IGNORECASE)
        if yes_no:
            return yes_no.group(1).lower()
    candidate_answer = _strip_leading_question_auxiliary(answer)
    normalized_answer = _norm_text(candidate_answer)
    for candidate in sorted(_candidate_phrases_from_question(original_question), key=len, reverse=True):
        normalized_candidate = _norm_text(candidate)
        if not normalized_candidate or not normalized_answer:
            continue
        if normalized_candidate in normalized_answer:
            return candidate
        if normalized_answer in normalized_candidate and len(normalized_answer) >= 6:
            return candidate
    return candidate_answer.strip()


def _strip_leading_question_auxiliary(answer: str) -> str:
    return re.sub(
        r"^(?:was|were|is|are|did|do|does)\s+",
        "",
        normalize_label(answer).strip(),
        flags=re.IGNORECASE,
    )


def _normalize_comparison_answer(answer: str) -> str:
    text = normalize_label(answer).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_label(text).lower()).strip()
