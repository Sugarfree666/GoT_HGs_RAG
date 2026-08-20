"""从原子问题提取轻量检索提示；可用时调用 LLM。"""

from __future__ import annotations

import re
from typing import Any

from ..llm.service import AtomicLLMService
from ..utils import ensure_list, normalize_label
from .models import AtomicQuestionAnalysis


CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9'&.-]*)(?:\s+(?:[A-Z][a-zA-Z0-9'&.-]*|of|the|and|&))*"
)
APPOSITIVE_TITLE_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9'&.-]*(?:\s+[A-Z][a-zA-Z0-9'&.-]*)*,\s+"
    r"(?:[0-9]+(?:st|nd|rd|th)\s+)?"
    r"(?:Duke|Earl|Count|King|Queen|Prince|Princess|Lord|Lady|Baron|Bishop|Pope|Emperor|Empress|"
    r"Saint|Sir|Dame|Dr)\b(?:\s+(?:of|the|and|[A-Z][a-zA-Z0-9'&.-]+))*"
)
COMMA_TITLE_CUE_RE = re.compile(
    r"\b(?:song|album|mixtape|film|movie|book|novel|work|series|episode|single|release|"
    r"performed|performer|director|label|of|called|titled|named)\s+"
    r"(?P<title>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*|\s+of|\s+the|\s+and|\s+&)*,\s+"
    r"[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*|\s+of|\s+the|\s+and|\s+&)*)"
)
WH_WORDS = {"what", "which", "who", "where", "when", "why", "how"}
GENERIC_ENTITY_MENTIONS = {
    "album",
    "artist",
    "award",
    "battle",
    "championship",
    "championship series",
    "city",
    "company",
    "continent",
    "countries",
    "country",
    "district",
    "economic growth",
    "event",
    "film",
    "group",
    "house of representatives",
    "league",
    "location",
    "man",
    "organization",
    "party",
    "person",
    "place",
    "region",
    "school",
    "series",
    "song",
    "state",
    "team",
    "teams",
    "the championship series",
    "the house of representatives",
    "the tournament",
    "tournament",
    "university",
    "woman",
    "work",
}
POSSESSIVE_ROLE_TERMS = {
    "actor",
    "actress",
    "artist",
    "author",
    "brother",
    "child",
    "composer",
    "creator",
    "daughter",
    "director",
    "father",
    "founder",
    "grandfather",
    "grandmother",
    "husband",
    "mother",
    "parent",
    "performer",
    "producer",
    "sister",
    "son",
    "spouse",
    "wife",
    "writer",
}


class AtomicQuestionAnalyzer:
    """将 LLM 或启发式分析统一为实体提及和预期答案类型。"""

    def __init__(self, llm_service: AtomicLLMService | None = None) -> None:
        self.llm_service = llm_service

    def analyze(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]] | None = None,
    ) -> AtomicQuestionAnalysis:
        """优先通过服务分析问题；不可用时使用透明启发式方法。"""

        dependency_answers = dependency_answers or []
        if self.llm_service is not None:
            payload = self.llm_service.analyze_atomic_question(
                atomic_question=atomic_question,
                dependency_answers=dependency_answers,
            )
        else:
            payload = self._heuristic_analysis(atomic_question)
        return self._coerce_payload(payload, atomic_question)

    def _coerce_payload(self, payload: Any, atomic_question: str) -> AtomicQuestionAnalysis:
        """清洗服务 payload，并在字段缺失时回退到可解释的本地分析。"""

        if not isinstance(payload, dict):
            payload = self._heuristic_analysis(atomic_question)
        entities = self._clean_entity_mentions(payload.get("entities", []))
        answer_type = _infer_answer_type(atomic_question)
        return AtomicQuestionAnalysis(
            entities=entities,
            answer_type=answer_type,
        )

    def _heuristic_analysis(self, atomic_question: str) -> dict[str, Any]:
        """在没有 LLM 服务时，从文本中提取保守的实体和答案类型。"""

        entities = _extract_capitalized_entities(atomic_question)
        return {
            "entities": entities,
        }

    def _clean_entity_mentions(self, value: Any) -> list[str]:
        cleaned: list[str] = []
        for item in ensure_list(value):
            text = normalize_label(str(item).strip())
            if not text:
                continue
            entity = _strip_possessive_role_tail(text)
            if _is_generic_entity_mention(entity):
                continue
            if entity and entity not in cleaned:
                cleaned.append(entity)
        return cleaned


def _extract_capitalized_entities(question: str) -> list[str]:
    spans: list[tuple[int, int, str]] = []
    for match in CAPITALIZED_PHRASE_RE.finditer(question):
        spans.append((match.start(), match.end(), match.group(0)))
    for match in APPOSITIVE_TITLE_RE.finditer(question):
        spans.append((match.start(), match.end(), match.group(0)))
    for match in COMMA_TITLE_CUE_RE.finditer(question):
        spans.append((match.start("title"), match.end("title"), match.group("title")))

    entities: list[str] = []
    occupied: list[tuple[int, int]] = []
    for start, end, raw_text in sorted(spans, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start >= used_start and end <= used_end for used_start, used_end in occupied):
            continue
        text = normalize_label(raw_text)
        if not text:
            continue
        if text.lower() in WH_WORDS:
            continue
        text = _strip_possessive_role_tail(text)
        if _is_generic_entity_mention(text):
            continue
        if text not in entities:
            entities.append(text)
            occupied.append((start, end))
    return entities


def _strip_possessive_role_tail(text: str) -> str:
    match = re.match(r"^(.+?)'s\s+([A-Za-z][A-Za-z -]*)$", text)
    if not match:
        return text
    owner = normalize_label(match.group(1))
    tail = normalize_label(match.group(2)).lower()
    if tail in POSSESSIVE_ROLE_TERMS:
        return owner
    return text


def _is_generic_entity_mention(text: str) -> bool:
    normalized = normalize_label(text).lower().strip(" ?.,;:!\"'")
    if not normalized:
        return True
    if normalized in WH_WORDS or normalized in GENERIC_ENTITY_MENTIONS:
        return True
    if normalized.startswith(("which ", "what ", "who ", "where ", "when ", "how ")):
        return True
    return False


def _infer_answer_type(question: str) -> str:
    lowered = question.strip().lower()
    if lowered.startswith("which "):
        tokens = lowered.split()
        if len(tokens) > 1:
            return tokens[1].strip(" ?.,")
    if lowered.startswith("what "):
        tokens = lowered.split()
        if len(tokens) > 1 and tokens[1] not in {"is", "are", "was", "were", "did", "does", "do"}:
            return tokens[1].strip(" ?.,")
        return "entity, concept, or phrase"
    if lowered.startswith("who "):
        return "person or organization"
    if lowered.startswith("where "):
        return "location"
    if lowered.startswith("when "):
        return "time or date"
    return "grounded short answer"
