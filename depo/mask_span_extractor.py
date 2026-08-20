"""在 DEPO 掩码前校验 LLM 提议的显式实体 span。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from models import ExplicitEntity, ExplicitEntityResult, MaskSpan, MaskSpanResult
from prompts import (
    EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
    build_explicit_entity_extraction_prompt,
)

if TYPE_CHECKING:
    from llm_client import LLMClient


class ExplicitEntityExtractor:
    """通过 LLM 提取显式实体，并拒绝不可靠输出。

    本模块不会依据标题格式、大小写、领域词表或手工类型表猜测实体；
    只接受能在原问题中无歧义匹配、且不重叠的 LLM 返回文本。
    """

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.llm_client = llm_client

    def extract(self, question: str) -> ExplicitEntityResult:
        """请求显式实体；失败时返回空结果而不是猜测。"""

        if self.llm_client is None:
            return ExplicitEntityResult(
                entities=[],
                warnings=[
                    "Explicit entity extraction is unavailable because no LLM client was supplied; "
                    "returning no entities rather than guessing."
                ],
                raw_payload=None,
                normalized_question=question,
                normalization_changed=False,
                normalization_note="",
            )

        warnings: list[str] = []
        raw_payload: dict[str, Any] | None = None
        try:
            raw = self.llm_client.chat_json(
                EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
                build_explicit_entity_extraction_prompt(question),
            )
            raw_payload = raw if isinstance(raw, dict) else {}
            if not isinstance(raw, dict):
                warnings.append("Explicit entity LLM returned a non-object payload; returning no entities.")

            # 问题规范化元数据与实体列表是否有效相互独立。
            normalized_question, normalization_changed, normalization_note = self._parse_normalization_payload(
                question,
                raw_payload,
                warnings,
            )
            entities = self._parse_payload(question, raw_payload, warnings)
            return ExplicitEntityResult(
                entities=entities,
                warnings=warnings,
                raw_payload=raw_payload,
                normalized_question=normalized_question,
                normalization_changed=normalization_changed,
                normalization_note=normalization_note,
            )
        except Exception as exc:
            return ExplicitEntityResult(
                entities=[],
                warnings=[f"Explicit entity LLM failed; returning no entities rather than guessing: {exc}"],
                raw_payload=raw_payload,
                normalized_question=question,
                normalization_changed=False,
                normalization_note="",
            )

    @staticmethod
    def _parse_normalization_payload(
        question: str,
        payload: dict[str, Any],
        warnings: list[str],
    ) -> tuple[str, bool, str]:
        raw_normalized = payload.get("normalized_question")
        if isinstance(raw_normalized, str) and raw_normalized.strip():
            normalized_question = raw_normalized.strip()
        else:
            normalized_question = question
            if "normalized_question" in payload and raw_normalized not in (None, ""):
                warnings.append("Ignored invalid normalized_question from explicit entity payload.")

        changed = _coerce_bool(
            payload.get("normalization_changed"),
            default=normalized_question != question,
        )
        if normalized_question == question:
            changed = False
        elif "normalization_changed" not in payload:
            changed = True

        note = str(payload.get("normalization_note") or "").strip()
        return normalized_question, changed, note

    @staticmethod
    def _parse_payload(
        question: str,
        payload: dict[str, Any],
        warnings: list[str],
    ) -> list[ExplicitEntity]:
        """仅当全部候选文本都通过结构校验时，才接受该实体列表。"""

        raw_entities = payload.get("explicit_entities")
        if not isinstance(raw_entities, list):
            warnings.append("Explicit entity payload did not contain an explicit_entities list.")
            return []
        if not raw_entities:
            warnings.append("Explicit entity LLM returned no entities; no heuristic fallback was applied.")
            return []

        entities: list[ExplicitEntity] = []
        invalid_reasons: list[str] = []
        seen_surfaces: set[str] = set()
        spans: list[tuple[int, int, str]] = []
        for index, raw in enumerate(raw_entities, start=1):
            if not isinstance(raw, dict):
                invalid_reasons.append(f"explicit_entities[{index - 1}] must be an object")
                continue

            surface = raw.get("surface")
            if not isinstance(surface, str) or not surface.strip():
                invalid_reasons.append(
                    f"explicit_entities[{index - 1}].surface must be a non-empty string"
                )
                continue

            # 即使 LLM 仅改变大小写，也要恢复原文形式和原始字符位置。
            resolved_surface, matches, match_error = _find_surface_matches_case_relaxed(question, surface)
            if match_error is not None:
                invalid_reasons.append(match_error)
                continue
            if resolved_surface in seen_surfaces:
                invalid_reasons.append(f"duplicate explicit entity surface={resolved_surface!r}")
                continue
            seen_surfaces.add(resolved_surface)

            if any(
                match.start() < other_end and match.end() > other_start
                for match in matches
                for other_start, other_end, _ in spans
            ):
                invalid_reasons.append(
                    f"overlapping explicit entity span for surface={resolved_surface!r}"
                )
                continue

            spans.extend((match.start(), match.end(), resolved_surface) for match in matches)
            entities.append(
                ExplicitEntity(
                    text=resolved_surface,
                    start_char=matches[0].start(),
                    end_char=matches[0].end(),
                    semantic_type_hint=_normalize_entity_type(raw.get("type")),
                    confidence=1.0,
                    reason="validated LLM explicit entity",
                )
            )

        if invalid_reasons:
            # 部分接受会悄悄改变 LLM 原本给出的实体集合，因此整组拒绝。
            warnings.append(
                "Invalid LLM explicit_entities output; ignoring all entities: "
                + "; ".join(invalid_reasons)
            )
            return []
        return entities


class MaskSpanExtractor:
    """将 ``ExplicitEntityExtractor`` 的结果适配为旧版掩码 span 接口。"""

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.explicit_extractor = ExplicitEntityExtractor(llm_client)

    def extract(self, question: str) -> MaskSpanResult:
        result = self.explicit_extractor.extract(question)
        return MaskSpanResult(
            mask_spans=[
                MaskSpan(
                    text=entity.text,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    kind_hint="entity",
                    semantic_type_hint=entity.semantic_type_hint or "Entity",
                    reason=entity.reason,
                )
                for entity in result.entities
            ],
            warnings=result.warnings,
            raw_payload=result.raw_payload,
        )


def _find_surface_matches_case_relaxed(
    question: str,
    surface: str,
) -> tuple[str, list[re.Match[str]], str | None]:
    """查找原文精确文本；仅在唯一时允许大小写差异。"""

    matches = list(re.finditer(re.escape(surface), question))
    if matches:
        return surface, matches, None

    case_relaxed_matches = list(re.finditer(re.escape(surface), question, flags=re.IGNORECASE))
    if not case_relaxed_matches:
        return (
            surface,
            [],
            "explicit entity surface was not found in the original question with matching spaces "
            f"and punctuation: {surface!r}",
        )

    source_surfaces = {match.group(0) for match in case_relaxed_matches}
    if len(source_surfaces) != 1:
        return (
            surface,
            [],
            f"case-insensitive explicit entity surface matched multiple source casings: {surface!r}",
        )
    return case_relaxed_matches[0].group(0), case_relaxed_matches, None


def _normalize_entity_type(value: Any) -> str:
    """保留可选的 LLM 类型元数据，不映射到固定领域词表。"""

    normalized = " ".join(str(value or "").split())
    return normalized[:80] if normalized else "Entity"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    return default
