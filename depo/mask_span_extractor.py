"""Step 1: extract explicit entities from a question with the LLM."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from models import ExplicitEntity, ExplicitEntityResult
from prompts import (
    EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
    build_explicit_entity_extraction_prompt,
)

if TYPE_CHECKING:
    from llm_client import LLMClient


class ExplicitEntityExtractor:
    """Extract exact, non-overlapping entity surfaces from the original question."""

    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def extract(self, question: str) -> ExplicitEntityResult:
        raw_payload = self.llm_client.chat_json(
            EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
            build_explicit_entity_extraction_prompt(question),
        )
        normalized_question = raw_payload.get("normalized_question") or question
        return ExplicitEntityResult(
            entities=self._parse_entities(question, raw_payload),
            raw_payload=raw_payload,
            normalized_question=normalized_question,
            normalization_changed=normalized_question != question,
            normalization_note=raw_payload.get("normalization_note", "").strip(),
        )

    @staticmethod
    def _parse_entities(
        question: str,
        payload: dict[str, Any],
    ) -> list[ExplicitEntity]:
        entities: list[ExplicitEntity] = []
        seen_surfaces: set[str] = set()
        spans: list[tuple[int, int]] = []

        for raw_entity in payload["explicit_entities"]:
            surface = raw_entity["surface"]
            matches = list(re.finditer(re.escape(surface), question))
            if not matches:
                raise ValueError(
                    f"Explicit entity surface is not in the original question: {surface!r}"
                )
            if surface in seen_surfaces:
                raise ValueError(f"Duplicate explicit entity surface: {surface!r}")
            if any(
                match.start() < end and match.end() > start
                for match in matches
                for start, end in spans
            ):
                raise ValueError(f"Overlapping explicit entity span: {surface!r}")

            seen_surfaces.add(surface)
            spans.extend((match.start(), match.end()) for match in matches)
            entities.append(
                ExplicitEntity(
                    text=surface,
                    start_char=matches[0].start(),
                    end_char=matches[0].end(),
                    semantic_type_hint=raw_entity["type"],
                    reason="LLM explicit entity",
                )
            )
        return entities
