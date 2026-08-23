"""在 DEPO 掩码前校验 LLM 提议的显式实体 span。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
#导入数据结构
from models import ExplicitEntity, ExplicitEntityResult
from prompts import (
    EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
    build_explicit_entity_extraction_prompt,
)

if TYPE_CHECKING:
    from llm_client import LLMClient


class ExplicitEntityExtractor:
    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def extract(self, question: str) -> ExplicitEntityResult:
        """使用LLM从问题中提取实体"""
        raw_payload = self.llm_client.chat_json(
            EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
            build_explicit_entity_extraction_prompt(question),
        )
        normalized_question = raw_payload["normalized_question"].strip()
        entities = self._parse_payload(raw_payload)
        return ExplicitEntityResult(
            entities=entities,
            normalized_question=normalized_question,
        )

    @staticmethod
    #取出LLM返回的实体
    def _parse_payload(
        payload: dict[str, Any],
    ) -> list[ExplicitEntity]:
        return [
            ExplicitEntity(text=item["surface"])
            for item in payload["explicit_entities"]
        ]
