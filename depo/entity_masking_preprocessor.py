"""DEPO Step1--2：识别显式实体并构造 HanLP 掩码输入。"""

from __future__ import annotations

import re

from llm_client import LLMClient
from models import PreprocessedQuestion
from prompts import EXPLICIT_ENTITY_EXTRACTION_SYSTEM, build_explicit_entity_extraction_prompt


def preprocess_question(question: str, llm_client: LLMClient) -> PreprocessedQuestion:
    """将问题中的显式实体替换为 ENTITYA、ENTITYB 等占位符。"""

    payload = llm_client.chat_json(
        EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
        build_explicit_entity_extraction_prompt(question),
    )
    entities = [item["surface"] for item in payload["explicit_entities"]]
    mask_mapping = {
        f"ENTITY{_letter_suffix(index)}": entity
        for index, entity in enumerate(entities)
    }
    masked_question = payload["normalized_question"]
    for placeholder, entity in mask_mapping.items():
        masked_question = re.sub(re.escape(entity), placeholder, masked_question)

    return PreprocessedQuestion(
        entities=entities,
        masked_question=masked_question,
        mask_mapping=mask_mapping,
    )


def _letter_suffix(index: int) -> str:
    """将 0、1、... 转为 A、B、...、AA。"""

    label = ""
    while True:
        label = chr(ord("A") + index % 26) + label
        index = index // 26 - 1
        if index < 0:
            return label
