"""DEPO Step1--2：识别显式实体并构造 HanLP 掩码输入。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from hyper_branch.client import OpenAIClient
from models import PreprocessedQuestion

#找到提示词文件，以UTF-8 读完整个 Markdown 文件，去掉头尾空白。
TOPIC_ENTITY_RECOGNITION_PROMPT = (
    Path(__file__).resolve().parents[1] / "prompts" / "topic_entity_recognition.md"
).read_text(encoding="utf-8").strip()


def preprocess_question(question: str, llm_client: OpenAIClient) -> PreprocessedQuestion:
    """将问题中的显式实体替换为 ENTITYA、ENTITYB 等占位符。"""

    payload = llm_client.chat_json(
        #System：
        TOPIC_ENTITY_RECOGNITION_PROMPT,
        #User：
        #将问题转换成json格式输入LLM
        json.dumps({"question": question}, ensure_ascii=False, indent=2),
    )
    #获取实体
    masked_question = question
    entities: list[str] = []
    mask_mapping: dict[str, str] = {}
    for entity in payload["entities"]:
        if not isinstance(entity, str) or not entity.strip():
            continue
        placeholder = f"ENTITY{_letter_suffix(len(mask_mapping))}"
        masked_question, matched_text = _mask_entity(
            masked_question, entity.strip(), placeholder
        )
        if matched_text is None:
            continue
        entities.append(matched_text)
        mask_mapping[placeholder] = matched_text

    return PreprocessedQuestion(
        entities=entities,
        masked_question=masked_question,
        mask_mapping=mask_mapping,
    )


def _mask_entity(
    question: str,
    entity: str,
    placeholder: str,
) -> tuple[str, str | None]:
    """Mask one surface while keeping the placeholder a standalone token."""
    pattern = _entity_pattern(entity)
    matched_text: str | None = None

    def replace(match: re.Match[str]) -> str:
        nonlocal matched_text
        matched_text = matched_text or match.group(0)
        return placeholder

    masked_question, count = pattern.subn(replace, question)
    if count:
        return masked_question, matched_text
    return question, None


def _entity_pattern(entity: str) -> re.Pattern[str]:
    return re.compile(
        rf"{_left_boundary(entity)}{re.escape(entity)}{_right_boundary(entity)}",
        flags=re.IGNORECASE,
    )


def _left_boundary(entity: str) -> str:
    return r"(?<!\w)" if entity[0].isalnum() else ""


def _right_boundary(entity: str) -> str:
    return r"(?!\w)" if entity[-1].isalnum() else ""


def _letter_suffix(index: int) -> str:
    """将 0、1、... 转为 A、B、...、AA。"""

    label = ""
    while True:
        label = chr(ord("A") + index % 26) + label
        index = index // 26 - 1
        if index < 0:
            return label
