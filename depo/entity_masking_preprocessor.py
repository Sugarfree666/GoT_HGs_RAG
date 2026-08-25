"""DEPO Step1--2：识别显式实体并构造 HanLP 掩码输入。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm_client import LLMClient
from models import PreprocessedQuestion

#找到提示词文件，以UTF-8 读完整个 Markdown 文件，去掉头尾空白。
ENTITY_RECOGNITION_PROMPT = (
    Path(__file__).resolve().parents[1] / "prompts" / "entity_recognition.md"
).read_text(encoding="utf-8").strip()


def preprocess_question(question: str, llm_client: LLMClient) -> PreprocessedQuestion:
    """将问题中的显式实体替换为 ENTITYA、ENTITYB 等占位符。"""

    payload = llm_client.chat_json(
        #System：
        ENTITY_RECOGNITION_PROMPT,
        #User：
        #将问题转换成json格式输入LLM
        json.dumps({"question": question}, ensure_ascii=False, indent=2),
    )
    #获取实体
    entities = payload["entities"]
    mask_mapping = {
        f"ENTITY{_letter_suffix(index)}": entity
        for index, entity in enumerate(entities)
    }
    masked_question = question
    #实体mask
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
