"""DEPO 实体预处理：提取显式实体、规范化问题、掩码并建立映射。"""

from __future__ import annotations

# 本文件负责 DEPO 的实体预处理：
# 原问题 -> 显式实体识别 -> 问题规范化 -> 实体掩码 -> 生成占位符映射

import re
from typing import TYPE_CHECKING

# ExplicitEntityExtractor：调用 LLM 识别显式实体
# models 中的数据类用于保存实体、映射和最终预处理结果
from mask_span_extractor import ExplicitEntityExtractor
#导入数据结构
from models import ExplicitEntity, ExplicitEntityResult, HanLPSDPPreprocessResult, MaskMapping

# 为了避免循环导入
if TYPE_CHECKING:
    from llm_client import LLMClient


class EntityMaskingPreprocessor:

    def __init__(self, llm_client: "LLMClient") -> None:
        # 创建显式实体提取器
        self.explicit_extractor = ExplicitEntityExtractor(llm_client)

    def preprocess(self, question: str) -> HanLPSDPPreprocessResult:
        """对一个问题完成实体识别、规范化、掩码和校验。"""

        # Step1：识别问题中的显式实体
        explicit_entities = self.explicit_extractor.extract(question)
        # 若实体提取器给出了规范化问题，则优先使用规范化版本
        normalized_question = _normalized_question_from_entities(question, explicit_entities)
        # 问题修改了，但是其中没有原问题中的实体
        if normalized_question != question and any(entity.text not in normalized_question for entity in explicit_entities.entities):
            raise ValueError("Normalized question must preserve every explicit entity surface.")
        # Step2：把真实实体替换为 ENTITYA、ENTITYB 等占位符
        masked_question = _masked_question_from_entities(
            normalized_question,
            explicit_entities.entities,
        )
        # 建立占位符与原实体之间的对应关系
        mask_mappings = _mask_mappings_from_entities(explicit_entities.entities)
        # 将所有预处理结果统一封装，交给 main.py 后续 HanLP/Step4 使用
        return HanLPSDPPreprocessResult(
            explicit_entities=ExplicitEntityResult(
                entities=list(explicit_entities.entities),
            ),
            masked_question=masked_question,
            mask_mappings=mask_mappings,
        )


def _normalized_question_from_entities(question: str, explicit_entities: ExplicitEntityResult) -> str:
    """取得实体提取器返回的规范化问题；若没有，则使用原问题。"""
    normalized_question = explicit_entities.normalized_question
    #检查是否非空字符串并且去除首尾空格
    if isinstance(normalized_question, str) and normalized_question.strip():
        return normalized_question.strip()
    return question


def _masked_question_from_entities(
    question: str,
    entities: list[ExplicitEntity],
) -> str:
    """把问题中的显式实体替换成 ENTITYA、ENTITYB 等占位符。"""
    #先复制一份原问题
    masked = question
    # 创建一个空列表，保存每一次替换的位置：(开始位置, 结束位置, 占位符)
    replacements: list[tuple[int, int, str]] = []
    # 遍历每一个实体得到实体和编号。
    for index, entity in enumerate(entities):
        # 在问题中查找当前实体的所有出现位置
        matches = list(re.finditer(re.escape(entity.text), question))
        # 第一个实体 -> ENTITYA，第二个 -> ENTITYB，以此类推
        placeholder = f"ENTITY{_letter_suffix(index)}"
        for match in matches:
            replacements.append((match.start(), match.end(), placeholder))
    # 从字符串后往前替换，避免前面的替换改变后续字符位置
    for start, end, placeholder in sorted(replacements, key=lambda item: item[0], reverse=True):
        masked = masked[:start] + placeholder + masked[end:]
    return masked


def _mask_mappings_from_entities(entities: list[ExplicitEntity]) -> list[MaskMapping]:
    """生成 ENTITYA -> 原实体 的映射信息。"""
    mappings: list[MaskMapping] = []
    for index, entity in enumerate(entities):
        placeholder = f"ENTITY{_letter_suffix(index)}"
        mappings.append(
            MaskMapping(
                placeholder=placeholder,
                original_text=entity.text,
            )
        )
    return mappings


#将数字编号变成字母编号
def _letter_suffix(index: int) -> str:
    """把 0,1,2... 转成 A,B,C...，超过 Z 后继续 AA、AB...。"""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    current = index
    while True:
        label = alphabet[current % len(alphabet)] + label
        current = current // len(alphabet) - 1
        if current < 0:
            return label
