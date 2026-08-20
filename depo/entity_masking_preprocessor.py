"""DEPO 实体预处理：提取显式实体、规范化问题、掩码并建立映射。"""

from __future__ import annotations

# 本文件负责 DEPO 的实体预处理：
# 原问题 -> 显式实体识别 -> 问题规范化 -> 实体掩码 -> 生成占位符映射

import re
from typing import TYPE_CHECKING

# ExplicitEntityExtractor：调用 LLM 识别显式实体
# models 中的数据类用于保存实体、映射和最终预处理结果
from mask_span_extractor import ExplicitEntityExtractor
from models import ExplicitEntity, ExplicitEntityResult, HanLPSDPPreprocessResult, MaskMapping

# 仅用于类型检查，程序运行时不会在这里真正导入
if TYPE_CHECKING:
    from llm_client import LLMClient


class EntityMaskingPreprocessor:
    """DEPO 解析管线的通用 Step2 实体掩码器。

    LLM 只用于显式实体提取；占位符分配和掩码问题构造保持确定性。
    """

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        # 创建显式实体提取器，LLM 主要在实体识别阶段使用
        self.explicit_extractor = ExplicitEntityExtractor(llm_client)

    def preprocess(self, question: str) -> HanLPSDPPreprocessResult:
        """对一个问题完成实体识别、规范化、掩码和校验。"""

        # Step1：识别问题中的显式实体
        explicit_entities = self.explicit_extractor.extract(question)
        # 保存实体提取阶段产生的警告
        warnings = list(explicit_entities.warnings)

        # 若实体提取器给出了规范化问题，则优先使用规范化版本
        normalized_question = _normalized_question_from_entities(question, explicit_entities)
        # 安全检查：规范化后不能把原来的显式实体弄丢
        if normalized_question != question and any(entity.text not in normalized_question for entity in explicit_entities.entities):
            warnings.append("Normalized question did not preserve every explicit entity surface; using original question.")
            normalized_question = question
        # 记录问题是否真的发生了规范化修改
        normalization_changed = explicit_entities.normalization_changed or normalized_question != question
        if normalized_question == question:
            normalization_changed = False
        normalization_note = explicit_entities.normalization_note if normalization_changed else ""
        # Step2：把真实实体替换为 ENTITYA、ENTITYB 等占位符
        masked_question = _masked_question_from_entities(
            normalized_question,
            explicit_entities.entities,
            warnings=warnings,
        )
        # 建立占位符与原实体之间的对应关系
        mask_mappings = _mask_mappings_from_entities(masked_question, explicit_entities.entities)
        # 对掩码后的结果做基本一致性检查
        _validate_preprocess_result(
            question=question,
            masked_question=masked_question,
            mask_mappings=mask_mappings,
            warnings=warnings,
        )
        # 将所有预处理结果统一封装，交给 main.py 后续 HanLP/Step4 使用
        return HanLPSDPPreprocessResult(
            original_question=question,
            explicit_entities=ExplicitEntityResult(
                entities=list(explicit_entities.entities),
                warnings=list(warnings),
                raw_payload=explicit_entities.raw_payload,
                normalized_question=normalized_question,
                normalization_changed=normalization_changed,
                normalization_note=normalization_note,
            ),
            masked_question=masked_question,
            sdp_input_sentence=masked_question,
            mask_mappings=mask_mappings,
            warnings=list(warnings),
            raw_payload=explicit_entities.raw_payload,
            normalized_question=normalized_question,
            normalization_changed=normalization_changed,
            normalization_note=normalization_note,
        )


def _normalized_question_from_entities(question: str, explicit_entities: ExplicitEntityResult) -> str:
    """取得实体提取器返回的规范化问题；若没有，则使用原问题。"""
    normalized_question = explicit_entities.normalized_question
    if isinstance(normalized_question, str) and normalized_question.strip():
        return normalized_question.strip()
    return question


def _masked_question_from_entities(
    question: str,
    entities: list[ExplicitEntity],
    *,
    warnings: list[str] | None = None,
) -> str:
    """把问题中的显式实体替换成 ENTITYA、ENTITYB 等占位符。"""

    masked = question
    # 保存每一次替换的位置：(开始位置, 结束位置, 占位符)
    replacements: list[tuple[int, int, str]] = []
    for index, entity in enumerate(entities):
        # 在问题中查找当前实体的所有出现位置
        matches = list(re.finditer(re.escape(entity.text), question))
        if not matches:
            if warnings is not None:
                warnings.append(
                    f"Could not find explicit entity surface in normalized question; left unmasked: {entity.text!r}."
                )
            continue
        # 第一个实体 -> ENTITYA，第二个 -> ENTITYB，以此类推
        placeholder = f"ENTITY{_letter_suffix(index)}"
        for match in matches:
            replacements.append((match.start(), match.end(), placeholder))
    # 从字符串后往前替换，避免前面的替换改变后续字符位置
    for start, end, placeholder in sorted(replacements, key=lambda item: item[0], reverse=True):
        masked = masked[:start] + placeholder + masked[end:]
    return masked


def _mask_mappings_from_entities(masked_question: str, entities: list[ExplicitEntity]) -> list[MaskMapping]:
    """生成 ENTITYA -> 原实体 的映射信息。"""
    mappings: list[MaskMapping] = []
    for index, entity in enumerate(entities):
        placeholder = f"ENTITY{_letter_suffix(index)}"
        # 查找占位符在掩码后问题中的字符位置
        masked_span = _find_placeholder_span(masked_question, placeholder)

        # 保存原实体、占位符、语义类型和字符位置
        mappings.append(
            MaskMapping(
                placeholder=placeholder,
                original_text=entity.text,
                kind_hint="entity",
                #实体类型
                semantic_type_hint=entity.semantic_type_hint or "Entity",
                original_char_span=[entity.start_char, entity.end_char],
                #记录掩码在掩码问题中的位置
                masked_char_span=list(masked_span) if masked_span else [],
            )
        )
    return mappings


def _validate_preprocess_result(
    question: str,
    masked_question: str,
    mask_mappings: list[MaskMapping],
    warnings: list[str],
) -> None:
    """检查实体掩码结果是否合理。"""

    for mapping in mask_mappings:
        # 每个占位符都必须真正出现在 masked_question 中
        if mapping.placeholder not in masked_question:
            raise ValueError(f"masked_question is missing placeholder {mapping.placeholder}.")
        # 多词实体若仍残留在 masked_question 中，说明掩码不完整
        if len(mapping.original_text.split()) > 1 and mapping.original_text in masked_question:
            raise ValueError(f"masked_question still contains unmasked multi-word entity {mapping.original_text!r}.")
        # 映射中的原实体如果原问题里不存在，则记录警告
        if mapping.original_text not in question:
            warnings.append(f"Mask mapping original_text was not found in original question: {mapping.original_text!r}.")


def _find_placeholder_span(text: str, placeholder: str) -> tuple[int, int] | None:
    """查找占位符在文本中的字符区间。"""
    match = re.search(rf"\b{re.escape(placeholder)}\b", text)
    if not match:
        return None
    return match.start(), match.end()

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
