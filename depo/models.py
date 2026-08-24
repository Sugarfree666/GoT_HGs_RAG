"""DEPO 阶段之间传递的最小数据对象。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreprocessedQuestion:
    """实体掩码后的 HanLP 输入，以及占位符恢复映射。"""

    entities: list[str]
    masked_question: str
    mask_mapping: dict[str, str]


@dataclass(frozen=True)
class HanLPSDPEdge:
    """一条 HanLP PAS 边：head --relation--> dependent。"""

    head_idx: int
    relation: str
    dep_idx: int


@dataclass
class HanLPSDPResult:
    """Step3 所需的分词、PAS 边和句法头节点。"""

    tokens: list[str]
    edges: list[HanLPSDPEdge]
    syntax_heads: dict[str, int] = field(default_factory=dict)
