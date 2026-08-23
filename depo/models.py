"""DEPO 预处理、HanLP 解析和 Step4/Step5 之间传递的类型化记录。"""


from dataclasses import dataclass, field


@dataclass
class QuestionRecord:
    #问题文本
    question: str
    #问题ID
    qid: str | None = None

@dataclass
class ExplicitEntity:
    text: str


@dataclass
class ExplicitEntityResult:
    entities: list[ExplicitEntity] = field(default_factory=list)
    normalized_question: str = ""


@dataclass
class MaskMapping:
    placeholder: str
    original_text: str


@dataclass
class HanLPSDPEdge:
    head_idx: int
    relation: str
    dep_idx: int


@dataclass
class HanLPSDPResult:
    tokens: list[str]
    edges: list[HanLPSDPEdge]
    syntax_heads: dict[str, int] = field(default_factory=dict)


@dataclass
class HanLPSDPPreprocessResult:
    explicit_entities: ExplicitEntityResult
    masked_question: str
    mask_mappings: list[MaskMapping]
