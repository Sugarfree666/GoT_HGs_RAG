"""DEPO 预处理、HanLP 解析和 Step4/Step5 之间传递的类型化记录。"""


from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QuestionRecord:
    question: str
    qid: str | None = None

@dataclass
class ExplicitEntity:
    text: str
    start_char: int
    end_char: int
    semantic_type_hint: str | None = None
    confidence: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExplicitEntityResult:
    entities: list[ExplicitEntity] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None
    normalized_question: str | None = None
    normalization_changed: bool = False
    normalization_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaskMapping:
    placeholder: str
    original_text: str
    kind_hint: str
    semantic_type_hint: str | None = None
    original_char_span: list[int] = field(default_factory=list)
    masked_char_span: list[int] = field(default_factory=list)
    token_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HanLPSDPEdge:
    formalism: str
    head_idx: int
    head: str
    relation: str
    dep_idx: int
    dep: str

    def display(self) -> str:
        head_label = "ROOT[0]" if self.head_idx == 0 else _token_label(self.head, self.head_idx)
        return f"{head_label} --{self.relation}--> {_token_label(self.dep, self.dep_idx)}"


@dataclass
class HanLPSDPResult:
    text: str
    tokens: list[str]
    available_keys: list[str]
    sdp_graphs: dict[str, Any]
    edges: list[HanLPSDPEdge]
    raw: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    mask_token_checks: dict[str, str] = field(default_factory=dict)
    syntax_heads: dict[str, int] = field(default_factory=dict)
    syntax_head_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HanLPSDPPreprocessResult:
    original_question: str
    explicit_entities: ExplicitEntityResult
    masked_question: str
    sdp_input_sentence: str
    mask_mappings: list[MaskMapping]
    raw_payload: dict[str, Any] | None = None
    normalized_question: str = ""
    normalization_changed: bool = False
    normalization_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_label(word: str, index: int) -> str:
    if index <= 0:
        return word
    return f"{word}[{index}]"
