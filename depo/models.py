from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QuestionRecord:
    question: str
    qid: str | None = None


@dataclass
class ExtractedNode:
    placeholder: str
    text: str
    kind: str
    semantic_type: str
    start: int | None = None
    end: int | None = None
    occurrence: int | None = None

    @property
    def is_entity(self) -> bool:
        return self.kind == "entity"

    @property
    def is_type_variable(self) -> bool:
        return self.kind == "type_variable"


@dataclass
class ExtractionResult:
    entities: list[ExtractedNode] = field(default_factory=list)
    type_variables: list[ExtractedNode] = field(default_factory=list)

    @property
    def nodes(self) -> list[ExtractedNode]:
        return [*self.entities, *self.type_variables]

    @property
    def placeholder_to_node(self) -> dict[str, ExtractedNode]:
        return {node.placeholder: node for node in self.nodes}


@dataclass
class PlaceholderReplacement:
    question: str
    mapping: dict[str, str]
    original_question: str | None = None
    replacements: list[dict[str, Any]] = field(default_factory=list)
    mask_mapping: dict[str, dict[str, Any]] = field(default_factory=dict)
    mask_mappings: list["MaskMapping"] = field(default_factory=list)
    preserved_type_variables: list[dict[str, Any]] = field(default_factory=list)
    anchor_extraction: ExtractionResult | None = None

    @property
    def masked_question(self) -> str:
        return self.question

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["masked_question"] = self.masked_question
        return data


class MaskReplacement(PlaceholderReplacement):
    def __init__(
        self,
        question: str | None = None,
        mapping: dict[str, str] | None = None,
        original_question: str | None = None,
        masked_question: str | None = None,
        replacements: list[dict[str, Any]] | None = None,
        mask_mapping: dict[str, dict[str, Any]] | None = None,
        mask_mappings: list["MaskMapping"] | None = None,
        preserved_type_variables: list[dict[str, Any]] | None = None,
        anchor_extraction: ExtractionResult | None = None,
    ) -> None:
        resolved_question = question if question is not None else masked_question
        if resolved_question is None:
            raise TypeError("MaskReplacement requires question or masked_question.")
        super().__init__(
            question=resolved_question,
            mapping=mapping or {},
            original_question=original_question,
            replacements=replacements or [],
            mask_mapping=mask_mapping or {},
            mask_mappings=mask_mappings or [],
            preserved_type_variables=preserved_type_variables or [],
            anchor_extraction=anchor_extraction,
        )


@dataclass
class MaskSpan:
    text: str
    start_char: int
    end_char: int
    kind_hint: str = "entity"
    semantic_type_hint: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None
    normalized_question: str | None = None
    normalization_changed: bool = False
    normalization_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaskSpanResult:
    mask_spans: list[MaskSpan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None

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
    warnings: list[str] = field(default_factory=list)
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
