from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QuestionRecord:
    question: str
    qid: str | None = None


@dataclass
class SemanticNormalizationResult:
    original_question: str
    normalized_question: str
    changed: bool = False
    added_type_variables: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeclarativeView:
    id: str
    sentence: str
    purpose: str = "relation_carrier"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationCarrierViewResult:
    masked_question: str
    declarative_views: list[DeclarativeView] = field(default_factory=list)
    operator_intent: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """Replacement produced by the new mask-only Step 1 pipeline.

    This intentionally inherits the legacy ``PlaceholderReplacement`` fields so
    existing CLI/debug code and tests can continue to read ``question``,
    ``masked_question``, ``mask_mapping``, and ``mapping``.
    """

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
class CoreNLPToken:
    index: int
    word: str
    character_offset_begin: int = -1
    character_offset_end: int = -1
    lemma: str | None = None
    pos: str | None = None
    ner: str | None = None


@dataclass
class DependencyEdge:
    source: str
    relation: str
    target: str
    source_index: int
    target_index: int

    def display(self) -> str:
        return f"{_token_label(self.source, self.source_index)} --{self.relation}--> {_token_label(self.target, self.target_index)}"


@dataclass
class DependencyParse:
    tokens: list[CoreNLPToken]
    edges: list[DependencyEdge]
    raw: dict[str, Any] | None = None


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimplifiedSDPDMEdge:
    head: str
    relation: str
    dep: str
    head_idx: int | None = None
    dep_idx: int | None = None
    source_relation: str | None = None
    source_formalism: str = "sdp/dm"
    derived: bool = False
    rule: str | None = None
    provenance: list[str] = field(default_factory=list)

    def display(self) -> str:
        return f"{self.head} --{self.relation}--> {self.dep}"


@dataclass
class SimplifiedSDPDMGraph:
    nodes: list[str]
    edges: list[SimplifiedSDPDMEdge]
    removed_edges: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpenIETriple:
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    subject_span: list[int] = field(default_factory=list)
    relation_span: list[int] = field(default_factory=list)
    object_span: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoreNLPViewAnnotation:
    view_id: str
    text: str
    tokens: list[CoreNLPToken] = field(default_factory=list)
    edges: list[DependencyEdge] = field(default_factory=list)
    openie_triples: list[OpenIETriple] = field(default_factory=list)
    constituency_parse: str | None = None
    phrase_spans: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    def to_dependency_parse(self) -> DependencyParse:
        return DependencyParse(tokens=list(self.tokens), edges=list(self.edges), raw=self.raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphNodeCandidate:
    node_id: str
    token_index: int
    graph_text: str
    placeholder: str | None = None
    restored_text: str = ""
    display_text: str = ""
    is_mask_placeholder: bool = False
    pos: str | None = None
    lemma: str | None = None
    kind_hint: str = "context"
    semantic_type_hint: str | None = None
    char_span: list[int] | None = None
    source_token_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.restored_text:
            self.restored_text = self.graph_text
        if not self.display_text:
            self.display_text = self.restored_text
        if not self.source_token_indices:
            self.source_token_indices = [self.token_index]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_llm_view(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "node_id": self.node_id,
            "text": self.display_text,
            "pos": self.pos,
            "kind_hint": self.kind_hint,
        }
        if self.semantic_type_hint:
            view["semantic_type_hint"] = self.semantic_type_hint
        return view


@dataclass
class RestoredGraphNodeCandidate(GraphNodeCandidate):
    """Candidate object after placeholder text has been restored for LLM display."""

    text: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.text:
            self.text = self.display_text

    def to_llm_view(self) -> dict[str, Any]:
        view = super().to_llm_view()
        view["text"] = self.text or self.display_text
        return view


@dataclass
class SelectedAnchor:
    node_id: str
    graph_text: str
    restored_text: str
    display_text: str
    anchor_kind: str
    source: str = "graph_node"
    token_index: int | None = None
    placeholder: str | None = None
    semantic_type_hint: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_llm_view(self) -> dict[str, Any]:
        view = {
            "node_id": self.node_id,
            "text": self.display_text,
            "anchor_kind": self.anchor_kind,
        }
        if self.semantic_type_hint:
            view["semantic_type_hint"] = self.semantic_type_hint
        if self.reason:
            view["reason"] = self.reason
        return view


@dataclass
class AnchorSelectionResult:
    selected_anchors: list[SelectedAnchor] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None

    def __iter__(self):
        return iter(self.selected_anchors)

    def __len__(self) -> int:
        return len(self.selected_anchors)

    def __getitem__(self, index: int) -> SelectedAnchor:
        return self.selected_anchors[index]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnchorConnectedSubgraph:
    selected_anchor_node_ids: list[str] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    shortest_paths: list[dict[str, Any]] = field(default_factory=list)
    graph: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_anchor_node_ids": self.selected_anchor_node_ids,
            "nodes": self.nodes,
            "edges": self.edges,
            "shortest_paths": self.shortest_paths,
        }


@dataclass
class RestoredAnchorConnectedSubgraph:
    selected_anchor_node_ids: list[str] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    shortest_paths: list[dict[str, Any]] = field(default_factory=list)
    display_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateNode:
    """High-recall candidate node proposed after dependency parsing.

    Candidate nodes are not final AST nodes. They are a recall-oriented pool
    used by dependency graph projection utilities.
    """

    id: str
    text: str
    kind: str
    token_ids: list[int] = field(default_factory=list)
    graph_node_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Requirement:
    id: str
    root: str
    target: str
    description: str | None = None
    context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePath:
    path_id: str
    nodes: list[str]
    node_ids: list[str]
    candidate_for: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityStartNode:
    entity_id: str
    text: str
    graph_node_ids: list[str]
    token_ids: list[int] = field(default_factory=list)
    kind_hint: str = "entity"
    semantic_type_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityOriginPath:
    path_id: str
    entity_id: str
    entity_text: str
    nodes: list[str]
    node_ids: list[str]
    length: int
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AtomicEvidence:
    id: str
    type: str = ""
    source: str = "surface"
    view_id: str | None = None
    text: str = ""
    span: list[int] = field(default_factory=list)
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    head: str | None = None
    dependent: str | None = None
    dependency_relation: str | None = None
    aligned_entities: list[str] = field(default_factory=list)
    semantic_hint: str | None = None
    operator_hint: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Deprecated compatibility fields accepted from older evidence payloads.
    kind: str = ""
    anchor: str | None = None
    cue: str | None = None
    candidates: list[str] = field(default_factory=list)
    left: str | None = None
    right: str | None = None
    source_path_id: str | None = None
    source_path_set_id: str | None = None

    def __post_init__(self) -> None:
        if not self.type and self.kind:
            self.type = self.kind
        if not self.kind and self.type:
            self.kind = self.type
        if not self.type:
            self.type = "unknown"
            self.kind = "unknown"
        if not self.kind:
            self.kind = self.type

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind or self.type
        data["type"] = self.type or self.kind
        return data

    @property
    def path_set_id(self) -> str:
        return self.source_path_set_id or str(self.metadata.get("path_set_id") or "")

    @property
    def entity_id(self) -> str:
        return str(self.metadata.get("entity_id") or "")

    @property
    def entity_text(self) -> str:
        return str(self.metadata.get("entity_text") or "")

    @property
    def node_texts(self) -> list[str]:
        value = self.metadata.get("node_texts")
        return [str(item) for item in value] if isinstance(value, list) else []

    @property
    def node_ids(self) -> list[str]:
        value = self.metadata.get("node_ids")
        return [str(item) for item in value] if isinstance(value, list) else []


@dataclass
class SemanticReasoningNode:
    node_id: str
    label: str
    kind: str
    semantic_type: str | None = None
    source_path_id: str | None = None
    source_node_texts: list[str] = field(default_factory=list)
    source_node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticReasoningEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    answer_type: str | None = None
    is_one_hop: bool = True
    support: list[dict[str, Any]] = field(default_factory=list)
    atomic_question_template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticReasoningPath:
    branch_id: str
    entity_id: str
    source_path_id: str
    nodes: list[SemanticReasoningNode] = field(default_factory=list)
    edges: list[SemanticReasoningEdge] = field(default_factory=list)
    terminal_node_id: str | None = None
    score: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticReasoningPathResult:
    paths: list[SemanticReasoningPath] = field(default_factory=list)
    operator_intent: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticASTNode:
    id: str
    label: str
    kind: str
    semantic_type: str | None = None
    source: str = "derived"
    source_graph_nodes: list[str] = field(default_factory=list)
    source_token_indices: list[int] = field(default_factory=list)
    grounding_text: str = ""
    cue_text: str = ""
    branch_of: str | None = None
    expected_value_slot: str | None = None
    relation_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticASTEdge:
    source: str
    target: str
    edge_type: str = "attribute"
    relation_hint: str = ""
    support_path: list[str] = field(default_factory=list)
    support_dependency_relations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticASTPrimaryOperator:
    operator: str = "NONE"
    inputs: list[str] = field(default_factory=list)
    output: str = "answer"
    cue_text: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticASTResult:
    status: str
    primary_operator: SemanticASTPrimaryOperator = field(default_factory=SemanticASTPrimaryOperator)
    nodes: list[SemanticASTNode] = field(default_factory=list)
    edges: list[SemanticASTEdge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None
    coreference_links: list[dict[str, str]] = field(default_factory=list)
    canonical_node_map: dict[str, str] = field(default_factory=dict)
    validation_warnings: list[str] = field(default_factory=list)
    detected_cue_frame: dict[str, Any] = field(default_factory=dict)
    operator_inputs_before_validation: list[str] = field(default_factory=list)
    retry_count: int = 0
    fallback_repair_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_by_id(self) -> dict[str, SemanticASTNode]:
        return {node.id: node for node in self.nodes}


@dataclass
class ExecutionPlanStep:
    step_id: str
    step_type: str
    source_node: str | None = None
    target_node: str | None = None
    known: str = ""
    known_node_label: str = ""
    ask: str = ""
    relation_hint: str = ""
    answer_variable: str | None = None
    operator: str | None = None
    inputs: list[str] = field(default_factory=list)
    semantic_inputs: list[str] = field(default_factory=list)
    output: str = "answer"
    cue_text: str = ""
    ast_edge: dict[str, Any] | None = None
    operator_branches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    steps: list[ExecutionPlanStep] = field(default_factory=list)
    node_bindings: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_label(word: str, index: int) -> str:
    if index <= 0:
        return word
    return f"{word}[{index}]"


@dataclass
class AnchorEdge:
    source: str
    target: str
    weight: float
    token_path: list[Any]
    path_words: list[str]
    relations: list[str]

    def display(self) -> str:
        return f"{self.source} ---- {self.target}"


@dataclass
class AnchorGraph:
    graph: Any
    edges: list[AnchorEdge]
    anchor_positions: dict[str, list[int]]
    folded_graph: Any | None = None
    weighted_graph: Any | None = None
    anchor_subgraph: Any | None = None


@dataclass
class OperatorSelection:
    operator: str
    attach_to: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class ASTResult:
    graph: Any
    operators: list[OperatorSelection]
    label_by_placeholder: dict[str, str]

    def display_label(self, node: str) -> str:
        return self.label_by_placeholder.get(node, node)


@dataclass
class AtomicSubquestion:
    index: int
    question: str
    answer_variable: str | None = None
    source_node: str | None = None
    target_node: str | None = None
    operator: str | None = None
    type: str = "edge"
    source: str = "llm"
    ast_edge: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AtomicQuestionNode:
    id: str
    question: str
    type: str
    inputs: list[str] = field(default_factory=list)
    output: str = ""
    depends_on: list[str] = field(default_factory=list)
    source_node: str | None = None
    target_node: str | None = None
    ast_edge: dict[str, Any] | None = None
    operator: str | None = None
    candidate_bindings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node_id": self.id,
            "question": self.question,
            "dependencies": list(self.depends_on),
        }
        metadata: dict[str, Any] = {
            key: value
            for key, value in self.metadata.items()
            if key not in {"operator", "candidates", "warning"}
        }
        if self.operator:
            metadata["operator"] = self.operator
        candidates = _external_candidates(self.candidate_bindings)
        if candidates:
            metadata["candidates"] = candidates
        warning = str(self.metadata.get("warning", "") or "").strip()
        if warning:
            metadata["warning"] = warning
        if metadata:
            payload["metadata"] = metadata
        return payload


@dataclass
class AtomicQuestionEdge:
    source: str
    target: str
    variable: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AtomicQuestionDAG:
    nodes: list[AtomicQuestionNode] = field(default_factory=list)
    edges: list[AtomicQuestionEdge] = field(default_factory=list)
    variable_to_question: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "nodes": [node.to_dict() for node in self.nodes],
        }
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload

    def to_subquestions(self) -> list[AtomicSubquestion]:
        subquestions: list[AtomicSubquestion] = []
        for index, node in enumerate(self.nodes, start=1):
            subquestions.append(
                AtomicSubquestion(
                    index=index,
                    question=node.question,
                    answer_variable=None if node.output == "FINAL" else node.output,
                    source_node=node.source_node,
                    target_node=node.target_node,
                    operator=node.operator,
                    type="operator_step" if node.operator else "edge",
                    source=node.source,
                    ast_edge=node.ast_edge,
                )
            )
        return subquestions


def _external_candidates(candidate_bindings: list[dict[str, Any]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for item in candidate_bindings:
        label = str(item.get("label") or item.get("candidate") or "").strip()
        source_node_id = str(item.get("source_node_id") or "").strip()
        if label and source_node_id:
            candidates.append({"label": label, "source_node_id": source_node_id})
    return candidates
