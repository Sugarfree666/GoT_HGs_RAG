from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from models import HanLPSDPEdge, HanLPSDPResult


SOURCE_WEIGHTS = {
    "sdp/dm": 1.00,
    "sdp/psd": 0.90,
    "sdp/pas": 0.85,
}

CLASS_WEIGHTS = {
    "CORE_ARG": 1.00,
    "RESTRICT": 0.85,
    "IDENTITY": 0.85,
    "COORD": 0.70,
    "BRIDGE": 0.35,
    "MODIFIER": 0.25,
    "UNKNOWN": 0.45,
}

DERIVED_PENALTIES = {
    "bridge_contraction": 0.15,
    "possessive_marker_contraction": 0.15,
    "restriction_closure": 0.25,
    "descriptor_lifting": 0.35,
    "candidate_expansion": 0.10,
    "candidate_slot_substitution": 0.10,
    "candidate_bare_wh_substitution": 0.10,
    "function_backbone_contraction": 0.20,
}

ENTITY_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d[\d,]*(?:\.\d+)?|\d{1,4}(?:[-/]\d{1,2}){1,2})%?$")

DETERMINERS = {"a", "an", "the"}
WH_WORDS = {"what", "which", "who", "whom", "whose", "where", "when"}
RELATIVE_PRONOUNS = {"that"}
LIGHT_VERBS = {
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
}
PREPOSITIONS = {
    "of",
    "at",
    "in",
    "on",
    "for",
    "by",
    "to",
    "from",
    "with",
    "about",
    "as",
    "into",
    "onto",
    "over",
    "under",
    "after",
    "before",
    "during",
    "through",
}
SCOPE_WORDS = {"and", "or", "among", "between", "than"}
FUNCTION_WORDS = DETERMINERS | WH_WORDS | RELATIVE_PRONOUNS | LIGHT_VERBS | PREPOSITIONS | SCOPE_WORDS
POSSESSIVE_MARKER_TOKENS = {"'", "’", "'s", "’s", "s"}
POSSESSIVE_OWNER_RELATIONS = {"poss_arg2"}
POSSESSIVE_POSSESSED_RELATIONS = {"poss_arg1", "adj_arg1", "noun_arg1", "modifier"}
ORDER_CUES = {"first", "earliest", "latest", "last", "older", "oldest", "younger", "youngest"}
APPROX_CUES = {"approximately", "about", "around", "roughly"}

ANSWER_ANCHOR_SOURCE_ORDER = {
    "typed_wh_slot": 0,
    "root_projection": 1,
    "modifier_projection": 2,
    "comparative_focus": 3,
    "bare_wh_predicate_root": 4,
    "explicit_entity": 5,
    "clause_predicate": 6,
}


@dataclass
class TokenReasoningNode:
    id: str
    text: str
    index: int
    kind: str
    is_anchor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenReasoningEdge:
    source: str
    target: str
    source_text: str
    target_text: str
    support: float = 0.0
    derived: bool = False
    rule: str = ""
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenReasoningPath:
    path_id: str
    nodes: list[str]
    node_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenReasoningStructureResult:
    nodes: list[TokenReasoningNode]
    edges: list[TokenReasoningEdge]
    paths: list[TokenReasoningPath]
    path_type: str
    anchor_path_results: list[dict[str, Any]] = field(default_factory=list)
    answer_anchor: str | None = None
    answer_anchor_id: str | None = None
    entity_anchors: list[str] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    candidate_sets: list[list[str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    debug_payload: dict[str, Any] = field(default_factory=dict)
    debug_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "paths": [path.to_dict() for path in self.paths],
            "path_type": self.path_type,
            "anchor_path_results": list(self.anchor_path_results),
            "answer_anchor": self.answer_anchor,
            "answer_anchor_id": self.answer_anchor_id,
            "entity_anchors": list(self.entity_anchors),
            "constraints": list(self.constraints),
            "candidate_sets": [list(candidate_set) for candidate_set in self.candidate_sets],
            "warnings": list(self.warnings),
            "debug_payload": self.debug_payload,
            "debug_file": self.debug_file,
        }


@dataclass
class _WorkingState:
    nodes: dict[str, TokenReasoningNode]
    raw_edges: dict[tuple[str, str], TokenReasoningEdge]
    edges: dict[tuple[str, str], TokenReasoningEdge]
    normalized_edges: list[dict[str, Any]]
    virtual_edges: list[dict[str, Any]]
    warnings: list[str]


@dataclass
class _AnswerAnchorCandidate:
    node_id: str
    text: str
    source_types: list[str]
    score: float
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_debug(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "source_types": list(self.source_types),
            "score": self.score,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class _QueryFocus:
    answer_anchor_id: str | None
    query_root_id: str | None
    slot_id: str | None
    terminal_id: str | None
    required_ids: tuple[str, ...]
    mode: str

    def to_dict(self, nodes: dict[str, TokenReasoningNode]) -> dict[str, Any]:
        return {
            "answer_anchor_id": self.answer_anchor_id,
            "answer_anchor": nodes[self.answer_anchor_id].text if self.answer_anchor_id in nodes else None,
            "query_root_id": self.query_root_id,
            "query_root": nodes[self.query_root_id].text if self.query_root_id in nodes else None,
            "slot_id": self.slot_id,
            "slot": nodes[self.slot_id].text if self.slot_id in nodes else None,
            "terminal_id": self.terminal_id,
            "terminal": nodes[self.terminal_id].text if self.terminal_id in nodes else None,
            "required_ids": list(self.required_ids),
            "required": [nodes[node_id].text for node_id in self.required_ids if node_id in nodes],
            "mode": self.mode,
        }


@dataclass(frozen=True)
class _CandidatePath:
    source_entity_id: str | None
    node_ids: tuple[str, ...]
    search_pass: str
    rank: tuple[Any, ...]
    rank_components: dict[str, Any]
    selected: bool = False
    rejected_reason: str = ""

    def with_selection(self, *, selected: bool, rejected_reason: str = "") -> "_CandidatePath":
        return _CandidatePath(
            source_entity_id=self.source_entity_id,
            node_ids=self.node_ids,
            search_pass=self.search_pass,
            rank=self.rank,
            rank_components=dict(self.rank_components),
            selected=selected,
            rejected_reason=rejected_reason,
        )

    def to_debug(self, nodes: dict[str, TokenReasoningNode]) -> dict[str, Any]:
        payload = {
            "source_entity_id": self.source_entity_id,
            "source_entity": nodes[self.source_entity_id].text if self.source_entity_id in nodes else None,
            "node_ids": list(self.node_ids),
            "nodes": [nodes[node_id].text for node_id in self.node_ids if node_id in nodes],
            "search_pass": self.search_pass,
            "rank_components": dict(self.rank_components),
            "selected": self.selected,
        }
        if self.rejected_reason:
            payload["rejected_reason"] = self.rejected_reason
        return payload


def compile_token_reasoning_structure(
    hanlp_sdp_result: HanLPSDPResult,
    explicit_entities: list[str],
    *,
    masked_question: str | None = None,
    question_id: str | None = None,
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> TokenReasoningStructureResult:
    """Compile three HanLP SDP graph views into a query-focused token structure.

    The compiler is intentionally symbolic and deterministic: it consumes DM,
    PAS, and PSD edge evidence, adds generic virtual edges by graph operations,
    selects one main entity-to-focus path or a controlled parallel path cover,
    and emits only the graph induced by those selected paths. It does not call
    an LLM and it does not introduce semantic relation labels.
    """

    state = build_evidence_graph(hanlp_sdp_result)
    explicit_entity_ids = _resolve_explicit_entity_ids(state.nodes, explicit_entities)
    answer_anchor_candidates = collect_answer_anchor_candidates(
        state.nodes,
        state.raw_edges,
        state.warnings,
        explicit_entity_ids=explicit_entity_ids,
    )

    add_possessive_marker_contraction_edges(state)
    add_bridge_contraction_edges(state)
    add_restriction_closure_edges(state)

    per_anchor_results: list[dict[str, Any]] = []
    paths: list[TokenReasoningPath] = []
    result_edge_map: dict[tuple[str, str], TokenReasoningEdge] = {}
    aggregated_constraints: list[dict[str, Any]] = []
    aggregated_candidate_sets: list[list[str]] = []

    for anchor_index, anchor in enumerate(answer_anchor_candidates, start=1):
        anchor_state = _copy_working_state(state)
        _mark_anchors(anchor_state.nodes, explicit_entity_ids, anchor.node_id)

        constraints = detect_constraints(anchor_state.nodes, anchor_state.raw_edges, anchor.node_id)
        query_focus = _build_query_focus(anchor_state.nodes, anchor_state.raw_edges, anchor.node_id, constraints)
        direct_candidate_sets = detect_candidate_sets(anchor_state.nodes, anchor_state.raw_edges, explicit_entity_ids)
        parallel_entity_sets = _detect_parallel_entity_sets(
            anchor_state,
            explicit_entity_ids,
            direct_candidate_sets,
            query_focus,
        )

        add_descriptor_lifting_edges(anchor_state, explicit_entity_ids, anchor.node_id)

        anchor_paths, path_type, candidate_path_records, selection_mode = _select_query_focused_paths(
            state=anchor_state,
            explicit_entity_ids=explicit_entity_ids,
            query_focus=query_focus,
            constraints=constraints,
            direct_candidate_sets=direct_candidate_sets,
            parallel_entity_sets=parallel_entity_sets,
        )

        final_node_ids, final_pairs = _graph_from_selected_paths(anchor_paths)
        active_entity_ids = _active_entity_ids(anchor_state.nodes, anchor_paths)
        final_edges = _final_edges(
            anchor_state.nodes,
            anchor_state.edges,
            final_pairs,
            anchor_paths,
            active_entity_ids,
            anchor.node_id,
        )
        for edge in final_edges:
            result_edge_map.setdefault(_edge_key(edge.source, edge.target), edge)

        for path_index, path in enumerate(anchor_paths, start=1):
            paths.append(
                TokenReasoningPath(
                    path_id=f"A{anchor_index}.P{path_index}",
                    nodes=list(path.nodes),
                    node_ids=list(path.node_ids),
                )
            )

        candidate_sets = _candidate_sets_for_result(anchor_state.nodes, direct_candidate_sets, parallel_entity_sets)
        aggregated_constraints = _merge_constraint_debug_lists(aggregated_constraints, constraints)
        aggregated_candidate_sets = _merge_candidate_set_lists(aggregated_candidate_sets, candidate_sets)

        per_anchor_results.append(
            {
                "anchor_id": anchor.node_id,
                "anchor_text": anchor.text,
                "source_types": list(anchor.source_types),
                "score": anchor.score,
                "evidence": list(anchor.evidence),
                "query_focus": query_focus.to_dict(anchor_state.nodes),
                "constraints": constraints,
                "path_type": path_type,
                "paths": [path.to_dict() for path in anchor_paths],
                "selection_mode": selection_mode,
                "candidate_paths": [record.to_debug(anchor_state.nodes) for record in candidate_path_records],
                "candidate_sets": candidate_sets,
                "parallel_entity_sets": list(parallel_entity_sets),
                "virtual_edges": list(anchor_state.virtual_edges),
            }
        )

    result_nodes = _copy_node_map(state.nodes)
    for anchor in answer_anchor_candidates:
        _mark_anchors(result_nodes, explicit_entity_ids, anchor.node_id)

    final_node_ids, final_pairs = _graph_from_selected_paths(paths)
    result_state = _copy_working_state(state)
    result_state.nodes = result_nodes
    result_state.edges = {**result_state.edges, **result_edge_map}
    backbone_before = _graph_snapshot(final_node_ids, final_pairs, result_nodes, result_state.edges)
    backbone_after = backbone_before
    final_nodes = _final_nodes(result_nodes, final_node_ids)
    active_entity_ids = _active_entity_ids(result_nodes, paths)
    final_edges = _final_edges(result_nodes, result_state.edges, final_pairs, paths, active_entity_ids, None)

    candidate_sets = aggregated_candidate_sets
    entity_anchors = [result_nodes[node_id].text for node_id in active_entity_ids if node_id in result_nodes]
    debug_payload = _build_debug_payload(
        question_id=question_id,
        masked_question=masked_question or hanlp_sdp_result.text,
        hanlp_sdp_result=hanlp_sdp_result,
        explicit_entities=explicit_entities,
        state=result_state,
        answer_anchor_id=None,
        entity_ids=explicit_entity_ids,
        constraints=aggregated_constraints,
        candidate_sets=candidate_sets,
        terminals=[],
        backbone_before=backbone_before,
        backbone_after=backbone_after,
        final_nodes=final_nodes,
        final_edges=final_edges,
        paths=paths,
        query_focus=None,
        entity_candidates=explicit_entity_ids,
        active_entity_ids=active_entity_ids,
        parallel_entity_sets=[],
        candidate_paths=[],
        selected_paths=paths,
        selection_mode="multi_anchor_candidates",
    )
    debug_payload["answer_anchor_candidates"] = [candidate.to_debug() for candidate in answer_anchor_candidates]
    debug_payload["per_anchor_results"] = list(per_anchor_results)
    debug_file = None
    if debug:
        debug_file = write_debug_json(debug_payload, question_id=question_id, debug_dir=debug_dir)

    return TokenReasoningStructureResult(
        nodes=final_nodes,
        edges=final_edges,
        paths=paths,
        path_type="multi_anchor_candidates",
        anchor_path_results=per_anchor_results,
        answer_anchor=None,
        answer_anchor_id=None,
        entity_anchors=entity_anchors,
        constraints=aggregated_constraints,
        candidate_sets=candidate_sets,
        warnings=list(state.warnings),
        debug_payload=debug_payload,
        debug_file=debug_file,
    )


def classify_label(relation: str) -> str:
    normalized = _normalize_relation(relation)
    compact = normalized.replace("-", "_").replace(".", "_")
    if any(marker in compact for marker in ("coord", "conj_member", "disj_member", "_and_c", "_or_c")):
        return "COORD"
    if any(marker in compact for marker in ("prep_arg", "bv", "det", "aux", "root", "punct", "case", "cop")):
        return "BRIDGE"
    if any(marker in compact for marker in ("rstr", "descr", "relative_arg")):
        return "RESTRICT"
    if any(marker in compact for marker in ("compound", "flat", "app")) or compact in {"id"} or compact.endswith("_id"):
        return "IDENTITY"
    if any(marker in compact for marker in ("adj_arg", "noun_arg", "nummod", "numeric", "amod", "modifier")):
        return "MODIFIER"
    if any(
        marker in compact
        for marker in (
            "arg",
            "act_arg",
            "pat_arg",
            "eff_arg",
            "auth",
            "compl",
            "loc",
            "twhen",
            "ext",
            "part",
        )
    ):
        return "CORE_ARG"
    return "UNKNOWN"


def classify_node(text: str, index: int) -> str:
    stripped = str(text or "").strip()
    lower = stripped.lower()
    if index <= 0 or lower == "root":
        return "function"
    if ENTITY_RE.fullmatch(stripped):
        return "entity"
    if not stripped or _is_punctuation(stripped):
        return "function"
    if NUMERIC_RE.fullmatch(stripped) or lower in ORDER_CUES or lower in APPROX_CUES:
        return "constraint"
    if lower in FUNCTION_WORDS:
        return "function"
    return "content"


def build_evidence_graph(hanlp_sdp_result: HanLPSDPResult) -> _WorkingState:
    nodes = _build_token_nodes(hanlp_sdp_result)
    raw_edges: dict[tuple[str, str], TokenReasoningEdge] = {}
    normalized_edges: list[dict[str, Any]] = []
    warnings: list[str] = []

    for raw_edge in hanlp_sdp_result.edges:
        _ensure_edge_nodes(nodes, raw_edge)
        source_id = str(raw_edge.head_idx)
        target_id = str(raw_edge.dep_idx)
        label_class = classify_label(raw_edge.relation)
        source_weight = SOURCE_WEIGHTS.get(raw_edge.formalism, 0.75)
        class_weight = CLASS_WEIGHTS[label_class]
        support = source_weight * class_weight
        provenance = {
            "formalism": raw_edge.formalism,
            "head_idx": raw_edge.head_idx,
            "head": raw_edge.head,
            "relation": raw_edge.relation,
            "dep_idx": raw_edge.dep_idx,
            "dep": raw_edge.dep,
            "direction": "head_to_dep",
            "normalized_relation": _normalize_relation(raw_edge.relation),
            "label_class": label_class,
            "source_weight": source_weight,
            "class_weight": class_weight,
            "support": support,
        }
        normalized_edges.append(provenance)
        _merge_edge(
            raw_edges,
            nodes,
            source_id,
            target_id,
            support=support,
            derived=False,
            rule="raw_evidence",
            provenance=[provenance],
        )

    _mark_possessive_marker_nodes(nodes, raw_edges)
    edges = {key: _copy_edge(edge) for key, edge in raw_edges.items()}
    return _WorkingState(
        nodes=nodes,
        raw_edges=raw_edges,
        edges=edges,
        normalized_edges=normalized_edges,
        virtual_edges=[],
        warnings=warnings,
    )


def detect_answer_anchor(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    warnings: list[str],
) -> str | None:
    candidates = collect_answer_anchor_candidates(nodes, raw_edges, explicit_entity_ids=[])
    if candidates:
        warnings.append(f"answer anchor selected by {','.join(candidates[0].source_types)}")
        return candidates[0].node_id
    warnings.append("answer anchor fallback failed")
    return None


def collect_answer_anchor_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    warnings: list[str] | None = None,
    *,
    explicit_entity_ids: list[str] | None = None,
) -> list[_AnswerAnchorCandidate]:
    """Recall answer-anchor candidates without committing to a final global anchor."""

    explicit_entity_ids = explicit_entity_ids or []
    query_root = _find_query_root(nodes, raw_edges)
    candidates: list[_AnswerAnchorCandidate] = []

    candidates.extend(_find_typed_wh_slot_candidates(nodes, raw_edges))
    if query_root:
        candidates.extend(_collect_root_projection_candidates(nodes, raw_edges, query_root))
        candidates.extend(_find_modifier_projection_candidates(nodes, raw_edges, query_root))
    candidates.extend(_find_comparative_focus_candidates(nodes, raw_edges, query_root))
    candidates.extend(_find_bare_wh_predicate_root_candidates(nodes, raw_edges, query_root))
    candidates.extend(_collect_explicit_entity_anchor_candidates(nodes, raw_edges, explicit_entity_ids))
    candidates.extend(_collect_clause_predicate_anchor_candidates(nodes, raw_edges, explicit_entity_ids, query_root))

    merged = _merge_answer_anchor_candidates(nodes, candidates)
    if warnings is not None:
        if merged:
            rendered = ", ".join(
                f"{candidate.text}[{candidate.node_id}]/{'+'.join(candidate.source_types)}"
                for candidate in merged
            )
            warnings.append(f"answer anchor candidates collected: {rendered}")
        else:
            warnings.append("answer anchor candidates collection found no candidates")
    return merged


def _find_typed_wh_slot_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> list[_AnswerAnchorCandidate]:
    wh_ids = [node.id for node in _sorted_nodes(nodes.values()) if node.text.lower() in WH_WORDS]
    adjacency = _adjacency(raw_edges)
    candidates: list[_AnswerAnchorCandidate] = []

    for wh_id in wh_ids:
        wh_text = nodes[wh_id].text.lower()
        neighbors = _neighbor_edges(wh_id, adjacency, raw_edges)
        if wh_text not in {"what", "which"}:
            continue
        for neighbor_id, edge in neighbors:
            neighbor = nodes[neighbor_id]
            if neighbor.kind != "content":
                continue
            classes = _edge_label_classes(edge)
            relations = _edge_relations(edge)
            class_match = bool(classes & {"BRIDGE", "RESTRICT", "IDENTITY", "MODIFIER"})
            relation_match = any(
                fragment in relation
                for relation in relations
                for fragment in {"bv", "det", "rstr", "adj", "noun", "compound", "id", "flat", "app"}
            )
            if not class_match and not relation_match:
                continue
            candidates.append(
                _answer_anchor_candidate(
                    nodes,
                    neighbor_id,
                    "typed_wh_slot",
                    {
                        "rule": "typed_wh_neighbor",
                        "wh_id": wh_id,
                        "wh": nodes[wh_id].text,
                        "edge": edge.to_dict(),
                        "support": edge.support,
                    },
                    edge.support,
                )
            )

    adjacent = _surface_adjacent_typed_wh_slot(nodes)
    if adjacent:
        previous = next((node for node in nodes.values() if node.index == nodes[adjacent].index - 1), None)
        candidates.append(
            _answer_anchor_candidate(
                nodes,
                adjacent,
                "typed_wh_slot",
                {
                    "rule": "surface_typed_wh_adjacency",
                    "wh_id": previous.id if previous else None,
                    "wh": previous.text if previous else None,
                    "slot_id": adjacent,
                    "slot": nodes[adjacent].text,
                    "support": 0.75,
                },
                0.75,
            )
        )
    return _merge_answer_anchor_candidates(nodes, candidates)


def _find_typed_wh_slot(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> str | None:
    wh_ids = [node.id for node in _sorted_nodes(nodes.values()) if node.text.lower() in WH_WORDS]
    adjacency = _adjacency(raw_edges)

    for wh_id in wh_ids:
        wh_text = nodes[wh_id].text.lower()
        neighbors = _neighbor_edges(wh_id, adjacency, raw_edges)
        if wh_text in {"what", "which"}:
            nominal = _best_neighbor(
                nodes,
                neighbors,
                allowed_classes={"BRIDGE", "RESTRICT", "IDENTITY", "MODIFIER"},
                allowed_rel_fragments={"bv", "det", "rstr", "adj", "noun", "compound", "id", "flat", "app"},
                kinds={"content"},
            )
            if nominal:
                return nominal
    adjacent = _surface_adjacent_typed_wh_slot(nodes)
    if adjacent:
        return adjacent
    return None


def _surface_adjacent_typed_wh_slot(nodes: dict[str, TokenReasoningNode]) -> str | None:
    by_index = {node.index: node for node in nodes.values()}
    for node in _sorted_nodes(nodes.values()):
        if node.text.lower() not in {"what", "which"}:
            continue
        next_node = by_index.get(node.index + 1)
        if not next_node or next_node.kind != "content":
            continue
        if _is_punctuation(next_node.text) or next_node.text.lower() in WH_WORDS:
            continue
        return next_node.id
    return None


def _find_wh_fallback_anchor(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> str | None:
    wh_ids = [node.id for node in _sorted_nodes(nodes.values()) if node.text.lower() in WH_WORDS]
    adjacency = _adjacency(raw_edges)

    for wh_id in wh_ids:
        wh_text = nodes[wh_id].text.lower()
        neighbors = _neighbor_edges(wh_id, adjacency, raw_edges)
        if wh_text == "what" and _has_predicate_object_edge(nodes, neighbors):
            return wh_id
        if wh_text == "which":
            continue
        elif wh_text == "where":
            located = _best_neighbor(
                nodes,
                neighbors,
                allowed_classes={"CORE_ARG", "MODIFIER", "UNKNOWN"},
                allowed_rel_fragments={"loc", "where", "adj"},
                kinds={"content"},
            )
            if located:
                return located
        elif wh_text == "when":
            temporal = _best_neighbor(
                nodes,
                neighbors,
                allowed_classes={"CORE_ARG", "MODIFIER", "UNKNOWN"},
                allowed_rel_fragments={"twhen", "time", "when", "temporal"},
                kinds={"content"},
            )
            if temporal:
                return temporal

    nearest = _nearest_content_to_wh(nodes, raw_edges, wh_ids)
    if nearest:
        # The caller records root-level failures; keep this fallback quiet except
        # for the existing high-level strategy warning.
        return nearest
    return None


def _find_query_root(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> str | None:
    candidates: list[tuple[float, int, tuple[int, str], str]] = []
    for edge in raw_edges.values():
        for item in _raw_provenance(edge):
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            relation = str(item.get("normalized_relation") or item.get("relation") or "")
            if head_idx != 0 or dep_idx is None or str(dep_idx) not in nodes:
                continue
            if _normalized_relation_key(relation) != "root":
                continue
            node = nodes[str(dep_idx)]
            if node.kind == "function":
                continue
            support = _coerce_float_value(item.get("support"), edge.support)
            candidates.append((-support, node.index, _node_sort_key(node), node.id))
    return sorted(candidates)[0][3] if candidates else None


def _collect_root_projection_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str,
) -> list[_AnswerAnchorCandidate]:
    if query_root_id not in nodes:
        return []
    root_idx = nodes[query_root_id].index
    candidates: dict[str, dict[str, Any]] = {}
    for edge in raw_edges.values():
        for item in _raw_provenance(edge):
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if head_idx is None or dep_idx is None:
                continue
            if head_idx != root_idx:
                continue
            candidate_id = str(dep_idx)
            if not _is_projection_candidate_node(nodes, candidate_id, query_root_id):
                continue
            polarity = _projection_relation_polarity(str(item.get("normalized_relation") or item.get("relation") or ""))
            if polarity is None:
                continue
            support = _coerce_float_value(item.get("support"), edge.support)
            formalism = str(item.get("formalism") or "")
            candidate = candidates.setdefault(
                candidate_id,
                {"positive": 0.0, "negative": 0.0, "formalisms": set(), "positive_count": 0, "evidence": []},
            )
            if polarity == "forward":
                candidate["positive"] += support
                candidate["positive_count"] += 1
                if formalism:
                    candidate["formalisms"].add(formalism)
                candidate["evidence"].append(
                    {
                        "rule": "root_projection",
                        "query_root_id": query_root_id,
                        "query_root": nodes[query_root_id].text,
                        "candidate_id": candidate_id,
                        "candidate": nodes[candidate_id].text,
                        "provenance": dict(item),
                        "support": support,
                    }
                )
            elif polarity == "subject":
                candidate["negative"] += support

    scored: list[tuple[float, int, int, str, list[dict[str, Any]]]] = []
    for candidate_id, data in candidates.items():
        positive = float(data["positive"])
        if positive <= 0.0 or int(data["positive_count"]) <= 0:
            continue
        total = positive - float(data["negative"])
        if total <= 0.0:
            continue
        formalisms = data["formalisms"]
        node = nodes[candidate_id]
        scored.append((-total, -len(formalisms), node.index, candidate_id, list(data["evidence"])))
    return [
        _answer_anchor_candidate(nodes, candidate_id, "root_projection", {"evidence": evidence, "support": -score}, -score)
        for score, _formalism_count, _node_index, candidate_id, evidence in sorted(scored)
    ]


def _find_modifier_projection_candidate(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str,
) -> str | None:
    candidates = _find_modifier_projection_candidates(nodes, raw_edges, query_root_id)
    return candidates[0].node_id if candidates else None


def _find_modifier_projection_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str,
) -> list[_AnswerAnchorCandidate]:
    if query_root_id not in nodes:
        return []
    adjacency = _adjacency(raw_edges)
    queue: list[tuple[str, list[str]]] = [(query_root_id, [query_root_id])]
    best_by_candidate: dict[str, dict[str, Any]] = {}
    while queue:
        node_id, path = queue.pop(0)
        if len(path) > 3:
            continue
        if len(path) > 1 and _is_projection_candidate_node(nodes, node_id, query_root_id):
            score, formalisms = _score_modifier_projection_path(raw_edges, path)
            if score > 0.0:
                existing = best_by_candidate.get(node_id)
                if existing is None or (score, len(formalisms)) > (float(existing["score"]), len(existing["formalisms"])):
                    best_by_candidate[node_id] = {"score": score, "formalisms": formalisms, "path": list(path)}
        if len(path) == 3:
            continue
        for neighbor_id in sorted(adjacency.get(node_id, {}), key=lambda item: _node_sort_key(nodes[item])):
            if neighbor_id in path or neighbor_id == "0":
                continue
            queue.append((neighbor_id, [*path, neighbor_id]))

    scored: list[tuple[float, int, int, str]] = []
    for candidate_id, data in best_by_candidate.items():
        node = nodes[candidate_id]
        scored.append((-float(data["score"]), -len(data["formalisms"]), node.index, candidate_id))
    result: list[_AnswerAnchorCandidate] = []
    for score, _formalism_count, _node_index, candidate_id in sorted(scored):
        data = best_by_candidate[candidate_id]
        path = list(data.get("path") or [])
        result.append(
            _answer_anchor_candidate(
                nodes,
                candidate_id,
                "modifier_projection",
                {
                    "rule": "modifier_projection_path",
                    "query_root_id": query_root_id,
                    "query_root": nodes[query_root_id].text,
                    "path_ids": path,
                    "path": [nodes[node_id].text for node_id in path if node_id in nodes],
                    "formalisms": sorted(data.get("formalisms") or []),
                    "support": -score,
                },
                -score,
            )
        )
    return result


def _score_modifier_projection_path(
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    path: list[str],
) -> tuple[float, set[str]]:
    score = 0.0
    formalisms: set[str] = set()
    for index in range(len(path) - 1):
        edge = raw_edges.get(_edge_key(path[index], path[index + 1]))
        if edge is None:
            continue
        for item in _raw_provenance(edge):
            relation = str(item.get("normalized_relation") or item.get("relation") or "")
            label_class = str(item.get("label_class") or classify_label(relation))
            polarity = _projection_relation_polarity(relation)
            support = _coerce_float_value(item.get("support"), edge.support)
            if label_class in {"RESTRICT", "MODIFIER"}:
                score += support
            elif polarity == "forward":
                score += support * 0.75
            elif polarity == "subject":
                score -= support * 0.50
            else:
                continue
            formalism = str(item.get("formalism") or "")
            if formalism:
                formalisms.add(formalism)
    return score, formalisms


def _find_comparative_focus_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str | None,
) -> list[_AnswerAnchorCandidate]:
    candidates: list[_AnswerAnchorCandidate] = []
    for node in _sorted_nodes(nodes.values()):
        if node.text.lower() not in ORDER_CUES:
            continue
        incident = _anchor_incident_evidence(raw_edges, node.id, rule="comparative_focus")
        support = sum(float(item.get("support") or 0.0) for item in incident)
        if node.id == query_root_id:
            support += 1.0
        candidates.append(
            _answer_anchor_candidate(
                nodes,
                node.id,
                "comparative_focus",
                {
                    "rule": "comparative_focus",
                    "query_root_id": query_root_id,
                    "incident_evidence": incident,
                    "support": support,
                },
                support,
            )
        )
    return candidates


def _find_bare_wh_predicate_root_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str | None,
) -> list[_AnswerAnchorCandidate]:
    candidates: list[_AnswerAnchorCandidate] = []
    typed_wh = _find_typed_wh_slot(nodes, raw_edges)
    if query_root_id and not typed_wh:
        direct_wh_ids = _bare_wh_direct_argument_ids(nodes, raw_edges, query_root_id)
        suffix_id = _bare_wh_temporal_or_modifier_suffix_id(nodes, raw_edges, query_root_id)
        if direct_wh_ids and suffix_id:
            candidates.append(
                _answer_anchor_candidate(
                    nodes,
                    query_root_id,
                    "bare_wh_predicate_root",
                    {
                        "rule": "bare_wh_query_root",
                        "direct_wh_ids": direct_wh_ids,
                        "direct_wh": [nodes[node_id].text for node_id in direct_wh_ids if node_id in nodes],
                        "suffix_id": suffix_id,
                        "suffix": nodes[suffix_id].text if suffix_id in nodes else None,
                        "support": 1.0,
                    },
                    1.0,
                )
            )

    inferred = _infer_bare_wh_query_predicate(nodes, raw_edges)
    if inferred is not None:
        predicate_id, wh_id = inferred
        suffix_id = _bare_wh_temporal_or_modifier_suffix_id(nodes, raw_edges, predicate_id)
        candidates.append(
            _answer_anchor_candidate(
                nodes,
                predicate_id,
                "bare_wh_predicate_root",
                {
                    "rule": "inferred_bare_wh_predicate_root",
                    "wh_id": wh_id,
                    "wh": nodes[wh_id].text if wh_id in nodes else None,
                    "suffix_id": suffix_id,
                    "suffix": nodes[suffix_id].text if suffix_id in nodes else None,
                    "support": 0.75,
                },
                0.75,
            )
        )
    return _merge_answer_anchor_candidates(nodes, candidates)


def _collect_explicit_entity_anchor_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    explicit_entity_ids: list[str],
) -> list[_AnswerAnchorCandidate]:
    candidates: list[_AnswerAnchorCandidate] = []
    for entity_id in _sort_node_ids([node_id for node_id in explicit_entity_ids if node_id in nodes], nodes):
        incident = _anchor_incident_evidence(raw_edges, entity_id, rule="explicit_entity")
        support = sum(float(item.get("support") or 0.0) for item in incident)
        candidates.append(
            _answer_anchor_candidate(
                nodes,
                entity_id,
                "explicit_entity",
                {
                    "rule": "explicit_entity",
                    "incident_evidence": incident,
                    "support": support,
                },
                support,
            )
        )
    return candidates


def _collect_clause_predicate_anchor_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    explicit_entity_ids: list[str],
    query_root_id: str | None,
) -> list[_AnswerAnchorCandidate]:
    explicit = set(explicit_entity_ids)
    scored: dict[str, dict[str, Any]] = {}

    if query_root_id and _is_clause_predicate_anchor_node(nodes, query_root_id):
        entry = scored.setdefault(query_root_id, {"support": 0.0, "evidence": []})
        entry["support"] += 1.0
        entry["evidence"].append(
            {
                "rule": "clause_query_root",
                "query_root_id": query_root_id,
                "query_root": nodes[query_root_id].text,
                "support": 1.0,
            }
        )

    for key in _sorted_edge_keys(raw_edges, nodes):
        edge = raw_edges[key]
        source_id, target_id = key
        classes = _edge_label_classes(edge)
        if not (classes & {"CORE_ARG", "RESTRICT", "IDENTITY", "MODIFIER", "UNKNOWN"}):
            continue
        for node_id, other_id in ((source_id, target_id), (target_id, source_id)):
            if not _is_clause_predicate_anchor_node(nodes, node_id):
                continue
            support = edge.support
            if other_id in explicit:
                support += 0.50
            if node_id == query_root_id:
                support += 0.50
            entry = scored.setdefault(node_id, {"support": 0.0, "evidence": []})
            entry["support"] += support
            entry["evidence"].append(
                {
                    "rule": "clause_predicate_edge",
                    "other_id": other_id,
                    "other": nodes[other_id].text if other_id in nodes else None,
                    "edge": edge.to_dict(),
                    "support": support,
                }
            )

    return [
        _answer_anchor_candidate(
            nodes,
            node_id,
            "clause_predicate",
            {"rule": "clause_predicate", "evidence": list(data["evidence"]), "support": float(data["support"])},
            float(data["support"]),
        )
        for node_id, data in sorted(
            scored.items(),
            key=lambda item: (-float(item[1]["support"]), _node_sort_key(nodes[item[0]])),
        )
    ]


def _is_clause_predicate_anchor_node(nodes: dict[str, TokenReasoningNode], node_id: str) -> bool:
    if node_id not in nodes or node_id == "0":
        return False
    node = nodes[node_id]
    if node.text.lower() in WH_WORDS:
        return False
    return node.kind in {"content", "answer", "constraint"}


def _answer_anchor_candidate(
    nodes: dict[str, TokenReasoningNode],
    node_id: str,
    source_type: str,
    evidence: dict[str, Any],
    score: float,
) -> _AnswerAnchorCandidate:
    return _AnswerAnchorCandidate(
        node_id=node_id,
        text=nodes[node_id].text if node_id in nodes else "",
        source_types=[source_type],
        score=score,
        evidence=[evidence],
    )


def _merge_answer_anchor_candidates(
    nodes: dict[str, TokenReasoningNode],
    candidates: list[_AnswerAnchorCandidate],
) -> list[_AnswerAnchorCandidate]:
    by_id: dict[str, _AnswerAnchorCandidate] = {}
    for candidate in candidates:
        if candidate.node_id not in nodes:
            continue
        existing = by_id.get(candidate.node_id)
        if existing is None:
            by_id[candidate.node_id] = _AnswerAnchorCandidate(
                node_id=candidate.node_id,
                text=nodes[candidate.node_id].text,
                source_types=[],
                score=0.0,
                evidence=[],
            )
            existing = by_id[candidate.node_id]
        for source_type in candidate.source_types:
            if source_type not in existing.source_types:
                existing.source_types.append(source_type)
        _append_unique_evidence(existing.evidence, candidate.evidence)

    for candidate in by_id.values():
        candidate.source_types.sort(key=lambda source: ANSWER_ANCHOR_SOURCE_ORDER.get(source, 99))
        candidate.score = _score_answer_anchor_candidate(nodes[candidate.node_id], candidate.source_types, candidate.evidence)

    return sorted(
        by_id.values(),
        key=lambda candidate: (
            -candidate.score,
            min((ANSWER_ANCHOR_SOURCE_ORDER.get(source, 99) for source in candidate.source_types), default=99),
            _node_sort_key(nodes[candidate.node_id]),
        ),
    )


def _score_answer_anchor_candidate(
    node: TokenReasoningNode,
    source_types: list[str],
    evidence: list[dict[str, Any]],
) -> float:
    kind_bonus = {
        "content": 4.0,
        "entity": 3.0,
        "answer": 4.0,
        "constraint": 3.0,
        "function": 0.0,
    }.get(node.kind, 1.0)
    source_weights = {
        "typed_wh_slot": 16.0,
        "root_projection": 14.0,
        "modifier_projection": 12.0,
        "comparative_focus": 10.0,
        "bare_wh_predicate_root": 10.0,
        "clause_predicate": 8.0,
        "explicit_entity": 2.0,
    }
    source_bonus = sum(source_weights.get(source_type, 1.0) for source_type in set(source_types))
    support_bonus = min(2.0, sum(_anchor_evidence_support(item) for item in evidence))
    return round(source_bonus + kind_bonus + support_bonus, 6)


def _anchor_evidence_support(item: Any) -> float:
    if isinstance(item, dict):
        total = _coerce_float_value(item.get("support"), 0.0)
        for value in item.values():
            if isinstance(value, (dict, list)):
                total += _anchor_evidence_support(value)
        return total
    if isinstance(item, list):
        return sum(_anchor_evidence_support(value) for value in item)
    return 0.0


def _append_unique_evidence(target: list[dict[str, Any]], items: list[dict[str, Any]]) -> None:
    seen = {json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) for item in target}
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        target.append(item)


def _anchor_incident_evidence(
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    node_id: str,
    *,
    rule: str,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for edge in raw_edges.values():
        if node_id not in (edge.source, edge.target):
            continue
        evidence.append({"rule": rule, "edge": edge.to_dict(), "support": edge.support})
    return evidence


def detect_candidate_sets(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    explicit_entity_ids: list[str],
) -> list[list[str]]:
    if len(explicit_entity_ids) < 2:
        return []

    explicit = set(explicit_entity_ids)
    parent = {node_id: node_id for node_id in explicit_entity_ids}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if _node_sort_key(nodes[left_root]) <= _node_sort_key(nodes[right_root]):
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    connector_to_entities: dict[str, set[str]] = {}
    for key, edge in raw_edges.items():
        source, target = key
        classes = _edge_label_classes(edge)
        source_node = nodes[source]
        target_node = nodes[target]
        if "COORD" in classes and source in explicit and target in explicit:
            union(source, target)
        if "COORD" in classes or _is_scope_node(source_node) or _is_scope_node(target_node):
            if _is_scope_node(source_node) and target in explicit:
                connector_to_entities.setdefault(source, set()).add(target)
            if _is_scope_node(target_node) and source in explicit:
                connector_to_entities.setdefault(target, set()).add(source)

    for connector_id, entity_ids in connector_to_entities.items():
        if len(entity_ids) >= 2:
            ordered = sorted(entity_ids, key=lambda node_id: _node_sort_key(nodes[node_id]))
            first = ordered[0]
            for other in ordered[1:]:
                union(first, other)
        else:
            del connector_id

    groups: dict[str, list[str]] = {}
    for entity_id in explicit_entity_ids:
        groups.setdefault(find(entity_id), []).append(entity_id)

    candidate_sets = []
    seen: set[tuple[str, ...]] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda node_id: _node_sort_key(nodes[node_id]))
        texts = [nodes[node_id].text for node_id in ordered]
        key = tuple(texts)
        if key not in seen:
            seen.add(key)
            candidate_sets.append(texts)
    return candidate_sets


def detect_constraints(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    answer_anchor_id: str | None,
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    adjacency = _adjacency(raw_edges)
    seen: set[tuple[str, str, str]] = set()
    for node in _sorted_nodes(nodes.values()):
        lower = node.text.lower()
        if node.kind != "constraint":
            continue
        neighbors = _neighbor_edges(node.id, adjacency, raw_edges)
        if lower in ORDER_CUES:
            target_id = _best_constraint_target(nodes, neighbors, answer_anchor_id)
            key = ("order", node.text, target_id or "")
            if key not in seen:
                seen.add(key)
                constraints.append(
                    {
                        "type": "order",
                        "text": node.text,
                        "node_id": node.id,
                        "target": nodes[target_id].text if target_id else None,
                        "target_id": target_id,
                    }
                )
        elif NUMERIC_RE.fullmatch(node.text):
            target_id = _best_constraint_target(nodes, neighbors, answer_anchor_id)
            key = ("numeric", node.text, target_id or "")
            if key not in seen:
                seen.add(key)
                constraints.append(
                    {
                        "type": "numeric",
                        "text": node.text,
                        "node_id": node.id,
                        "target": nodes[target_id].text if target_id else None,
                        "target_id": target_id,
                    }
                )
        elif lower in APPROX_CUES:
            target_id = _best_constraint_target(nodes, neighbors, answer_anchor_id)
            key = ("approximation", node.text, target_id or "")
            if key not in seen:
                seen.add(key)
                constraints.append(
                    {
                        "type": "approximation",
                        "text": node.text,
                        "node_id": node.id,
                        "target": nodes[target_id].text if target_id else None,
                        "target_id": target_id,
                    }
                )
    return constraints


def add_bridge_contraction_edges(state: _WorkingState) -> None:
    adjacency = _adjacency(state.edges)
    for bridge in _sorted_nodes(state.nodes.values()):
        if not _is_bridge_node(bridge):
            continue
        if _is_contextual_possessive_marker(bridge.id, state.nodes, state.raw_edges):
            continue
        neighbors = [
            neighbor_id
            for neighbor_id in adjacency.get(bridge.id, {})
            if _is_high_salience_node(state.nodes[neighbor_id], include_order_constraints=False)
        ]
        neighbors = sorted(set(neighbors), key=lambda node_id: _node_sort_key(state.nodes[node_id]))
        for left_index, left_id in enumerate(neighbors):
            for right_id in neighbors[left_index + 1 :]:
                left_edge = state.edges[_edge_key(left_id, bridge.id)]
                right_edge = state.edges[_edge_key(bridge.id, right_id)]
                support = max(0.05, min(left_edge.support, right_edge.support) * 0.80)
                provenance = {
                    "rule": "bridge_contraction",
                    "bridge": bridge.text,
                    "bridge_id": bridge.id,
                    "collapsed_path": [
                        state.nodes[left_id].text,
                        bridge.text,
                        state.nodes[right_id].text,
                    ],
                    "source_edges": [left_edge.to_dict(), right_edge.to_dict()],
                    "support": support,
                }
                virtual = _merge_edge(
                    state.edges,
                    state.nodes,
                    left_id,
                    right_id,
                    support=support,
                    derived=True,
                    rule="bridge_contraction",
                    provenance=[provenance],
                )
                state.virtual_edges.append(virtual.to_dict())


def add_possessive_marker_contraction_edges(state: _WorkingState) -> None:
    """Collapse parser-introduced possessive clitics without treating all "s" as function words."""

    for marker in _sorted_nodes(state.nodes.values()):
        if not _is_contextual_possessive_marker(marker.id, state.nodes, state.raw_edges):
            continue
        owners, possessed = _possessive_marker_role_edges(marker.id, state)
        for owner_id, owner_edge in owners:
            if not _is_high_salience_node(state.nodes[owner_id], include_order_constraints=False):
                continue
            for possessed_id, possessed_edge in possessed:
                if owner_id == possessed_id:
                    continue
                if not _is_high_salience_node(state.nodes[possessed_id], include_order_constraints=False):
                    continue
                support = max(0.05, min(owner_edge.support, possessed_edge.support) * 0.90)
                provenance = {
                    "rule": "possessive_marker_contraction",
                    "marker": marker.text,
                    "marker_id": marker.id,
                    "collapsed_path": [
                        state.nodes[owner_id].text,
                        marker.text,
                        state.nodes[possessed_id].text,
                    ],
                    "source_edges": [owner_edge.to_dict(), possessed_edge.to_dict()],
                    "support": support,
                }
                virtual = _merge_edge(
                    state.edges,
                    state.nodes,
                    owner_id,
                    possessed_id,
                    support=support,
                    derived=True,
                    rule="possessive_marker_contraction",
                    provenance=[provenance],
                )
                state.virtual_edges.append(virtual.to_dict())


def add_restriction_closure_edges(state: _WorkingState) -> None:
    adjacency = _adjacency(state.edges)
    restrict_pairs = [
        (source, target, state.edges[key])
        for key in _sorted_edge_keys(state.edges, state.nodes)
        for source, target in [key]
        if "RESTRICT" in _edge_label_classes(state.edges[key])
    ]

    for head_id, predicate_id, restrict_edge in restrict_pairs:
        head_candidates = [head_id, predicate_id]
        for candidate_head_id in head_candidates:
            predicate = predicate_id if candidate_head_id == head_id else head_id
            if not _is_high_salience_node(state.nodes[candidate_head_id], include_order_constraints=False):
                continue
            for neighbor_id, neighbor_key in adjacency.get(predicate, {}).items():
                if neighbor_id == candidate_head_id:
                    continue
                neighbor_node = state.nodes[neighbor_id]
                if not _is_high_salience_node(neighbor_node, include_order_constraints=False):
                    continue
                neighbor_edge = state.edges[neighbor_key]
                if not (_edge_label_classes(neighbor_edge) & {"CORE_ARG", "IDENTITY", "RESTRICT", "UNKNOWN"}):
                    continue
                support = max(0.05, min(restrict_edge.support, neighbor_edge.support) * 0.70)
                provenance = {
                    "rule": "restriction_closure",
                    "collapsed_path": [
                        state.nodes[candidate_head_id].text,
                        state.nodes[predicate].text,
                        neighbor_node.text,
                    ],
                    "source_edges": [restrict_edge.to_dict(), neighbor_edge.to_dict()],
                    "support": support,
                }
                virtual = _merge_edge(
                    state.edges,
                    state.nodes,
                    candidate_head_id,
                    neighbor_id,
                    support=support,
                    derived=True,
                    rule="restriction_closure",
                    provenance=[provenance],
                )
                state.virtual_edges.append(virtual.to_dict())

    predicate_to_heads: dict[str, list[str]] = {}
    for left_id, right_id, _edge in restrict_pairs:
        if _is_high_salience_node(state.nodes[left_id], include_order_constraints=False):
            predicate_to_heads.setdefault(right_id, []).append(left_id)
        if _is_high_salience_node(state.nodes[right_id], include_order_constraints=False):
            predicate_to_heads.setdefault(left_id, []).append(right_id)

    for predicate_id, head_ids in predicate_to_heads.items():
        ordered_heads = sorted(set(head_ids), key=lambda node_id: _node_sort_key(state.nodes[node_id]))
        for left_index, left_id in enumerate(ordered_heads):
            for right_id in ordered_heads[left_index + 1 :]:
                left_edge = state.edges[_edge_key(left_id, predicate_id)]
                right_edge = state.edges[_edge_key(right_id, predicate_id)]
                support = max(0.05, min(left_edge.support, right_edge.support) * 0.65)
                provenance = {
                    "rule": "restriction_closure",
                    "shared_predicate": state.nodes[predicate_id].text,
                    "collapsed_path": [
                        state.nodes[left_id].text,
                        state.nodes[predicate_id].text,
                        state.nodes[right_id].text,
                    ],
                    "source_edges": [left_edge.to_dict(), right_edge.to_dict()],
                    "support": support,
                }
                virtual = _merge_edge(
                    state.edges,
                    state.nodes,
                    left_id,
                    right_id,
                    support=support,
                    derived=True,
                    rule="restriction_closure",
                    provenance=[provenance],
                )
                state.virtual_edges.append(virtual.to_dict())


def add_descriptor_lifting_edges(
    state: _WorkingState,
    entity_ids: list[str],
    answer_anchor_id: str | None,
) -> None:
    graph = _weighted_adjacency(state.nodes, state.edges, include_scope=False)
    anchor_targets = set(entity_ids)
    if answer_anchor_id:
        anchor_targets.add(answer_anchor_id)

    for entity_id in entity_ids:
        if entity_id not in state.nodes:
            continue
        candidates = _bounded_paths_from_entity(state.nodes, state.edges, graph, entity_id, max_depth=4)
        for target_id, path in candidates:
            if len(path) < 4:
                continue
            if target_id in anchor_targets or state.nodes[target_id].kind not in {"content", "answer", "constraint"}:
                continue
            if not _path_has_descriptor_evidence(path, state.edges, state.nodes):
                continue
            if not _target_reaches_anchor(target_id, entity_id, anchor_targets, graph):
                continue
            source_edges = [
                state.edges[_edge_key(path[index], path[index + 1])].to_dict()
                for index in range(len(path) - 1)
                if _edge_key(path[index], path[index + 1]) in state.edges
            ]
            if not source_edges:
                continue
            support = max(0.05, min(float(edge["support"]) for edge in source_edges) * 0.60)
            provenance = {
                "rule": "descriptor_lifting",
                "collapsed_path": [state.nodes[node_id].text for node_id in path],
                "source_edges": source_edges,
                "support": support,
            }
            virtual = _merge_edge(
                state.edges,
                state.nodes,
                entity_id,
                target_id,
                support=support,
                derived=True,
                rule="descriptor_lifting",
                provenance=[provenance],
            )
            state.virtual_edges.append(virtual.to_dict())


# Legacy terminal-cover utilities retained for offline comparison only. The
# formal Step4 path no longer calls these functions.
def extract_steiner_backbone(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    terminals: list[str],
) -> set[tuple[str, str]]:
    terminals = [terminal for terminal in _sort_node_ids(terminals, nodes) if terminal in nodes]
    if len(terminals) < 2:
        return set()
    graph = _weighted_adjacency(nodes, edges, include_scope=False)
    pair_paths: list[tuple[float, tuple[int, ...], str, str, list[str]]] = []
    for left_index, left_id in enumerate(terminals):
        for right_id in terminals[left_index + 1 :]:
            path, cost = _shortest_path(graph, nodes, left_id, right_id)
            if not path:
                continue
            pair_paths.append((cost, _path_index_tuple(path, nodes), left_id, right_id, path))

    parent = {terminal: terminal for terminal in terminals}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: str, right: str) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if _node_sort_key(nodes[left_root]) <= _node_sort_key(nodes[right_root]):
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root
        return True

    backbone_pairs: set[tuple[str, str]] = set()
    for _cost, _path_indices, left_id, right_id, path in sorted(pair_paths):
        if not union(left_id, right_id):
            continue
        for index in range(len(path) - 1):
            backbone_pairs.add(_edge_key(path[index], path[index + 1]))
    return backbone_pairs


def prune_backbone(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    backbone_pairs: set[tuple[str, str]],
    terminals: list[str],
    answer_anchor_id: str | None,
) -> set[tuple[str, str]]:
    retained = set(backbone_pairs)
    terminal_set = set(terminals)

    retained = _contract_function_backbone_nodes(nodes, edges, retained, terminal_set)

    changed = True
    while changed:
        changed = False
        degree = _degree_map(retained)
        for node_id, degree_value in list(degree.items()):
            if degree_value != 1 or node_id in terminal_set:
                continue
            node = nodes[node_id]
            lower = node.text.lower()
            if node.kind == "function" or (lower in WH_WORDS and node_id != answer_anchor_id):
                retained = {pair for pair in retained if node_id not in pair}
                changed = True
                break
            if node.kind == "constraint" and not NUMERIC_RE.fullmatch(node.text):
                retained = {pair for pair in retained if node_id not in pair}
                changed = True
                break
    return retained


def linearize_paths(
    nodes: dict[str, TokenReasoningNode],
    final_pairs: set[tuple[str, str]],
    entity_ids: list[str],
    answer_anchor_id: str | None,
    terminals: list[str],
) -> tuple[list[TokenReasoningPath], str]:
    if not final_pairs:
        fallback_ids = [node_id for node_id in _sort_node_ids(entity_ids, nodes) if node_id in nodes]
        if answer_anchor_id and answer_anchor_id in nodes and answer_anchor_id not in fallback_ids:
            fallback_ids.append(answer_anchor_id)
        paths = [
            TokenReasoningPath(f"P{index}", [nodes[node_id].text], [node_id])
            for index, node_id in enumerate(fallback_ids, start=1)
        ]
        return paths, "empty"

    adjacency = _plain_adjacency(final_pairs)
    if _is_simple_connected_path(adjacency):
        start_id = _path_start_node(nodes, adjacency, entity_ids, answer_anchor_id)
        target_id = _path_target_node(nodes, adjacency, start_id, terminals, answer_anchor_id)
        path_ids = _unweighted_path(adjacency, nodes, start_id, target_id)
        return [_path_from_ids("P1", nodes, path_ids)], "single_path"

    root_id = _walk_root(nodes, adjacency, entity_ids, answer_anchor_id)
    walk_ids = _dfs_walk(nodes, adjacency, root_id, parent_id=None)
    return [_path_from_ids("P1", nodes, walk_ids)], "graph_walk"


def _build_query_focus(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    answer_anchor_id: str | None,
    constraints: list[dict[str, Any]],
) -> _QueryFocus:
    typed_slot_id = _find_typed_wh_slot(nodes, raw_edges)
    query_root_id = _find_query_root(nodes, raw_edges)
    if query_root_id:
        bare_wh = _find_bare_wh_root_argument(nodes, raw_edges, query_root_id)
        if bare_wh:
            wh_id, projection_slot_id = bare_wh
            suffix_id = _bare_wh_temporal_or_modifier_suffix_id(nodes, raw_edges, query_root_id)
            if nodes[wh_id].text.lower() in {"what", "which"} or suffix_id:
                required = _unique_node_ids([projection_slot_id, query_root_id, wh_id], nodes)
                return _QueryFocus(
                    answer_anchor_id=answer_anchor_id,
                    query_root_id=query_root_id,
                    slot_id=projection_slot_id,
                    terminal_id=wh_id,
                    required_ids=tuple(required),
                    mode="bare_wh_root_argument",
                )

    required_ids: list[str] = []
    mode = "answer_anchor"
    terminal_id = answer_anchor_id
    slot_id = typed_slot_id
    if typed_slot_id:
        mode = "typed_wh_slot"
        order_terminal_id = _order_constraint_terminal_id(nodes, raw_edges, constraints)
        terminal_id = order_terminal_id or typed_slot_id
        required_ids.append(typed_slot_id)
        if order_terminal_id:
            required_ids.append(order_terminal_id)
        if query_root_id and query_root_id != typed_slot_id:
            required_ids.append(query_root_id)
    elif query_root_id and answer_anchor_id and query_root_id != answer_anchor_id:
        mode = "root_projection"
        required_ids.extend([query_root_id, answer_anchor_id])
    elif answer_anchor_id:
        required_ids.append(answer_anchor_id)
    elif query_root_id:
        mode = "root_fallback"
        terminal_id = query_root_id
        required_ids.append(query_root_id)

    required = _unique_node_ids(required_ids or ([terminal_id] if terminal_id else []), nodes)
    return _QueryFocus(
        answer_anchor_id=answer_anchor_id,
        query_root_id=query_root_id,
        slot_id=slot_id,
        terminal_id=terminal_id,
        required_ids=tuple(required),
        mode=mode,
    )


def _find_bare_wh_root_argument(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str,
) -> tuple[str, str] | None:
    if query_root_id not in nodes:
        return None
    root_idx = nodes[query_root_id].index
    wh_ids = {
        node.id
        for node in nodes.values()
        if node.text.lower() in {"what", "who", "whom", "where", "when"} and node.id != query_root_id
    }
    direct_wh_ids: set[str] = set()
    projection_candidates: dict[str, float] = {}
    for edge in raw_edges.values():
        for item in _raw_provenance(edge):
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if head_idx != root_idx or dep_idx is None:
                continue
            dep_id = str(dep_idx)
            relation = str(item.get("normalized_relation") or item.get("relation") or "")
            polarity = _projection_relation_polarity(relation)
            if dep_id in wh_ids:
                direct_wh_ids.add(dep_id)
                continue
            if (
                dep_id in nodes
                and nodes[dep_id].kind == "content"
                and polarity == "forward"
            ):
                projection_candidates[dep_id] = projection_candidates.get(dep_id, 0.0) + _coerce_float_value(
                    item.get("support"),
                    edge.support,
                )
    if not direct_wh_ids or not projection_candidates:
        return None
    wh_id = _sort_node_ids(direct_wh_ids, nodes)[0]
    projection_id = sorted(
        projection_candidates,
        key=lambda node_id: (-projection_candidates[node_id], _node_sort_key(nodes[node_id])),
    )[0]
    return wh_id, projection_id


def _bare_wh_direct_argument_ids(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str,
) -> list[str]:
    if query_root_id not in nodes or _find_typed_wh_slot(nodes, raw_edges):
        return []
    root_idx = nodes[query_root_id].index
    wh_ids = {
        node.id
        for node in nodes.values()
        if node.text.lower() in WH_WORDS and node.id != query_root_id
    }
    candidates: dict[str, dict[str, Any]] = {}
    for edge in raw_edges.values():
        for item in _raw_provenance(edge):
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if head_idx is None or dep_idx is None:
                continue
            wh_id: str | None = None
            if head_idx == root_idx and str(dep_idx) in wh_ids:
                wh_id = str(dep_idx)
            elif dep_idx == root_idx and str(head_idx) in wh_ids:
                wh_id = str(head_idx)
            if wh_id is None:
                continue
            relation = str(item.get("normalized_relation") or item.get("relation") or "")
            if not _is_direct_wh_core_argument_relation(relation):
                continue
            support = _coerce_float_value(item.get("support"), edge.support)
            formalism = str(item.get("formalism") or "")
            entry = candidates.setdefault(wh_id, {"support": 0.0, "formalisms": set()})
            entry["support"] += support
            if formalism:
                entry["formalisms"].add(formalism)
    return sorted(
        candidates,
        key=lambda node_id: (
            -float(candidates[node_id]["support"]),
            -len(candidates[node_id]["formalisms"]),
            _node_sort_key(nodes[node_id]),
        ),
    )


def _infer_bare_wh_query_predicate(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> tuple[str, str] | None:
    """Infer a bare-WH query predicate when HanLP omits an SDP root edge."""

    if _find_typed_wh_slot(nodes, raw_edges):
        return None
    wh_ids = {
        node.id
        for node in nodes.values()
        if node.text.lower() in WH_WORDS
    }
    if not wh_ids:
        return None

    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in raw_edges.values():
        for item in _raw_provenance(edge):
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if head_idx is None or dep_idx is None:
                continue
            relation = str(item.get("normalized_relation") or item.get("relation") or "")
            if not _is_direct_wh_core_argument_relation(relation):
                continue

            head_id = str(head_idx)
            dep_id = str(dep_idx)
            if head_id in wh_ids and _is_bare_wh_predicate_candidate(nodes, dep_id):
                wh_id = head_id
                predicate_id = dep_id
            elif dep_id in wh_ids and _is_bare_wh_predicate_candidate(nodes, head_id):
                wh_id = dep_id
                predicate_id = head_id
            else:
                continue

            suffix_id = _bare_wh_temporal_or_modifier_suffix_id(nodes, raw_edges, predicate_id)
            if not suffix_id:
                continue

            support = _coerce_float_value(item.get("support"), edge.support)
            formalism = str(item.get("formalism") or "")
            suffix_edge = raw_edges.get(_edge_key(predicate_id, suffix_id))
            suffix_support = suffix_edge.support if suffix_edge else 0.0
            key = (predicate_id, wh_id)
            entry = candidates.setdefault(
                key,
                {"support": 0.0, "formalisms": set(), "suffix_id": suffix_id, "suffix_support": 0.0},
            )
            entry["support"] += support
            entry["suffix_support"] = max(float(entry["suffix_support"]), suffix_support)
            if formalism:
                entry["formalisms"].add(formalism)

    if not candidates:
        return None
    predicate_id, wh_id = sorted(
        candidates,
        key=lambda key: (
            -float(candidates[key]["support"]),
            -float(candidates[key]["suffix_support"]),
            -len(candidates[key]["formalisms"]),
            _node_sort_key(nodes[key[0]]),
            _node_sort_key(nodes[key[1]]),
        ),
    )[0]
    return predicate_id, wh_id


def _is_bare_wh_predicate_candidate(nodes: dict[str, TokenReasoningNode], node_id: str) -> bool:
    return node_id in nodes and nodes[node_id].kind == "content" and nodes[node_id].text.lower() not in WH_WORDS


def _is_direct_wh_core_argument_relation(relation: str) -> bool:
    key = _normalized_relation_key(relation)
    return (
        key == "arg"
        or key.startswith("arg")
        or key.startswith("verb_arg")
        or key in {"pat_arg", "act_arg", "eff_arg"}
    )


def _bare_wh_temporal_or_modifier_suffix_id(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str,
) -> str | None:
    if query_root_id not in nodes:
        return None
    root_idx = nodes[query_root_id].index
    candidates: dict[str, dict[str, Any]] = {}
    for edge in raw_edges.values():
        for item in _raw_provenance(edge):
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if head_idx is None or dep_idx is None:
                continue
            other_idx: int | None = None
            if head_idx == root_idx:
                other_idx = dep_idx
            elif dep_idx == root_idx:
                other_idx = head_idx
            if other_idx is None:
                continue
            other_id = str(other_idx)
            if not _is_bare_wh_suffix_candidate_node(nodes, other_id, query_root_id):
                continue
            score = _bare_wh_suffix_relation_score(item, edge.support)
            if score <= 0.0:
                continue
            formalism = str(item.get("formalism") or "")
            entry = candidates.setdefault(other_id, {"score": 0.0, "formalisms": set()})
            entry["score"] += score
            if formalism:
                entry["formalisms"].add(formalism)
    return sorted(
        candidates,
        key=lambda node_id: (
            -float(candidates[node_id]["score"]),
            -len(candidates[node_id]["formalisms"]),
            _node_sort_key(nodes[node_id]),
        ),
    )[0] if candidates else None


def _is_bare_wh_suffix_candidate_node(
    nodes: dict[str, TokenReasoningNode],
    node_id: str,
    query_root_id: str,
) -> bool:
    if node_id not in nodes or node_id == query_root_id:
        return False
    node = nodes[node_id]
    if node.kind in {"entity", "function"}:
        return False
    if node.text.lower() in WH_WORDS:
        return False
    return node.kind in {"content", "constraint", "answer"}


def _bare_wh_suffix_relation_score(item: dict[str, Any], default_support: float) -> float:
    relation = str(item.get("normalized_relation") or item.get("relation") or "")
    key = _normalized_relation_key(relation)
    label_class = str(item.get("label_class") or classify_label(relation))
    support = _coerce_float_value(item.get("support"), default_support)
    if key in {"twhen", "tmp", "tloc", "loc"} or "time" in key or "temporal" in key:
        return support * 2.0
    if label_class == "MODIFIER" or key.startswith("adj_arg") or key.startswith("noun_arg"):
        return support * 1.5
    if label_class == "RESTRICT":
        return support
    return 0.0


def _select_query_focused_paths(
    *,
    state: _WorkingState,
    explicit_entity_ids: list[str],
    query_focus: _QueryFocus,
    constraints: list[dict[str, Any]],
    direct_candidate_sets: list[list[str]],
    parallel_entity_sets: list[dict[str, Any]],
) -> tuple[list[TokenReasoningPath], str, list[_CandidatePath], str]:
    candidate_records: list[_CandidatePath] = []

    parallel_paths = _extract_actual_parallel_path_cover(
        state,
        query_focus,
        parallel_entity_sets,
    )
    if parallel_paths:
        paths, selected_records, all_records = parallel_paths
        candidate_records.extend(all_records)
        return paths, "candidate_path_cover", candidate_records, "parallel_entity_paths"

    typed_paths = _extract_typed_slot_candidate_path_cover(
        state,
        query_focus,
        constraints,
        direct_candidate_sets,
    )
    if typed_paths:
        paths, selected_records = typed_paths
        candidate_records.extend(selected_records)
        return paths, "candidate_path_cover", candidate_records, "candidate_slot_substitution"

    bare_wh_paths = _extract_bare_wh_candidate_path_cover(
        state,
        query_focus,
        direct_candidate_sets,
    )
    if bare_wh_paths:
        paths, selected_records = bare_wh_paths
        candidate_records.extend(selected_records)
        return paths, "candidate_path_cover", candidate_records, "candidate_bare_wh_substitution"

    single_path, path_type, selected_record, all_records = _select_single_main_path(
        state,
        explicit_entity_ids,
        query_focus,
    )
    candidate_records.extend(all_records)
    if selected_record:
        candidate_records = [
            record.with_selection(selected=record == selected_record, rejected_reason="" if record == selected_record else "lower ranked")
            for record in candidate_records
        ]
    if single_path:
        return [single_path], path_type, candidate_records, path_type

    fallback_id = _fallback_single_node_id(state.nodes, explicit_entity_ids, query_focus)
    if fallback_id:
        path = _path_from_ids("P1", state.nodes, [fallback_id])
        return [path], "empty", candidate_records, "single_node_fallback"
    return [], "empty", candidate_records, "empty"


def _extract_typed_slot_candidate_path_cover(
    state: _WorkingState,
    query_focus: _QueryFocus,
    constraints: list[dict[str, Any]],
    direct_candidate_sets: list[list[str]],
) -> tuple[list[TokenReasoningPath], list[_CandidatePath]] | None:
    if query_focus.mode != "typed_wh_slot" or not query_focus.slot_id or not direct_candidate_sets:
        return None
    candidate_ids = _candidate_text_set_to_ids(state.nodes, direct_candidate_sets[0])
    if len(candidate_ids) < 2:
        return None

    focus_id = _schema_focus_id(state, query_focus, constraints)
    if not focus_id or focus_id == query_focus.slot_id:
        return None

    schema_path = _best_schema_path(state, query_focus.slot_id, focus_id)
    if len(schema_path) < 2:
        return None

    paths: list[TokenReasoningPath] = []
    selected_records: list[_CandidatePath] = []
    for path_index, candidate_id in enumerate(candidate_ids, start=1):
        branch = _dedupe_adjacent([candidate_id, *schema_path[1:]])
        if len(branch) != len(set(branch)):
            continue
        if len(branch) >= 2:
            _ensure_candidate_slot_edge(state, candidate_id, branch[1], query_focus.slot_id, direct_candidate_sets[0], schema_path)
        record = _rank_candidate_path(
            state.nodes,
            state.edges,
            branch,
            query_focus,
            candidate_id,
            "candidate_slot_substitution",
            required_ids=schema_path[1:],
        ).with_selection(selected=True)
        selected_records.append(record)
        paths.append(_path_from_ids(f"P{path_index}", state.nodes, branch))
    if len(paths) != len(candidate_ids):
        return None
    return paths, selected_records


def _extract_bare_wh_candidate_path_cover(
    state: _WorkingState,
    query_focus: _QueryFocus,
    direct_candidate_sets: list[list[str]],
) -> tuple[list[TokenReasoningPath], list[_CandidatePath]] | None:
    if not direct_candidate_sets or _find_typed_wh_slot(state.nodes, state.raw_edges):
        return None

    schema = _bare_wh_candidate_schema(state, query_focus)
    if schema is None:
        return None
    wh_id = schema["wh_id"]
    query_root_id = schema["query_root_id"]
    schema_path = list(schema["schema_path"])
    if len(schema_path) < 2 or schema_path[0] != wh_id or query_root_id not in schema_path:
        return None

    for candidate_set in direct_candidate_sets:
        candidate_ids = _candidate_text_set_to_ids(state.nodes, candidate_set)
        if len(candidate_ids) < 2:
            continue
        paths: list[TokenReasoningPath] = []
        selected_records: list[_CandidatePath] = []
        for path_index, candidate_id in enumerate(candidate_ids, start=1):
            branch = _dedupe_adjacent([candidate_id, *schema_path[1:]])
            if len(branch) != len(set(branch)) or len(branch) < 2:
                break
            _ensure_candidate_bare_wh_edge(
                state,
                candidate_id,
                branch[1],
                wh_id,
                query_root_id,
                candidate_set,
                schema_path,
            )
            record = _rank_candidate_path(
                state.nodes,
                state.edges,
                branch,
                query_focus,
                candidate_id,
                "candidate_bare_wh_substitution",
                required_ids=schema_path[1:],
            ).with_selection(selected=True)
            selected_records.append(record)
            paths.append(_path_from_ids(f"P{path_index}", state.nodes, branch))
        if len(paths) == len(candidate_ids):
            return paths, selected_records
    return None


def _extract_actual_parallel_path_cover(
    state: _WorkingState,
    query_focus: _QueryFocus,
    parallel_entity_sets: list[dict[str, Any]],
) -> tuple[list[TokenReasoningPath], list[_CandidatePath], list[_CandidatePath]] | None:
    all_records: list[_CandidatePath] = []
    for parallel_set in _sort_parallel_entity_sets(state.nodes, parallel_entity_sets):
        if parallel_set.get("kind") == "direct_entity_set" and query_focus.slot_id:
            continue
        entity_ids = [node_id for node_id in parallel_set.get("entity_ids", []) if node_id in state.nodes]
        if len(entity_ids) < 2:
            continue
        branch_heads = {
            str(entity_id): str(head_id)
            for entity_id, head_id in dict(parallel_set.get("branch_heads") or {}).items()
            if str(head_id) in state.nodes
        }
        selected: list[_CandidatePath] = []
        viable = True
        for entity_id in entity_ids:
            required_ids = list(query_focus.required_ids)
            if entity_id in branch_heads:
                required_ids.insert(0, branch_heads[entity_id])
            candidates = _enumerate_entity_focus_paths(
                state,
                entity_id,
                query_focus,
                forbidden_entity_ids=set(entity_ids) - {entity_id},
                required_ids=_unique_node_ids(required_ids, state.nodes),
            )
            all_records.extend(candidates)
            if not candidates:
                viable = False
                break
            if entity_id in branch_heads:
                candidates = [candidate for candidate in candidates if branch_heads[entity_id] in candidate.node_ids]
                if not candidates:
                    viable = False
                    break
            selected.append(candidates[0].with_selection(selected=True))
        if not viable or len(selected) != len(entity_ids):
            continue
        selected_paths = [
            _path_from_ids(f"P{index}", state.nodes, list(record.node_ids))
            for index, record in enumerate(selected, start=1)
        ]
        selected_keys = {record.node_ids for record in selected}
        marked_records = [
            record.with_selection(selected=record.node_ids in selected_keys, rejected_reason="" if record.node_ids in selected_keys else "lower ranked")
            for record in all_records
        ]
        return selected_paths, selected, marked_records
    return None


def _select_single_main_path(
    state: _WorkingState,
    explicit_entity_ids: list[str],
    query_focus: _QueryFocus,
) -> tuple[TokenReasoningPath | None, str, _CandidatePath | None, list[_CandidatePath]]:
    all_records: list[_CandidatePath] = []
    entity_best: list[_CandidatePath] = []
    for entity_id in explicit_entity_ids:
        candidates = _enumerate_entity_focus_paths(
            state,
            entity_id,
            query_focus,
            forbidden_entity_ids=set(explicit_entity_ids) - {entity_id},
            required_ids=list(query_focus.required_ids),
        )
        all_records.extend(candidates)
        if candidates:
            entity_best.append(candidates[0])

    if not entity_best:
        return None, "empty", None, all_records

    selected = sorted(entity_best, key=lambda record: record.rank)[0]
    path_type = "single_main_path"
    if selected.rank_components.get("missing_required_focus_count", 0) or selected.search_pass != "strong":
        path_type = "fallback_main_path"
    return _path_from_ids("P1", state.nodes, list(selected.node_ids)), path_type, selected, all_records


def _enumerate_entity_focus_paths(
    state: _WorkingState,
    entity_id: str,
    query_focus: _QueryFocus,
    *,
    forbidden_entity_ids: set[str],
    required_ids: list[str],
) -> list[_CandidatePath]:
    if entity_id not in state.nodes:
        return []
    target_id = query_focus.terminal_id
    if not target_id or target_id not in state.nodes:
        return []
    candidates: list[_CandidatePath] = []
    for search_pass, allow_weak in (("strong", False), ("weak", True)):
        raw_paths = _bounded_k_simple_paths(
            state.nodes,
            state.edges,
            source_id=entity_id,
            target_id=target_id,
            forbidden_nodes=forbidden_entity_ids,
            required_ids=set(required_ids),
            allow_weak=allow_weak,
            max_nodes=12,
            top_k=12,
        )
        for path in raw_paths:
            candidates.append(
                _rank_candidate_path(
                    state.nodes,
                    state.edges,
                    path,
                    query_focus,
                    entity_id,
                    search_pass,
                    required_ids=required_ids,
                )
            )
        if candidates:
            break
    candidates.sort(key=lambda record: record.rank)
    return candidates[:12]


def _bounded_k_simple_paths(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    *,
    source_id: str,
    target_id: str,
    forbidden_nodes: set[str],
    required_ids: set[str],
    allow_weak: bool,
    max_nodes: int,
    top_k: int,
) -> list[list[str]]:
    if source_id == target_id:
        return [[source_id]]
    adjacency = _adjacency(edges)
    queue: list[list[str]] = [[source_id]]
    results: list[list[str]] = []
    expansions = 0
    while queue and expansions < 5000:
        path = queue.pop(0)
        expansions += 1
        current = path[-1]
        if len(path) > max_nodes:
            continue
        if current == target_id:
            results.append(path)
            continue
        if len(path) == max_nodes:
            continue
        next_paths: list[list[str]] = []
        for neighbor_id, edge in _search_neighbor_edges(nodes, edges, adjacency, current):
            if neighbor_id in path or neighbor_id in forbidden_nodes:
                continue
            if not _edge_allowed_for_path(nodes, edge, neighbor_id, target_id, required_ids, allow_weak):
                continue
            next_paths.append([*path, neighbor_id])
        next_paths.sort(key=lambda item: (len(item), _path_index_tuple(item, nodes), item[-1]))
        queue.extend(next_paths)
    unique: dict[tuple[str, ...], list[str]] = {}
    for path in results:
        unique.setdefault(tuple(path), path)
    return sorted(unique.values(), key=lambda path: _path_search_sort_key(nodes, edges, path))[: max(top_k, 1)]


def _path_search_sort_key(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    path: list[str],
) -> tuple[float, int, int, tuple[int, ...]]:
    total_derived_penalty = 0.0
    descriptor_lifting_count = 0
    for left, right in zip(path, path[1:]):
        edge = edges.get(_edge_key(left, right))
        if edge is None:
            total_derived_penalty += 1_000_000.0
            continue
        total_derived_penalty += _edge_rule_penalty(edge)
        if "descriptor_lifting" in edge.rule:
            descriptor_lifting_count += 1
    return (total_derived_penalty, descriptor_lifting_count, len(path), _path_index_tuple(path, nodes))


def _search_neighbor_edges(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    adjacency: dict[str, dict[str, tuple[str, str]]],
    node_id: str,
) -> list[tuple[str, TokenReasoningEdge]]:
    neighbors = [(neighbor_id, edges[key]) for neighbor_id, key in adjacency.get(node_id, {}).items()]
    neighbors.sort(key=lambda item: (_edge_cost(item[1], nodes[node_id], nodes[item[0]]), _node_sort_key(nodes[item[0]]), item[0]))
    return neighbors


def _edge_allowed_for_path(
    nodes: dict[str, TokenReasoningNode],
    edge: TokenReasoningEdge,
    neighbor_id: str,
    target_id: str,
    required_ids: set[str],
    allow_weak: bool,
) -> bool:
    if "0" in (edge.source, edge.target):
        return False
    if _is_scope_node(nodes[edge.source]) or _is_scope_node(nodes[edge.target]):
        return False
    if "COORD" in _edge_label_classes_deep(edge):
        return False
    if (
        "candidate_expansion" in edge.rule
        or "candidate_slot_substitution" in edge.rule
        or "candidate_bare_wh_substitution" in edge.rule
    ):
        return False
    if not allow_weak and "descriptor_lifting" in edge.rule:
        return False
    neighbor = nodes[neighbor_id]
    if neighbor.kind == "function" and neighbor_id != target_id:
        return False
    if neighbor.kind == "constraint" and neighbor_id not in required_ids and neighbor_id != target_id:
        return False
    return True


def _rank_candidate_path(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    path_ids: list[str],
    query_focus: _QueryFocus,
    source_entity_id: str | None,
    search_pass: str,
    *,
    required_ids: Iterable[str],
) -> _CandidatePath:
    required = [node_id for node_id in _unique_node_ids(required_ids, nodes) if node_id in nodes]
    path_set = set(path_ids)
    missing_required_count = len([node_id for node_id in required if node_id not in path_set])
    descriptor_lifting_count = 0
    total_derived_penalty = 0.0
    evidence_cost = 0.0
    for left, right in zip(path_ids, path_ids[1:]):
        edge = edges.get(_edge_key(left, right))
        if edge is None:
            evidence_cost += 1_000_000.0
            continue
        if "descriptor_lifting" in edge.rule:
            descriptor_lifting_count += 1
        penalty = _edge_rule_penalty(edge)
        total_derived_penalty += penalty
        evidence_cost += 1.0 / max(edge.support, 1e-6) + penalty

    function_node_count = sum(
        1
        for node_id in path_ids
        if nodes[node_id].kind == "function" and node_id != query_focus.terminal_id
    )
    components = {
        "missing_required_focus_count": missing_required_count,
        "source_entry_tier": _source_entry_tier(edges, path_ids),
        "low_salience_prefix_length": _low_salience_prefix_length(edges, path_ids),
        "descriptor_lifting_count": descriptor_lifting_count,
        "total_derived_penalty": round(total_derived_penalty, 6),
        "function_node_count": function_node_count,
        "evidence_cost": round(evidence_cost, 6),
        "path_length": len(path_ids),
        "token_index_tuple": list(_path_index_tuple(path_ids, nodes)),
    }
    rank = (
        components["missing_required_focus_count"],
        components["source_entry_tier"],
        components["low_salience_prefix_length"],
        components["descriptor_lifting_count"],
        components["total_derived_penalty"],
        components["function_node_count"],
        components["evidence_cost"],
        components["path_length"],
        tuple(components["token_index_tuple"]),
    )
    return _CandidatePath(
        source_entity_id=source_entity_id,
        node_ids=tuple(path_ids),
        search_pass=search_pass,
        rank=rank,
        rank_components=components,
    )


def _source_entry_tier(edges: dict[tuple[str, str], TokenReasoningEdge], path_ids: list[str]) -> int:
    tiers: list[int] = []
    for index in range(min(2, len(path_ids) - 1)):
        edge = edges.get(_edge_key(path_ids[index], path_ids[index + 1]))
        if edge is None:
            continue
        classes = _edge_label_classes_deep(edge)
        if "CORE_ARG" in classes:
            tiers.append(0)
        elif "RESTRICT" in classes:
            tiers.append(1)
        elif classes & {"IDENTITY", "MODIFIER"}:
            tiers.append(2)
        else:
            tiers.append(3)
    return min(tiers) if tiers else 3


def _low_salience_prefix_length(edges: dict[tuple[str, str], TokenReasoningEdge], path_ids: list[str]) -> int:
    count = 0
    for left, right in zip(path_ids, path_ids[1:]):
        edge = edges.get(_edge_key(left, right))
        if edge is None:
            break
        classes = _edge_label_classes_deep(edge)
        if classes & {"CORE_ARG", "RESTRICT"}:
            break
        if classes & {"IDENTITY", "MODIFIER", "BRIDGE"} or not classes:
            count += 1
            continue
        break
    return count


def _edge_rule_penalty(edge: TokenReasoningEdge) -> float:
    total = 0.0
    for rule, penalty in DERIVED_PENALTIES.items():
        if rule in edge.rule:
            total += penalty
    return total


def _edge_label_classes_deep(edge: TokenReasoningEdge) -> set[str]:
    classes = set(_edge_label_classes(edge))

    def visit(payload: Any) -> None:
        if isinstance(payload, dict):
            label_class = payload.get("label_class")
            if label_class:
                classes.add(str(label_class))
            for source_edge in payload.get("source_edges") or []:
                visit(source_edge)
            for provenance in payload.get("provenance") or []:
                visit(provenance)
        elif isinstance(payload, list):
            for item in payload:
                visit(item)

    for item in edge.provenance:
        visit(item)
    return classes


def _detect_parallel_entity_sets(
    state: _WorkingState,
    explicit_entity_ids: list[str],
    direct_candidate_sets: list[list[str]],
    query_focus: _QueryFocus,
) -> list[dict[str, Any]]:
    parallel_sets: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate_set in direct_candidate_sets:
        entity_ids = _candidate_text_set_to_ids(state.nodes, candidate_set)
        if len(entity_ids) < 2:
            continue
        key = tuple(entity_ids)
        if key in seen:
            continue
        seen.add(key)
        parallel_sets.append(
            {
                "kind": "direct_entity_set",
                "entity_ids": entity_ids,
                "branch_heads": {},
                "evidence": {"candidate_set": candidate_set},
            }
        )

    for group in _content_coordination_groups(state):
        branch_heads = _sort_node_ids(group["member_ids"], state.nodes)
        member_set = set(branch_heads)
        entity_to_branch: dict[str, str] = {}
        for branch_head_id in branch_heads:
            bound_entity = _unique_bound_entity_for_branch(
                state,
                branch_head_id,
                explicit_entity_ids,
                query_focus=query_focus,
                coordination_member_ids=member_set,
            )
            if not bound_entity or bound_entity in entity_to_branch:
                continue
            if not _raw_branch_reaches_query_focus(
                state,
                branch_head_id,
                query_focus,
                blocked_ids=(member_set - {branch_head_id}) | (set(explicit_entity_ids) - {bound_entity}),
            ):
                continue
            entity_to_branch[bound_entity] = branch_head_id
        if len(entity_to_branch) < 2:
            continue
        entity_ids = _sort_node_ids(entity_to_branch, state.nodes)
        key = tuple(entity_ids)
        if key in seen:
            continue
        seen.add(key)
        parallel_sets.append(
            {
                "kind": "lifted_coordination",
                "entity_ids": entity_ids,
                "branch_heads": {entity_id: entity_to_branch[entity_id] for entity_id in entity_ids},
                "evidence": group["evidence"],
            }
        )
    return parallel_sets


def _content_coordination_groups(state: _WorkingState) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for key in _sorted_edge_keys(state.raw_edges, state.nodes):
        edge = state.raw_edges[key]
        source, target = key
        if "COORD" in _edge_label_classes(edge) and _is_content_branch_head(state.nodes[source]) and _is_content_branch_head(state.nodes[target]):
            groups.append(
                {
                    "member_ids": [source, target],
                    "evidence": {"rule": "content_coordination_edge", "edge": edge.to_dict()},
                }
            )

    adjacency = _adjacency(state.raw_edges)
    for connector in _sorted_nodes(state.nodes.values()):
        if connector.id == "0":
            continue
        if not (_is_scope_node(connector) or connector.text.lower() in {"and", "or"}):
            continue
        member_ids: list[str] = []
        source_edges: list[dict[str, Any]] = []
        for neighbor_id, edge in _neighbor_edges(connector.id, adjacency, state.raw_edges):
            if "COORD" not in _edge_label_classes(edge):
                continue
            if not _is_content_branch_head(state.nodes[neighbor_id]):
                continue
            member_ids.append(neighbor_id)
            source_edges.append(edge.to_dict())
        ordered = _sort_node_ids(member_ids, state.nodes)
        if len(ordered) >= 2:
            groups.append(
                {
                    "member_ids": ordered,
                    "evidence": {
                        "rule": "content_coordination_connector",
                        "connector_id": connector.id,
                        "connector": connector.text,
                        "source_edges": source_edges,
                    },
                }
            )
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for group in groups:
        ordered = tuple(_sort_node_ids(group["member_ids"], state.nodes))
        if len(ordered) < 2:
            continue
        unique.setdefault(ordered, {"member_ids": list(ordered), "evidence": group["evidence"]})
    return [
        unique[key]
        for key in sorted(unique, key=lambda item: tuple(_node_sort_key(state.nodes[node_id]) for node_id in item))
    ]


def _content_coordination_pairs(state: _WorkingState) -> list[tuple[str, str, dict[str, Any]]]:
    pairs: list[tuple[str, str, dict[str, Any]]] = []
    for group in _content_coordination_groups(state):
        ordered = _sort_node_ids(group["member_ids"], state.nodes)
        for index, left_id in enumerate(ordered):
            for right_id in ordered[index + 1 :]:
                pairs.append((left_id, right_id, group["evidence"]))
    return pairs


def _unique_bound_entity_for_branch(
    state: _WorkingState,
    branch_head_id: str,
    explicit_entity_ids: list[str],
    *,
    query_focus: _QueryFocus,
    coordination_member_ids: set[str],
) -> str | None:
    if branch_head_id not in state.nodes:
        return None
    explicit = set(explicit_entity_ids)
    adjacency = _adjacency(state.raw_edges)
    blocked_ids = {
        node_id
        for node_id in (query_focus.query_root_id, query_focus.answer_anchor_id, query_focus.terminal_id)
        if node_id and node_id != branch_head_id
    }
    blocked_ids.update(coordination_member_ids - {branch_head_id})

    direct_matches = {
        neighbor_id
        for neighbor_id, edge in _neighbor_edges(branch_head_id, adjacency, state.raw_edges)
        if neighbor_id in explicit and _raw_binding_edge_allowed(edge)
    }
    ordered_direct = _sort_node_ids(direct_matches, state.nodes)
    if len(ordered_direct) == 1:
        return ordered_direct[0]
    if len(ordered_direct) > 1:
        return None

    queue: list[list[str]] = [[branch_head_id]]
    matches: set[str] = set()
    while queue:
        path = queue.pop(0)
        current = path[-1]
        if len(path) > 3:
            continue
        if current != branch_head_id and current in blocked_ids:
            continue
        if current in explicit and current != branch_head_id:
            matches.add(current)
            continue
        if len(path) == 3:
            continue
        for neighbor_id, edge in _raw_neighbor_edges(state.nodes, state.raw_edges, adjacency, current):
            if neighbor_id in path:
                continue
            if neighbor_id in blocked_ids:
                continue
            if "COORD" in _edge_label_classes(edge):
                continue
            if _is_scope_node(state.nodes[neighbor_id]):
                continue
            if not _raw_binding_edge_allowed(edge):
                continue
            if state.nodes[neighbor_id].kind == "function" and neighbor_id not in explicit:
                continue
            queue.append([*path, neighbor_id])
    ordered = _sort_node_ids(matches, state.nodes)
    return ordered[0] if len(ordered) == 1 else None


def _is_content_branch_head(node: TokenReasoningNode) -> bool:
    return node.kind in {"content", "constraint", "answer"}


def _raw_binding_edge_allowed(edge: TokenReasoningEdge) -> bool:
    return bool(_edge_label_classes(edge) & {"CORE_ARG", "RESTRICT", "IDENTITY", "MODIFIER", "BRIDGE"})


def _raw_neighbor_edges(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    adjacency: dict[str, dict[str, tuple[str, str]]],
    node_id: str,
) -> list[tuple[str, TokenReasoningEdge]]:
    neighbors = [(neighbor_id, edges[key]) for neighbor_id, key in adjacency.get(node_id, {}).items()]
    neighbors.sort(key=lambda item: (-item[1].support, _node_sort_key(nodes[item[0]]), item[0]))
    return neighbors


def _raw_branch_reaches_query_focus(
    state: _WorkingState,
    branch_head_id: str,
    query_focus: _QueryFocus,
    *,
    blocked_ids: set[str],
) -> bool:
    targets = [node_id for node_id in (query_focus.terminal_id, query_focus.answer_anchor_id) if node_id in state.nodes]
    if not targets:
        return False
    target_set = set(targets)
    adjacency = _adjacency(state.raw_edges)
    queue: list[list[str]] = [[branch_head_id]]
    while queue:
        path = queue.pop(0)
        current = path[-1]
        if len(path) > 7:
            continue
        if current in target_set and current != branch_head_id:
            return True
        if len(path) == 7:
            continue
        for neighbor_id, edge in _raw_neighbor_edges(state.nodes, state.raw_edges, adjacency, current):
            if neighbor_id in path or neighbor_id in blocked_ids:
                continue
            if "COORD" in _edge_label_classes(edge):
                continue
            if _is_scope_node(state.nodes[neighbor_id]):
                continue
            if state.nodes[neighbor_id].kind == "function":
                continue
            if not (_edge_label_classes(edge) & {"CORE_ARG", "RESTRICT", "IDENTITY", "MODIFIER", "BRIDGE", "UNKNOWN"}):
                continue
            queue.append([*path, neighbor_id])
    return False


def _schema_focus_id(
    state: _WorkingState,
    query_focus: _QueryFocus,
    constraints: list[dict[str, Any]],
) -> str | None:
    order_terminal_id = _order_constraint_terminal_id(state.nodes, state.raw_edges, constraints)
    if order_terminal_id:
        return order_terminal_id
    for constraint in constraints:
        if constraint.get("type") == "order" and constraint.get("target_id") in state.nodes:
            return str(constraint["target_id"])
    if query_focus.query_root_id and query_focus.query_root_id in state.nodes and query_focus.query_root_id != query_focus.slot_id:
        return query_focus.query_root_id
    if query_focus.answer_anchor_id and query_focus.answer_anchor_id != query_focus.slot_id:
        return query_focus.answer_anchor_id
    if query_focus.slot_id:
        return _candidate_focus(state.nodes, state.edges, query_focus.slot_id, constraints)
    return query_focus.terminal_id


def _order_constraint_terminal_id(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    constraints: list[dict[str, Any]],
) -> str | None:
    candidates: list[tuple[float, tuple[int, str], str]] = []
    for constraint in constraints:
        if constraint.get("type") != "order":
            continue
        cue_id = str(constraint.get("node_id") or "")
        target_id = str(constraint.get("target_id") or "")
        if cue_id not in nodes or target_id not in nodes:
            continue
        score = _predicative_order_cue_score(nodes, raw_edges, cue_id, target_id)
        if score <= 0.0:
            continue
        candidates.append((-score, _node_sort_key(nodes[cue_id]), cue_id))
    return sorted(candidates)[0][2] if candidates else None


def _predicative_order_cue_score(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    cue_id: str,
    target_id: str,
) -> float:
    cue_target_score = 0.0
    complement_score = 0.0
    cue_idx = nodes[cue_id].index
    target_idx = nodes[target_id].index
    for edge in raw_edges.values():
        for item in _raw_provenance(edge):
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if head_idx is None or dep_idx is None:
                continue
            relation = str(item.get("normalized_relation") or item.get("relation") or "")
            support = _coerce_float_value(item.get("support"), edge.support)
            if _is_order_cue_target_evidence(relation, head_idx, dep_idx, cue_idx, target_idx):
                cue_target_score += support
            if _is_predicative_complement_evidence(relation, head_idx, dep_idx, cue_idx, nodes):
                complement_score += support
    if cue_target_score <= 0.0 or complement_score <= 0.0:
        return 0.0
    return cue_target_score + complement_score


def _is_order_cue_target_evidence(
    relation: str,
    head_idx: int,
    dep_idx: int,
    cue_idx: int,
    target_idx: int,
) -> bool:
    if head_idx != cue_idx or dep_idx != target_idx:
        return False
    key = _normalized_relation_key(relation)
    return key in {"arg1", "verb_arg1", "adj_arg1", "act_arg"} or key.endswith("_arg1")


def _is_predicative_complement_evidence(
    relation: str,
    head_idx: int,
    dep_idx: int,
    cue_idx: int,
    nodes: dict[str, TokenReasoningNode],
) -> bool:
    if dep_idx != cue_idx:
        return False
    key = _normalized_relation_key(relation)
    if key not in {"arg2", "verb_arg2", "pat_arg", "eff_arg", "compl"} and not key.endswith("_arg2"):
        return False
    head_node = nodes.get(str(head_idx))
    return head_node is not None and head_node.kind in {"function", "content"}


def _best_schema_path(state: _WorkingState, slot_id: str, focus_id: str) -> list[str]:
    paths = _bounded_k_simple_paths(
        state.nodes,
        state.edges,
        source_id=slot_id,
        target_id=focus_id,
        forbidden_nodes=set(),
        required_ids={focus_id},
        allow_weak=False,
        max_nodes=8,
        top_k=8,
    )
    if not paths:
        paths = _bounded_k_simple_paths(
            state.nodes,
            state.edges,
            source_id=slot_id,
            target_id=focus_id,
            forbidden_nodes=set(),
            required_ids={focus_id},
            allow_weak=True,
            max_nodes=8,
            top_k=8,
        )
    if not paths:
        return []
    ranked = [
        _rank_candidate_path(
            state.nodes,
            state.edges,
            path,
            _QueryFocus(focus_id, None, slot_id, focus_id, tuple([focus_id]), "schema_path"),
            source_entity_id=None,
            search_pass="schema",
            required_ids=[focus_id],
        )
        for path in paths
    ]
    return list(sorted(ranked, key=lambda record: record.rank)[0].node_ids)


def _bare_wh_candidate_schema(
    state: _WorkingState,
    query_focus: _QueryFocus,
) -> dict[str, Any] | None:
    query_root_id = query_focus.query_root_id or _find_query_root(state.nodes, state.raw_edges)
    wh_id: str | None = None
    if query_root_id and query_root_id in state.nodes:
        wh_ids = _bare_wh_direct_argument_ids(state.nodes, state.raw_edges, query_root_id)
        if not wh_ids:
            return None
        wh_id = wh_ids[0]
    else:
        inferred = _infer_bare_wh_query_predicate(state.nodes, state.raw_edges)
        if inferred is None:
            return None
        query_root_id, wh_id = inferred
    if not query_root_id or not wh_id or query_root_id not in state.nodes or wh_id not in state.nodes:
        return None
    suffix_id = _bare_wh_temporal_or_modifier_suffix_id(state.nodes, state.raw_edges, query_root_id)
    focus_id = suffix_id or query_root_id
    schema_path = _schema_path_through_query_root(state, wh_id, query_root_id, focus_id)
    if len(schema_path) < 2:
        return None
    return {
        "wh_id": wh_id,
        "query_root_id": query_root_id,
        "focus_id": focus_id,
        "schema_path": schema_path,
    }


def _schema_path_through_query_root(
    state: _WorkingState,
    wh_id: str,
    query_root_id: str,
    focus_id: str,
) -> list[str]:
    prefix = _best_schema_path(state, wh_id, query_root_id)
    if not prefix or query_root_id not in prefix:
        return []
    if focus_id == query_root_id:
        return prefix
    suffix = _best_schema_path(state, query_root_id, focus_id)
    if not suffix or suffix[0] != query_root_id:
        return []
    path = _dedupe_adjacent([*prefix, *suffix[1:]])
    return path if len(path) == len(set(path)) else []


def _ensure_candidate_slot_edge(
    state: _WorkingState,
    candidate_id: str,
    successor_id: str,
    slot_id: str,
    candidate_set: list[str],
    schema_path: list[str],
) -> None:
    if _edge_key(candidate_id, successor_id) in state.edges:
        return
    schema_edge = state.edges.get(_edge_key(slot_id, successor_id))
    provenance = {
        "rule": "candidate_slot_substitution",
        "typed_wh_slot_id": slot_id,
        "typed_wh_slot": state.nodes[slot_id].text,
        "typed_wh_evidence": _typed_wh_slot_evidence(state, slot_id),
        "candidate_id": candidate_id,
        "candidate": state.nodes[candidate_id].text,
        "candidate_set": list(candidate_set),
        "candidate_set_entity_ids": _candidate_text_set_to_ids(state.nodes, candidate_set),
        "candidate_set_evidence": _candidate_set_coordination_evidence(state, candidate_set),
        "schema_path_ids": list(schema_path),
        "schema_path": [state.nodes[node_id].text for node_id in schema_path if node_id in state.nodes],
        "schema_edge": schema_edge.to_dict() if schema_edge else None,
    }
    virtual = _merge_edge(
        state.edges,
        state.nodes,
        candidate_id,
        successor_id,
        support=0.80,
        derived=True,
        rule="candidate_slot_substitution",
        provenance=[provenance],
    )
    state.virtual_edges.append(virtual.to_dict())


def _typed_wh_slot_evidence(state: _WorkingState, slot_id: str) -> dict[str, Any]:
    raw_edges: list[dict[str, Any]] = []
    for edge in state.raw_edges.values():
        if slot_id not in (edge.source, edge.target):
            continue
        if any(
            isinstance(item, dict)
            and (
                str(item.get("head", "")).lower() in {"what", "which"}
                or str(item.get("dep", "")).lower() in {"what", "which"}
            )
            for item in edge.provenance
        ):
            raw_edges.append(edge.to_dict())

    surface_adjacency: dict[str, Any] | None = None
    slot = state.nodes.get(slot_id)
    if slot is not None:
        previous = next((node for node in state.nodes.values() if node.index == slot.index - 1), None)
        if previous is not None and previous.text.lower() in {"what", "which"}:
            surface_adjacency = {
                "rule": "surface_typed_wh_adjacency",
                "wh_id": previous.id,
                "wh": previous.text,
                "slot_id": slot.id,
                "slot": slot.text,
            }
    return {
        "raw_edges": raw_edges,
        "surface_adjacency": surface_adjacency,
    }


def _ensure_candidate_bare_wh_edge(
    state: _WorkingState,
    candidate_id: str,
    successor_id: str,
    wh_id: str,
    query_root_id: str,
    candidate_set: list[str],
    schema_path: list[str],
) -> None:
    if _edge_key(candidate_id, successor_id) in state.edges:
        return
    schema_edge = state.edges.get(_edge_key(wh_id, successor_id))
    wh_predicate_edge = state.raw_edges.get(_edge_key(wh_id, query_root_id))
    support = max(0.05, min((schema_edge.support if schema_edge else 0.80), 0.80))
    provenance = {
        "rule": "candidate_bare_wh_substitution",
        "candidate_id": candidate_id,
        "candidate": state.nodes[candidate_id].text,
        "bare_wh_slot_id": wh_id,
        "bare_wh_slot": state.nodes[wh_id].text,
        "query_predicate_id": query_root_id,
        "query_predicate": state.nodes[query_root_id].text,
        "candidate_set": list(candidate_set),
        "candidate_set_entity_ids": _candidate_text_set_to_ids(state.nodes, candidate_set),
        "candidate_set_evidence": _candidate_set_coordination_evidence(state, candidate_set),
        "schema_path_ids": list(schema_path),
        "schema_path": [state.nodes[node_id].text for node_id in schema_path if node_id in state.nodes],
        "schema_edge": schema_edge.to_dict() if schema_edge else None,
        "wh_predicate_edge": wh_predicate_edge.to_dict() if wh_predicate_edge else None,
        "support": support,
    }
    virtual = _merge_edge(
        state.edges,
        state.nodes,
        candidate_id,
        successor_id,
        support=support,
        derived=True,
        rule="candidate_bare_wh_substitution",
        provenance=[provenance],
    )
    state.virtual_edges.append(virtual.to_dict())


def _candidate_set_coordination_evidence(
    state: _WorkingState,
    candidate_set: list[str],
) -> list[dict[str, Any]]:
    candidate_ids = set(_candidate_text_set_to_ids(state.nodes, candidate_set))
    evidence: list[dict[str, Any]] = []
    for key in _sorted_edge_keys(state.raw_edges, state.nodes):
        edge = state.raw_edges[key]
        source, target = key
        classes = _edge_label_classes(edge)
        if "COORD" not in classes and not (_is_scope_node(state.nodes[source]) or _is_scope_node(state.nodes[target])):
            continue
        if source in candidate_ids or target in candidate_ids:
            evidence.append(edge.to_dict())
    return evidence


def _graph_from_selected_paths(paths: list[TokenReasoningPath]) -> tuple[set[str], set[tuple[str, str]]]:
    node_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for path in paths:
        node_ids.update(path.node_ids)
        for left, right in zip(path.node_ids, path.node_ids[1:]):
            pairs.add(_edge_key(left, right))
    return node_ids, pairs


def _active_entity_ids(nodes: dict[str, TokenReasoningNode], paths: list[TokenReasoningPath]) -> list[str]:
    active: list[str] = []
    for path in paths:
        if path.node_ids and path.node_ids[0] in nodes and nodes[path.node_ids[0]].kind == "entity":
            active.append(path.node_ids[0])
    return _sort_node_ids(active, nodes)


def _candidate_text_set_to_ids(nodes: dict[str, TokenReasoningNode], candidate_set: list[str]) -> list[str]:
    wanted = set(candidate_set)
    return _sort_node_ids(
        [node.id for node in nodes.values() if node.kind == "entity" and node.text in wanted],
        nodes,
    )


def _candidate_sets_for_result(
    nodes: dict[str, TokenReasoningNode],
    direct_candidate_sets: list[list[str]],
    parallel_entity_sets: list[dict[str, Any]],
) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    result: list[list[str]] = []
    for candidate_set in direct_candidate_sets:
        ordered = tuple(candidate_set)
        if len(ordered) >= 2 and ordered not in seen:
            seen.add(ordered)
            result.append(list(ordered))
    for parallel_set in parallel_entity_sets:
        entity_ids = [node_id for node_id in parallel_set.get("entity_ids", []) if node_id in nodes]
        texts = tuple(nodes[node_id].text for node_id in _sort_node_ids(entity_ids, nodes))
        if len(texts) >= 2 and texts not in seen:
            seen.add(texts)
            result.append(list(texts))
    return result


def _sort_parallel_entity_sets(
    nodes: dict[str, TokenReasoningNode],
    parallel_entity_sets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    priority = {"direct_entity_set": 0, "lifted_coordination": 1}
    return sorted(
        parallel_entity_sets,
        key=lambda item: (
            priority.get(str(item.get("kind")), 9),
            tuple(nodes[node_id].index for node_id in item.get("entity_ids", []) if node_id in nodes),
        ),
    )


def _fallback_single_node_id(
    nodes: dict[str, TokenReasoningNode],
    explicit_entity_ids: list[str],
    query_focus: _QueryFocus,
) -> str | None:
    for node_id in explicit_entity_ids:
        if node_id in nodes:
            return node_id
    for node_id in (query_focus.terminal_id, query_focus.answer_anchor_id, query_focus.query_root_id):
        if node_id in nodes:
            return node_id
    return _first_content_node(nodes)


def _unique_node_ids(node_ids: Iterable[str | None], nodes: dict[str, TokenReasoningNode]) -> list[str]:
    result: list[str] = []
    for node_id in node_ids:
        if not node_id or node_id not in nodes or node_id in result:
            continue
        result.append(node_id)
    return result


def write_debug_json(payload: dict[str, Any], *, question_id: str | None, debug_dir: str | Path | None) -> str:
    directory = Path(debug_dir) if debug_dir is not None else Path("debug") / "hanlp_sdp"
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_filename(question_id) if question_id else 'q1'}_tri_sdp_reasoning.json"
    path = directory / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


# Legacy generic candidate expansion retained for comparison only. The formal
# Step4 path uses the controlled candidate_slot_substitution implementation.
def _extract_candidate_path_cover(
    state: _WorkingState,
    candidate_sets: list[list[str]],
    answer_anchor_id: str | None,
    constraints: list[dict[str, Any]],
) -> tuple[set[str], set[tuple[str, str]], list[TokenReasoningPath], str]:
    first_candidate_set = candidate_sets[0]
    candidate_ids = [
        node.id
        for node in _sorted_nodes(state.nodes.values())
        if node.text in set(first_candidate_set) and node.kind == "entity"
    ]
    focus_id = _candidate_focus(state.nodes, state.edges, answer_anchor_id, constraints)
    graph = _weighted_adjacency(state.nodes, state.edges, include_scope=False)
    schema_path: list[str] = []
    if answer_anchor_id and focus_id:
        schema_path, _cost = _shortest_path(graph, state.nodes, answer_anchor_id, focus_id)
    elif focus_id:
        schema_path = [focus_id]

    final_pairs: set[tuple[str, str]] = set()
    final_node_ids: set[str] = set()
    paths: list[TokenReasoningPath] = []

    for path_index, candidate_id in enumerate(candidate_ids, start=1):
        if schema_path:
            if answer_anchor_id and schema_path and schema_path[0] == answer_anchor_id:
                branch = [candidate_id, *schema_path[1:]]
                if len(branch) == 1 and focus_id and focus_id != candidate_id:
                    branch.append(focus_id)
            else:
                branch = [candidate_id, *schema_path]
        elif focus_id:
            branch = [candidate_id, focus_id]
        else:
            branch = [candidate_id]

        branch = _dedupe_adjacent(branch)
        if len(branch) >= 2:
            first_pair = _edge_key(branch[0], branch[1])
            if first_pair not in state.edges:
                provenance = {
                    "rule": "candidate_expansion",
                    "candidate": state.nodes[candidate_id].text,
                    "schema_path": [state.nodes[node_id].text for node_id in schema_path],
                }
                virtual = _merge_edge(
                    state.edges,
                    state.nodes,
                    branch[0],
                    branch[1],
                    support=0.80,
                    derived=True,
                    rule="candidate_expansion",
                    provenance=[provenance],
                )
                state.virtual_edges.append(virtual.to_dict())
            for index in range(len(branch) - 1):
                final_pairs.add(_edge_key(branch[index], branch[index + 1]))
        final_node_ids.update(branch)
        paths.append(_path_from_ids(f"P{path_index}", state.nodes, branch))

    return final_node_ids, final_pairs, paths, "candidate_path_cover"


def _candidate_focus(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    answer_anchor_id: str | None,
    constraints: list[dict[str, Any]],
) -> str | None:
    for constraint in constraints:
        if constraint.get("type") == "order" and constraint.get("target_id") in nodes:
            return str(constraint["target_id"])
    if answer_anchor_id not in nodes:
        return _first_content_node(nodes)

    graph = _weighted_adjacency(nodes, edges, include_scope=False)
    best: tuple[float, tuple[int, str], str] | None = None
    for node in nodes.values():
        if node.id == answer_anchor_id or node.kind not in {"content", "answer"}:
            continue
        path, cost = _shortest_path(graph, nodes, answer_anchor_id, node.id)
        if not path:
            continue
        # Prefer farther schema content from the answer type, then deterministic order.
        candidate = (-len(path), _node_sort_key(node), node.id)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best else answer_anchor_id


def _select_terminals(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    entity_ids: list[str],
    answer_anchor_id: str | None,
    constraints: list[dict[str, Any]],
) -> list[str]:
    terminals: list[str] = [node_id for node_id in entity_ids if node_id in nodes]
    if answer_anchor_id and answer_anchor_id in nodes and answer_anchor_id not in terminals:
        terminals.append(answer_anchor_id)
    for constraint in constraints:
        if constraint.get("type") != "numeric":
            continue
        node_id = str(constraint.get("node_id") or "")
        target_id = str(constraint.get("target_id") or "")
        if node_id in nodes and (target_id == answer_anchor_id or _edge_key(node_id, answer_anchor_id or "") in edges):
            if node_id not in terminals:
                terminals.append(node_id)
    if not terminals:
        first_content = _first_content_node(nodes)
        if first_content:
            terminals.append(first_content)
    return _sort_node_ids(terminals, nodes)


def _build_debug_payload(
    *,
    question_id: str | None,
    masked_question: str,
    hanlp_sdp_result: HanLPSDPResult,
    explicit_entities: list[str],
    state: _WorkingState,
    answer_anchor_id: str | None,
    entity_ids: list[str],
    constraints: list[dict[str, Any]],
    candidate_sets: list[list[str]],
    terminals: list[str],
    backbone_before: dict[str, Any],
    backbone_after: dict[str, Any],
    final_nodes: list[TokenReasoningNode],
    final_edges: list[TokenReasoningEdge],
    paths: list[TokenReasoningPath],
    query_focus: _QueryFocus | None = None,
    entity_candidates: list[str] | None = None,
    active_entity_ids: list[str] | None = None,
    parallel_entity_sets: list[dict[str, Any]] | None = None,
    candidate_paths: list[_CandidatePath] | None = None,
    selected_paths: list[TokenReasoningPath] | None = None,
    selection_mode: str | None = None,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "masked_question": masked_question,
        "tokens": list(hanlp_sdp_result.tokens),
        "explicit_entities": list(explicit_entities),
        "raw_sdp_edges": [_raw_edge_to_dict(edge) for edge in hanlp_sdp_result.edges],
        "normalized_evidence_edges": list(state.normalized_edges),
        "aggregated_edges": [edge.to_dict() for edge in _sorted_edges(state.raw_edges.values(), state.nodes)],
        "virtual_edges": list(state.virtual_edges),
        "anchors": {
            "entities": [state.nodes[node_id].text for node_id in entity_ids if node_id in state.nodes],
            "answer_anchor": state.nodes[answer_anchor_id].text if answer_anchor_id in state.nodes else None,
            "answer_anchor_id": answer_anchor_id,
            "constraints": constraints,
        },
        "candidate_sets": candidate_sets,
        "terminals": [state.nodes[node_id].text for node_id in terminals if node_id in state.nodes],
        "backbone_before_pruning": backbone_before,
        "backbone_after_pruning": backbone_after,
        "final_nodes": [node.to_dict() for node in final_nodes],
        "final_edges": [edge.to_dict() for edge in final_edges],
        "paths": [path.to_dict() for path in paths],
        "query_focus": query_focus.to_dict(state.nodes) if query_focus else None,
        "entity_candidates": [
            {"id": node_id, "text": state.nodes[node_id].text}
            for node_id in (entity_candidates or [])
            if node_id in state.nodes
        ],
        "active_entity_anchors": [
            {"id": node_id, "text": state.nodes[node_id].text}
            for node_id in (active_entity_ids or [])
            if node_id in state.nodes
        ],
        "parallel_entity_sets": list(parallel_entity_sets or []),
        "candidate_paths": [record.to_debug(state.nodes) for record in (candidate_paths or [])],
        "selected_paths": [path.to_dict() for path in (selected_paths or paths)],
        "selection_mode": selection_mode,
        "warnings": list(state.warnings),
    }


def _build_token_nodes(hanlp_sdp_result: HanLPSDPResult) -> dict[str, TokenReasoningNode]:
    nodes = {"0": TokenReasoningNode(id="0", text="ROOT", index=0, kind="function")}
    for index, token in enumerate(hanlp_sdp_result.tokens, start=1):
        nodes[str(index)] = TokenReasoningNode(
            id=str(index),
            text=str(token),
            index=index,
            kind=classify_node(str(token), index),
        )
    return nodes


def _mark_possessive_marker_nodes(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> None:
    for node in nodes.values():
        if _is_contextual_possessive_marker(node.id, nodes, raw_edges):
            node.kind = "function"


def _ensure_edge_nodes(nodes: dict[str, TokenReasoningNode], edge: HanLPSDPEdge) -> None:
    for index, text in ((edge.head_idx, edge.head), (edge.dep_idx, edge.dep)):
        node_id = str(index)
        if node_id in nodes:
            continue
        nodes[node_id] = TokenReasoningNode(
            id=node_id,
            text="ROOT" if index == 0 else str(text),
            index=index,
            kind=classify_node("ROOT" if index == 0 else str(text), index),
        )


def _resolve_explicit_entity_ids(
    nodes: dict[str, TokenReasoningNode],
    explicit_entities: list[str],
) -> list[str]:
    entity_order = {entity: position for position, entity in enumerate(explicit_entities)}
    matched = [
        node.id
        for node in nodes.values()
        if node.text in entity_order and ENTITY_RE.fullmatch(node.text)
    ]
    return sorted(set(matched), key=lambda node_id: (entity_order.get(nodes[node_id].text, 9999), _node_sort_key(nodes[node_id])))


def _mark_anchors(nodes: dict[str, TokenReasoningNode], entity_ids: list[str], answer_anchor_id: str | None) -> None:
    for node_id in entity_ids:
        if node_id in nodes:
            nodes[node_id].is_anchor = True
    if answer_anchor_id and answer_anchor_id in nodes:
        nodes[answer_anchor_id].is_anchor = True
        if nodes[answer_anchor_id].kind == "function" and nodes[answer_anchor_id].text.lower() in WH_WORDS:
            nodes[answer_anchor_id].kind = "answer"


def _copy_node_map(nodes: dict[str, TokenReasoningNode]) -> dict[str, TokenReasoningNode]:
    return {
        node_id: TokenReasoningNode(
            id=node.id,
            text=node.text,
            index=node.index,
            kind=node.kind,
            is_anchor=node.is_anchor,
        )
        for node_id, node in nodes.items()
    }


def _copy_working_state(state: _WorkingState) -> _WorkingState:
    return _WorkingState(
        nodes=_copy_node_map(state.nodes),
        raw_edges={key: _copy_edge(edge) for key, edge in state.raw_edges.items()},
        edges={key: _copy_edge(edge) for key, edge in state.edges.items()},
        normalized_edges=[dict(item) for item in state.normalized_edges],
        virtual_edges=[dict(item) for item in state.virtual_edges],
        warnings=list(state.warnings),
    )


def _merge_constraint_debug_lists(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = [dict(item) for item in existing]
    seen = {json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) for item in result}
    for item in incoming:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def _merge_candidate_set_lists(
    existing: list[list[str]],
    incoming: list[list[str]],
) -> list[list[str]]:
    result = [list(item) for item in existing]
    seen = {tuple(item) for item in result}
    for item in incoming:
        key = tuple(item)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(list(item))
    return result


def _merge_edge(
    edge_map: dict[tuple[str, str], TokenReasoningEdge],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
    *,
    support: float,
    derived: bool,
    rule: str,
    provenance: list[dict[str, Any]],
) -> TokenReasoningEdge:
    if source_id == target_id:
        return TokenReasoningEdge(source_id, target_id, nodes[source_id].text, nodes[target_id].text)
    key = _edge_key(source_id, target_id)
    source, target = key
    if key not in edge_map:
        edge_map[key] = TokenReasoningEdge(
            source=source,
            target=target,
            source_text=nodes[source].text,
            target_text=nodes[target].text,
            support=0.0,
            derived=derived,
            rule=rule if derived else "raw_evidence",
            provenance=[],
        )
    edge = edge_map[key]
    edge.support += support
    edge.derived = edge.derived or derived
    if derived:
        edge.rule = _combine_rules(edge.rule, rule)
    edge.provenance.extend(provenance)
    return edge


def _copy_edge(edge: TokenReasoningEdge) -> TokenReasoningEdge:
    return TokenReasoningEdge(
        source=edge.source,
        target=edge.target,
        source_text=edge.source_text,
        target_text=edge.target_text,
        support=edge.support,
        derived=edge.derived,
        rule=edge.rule,
        provenance=[dict(item) for item in edge.provenance],
    )


def _combine_rules(existing: str, new_rule: str) -> str:
    if not existing or existing == "raw_evidence":
        return new_rule
    parts = existing.split("+")
    if new_rule not in parts:
        parts.append(new_rule)
    return "+".join(parts)


def _normalize_relation(relation: str) -> str:
    return str(relation or "").strip().lower()


def _normalized_relation_key(relation: str) -> str:
    return _normalize_relation(relation).replace("-", "_").replace(".", "_").replace("/", "_")


def _projection_relation_polarity(relation: str) -> str | None:
    key = _normalized_relation_key(relation)
    forward_roles = {
        "arg2",
        "verb_arg2",
        "pat_arg",
        "eff_arg",
        "compl",
        "twhen",
        "loc",
        "tloc",
        "tmp",
        "ext",
    }
    subject_roles = {
        "arg1",
        "verb_arg1",
        "act_arg",
        "auth",
    }
    if key in forward_roles:
        return "forward"
    if key in subject_roles:
        return "subject"
    return None


def _raw_provenance(edge: TokenReasoningEdge) -> list[dict[str, Any]]:
    return [item for item in edge.provenance if isinstance(item, dict) and "head_idx" in item and "dep_idx" in item]


def _is_contextual_possessive_marker(
    node_id: str,
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> bool:
    node = nodes.get(node_id)
    if node is None or not _is_possessive_marker_surface(node.text):
        return False
    incident = _incident_raw_provenance(node_id, raw_edges)
    if not incident:
        return False
    return any(_is_possessive_relation(item.get("normalized_relation") or item.get("relation")) for item in incident)


def _possessive_marker_role_edges(
    marker_id: str,
    state: _WorkingState,
) -> tuple[list[tuple[str, TokenReasoningEdge]], list[tuple[str, TokenReasoningEdge]]]:
    owners: dict[str, TokenReasoningEdge] = {}
    possessed: dict[str, TokenReasoningEdge] = {}
    for key, edge in state.raw_edges.items():
        if marker_id not in key:
            continue
        for item in _raw_provenance(edge):
            relation = _normalized_relation_key(str(item.get("normalized_relation") or item.get("relation") or ""))
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if str(head_idx) == marker_id:
                related_id = str(dep_idx)
            elif str(dep_idx) == marker_id:
                related_id = str(head_idx)
            else:
                continue
            if related_id not in state.nodes:
                continue
            if relation in POSSESSIVE_OWNER_RELATIONS:
                owners[related_id] = edge
            elif (
                relation in POSSESSIVE_POSSESSED_RELATIONS
                and state.nodes[related_id].index > state.nodes[marker_id].index
            ):
                possessed[related_id] = edge
    owner_items = sorted(owners.items(), key=lambda item: _node_sort_key(state.nodes[item[0]]))
    possessed_items = sorted(possessed.items(), key=lambda item: _node_sort_key(state.nodes[item[0]]))
    return owner_items, possessed_items


def _incident_raw_provenance(
    node_id: str,
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> list[dict[str, Any]]:
    incident: list[dict[str, Any]] = []
    for key, edge in raw_edges.items():
        if node_id not in key:
            continue
        incident.extend(_raw_provenance(edge))
    return incident


def _is_possessive_marker_surface(text: str) -> bool:
    return str(text or "").strip().lower() in POSSESSIVE_MARKER_TOKENS


def _is_possessive_relation(relation: object) -> bool:
    key = _normalized_relation_key(str(relation or ""))
    return key == "poss" or key.startswith("poss_") or key.endswith("_poss")


def _coerce_provenance_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_projection_candidate_node(
    nodes: dict[str, TokenReasoningNode],
    node_id: str,
    query_root_id: str,
) -> bool:
    if node_id == query_root_id or node_id == "0" or node_id not in nodes:
        return False
    return nodes[node_id].kind == "content"


def _is_punctuation(text: str) -> bool:
    return all(not char.isalnum() for char in text)


def _is_scope_node(node: TokenReasoningNode) -> bool:
    return node.text.lower() in SCOPE_WORDS


def _is_bridge_node(node: TokenReasoningNode) -> bool:
    return node.kind == "function" and not _is_scope_node(node) and node.text.lower() not in WH_WORDS


def _is_high_salience_node(node: TokenReasoningNode, *, include_order_constraints: bool) -> bool:
    if node.kind in {"entity", "content", "answer"}:
        return True
    if node.kind == "constraint":
        return include_order_constraints or NUMERIC_RE.fullmatch(node.text) is not None
    return False


def _edge_key(source_id: str, target_id: str) -> tuple[str, str]:
    return tuple(sorted((source_id, target_id), key=lambda item: int(item) if str(item).lstrip("-").isdigit() else 10**9))  # type: ignore[return-value]


def _node_sort_key(node: TokenReasoningNode) -> tuple[int, str]:
    return (node.index, node.text)


def _sort_node_ids(node_ids: Iterable[str], nodes: dict[str, TokenReasoningNode]) -> list[str]:
    return sorted(dict.fromkeys(node_ids), key=lambda node_id: _node_sort_key(nodes[node_id]))


def _sorted_nodes(nodes: Iterable[TokenReasoningNode]) -> list[TokenReasoningNode]:
    return sorted(nodes, key=_node_sort_key)


def _sorted_edge_keys(
    edges: dict[tuple[str, str], TokenReasoningEdge],
    nodes: dict[str, TokenReasoningNode],
) -> list[tuple[str, str]]:
    return sorted(edges, key=lambda key: (_node_sort_key(nodes[key[0]]), _node_sort_key(nodes[key[1]])))


def _sorted_edges(
    edges: Iterable[TokenReasoningEdge],
    nodes: dict[str, TokenReasoningNode],
) -> list[TokenReasoningEdge]:
    return sorted(edges, key=lambda edge: (_node_sort_key(nodes[edge.source]), _node_sort_key(nodes[edge.target])))


def _adjacency(edges: dict[tuple[str, str], TokenReasoningEdge]) -> dict[str, dict[str, tuple[str, str]]]:
    adjacency: dict[str, dict[str, tuple[str, str]]] = {}
    for key in edges:
        source, target = key
        adjacency.setdefault(source, {})[target] = key
        adjacency.setdefault(target, {})[source] = key
    return adjacency


def _neighbor_edges(
    node_id: str,
    adjacency: dict[str, dict[str, tuple[str, str]]],
    edges: dict[tuple[str, str], TokenReasoningEdge],
) -> list[tuple[str, TokenReasoningEdge]]:
    return sorted(
        [(neighbor_id, edges[key]) for neighbor_id, key in adjacency.get(node_id, {}).items()],
        key=lambda item: (int(item[0]) if item[0].lstrip("-").isdigit() else 10**9, item[0]),
    )


def _edge_label_classes(edge: TokenReasoningEdge) -> set[str]:
    return {
        str(item.get("label_class"))
        for item in edge.provenance
        if isinstance(item, dict) and item.get("label_class")
    }


def _edge_relations(edge: TokenReasoningEdge) -> set[str]:
    values: set[str] = set()
    for item in edge.provenance:
        if not isinstance(item, dict):
            continue
        relation = item.get("normalized_relation") or item.get("relation")
        if relation:
            values.add(str(relation).lower())
    return values


def _best_neighbor(
    nodes: dict[str, TokenReasoningNode],
    neighbors: list[tuple[str, TokenReasoningEdge]],
    *,
    allowed_classes: set[str],
    allowed_rel_fragments: set[str],
    kinds: set[str],
) -> str | None:
    candidates: list[tuple[tuple[int, str], float, str]] = []
    for neighbor_id, edge in neighbors:
        neighbor = nodes[neighbor_id]
        if neighbor.kind not in kinds:
            continue
        classes = _edge_label_classes(edge)
        relations = _edge_relations(edge)
        class_match = bool(classes & allowed_classes)
        relation_match = any(fragment in relation for relation in relations for fragment in allowed_rel_fragments)
        if not class_match and not relation_match:
            continue
        candidates.append((_node_sort_key(neighbor), -edge.support, neighbor_id))
    return sorted(candidates)[0][2] if candidates else None


def _has_predicate_object_edge(
    nodes: dict[str, TokenReasoningNode],
    neighbors: list[tuple[str, TokenReasoningEdge]],
) -> bool:
    for neighbor_id, edge in neighbors:
        if nodes[neighbor_id].kind == "content" and _edge_label_classes(edge) & {"CORE_ARG", "UNKNOWN"}:
            return True
    return False


def _nearest_content_to_wh(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    wh_ids: list[str],
) -> str | None:
    if not wh_ids:
        return None
    graph = _weighted_adjacency(nodes, raw_edges, include_scope=False)
    candidates: list[tuple[float, tuple[int, str], str]] = []
    for wh_id in wh_ids:
        for node in nodes.values():
            if node.kind not in {"content", "constraint"}:
                continue
            path, cost = _shortest_path(graph, nodes, wh_id, node.id)
            if path:
                candidates.append((cost, _node_sort_key(node), node.id))
    return sorted(candidates)[0][2] if candidates else None


def _root_candidate(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
) -> str | None:
    for key, edge in sorted(raw_edges.items(), key=lambda item: -item[1].support):
        if "0" not in key:
            continue
        other = key[1] if key[0] == "0" else key[0]
        if nodes[other].kind == "content":
            return other
    return _first_content_node(nodes)


def _first_content_node(nodes: dict[str, TokenReasoningNode]) -> str | None:
    for node in _sorted_nodes(nodes.values()):
        if node.kind == "content":
            return node.id
    return None


def _best_constraint_target(
    nodes: dict[str, TokenReasoningNode],
    neighbors: list[tuple[str, TokenReasoningEdge]],
    answer_anchor_id: str | None,
) -> str | None:
    candidates: list[tuple[int, tuple[int, str], str]] = []
    for neighbor_id, _edge in neighbors:
        neighbor = nodes[neighbor_id]
        if not _is_high_salience_node(neighbor, include_order_constraints=False):
            continue
        priority = 0 if neighbor_id == answer_anchor_id else 1
        candidates.append((priority, _node_sort_key(neighbor), neighbor_id))
    return sorted(candidates)[0][2] if candidates else None


def _weighted_adjacency(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    *,
    include_scope: bool,
) -> dict[str, list[tuple[str, float, tuple[str, str]]]]:
    adjacency: dict[str, list[tuple[str, float, tuple[str, str]]]] = {}
    for key in _sorted_edge_keys(edges, nodes):
        source, target = key
        if "0" in key:
            continue
        if not include_scope and (_is_scope_node(nodes[source]) or _is_scope_node(nodes[target])):
            continue
        edge = edges[key]
        cost = _edge_cost(edge, nodes[source], nodes[target])
        adjacency.setdefault(source, []).append((target, cost, key))
        adjacency.setdefault(target, []).append((source, cost, key))
    for items in adjacency.values():
        items.sort(key=lambda item: (item[1], _node_sort_key(nodes[item[0]]), item[0]))
    return adjacency


def _edge_cost(edge: TokenReasoningEdge, source: TokenReasoningNode, target: TokenReasoningNode) -> float:
    support = max(edge.support, 1e-6)
    cost = 1.0 / support
    if edge.derived:
        for rule, penalty in DERIVED_PENALTIES.items():
            if rule in edge.rule:
                cost += penalty
    cost += 0.01 * abs(source.index - target.index)
    if source.kind == "function" or target.kind == "function":
        cost += 0.20
    return cost


def _shortest_path(
    graph: dict[str, list[tuple[str, float, tuple[str, str]]]],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
) -> tuple[list[str], float]:
    if source_id == target_id:
        return [source_id], 0.0
    import heapq

    start_key = (_path_index_tuple([source_id], nodes),)
    heap: list[tuple[float, int, tuple[int, ...], str, list[str]]] = [(0.0, 1, start_key[0], source_id, [source_id])]
    best: dict[str, tuple[float, int, tuple[int, ...]]] = {source_id: (0.0, 1, start_key[0])}

    while heap:
        cost, length, path_key, node_id, path = heapq.heappop(heap)
        if node_id == target_id:
            return path, cost
        if best.get(node_id) != (cost, length, path_key):
            continue
        for neighbor_id, edge_cost, _edge_key_value in graph.get(node_id, []):
            if neighbor_id in path:
                continue
            next_path = [*path, neighbor_id]
            next_cost = cost + edge_cost
            next_length = length + 1
            next_key = _path_index_tuple(next_path, nodes)
            previous = best.get(neighbor_id)
            candidate = (next_cost, next_length, next_key)
            if previous is None or candidate < previous:
                best[neighbor_id] = candidate
                heapq.heappush(heap, (next_cost, next_length, next_key, neighbor_id, next_path))
    return [], math.inf


def _path_index_tuple(path: list[str], nodes: dict[str, TokenReasoningNode]) -> tuple[int, ...]:
    return tuple(nodes[node_id].index for node_id in path if node_id in nodes)


def _bounded_paths_from_entity(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    graph: dict[str, list[tuple[str, float, tuple[str, str]]]],
    entity_id: str,
    *,
    max_depth: int,
) -> list[tuple[str, list[str]]]:
    del edges
    results: list[tuple[str, list[str]]] = []
    queue: list[list[str]] = [[entity_id]]
    while queue:
        path = queue.pop(0)
        current = path[-1]
        if len(path) > max_depth + 1:
            continue
        if len(path) > 1 and nodes[current].kind in {"content", "answer", "constraint"}:
            interiors = path[1:-1]
            if interiors and all(not nodes[node_id].is_anchor and nodes[node_id].kind != "entity" for node_id in interiors):
                results.append((current, path))
        if len(path) == max_depth + 1:
            continue
        for neighbor_id, _cost, _key in graph.get(current, []):
            if neighbor_id in path:
                continue
            if nodes[neighbor_id].kind == "entity" and neighbor_id != entity_id:
                continue
            queue.append([*path, neighbor_id])
    results.sort(key=lambda item: (len(item[1]), _path_index_tuple(item[1], nodes), item[0]))
    return results


def _path_has_descriptor_evidence(
    path: list[str],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    nodes: dict[str, TokenReasoningNode],
) -> bool:
    del nodes
    first_edge = edges.get(_edge_key(path[0], path[1]))
    if not first_edge:
        return False
    if _edge_label_classes(first_edge) & {"IDENTITY", "RESTRICT"}:
        return True
    for item in first_edge.provenance:
        if not isinstance(item, dict):
            continue
        bridge = str(item.get("bridge") or "").lower()
        if bridge in LIGHT_VERBS:
            return True
        for source_edge in item.get("source_edges") or []:
            if not isinstance(source_edge, dict):
                continue
            for source_provenance in source_edge.get("provenance") or []:
                relation = str(source_provenance.get("relation") or "").lower()
                token = str(source_provenance.get("head") or source_provenance.get("dep") or "").lower()
                if relation in {"cop", "bv"} or token in LIGHT_VERBS:
                    return True
    return False


def _target_reaches_anchor(
    target_id: str,
    source_entity_id: str,
    anchor_targets: set[str],
    graph: dict[str, list[tuple[str, float, tuple[str, str]]]],
) -> bool:
    queue = [target_id]
    visited = {source_entity_id}
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in anchor_targets and node_id != source_entity_id and node_id != target_id:
            return True
        for neighbor_id, _cost, _key in graph.get(node_id, []):
            if neighbor_id not in visited:
                queue.append(neighbor_id)
    return False


def _contract_function_backbone_nodes(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    retained: set[tuple[str, str]],
    terminals: set[str],
) -> set[tuple[str, str]]:
    retained = set(retained)
    changed = True
    while changed:
        changed = False
        adjacency = _plain_adjacency(retained)
        for node_id, neighbors in list(adjacency.items()):
            if node_id in terminals or nodes[node_id].kind != "function" or _is_scope_node(nodes[node_id]):
                continue
            if len(neighbors) != 2:
                continue
            left_id, right_id = sorted(neighbors, key=lambda item: _node_sort_key(nodes[item]))
            retained.discard(_edge_key(left_id, node_id))
            retained.discard(_edge_key(node_id, right_id))
            retained.add(_edge_key(left_id, right_id))
            if _edge_key(left_id, right_id) not in edges:
                left_edge = edges.get(_edge_key(left_id, node_id))
                right_edge = edges.get(_edge_key(node_id, right_id))
                source_edges = [
                    edge.to_dict()
                    for edge in (left_edge, right_edge)
                    if edge is not None
                ]
                support = max(0.05, min((edge.support for edge in (left_edge, right_edge) if edge is not None), default=0.2) * 0.70)
                _merge_edge(
                    edges,
                    nodes,
                    left_id,
                    right_id,
                    support=support,
                    derived=True,
                    rule="function_backbone_contraction",
                    provenance=[
                        {
                            "rule": "function_backbone_contraction",
                            "bridge": nodes[node_id].text,
                            "collapsed_path": [nodes[left_id].text, nodes[node_id].text, nodes[right_id].text],
                            "source_edges": source_edges,
                        }
                    ],
                )
            changed = True
            break
    return retained


def _degree_map(pairs: set[tuple[str, str]]) -> dict[str, int]:
    degree: dict[str, int] = {}
    for source, target in pairs:
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1
    return degree


def _plain_adjacency(pairs: set[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for source, target in pairs:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return adjacency


def _is_simple_connected_path(adjacency: dict[str, set[str]]) -> bool:
    if not adjacency:
        return False
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return False
    start = next(iter(adjacency))
    visited = _component_nodes(adjacency, start)
    return len(visited) == len(adjacency)


def _component_nodes(adjacency: dict[str, set[str]], start_id: str) -> set[str]:
    stack = [start_id]
    visited: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(sorted(adjacency.get(node_id, set()) - visited))
    return visited


def _path_start_node(
    nodes: dict[str, TokenReasoningNode],
    adjacency: dict[str, set[str]],
    entity_ids: list[str],
    answer_anchor_id: str | None,
) -> str:
    entity_candidates = [node_id for node_id in entity_ids if node_id in adjacency]
    if entity_candidates:
        return _sort_node_ids(entity_candidates, nodes)[0]
    endpoints = [node_id for node_id, neighbors in adjacency.items() if len(neighbors) == 1]
    if answer_anchor_id in endpoints:
        endpoints.remove(answer_anchor_id)  # type: ignore[arg-type]
    return _sort_node_ids(endpoints or list(adjacency), nodes)[0]


def _path_target_node(
    nodes: dict[str, TokenReasoningNode],
    adjacency: dict[str, set[str]],
    start_id: str,
    terminals: list[str],
    answer_anchor_id: str | None,
) -> str:
    candidates = [node_id for node_id in terminals if node_id in adjacency and node_id != start_id]
    if not candidates and answer_anchor_id in adjacency:
        candidates = [answer_anchor_id]  # type: ignore[list-item]
    if not candidates:
        candidates = [node_id for node_id, neighbors in adjacency.items() if len(neighbors) == 1 and node_id != start_id]
    scored = []
    for candidate in candidates:
        path = _unweighted_path(adjacency, nodes, start_id, candidate)
        scored.append((-len(path), _node_sort_key(nodes[candidate]), candidate))
    return sorted(scored)[0][2] if scored else start_id


def _unweighted_path(
    adjacency: dict[str, set[str]],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
) -> list[str]:
    queue: list[list[str]] = [[source_id]]
    visited = {source_id}
    while queue:
        path = queue.pop(0)
        current = path[-1]
        if current == target_id:
            return path
        for neighbor_id in sorted(adjacency.get(current, set()), key=lambda item: _node_sort_key(nodes[item])):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            queue.append([*path, neighbor_id])
    return [source_id]


def _walk_root(
    nodes: dict[str, TokenReasoningNode],
    adjacency: dict[str, set[str]],
    entity_ids: list[str],
    answer_anchor_id: str | None,
) -> str:
    entity_candidates = [node_id for node_id in entity_ids if node_id in adjacency]
    if entity_candidates:
        return _sort_node_ids(entity_candidates, nodes)[0]
    if answer_anchor_id in adjacency:
        return answer_anchor_id  # type: ignore[return-value]
    content = [node_id for node_id in adjacency if nodes[node_id].kind in {"content", "answer"}]
    return _sort_node_ids(content or list(adjacency), nodes)[0]


def _dfs_walk(
    nodes: dict[str, TokenReasoningNode],
    adjacency: dict[str, set[str]],
    node_id: str,
    parent_id: str | None,
) -> list[str]:
    walk = [node_id]
    children = [neighbor for neighbor in adjacency.get(node_id, set()) if neighbor != parent_id]
    children.sort(key=lambda child: _child_sort_key(nodes[child]))
    for index, child in enumerate(children):
        walk.extend(_dfs_walk(nodes, adjacency, child, node_id))
        if index != len(children) - 1:
            walk.append(node_id)
    return walk


def _child_sort_key(node: TokenReasoningNode) -> tuple[int, int, str]:
    if node.kind == "entity":
        group = 0
    elif node.kind in {"answer", "constraint"}:
        group = 1
    else:
        group = 2
    return (group, node.index, node.text)


def _path_from_ids(path_id: str, nodes: dict[str, TokenReasoningNode], node_ids: list[str]) -> TokenReasoningPath:
    return TokenReasoningPath(
        path_id=path_id,
        nodes=[nodes[node_id].text for node_id in node_ids if node_id in nodes],
        node_ids=[node_id for node_id in node_ids if node_id in nodes],
    )


def _dedupe_adjacent(node_ids: list[str]) -> list[str]:
    result: list[str] = []
    for node_id in node_ids:
        if result and result[-1] == node_id:
            continue
        result.append(node_id)
    return result


def _node_ids_from_pairs(pairs: set[tuple[str, str]], terminals: list[str]) -> set[str]:
    node_ids = set(terminals)
    for source, target in pairs:
        node_ids.add(source)
        node_ids.add(target)
    return node_ids


def _final_nodes(nodes: dict[str, TokenReasoningNode], node_ids: Iterable[str]) -> list[TokenReasoningNode]:
    return [
        TokenReasoningNode(
            id=node.id,
            text=node.text,
            index=node.index,
            kind=node.kind,
            is_anchor=node.is_anchor,
        )
        for node in _sorted_nodes(nodes[node_id] for node_id in set(node_ids) if node_id in nodes and node_id != "0")
    ]


def _final_edges(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    final_pairs: set[tuple[str, str]],
    paths: list[TokenReasoningPath],
    entity_ids: list[str],
    answer_anchor_id: str | None,
) -> list[TokenReasoningEdge]:
    orientations: dict[tuple[str, str], tuple[str, str]] = {}
    for path in paths:
        for index in range(len(path.node_ids) - 1):
            left = path.node_ids[index]
            right = path.node_ids[index + 1]
            key = _edge_key(left, right)
            orientations.setdefault(key, (left, right))

    root = None
    for node_id in _sort_node_ids(entity_ids, nodes):
        if node_id in _node_ids_from_pairs(final_pairs, []):
            root = node_id
            break
    if root is None and answer_anchor_id in _node_ids_from_pairs(final_pairs, []):
        root = answer_anchor_id
    distances = _graph_distances(_plain_adjacency(final_pairs), root) if root else {}

    final_edges: list[TokenReasoningEdge] = []
    for key in sorted(final_pairs, key=lambda item: _oriented_edge_sort_key(item, orientations, distances, nodes)):
        if key in orientations:
            source, target = orientations[key]
        else:
            source, target = _orient_pair(key, distances, nodes)
        base = edges.get(key)
        if base is None:
            base = TokenReasoningEdge(
                source=source,
                target=target,
                source_text=nodes[source].text,
                target_text=nodes[target].text,
                support=0.0,
            )
        final_edges.append(
            TokenReasoningEdge(
                source=source,
                target=target,
                source_text=nodes[source].text,
                target_text=nodes[target].text,
                support=base.support,
                derived=base.derived,
                rule=base.rule,
                provenance=list(base.provenance),
            )
        )
    return final_edges


def _graph_distances(adjacency: dict[str, set[str]], root: str | None) -> dict[str, int]:
    if root is None:
        return {}
    distances = {root: 0}
    queue = [root]
    while queue:
        node_id = queue.pop(0)
        for neighbor in sorted(adjacency.get(node_id, set())):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node_id] + 1
            queue.append(neighbor)
    return distances


def _orient_pair(
    pair: tuple[str, str],
    distances: dict[str, int],
    nodes: dict[str, TokenReasoningNode],
) -> tuple[str, str]:
    source, target = pair
    source_distance = distances.get(source, 10**9)
    target_distance = distances.get(target, 10**9)
    if source_distance < target_distance:
        return source, target
    if target_distance < source_distance:
        return target, source
    return tuple(sorted(pair, key=lambda node_id: _node_sort_key(nodes[node_id])))  # type: ignore[return-value]


def _oriented_edge_sort_key(
    pair: tuple[str, str],
    orientations: dict[tuple[str, str], tuple[str, str]],
    distances: dict[str, int],
    nodes: dict[str, TokenReasoningNode],
) -> tuple[int, int, int, str, str]:
    source, target = orientations.get(pair) or _orient_pair(pair, distances, nodes)
    return (
        min(distances.get(source, 10**9), distances.get(target, 10**9)),
        nodes[source].index,
        nodes[target].index,
        nodes[source].text,
        nodes[target].text,
    )


def _graph_snapshot(
    node_ids: Iterable[str],
    pairs: set[tuple[str, str]],
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
) -> dict[str, Any]:
    sorted_node_ids = _sort_node_ids([node_id for node_id in node_ids if node_id in nodes and node_id != "0"], nodes)
    return {
        "nodes": [nodes[node_id].to_dict() for node_id in sorted_node_ids],
        "edges": [
            edges[pair].to_dict() if pair in edges else {
                "source": pair[0],
                "target": pair[1],
                "source_text": nodes[pair[0]].text,
                "target_text": nodes[pair[1]].text,
                "support": 0.0,
            }
            for pair in sorted(pairs, key=lambda key: (_node_sort_key(nodes[key[0]]), _node_sort_key(nodes[key[1]])))
            if pair[0] in nodes and pair[1] in nodes
        ],
    }


def _raw_edge_to_dict(edge: HanLPSDPEdge) -> dict[str, Any]:
    return {
        "formalism": edge.formalism,
        "head_idx": edge.head_idx,
        "head": edge.head,
        "relation": edge.relation,
        "dep_idx": edge.dep_idx,
        "dep": edge.dep,
    }


def _safe_filename(value: str | None) -> str:
    if not value:
        return "q1"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "q1"
