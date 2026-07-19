from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from models import HanLPSDPEdge, HanLPSDPResult


EDGE_QUALITY_SCORES = {
    "STRONG": 1.0,
    "MEDIUM": 0.67,
    "WEAK": 0.33,
}

EDGE_QUALITY_RANK = {
    "WEAK": 0,
    "MEDIUM": 1,
    "STRONG": 2,
}

LABEL_CLASS_EDGE_QUALITY = {
    "CORE_ARG": "STRONG",
    "RESTRICT": "STRONG",
    "IDENTITY": "STRONG",
    "COORD": "MEDIUM",
    "MODIFIER": "MEDIUM",
    "BRIDGE": "WEAK",
    "UNKNOWN": "WEAK",
}

STRONG_BRIDGE_TOKENS = {"of", "in", "from", "by", "at", "on", "to", "with", "for", "as"}

ENTITY_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d[\d,]*(?:\.\d+)?|\d{1,4}(?:[-/]\d{1,2}){1,2})%?$")

DETERMINERS = {"a", "an", "the"}
WH_WORDS = {"what", "which", "who", "whom", "whose", "where", "when"}
WH_ANCHOR_WORDS = {"who", "whom", "whose", "what", "which", "when", "where", "why", "how"}
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
SEMANTIC_BOUNDARY_SCOPE_WORDS = SCOPE_WORDS | {"both", "either", "neither"}
FUNCTION_WORDS = DETERMINERS | WH_WORDS | RELATIVE_PRONOUNS | LIGHT_VERBS | PREPOSITIONS | SCOPE_WORDS
POSSESSIVE_MARKER_TOKENS = {"'", "’", "'s", "’s", "s"}
POSSESSIVE_OWNER_RELATIONS = {"poss_arg2"}
POSSESSIVE_POSSESSED_RELATIONS = {"poss_arg1", "adj_arg1", "noun_arg1", "modifier"}
ORDER_CUES = {"first", "earliest", "latest", "last", "older", "oldest", "younger", "youngest"}
APPROX_CUES = {"approximately", "about", "around", "roughly"}

ANSWER_ANCHOR_SOURCE_ORDER = {
    "typed_wh_slot": 0,
    "wh_anchor": 1,
    "root_projection": 2,
    "modifier_projection": 3,
    "comparative_focus": 4,
    "bare_wh_predicate_root": 5,
    "explicit_entity": 6,
    "clause_predicate": 7,
}


@dataclass
class TokenReasoningNode:
    id: str
    text: str
    index: int
    kind: str
    is_anchor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "index": self.index,
            "kind": self.kind,
            "is_anchor": self.is_anchor,
        }


@dataclass
class TokenReasoningEdge:
    source: str
    target: str
    source_text: str
    target_text: str
    support: float = 0.0
    edge_quality: str = "WEAK"
    consensus_count: int = 0
    derived: bool = False
    rule: str = ""
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "support": self.support,
            "edge_quality": self.edge_quality,
            "consensus_count": self.consensus_count,
            "derived": self.derived,
            "rule": self.rule,
            "provenance": [dict(item) if isinstance(item, dict) else item for item in self.provenance],
        }


@dataclass
class TokenReasoningPath:
    path_id: str
    nodes: list[str]
    node_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "nodes": list(self.nodes),
            "node_ids": list(self.node_ids),
        }


@dataclass
class TokenReasoningStructureResult:
    nodes: list[TokenReasoningNode]
    edges: list[TokenReasoningEdge]
    paths: list[TokenReasoningPath]
    path_type: str
    anchor_path_results: list[dict[str, Any]] = field(default_factory=list)
    global_selection: dict[str, Any] = field(default_factory=dict)
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
            "global_selection": dict(self.global_selection),
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


@dataclass
class _GlobalPathCandidate:
    anchor_index: int
    path_index: int
    anchor_id: str
    anchor_text: str
    anchor_source_types: list[str]
    path: TokenReasoningPath
    global_rank: tuple[int, int, int, int]
    global_rank_components: dict[str, Any]
    constraints: list[dict[str, Any]]
    candidate_sets: list[list[str]]
    query_focus: _QueryFocus
    anchor_state: _WorkingState
    path_type: str
    selection_mode: str
    conjunctive_cover: dict[str, Any] = field(default_factory=dict)

    def to_debug(self) -> dict[str, Any]:
        payload = {
            "anchor_index": self.anchor_index,
            "path_index": self.path_index,
            "anchor_id": self.anchor_id,
            "anchor_text": self.anchor_text,
            "source_types": list(self.anchor_source_types),
            "path_id": self.path.path_id,
            "node_ids": list(self.path.node_ids),
            "nodes": list(self.path.nodes),
            "global_rank": list(self.global_rank),
            "global_rank_components": dict(self.global_rank_components),
            "path_type": self.path_type,
            "selection_mode": self.selection_mode,
        }
        if self.conjunctive_cover:
            payload["conjunctive_constraint_cover"] = dict(self.conjunctive_cover)
        return payload


@dataclass
class _GlobalPathSelection:
    selection_type: str
    candidates: list[_GlobalPathCandidate]
    winning_candidate: _GlobalPathCandidate
    candidate_set: list[str] = field(default_factory=list)

    def to_debug(self) -> dict[str, Any]:
        payload = self.winning_candidate.to_debug()
        payload["selection_type"] = self.selection_type
        payload["selected_paths"] = [candidate.to_debug() for candidate in self.candidates]
        if self.selection_type == "global_best_path_cover":
            payload["candidate_set"] = list(self.candidate_set)
            payload["paths"] = [candidate.to_debug() for candidate in self.candidates]
            payload["winning_path"] = self.winning_candidate.to_debug()
        elif self.selection_type == "global_conjunctive_constraint_cover":
            payload["paths"] = [candidate.to_debug() for candidate in self.candidates]
            payload["winning_path"] = self.winning_candidate.to_debug()
            if self.winning_candidate.conjunctive_cover:
                payload["conjunctive_constraint_cover"] = dict(self.winning_candidate.conjunctive_cover)
        return payload


@dataclass(frozen=True)
class _ConjunctiveConstraintCover:
    cover_id: str
    predicate_id: str
    target_id: str
    role_family: str
    marker_id: str | None
    member_head_ids: tuple[str, ...]
    branch_entity_ids: tuple[str, ...]
    branch_paths: tuple[tuple[str, ...], ...]
    evidence: dict[str, Any]

    def to_debug(self, nodes: dict[str, TokenReasoningNode]) -> dict[str, Any]:
        return {
            "cover_id": self.cover_id,
            "predicate_id": self.predicate_id,
            "predicate": nodes[self.predicate_id].text if self.predicate_id in nodes else None,
            "target_id": self.target_id,
            "target": nodes[self.target_id].text if self.target_id in nodes else None,
            "role_family": self.role_family,
            "marker_id": self.marker_id,
            "marker": nodes[self.marker_id].text if self.marker_id in nodes else None,
            "member_head_ids": list(self.member_head_ids),
            "member_heads": [nodes[node_id].text for node_id in self.member_head_ids if node_id in nodes],
            "branch_entity_ids": list(self.branch_entity_ids),
            "branch_entities": [nodes[node_id].text for node_id in self.branch_entity_ids if node_id in nodes],
            "branch_paths": [list(path) for path in self.branch_paths],
            "branch_path_nodes": [
                [nodes[node_id].text for node_id in path if node_id in nodes]
                for path in self.branch_paths
            ],
            "evidence": dict(self.evidence),
        }


def compile_token_reasoning_structure(
    hanlp_sdp_result: HanLPSDPResult,
    explicit_entities: list[str],
    *,
    masked_question: str | None = None,
    original_question: str | None = None,
    normalized_question: str | None = None,
    normalization_changed: bool | None = None,
    normalization_note: str | None = None,
    question_id: str | None = None,
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> TokenReasoningStructureResult:
    """Compile three HanLP SDP graph views into entity-branch token paths.

    Step 4 no longer searches over answer-anchor candidates.  Explicit masked
    entities are fixed branch starts; each branch keeps the best Dijkstra path
    to one semantic boundary node under the existing path ranking components.
    """

    state = build_evidence_graph(hanlp_sdp_result)
    explicit_entity_ids = _resolve_explicit_entity_ids(state.nodes, explicit_entities)
    _mark_anchors(state.nodes, explicit_entity_ids, None)

    add_possessive_marker_contraction_edges(state)
    add_bridge_contraction_edges(state)

    branch_selection = _select_entity_branch_best_paths(state, explicit_entity_ids)
    paths = branch_selection["paths"]
    path_type = "entity_branch_best_paths" if paths else "no_entity_branch_path"
    constraints: list[dict[str, Any]] = []
    candidate_sets: list[list[str]] = []
    answer_anchor = None
    answer_anchor_id = None

    final_node_ids, final_pairs = _graph_from_selected_paths(paths)
    backbone_before = _graph_snapshot(final_node_ids, final_pairs, state.nodes, state.edges)
    backbone_after = backbone_before
    final_nodes = _final_nodes(state.nodes, final_node_ids)
    active_entity_ids = _active_entity_ids(state.nodes, paths)
    final_edges = _final_edges(state.nodes, state.edges, final_pairs, paths, active_entity_ids, answer_anchor_id)

    entity_anchors = [state.nodes[node_id].text for node_id in active_entity_ids if node_id in state.nodes]
    result_warnings = _unique_warnings(state.warnings)
    global_selection = _entity_branch_global_selection_payload(
        state,
        path_type=path_type,
        branch_selection=branch_selection,
    )
    debug_payload = _build_debug_payload(
        question_id=question_id,
        masked_question=masked_question or hanlp_sdp_result.text,
        original_question=original_question or hanlp_sdp_result.text,
        normalized_question=normalized_question or masked_question or hanlp_sdp_result.text,
        normalization_changed=bool(normalization_changed),
        normalization_note=normalization_note or "",
        hanlp_sdp_result=hanlp_sdp_result,
        explicit_entities=explicit_entities,
        state=state,
        answer_anchor_id=answer_anchor_id,
        entity_ids=explicit_entity_ids,
        constraints=constraints,
        candidate_sets=candidate_sets,
        terminals=branch_selection["boundary_ids"],
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
        selection_mode=path_type,
    )
    debug_payload["step4_path_extraction"] = "entity_branch_best_paths"
    debug_payload["semantic_boundary_nodes"] = list(global_selection["semantic_boundary_nodes"])
    debug_payload["semantic_node_ids"] = list(branch_selection["semantic_node_ids"])
    debug_payload["semantic_nodes"] = [
        state.nodes[node_id].text for node_id in branch_selection["semantic_node_ids"] if node_id in state.nodes
    ]
    debug_payload["entity_branch_results"] = list(global_selection["entity_branch_results"])
    debug_payload["answer_anchor_candidates"] = []
    debug_payload["global_candidates"] = []
    debug_payload["selected_global_candidate"] = None
    debug_payload["selected_global_candidates"] = []
    debug_payload["selected_global_selection"] = global_selection
    debug_payload["warnings"] = result_warnings
    debug_file = None
    if debug:
        debug_file = write_debug_json(debug_payload, question_id=question_id, debug_dir=debug_dir)

    return TokenReasoningStructureResult(
        nodes=final_nodes,
        edges=final_edges,
        paths=paths,
        path_type=path_type,
        anchor_path_results=[],
        global_selection=global_selection,
        answer_anchor=answer_anchor,
        answer_anchor_id=answer_anchor_id,
        entity_anchors=entity_anchors,
        constraints=constraints,
        candidate_sets=candidate_sets,
        warnings=result_warnings,
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
        edge_quality = LABEL_CLASS_EDGE_QUALITY[label_class]
        support = EDGE_QUALITY_SCORES[edge_quality]
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
            "edge_quality": edge_quality,
            "consensus_count": 1,
            "derived": False,
            "rule": "raw_evidence",
            "support": support,
        }
        normalized_edges.append(provenance)
        _merge_edge(
            raw_edges,
            nodes,
            source_id,
            target_id,
            support=support,
            edge_quality=edge_quality,
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


def _select_entity_branch_best_paths(
    state: _WorkingState,
    explicit_entity_ids: list[str],
) -> dict[str, Any]:
    boundary_degree_graph = _semantic_boundary_degree_graph(state.nodes, state.edges)
    boundary_ids = _semantic_boundary_node_ids(state.nodes, boundary_degree_graph)
    semantic_node_ids = _semantic_degree_node_ids(state.nodes, boundary_degree_graph)
    search_graph = _semantic_path_search_graph(state.nodes, state.edges)

    if not explicit_entity_ids:
        state.warnings.append("entity branch path extraction found no explicit entity starts")
    if not boundary_ids:
        state.warnings.append("entity branch path extraction found no semantic boundary nodes")

    branch_results: list[dict[str, Any]] = []
    selected_paths: list[TokenReasoningPath] = []
    selected_payloads: list[dict[str, Any]] = []
    selected_node_paths: set[tuple[str, ...]] = set()

    for entity_id in _sort_node_ids(explicit_entity_ids, state.nodes):
        if entity_id not in state.nodes:
            continue
        blocked_internal_ids = set(explicit_entity_ids) - {entity_id}
        candidates: list[dict[str, Any]] = []
        for boundary_id in boundary_ids:
            if boundary_id == entity_id:
                continue
            path_ids, dijkstra_cost = _shortest_semantic_boundary_path(
                search_graph,
                state.nodes,
                entity_id,
                boundary_id,
                blocked_internal_ids=blocked_internal_ids,
            )
            if not path_ids:
                continue
            rank, rank_components = _rank_entity_branch_boundary_path(
                state,
                path_ids,
                semantic_node_ids,
            )
            candidates.append(
                {
                    "entity_id": entity_id,
                    "entity": state.nodes[entity_id].text,
                    "boundary_id": boundary_id,
                    "boundary": state.nodes[boundary_id].text,
                    "node_ids": list(path_ids),
                    "nodes": [state.nodes[node_id].text for node_id in path_ids if node_id in state.nodes],
                    "dijkstra_cost": dijkstra_cost,
                    "rank": rank,
                    "rank_components": rank_components,
                }
            )

        candidates.sort(key=lambda item: item["rank"])
        selected = dict(candidates[0]) if candidates else None
        if selected is None:
            state.warnings.append(
                f"entity branch path extraction found no reachable semantic boundary for {state.nodes[entity_id].text}[{entity_id}]"
            )
        else:
            path_key = tuple(selected["node_ids"])
            if path_key not in selected_node_paths:
                selected_node_paths.add(path_key)
                path_id = f"P{len(selected_paths) + 1}"
                selected_path = _path_from_ids(path_id, state.nodes, list(selected["node_ids"]))
                selected_paths.append(selected_path)
                selected["path_id"] = path_id
                selected["path"] = selected_path.to_dict()
                selected_payloads.append(_entity_branch_selected_path_payload(selected))
            else:
                selected["path_id"] = None
                selected["path"] = None

        branch_results.append(
            {
                "entity_id": entity_id,
                "entity": state.nodes[entity_id].text,
                "candidate_count": len(candidates),
                "candidates": [_entity_branch_candidate_payload(candidate) for candidate in candidates],
                "selected": _entity_branch_candidate_payload(selected) if selected is not None else None,
            }
        )

    return {
        "paths": selected_paths,
        "selected_paths": selected_payloads,
        "boundary_ids": boundary_ids,
        "semantic_node_ids": semantic_node_ids,
        "boundary_degree_graph": {
            node_id: _sort_node_ids(neighbors, state.nodes)
            for node_id, neighbors in sorted(boundary_degree_graph.items(), key=lambda item: _node_sort_key(state.nodes[item[0]]))
        },
        "entity_branch_results": branch_results,
    }


def _semantic_boundary_degree_graph(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for key in _sorted_edge_keys(edges, nodes):
        source_id, target_id = key
        if source_id not in nodes or target_id not in nodes:
            continue
        source = nodes[source_id]
        target = nodes[target_id]
        if not _semantic_degree_node_allowed(source) or not _semantic_degree_node_allowed(target):
            continue
        edge = edges[key]
        if _edge_is_pure_coordination(edge):
            continue
        graph.setdefault(source_id, set()).add(target_id)
        graph.setdefault(target_id, set()).add(source_id)
    return graph


def _semantic_path_search_graph(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
) -> dict[str, list[tuple[str, float, tuple[str, str]]]]:
    graph: dict[str, list[tuple[str, float, tuple[str, str]]]] = {}
    for key in _sorted_edge_keys(edges, nodes):
        source_id, target_id = key
        if source_id not in nodes or target_id not in nodes:
            continue
        source = nodes[source_id]
        target = nodes[target_id]
        if not _semantic_search_node_allowed(source) or not _semantic_search_node_allowed(target):
            continue
        edge = edges[key]
        if _edge_is_pure_coordination(edge):
            continue
        cost = _edge_cost(edge, source, target)
        graph.setdefault(source_id, []).append((target_id, cost, key))
        graph.setdefault(target_id, []).append((source_id, cost, key))
    for items in graph.values():
        items.sort(key=lambda item: (item[1], _node_sort_key(nodes[item[0]]), item[0]))
    return graph


def _semantic_boundary_node_ids(
    nodes: dict[str, TokenReasoningNode],
    boundary_degree_graph: dict[str, set[str]],
) -> list[str]:
    return _sort_node_ids(
        [
            node_id
            for node_id, neighbors in boundary_degree_graph.items()
            if node_id in nodes
            and len(neighbors) == 1
            and _is_semantic_boundary_node(nodes[node_id])
        ],
        nodes,
    )


def _semantic_degree_node_ids(
    nodes: dict[str, TokenReasoningNode],
    boundary_degree_graph: dict[str, set[str]],
) -> list[str]:
    return _sort_node_ids(
        [
            node_id
            for node_id, neighbors in boundary_degree_graph.items()
            if node_id in nodes
            and neighbors
            and _is_semantic_boundary_node(nodes[node_id])
        ],
        nodes,
    )


def _semantic_degree_node_allowed(node: TokenReasoningNode) -> bool:
    lower = node.text.lower()
    if node.id == "0" or node.index <= 0 or lower == "root":
        return False
    if _is_punctuation(node.text):
        return False
    if lower in SEMANTIC_BOUNDARY_SCOPE_WORDS:
        return False
    if lower in DETERMINERS or lower in PREPOSITIONS or lower in LIGHT_VERBS or lower in RELATIVE_PRONOUNS:
        return False
    if node.kind == "function" and lower not in WH_ANCHOR_WORDS and node.kind != "entity":
        return False
    return True


def _semantic_search_node_allowed(node: TokenReasoningNode) -> bool:
    lower = node.text.lower()
    if node.id == "0" or node.index <= 0 or lower == "root":
        return False
    if _is_punctuation(node.text):
        return False
    if lower in SEMANTIC_BOUNDARY_SCOPE_WORDS:
        return False
    return True


def _is_semantic_boundary_node(node: TokenReasoningNode) -> bool:
    lower = node.text.lower()
    if node.kind == "entity" or ENTITY_RE.fullmatch(node.text):
        return False
    if lower in WH_ANCHOR_WORDS:
        return True
    if not _semantic_degree_node_allowed(node):
        return False
    return node.kind in {"content", "constraint", "answer"}


def _edge_is_pure_coordination(edge: TokenReasoningEdge) -> bool:
    classes = _edge_label_classes_deep(edge)
    if not classes:
        classes = _edge_label_classes(edge)
    return bool(classes) and all(label_class == "COORD" for label_class in classes)


def _shortest_semantic_boundary_path(
    graph: dict[str, list[tuple[str, float, tuple[str, str]]]],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
    *,
    blocked_internal_ids: set[str],
) -> tuple[list[str], float]:
    if source_id == target_id:
        return [], math.inf
    if source_id not in graph or target_id not in graph:
        return [], math.inf

    import heapq

    start_key = _path_index_tuple([source_id], nodes)
    heap: list[tuple[float, int, tuple[int, ...], str, list[str]]] = [
        (0.0, 0, start_key, source_id, [source_id])
    ]
    best: dict[str, tuple[float, int, tuple[int, ...]]] = {source_id: (0.0, 0, start_key)}

    while heap:
        cost, edge_count, path_key, node_id, path = heapq.heappop(heap)
        if best.get(node_id) != (cost, edge_count, path_key):
            continue
        if node_id == target_id:
            return path, cost
        for neighbor_id, edge_cost, _edge_key_value in graph.get(node_id, []):
            if neighbor_id in path:
                continue
            if neighbor_id in blocked_internal_ids and neighbor_id != target_id:
                continue
            next_path = [*path, neighbor_id]
            next_cost = cost + edge_cost
            next_edge_count = edge_count + 1
            next_key = _path_index_tuple(next_path, nodes)
            candidate = (next_cost, next_edge_count, next_key)
            previous = best.get(neighbor_id)
            if previous is None or candidate < previous:
                best[neighbor_id] = candidate
                heapq.heappush(heap, (next_cost, next_edge_count, next_key, neighbor_id, next_path))
    return [], math.inf


def _rank_entity_branch_boundary_path(
    state: _WorkingState,
    path_ids: list[str],
    semantic_node_ids: list[str],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    path_set = set(path_ids)
    semantic_ids = _sort_node_ids([node_id for node_id in semantic_node_ids if node_id in state.nodes], state.nodes)
    covered_semantic_ids = _sort_node_ids([node_id for node_id in semantic_ids if node_id in path_set], state.nodes)
    missing_semantic_ids = _sort_node_ids([node_id for node_id in semantic_ids if node_id not in path_set], state.nodes)
    weak_edge_count = 0
    medium_edge_count = 0
    derived_edge_count = 0
    strong_edge_count = 0
    consensus_count = 0
    missing_edge_pairs: list[dict[str, Any]] = []

    for left, right in zip(path_ids, path_ids[1:]):
        edge = state.edges.get(_edge_key(left, right))
        if edge is None:
            weak_edge_count += 1
            missing_edge_pairs.append(
                {
                    "source": left,
                    "target": right,
                    "source_text": state.nodes[left].text if left in state.nodes else left,
                    "target_text": state.nodes[right].text if right in state.nodes else right,
                }
            )
            continue
        quality = _normalize_edge_quality(edge.edge_quality)
        if quality == "STRONG":
            strong_edge_count += 1
        elif quality == "MEDIUM":
            medium_edge_count += 1
        else:
            weak_edge_count += 1
        if edge.derived:
            derived_edge_count += 1
        consensus_count += edge.consensus_count

    function_node_count = sum(
        1 for node_id in path_ids if node_id in state.nodes and state.nodes[node_id].kind == "function"
    )
    dirty_path_count = weak_edge_count + medium_edge_count + function_node_count + derived_edge_count
    path_length = max(0, len(path_ids) - 1)
    token_index_sequence = _path_index_tuple(path_ids, state.nodes)
    rank = (
        len(missing_semantic_ids),
        dirty_path_count,
        path_length,
        token_index_sequence,
    )
    components: dict[str, Any] = {
        "semantic_node_ids": semantic_ids,
        "semantic_nodes": [state.nodes[node_id].text for node_id in semantic_ids if node_id in state.nodes],
        "covered_semantic_node_ids": covered_semantic_ids,
        "covered_semantic_nodes": [
            state.nodes[node_id].text for node_id in covered_semantic_ids if node_id in state.nodes
        ],
        "missing_semantic_node_ids": missing_semantic_ids,
        "missing_semantic_nodes": [
            state.nodes[node_id].text for node_id in missing_semantic_ids if node_id in state.nodes
        ],
        "missing_semantic_nodes_count": len(missing_semantic_ids),
        "weak_edge_count": weak_edge_count,
        "medium_edge_count": medium_edge_count,
        "function_node_count": function_node_count,
        "derived_edge_count": derived_edge_count,
        "dirty_path_count": dirty_path_count,
        "path_length": path_length,
        "path_token_index_sequence": list(token_index_sequence),
        "strong_edge_count": strong_edge_count,
        "consensus_count": consensus_count,
        "rank": _jsonable_rank(rank),
    }
    if missing_edge_pairs:
        components["missing_edge_pairs"] = missing_edge_pairs
    return rank, components


def _entity_branch_candidate_payload(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    payload = {
        "entity_id": candidate["entity_id"],
        "entity": candidate["entity"],
        "boundary_id": candidate["boundary_id"],
        "boundary": candidate["boundary"],
        "node_ids": list(candidate["node_ids"]),
        "nodes": list(candidate["nodes"]),
        "dijkstra_cost": candidate["dijkstra_cost"],
        "rank": _jsonable_rank(candidate["rank"]),
        "rank_components": dict(candidate["rank_components"]),
    }
    if "path_id" in candidate:
        payload["path_id"] = candidate["path_id"]
    if "path" in candidate:
        payload["path"] = candidate["path"]
    return payload


def _entity_branch_selected_path_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = _entity_branch_candidate_payload(candidate) or {}
    payload["global_rank"] = payload.get("rank", [])
    return payload


def _entity_branch_global_selection_payload(
    state: _WorkingState,
    *,
    path_type: str,
    branch_selection: dict[str, Any],
) -> dict[str, Any]:
    boundary_ids = branch_selection["boundary_ids"]
    semantic_node_ids = branch_selection["semantic_node_ids"]
    return {
        "selection_type": path_type,
        "path_type": path_type,
        "anchor_id": None,
        "anchor_text": None,
        "source_types": [],
        "semantic_boundary_node_ids": list(boundary_ids),
        "semantic_boundary_nodes": [
            {
                "id": node_id,
                "text": state.nodes[node_id].text,
                "index": state.nodes[node_id].index,
                "kind": state.nodes[node_id].kind,
                "degree": len(branch_selection["boundary_degree_graph"].get(node_id, [])),
            }
            for node_id in boundary_ids
            if node_id in state.nodes
        ],
        "semantic_node_ids": list(semantic_node_ids),
        "semantic_nodes": [state.nodes[node_id].text for node_id in semantic_node_ids if node_id in state.nodes],
        "entity_branch_results": list(branch_selection["entity_branch_results"]),
        "paths": list(branch_selection["selected_paths"]),
        "selected_paths": list(branch_selection["selected_paths"]),
        "warnings": list(state.warnings),
    }


def _jsonable_rank(rank: tuple[Any, ...]) -> list[Any]:
    result: list[Any] = []
    for item in rank:
        if isinstance(item, tuple):
            result.append(list(item))
        else:
            result.append(item)
    return result


def detect_answer_anchor(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    warnings: list[str],
) -> str | None:
    del nodes, raw_edges
    warnings.append("answer anchor selection is disabled in Step4 entity-branch mode")
    return None


def collect_answer_anchor_candidates(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    warnings: list[str] | None = None,
    *,
    explicit_entity_ids: list[str] | None = None,
) -> list[_AnswerAnchorCandidate]:
    """Disabled: Step 4 now starts from explicit entity branches."""

    del nodes, raw_edges, explicit_entity_ids
    if warnings is not None:
        warnings.append("answer anchor candidate collection is disabled in Step4 entity-branch mode")
    return []


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
                        "edge": _edge_provenance_summary(edge),
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


def _find_wh_anchor_candidates(
    nodes: dict[str, TokenReasoningNode],
) -> list[_AnswerAnchorCandidate]:
    candidates: list[_AnswerAnchorCandidate] = []
    for node in _sorted_nodes(nodes.values()):
        if node.id == "0" or node.text.lower() not in WH_ANCHOR_WORDS:
            continue
        candidates.append(
            _answer_anchor_candidate(
                nodes,
                node.id,
                "wh_anchor",
                {
                    "rule": "wh_token_anchor",
                    "wh_id": node.id,
                    "wh": node.text,
                    "support": 1.0,
                },
                1.0,
            )
        )
    return candidates


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
                    "edge": _edge_provenance_summary(edge),
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
    anchor_source_weights = {
        "typed_wh_slot": 16.0,
        "wh_anchor": 15.0,
        "root_projection": 14.0,
        "modifier_projection": 12.0,
        "comparative_focus": 10.0,
        "bare_wh_predicate_root": 10.0,
        "clause_predicate": 8.0,
        "explicit_entity": 2.0,
    }
    source_bonus = sum(anchor_source_weights.get(source_type, 1.0) for source_type in set(source_types))
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
        evidence.append({"rule": rule, "edge": _edge_provenance_summary(edge), "support": edge.support})
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
                provenance = {
                    "rule": "bridge_contraction",
                    "bridge": bridge.text,
                    "bridge_id": bridge.id,
                    "collapsed_path": [
                        state.nodes[left_id].text,
                        bridge.text,
                        state.nodes[right_id].text,
                    ],
                    "source_edges": _edge_provenance_summaries([left_edge, right_edge]),
                }
                edge_quality = _infer_edge_quality(state.nodes, left_id, right_id, "bridge_contraction", [provenance])
                support = EDGE_QUALITY_SCORES[edge_quality]
                provenance["edge_quality"] = edge_quality
                provenance["derived"] = True
                provenance["support"] = support
                virtual = _merge_edge(
                    state.edges,
                    state.nodes,
                    left_id,
                    right_id,
                    support=support,
                    edge_quality=edge_quality,
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
                edge_quality = "STRONG"
                support = EDGE_QUALITY_SCORES[edge_quality]
                provenance = {
                    "rule": "possessive_marker_contraction",
                    "edge_quality": edge_quality,
                    "derived": True,
                    "marker": marker.text,
                    "marker_id": marker.id,
                    "collapsed_path": [
                        state.nodes[owner_id].text,
                        marker.text,
                        state.nodes[possessed_id].text,
                    ],
                    "source_edges": _edge_provenance_summaries([owner_edge, possessed_edge]),
                    "support": support,
                }
                virtual = _merge_edge(
                    state.edges,
                    state.nodes,
                    owner_id,
                    possessed_id,
                    support=support,
                    edge_quality=edge_quality,
                    derived=True,
                    rule="possessive_marker_contraction",
                    provenance=[provenance],
                )
                state.virtual_edges.append(virtual.to_dict())


def add_candidate_typed_slot_instantiation_edges(
    state: _WorkingState,
    query_focus: _QueryFocus,
    direct_candidate_sets: list[list[str]],
) -> None:
    if query_focus.mode != "typed_wh_slot" or not query_focus.slot_id:
        return
    for candidate_set in direct_candidate_sets:
        candidate_ids = _candidate_text_set_to_ids(state.nodes, candidate_set)
        if len(candidate_ids) < 2:
            continue
        for candidate_id in candidate_ids:
            _ensure_candidate_typed_slot_instantiation_edge(
                state,
                candidate_id,
                query_focus.slot_id,
                candidate_set,
            )


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
) -> tuple[list[TokenReasoningPath], str, list[_CandidatePath], str, dict[str, Any]]:
    candidate_records: list[_CandidatePath] = []

    parallel_paths = _extract_actual_parallel_path_cover(
        state,
        query_focus,
        constraints,
        parallel_entity_sets,
    )
    if parallel_paths:
        paths, selected_records, all_records = parallel_paths
        candidate_records.extend(all_records)
        return paths, "candidate_path_cover", candidate_records, "parallel_entity_paths", {}

    typed_paths = _extract_typed_slot_candidate_path_cover(
        state,
        query_focus,
        constraints,
        direct_candidate_sets,
    )
    if typed_paths:
        paths, selected_records = typed_paths
        candidate_records.extend(selected_records)
        return paths, "candidate_path_cover", candidate_records, "candidate_slot_substitution", {}

    bare_wh_paths = _extract_bare_wh_candidate_path_cover(
        state,
        query_focus,
        direct_candidate_sets,
    )
    if bare_wh_paths:
        paths, selected_records = bare_wh_paths
        candidate_records.extend(selected_records)
        return paths, "candidate_path_cover", candidate_records, "candidate_bare_wh_substitution", {}

    conjunctive_covers = detect_conjunctive_constraint_covers(
        state,
        explicit_entity_ids,
        query_focus,
        direct_candidate_sets=direct_candidate_sets,
    )
    if conjunctive_covers:
        add_conjunctive_constraint_edges(state, conjunctive_covers)
        conjunctive_paths = _select_conjunctive_constraint_path_cover(
            state,
            query_focus,
            conjunctive_covers,
        )
        if conjunctive_paths:
            paths, selected_records, cover = conjunctive_paths
            candidate_records.extend(selected_records)
            return (
                paths,
                "conjunctive_constraint_path_cover",
                candidate_records,
                "conjunctive_constraint_path_cover",
                cover.to_debug(state.nodes),
            )

    single_path, path_type, selected_record, all_records = _select_single_main_path(
        state,
        explicit_entity_ids,
        query_focus,
        conjunctive_covers=conjunctive_covers,
    )
    candidate_records.extend(all_records)
    if selected_record:
        candidate_records = [
            record.with_selection(selected=record == selected_record, rejected_reason="" if record == selected_record else "lower ranked")
            for record in candidate_records
        ]
    if single_path:
        return [single_path], path_type, candidate_records, path_type, {}

    fallback_id = _fallback_single_node_id(state.nodes, explicit_entity_ids, query_focus)
    if fallback_id:
        path = _path_from_ids("P1", state.nodes, [fallback_id])
        return [path], "empty", candidate_records, "single_node_fallback", {}
    return [], "empty", candidate_records, "empty", {}


def _build_global_path_candidate(
    *,
    anchor_index: int,
    path_index: int,
    anchor: _AnswerAnchorCandidate,
    path: TokenReasoningPath,
    question_semantic_node_ids: set[str],
    anchor_state: _WorkingState,
    constraints: list[dict[str, Any]],
    candidate_sets: list[list[str]],
    query_focus: _QueryFocus,
    path_type: str,
    selection_mode: str,
    conjunctive_cover: dict[str, Any],
    warnings: list[str],
) -> _GlobalPathCandidate:
    global_rank, components = _rank_global_path_candidate(
        path,
        anchor,
        anchor_state,
        question_semantic_node_ids,
        warnings=warnings,
    )
    return _GlobalPathCandidate(
        anchor_index=anchor_index,
        path_index=path_index,
        anchor_id=anchor.node_id,
        anchor_text=anchor.text,
        anchor_source_types=list(anchor.source_types),
        path=path,
        global_rank=global_rank,
        global_rank_components=components,
        constraints=[dict(item) for item in constraints],
        candidate_sets=[list(item) for item in candidate_sets],
        query_focus=query_focus,
        anchor_state=anchor_state,
        path_type=path_type,
        selection_mode=selection_mode,
        conjunctive_cover=dict(conjunctive_cover),
    )


def _rank_global_path_candidate(
    path: TokenReasoningPath,
    anchor: _AnswerAnchorCandidate,
    state: _WorkingState,
    question_semantic_node_ids: set[str],
    *,
    warnings: list[str],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    covered_node_ids = set(path.node_ids)
    missing_semantic_node_ids = set(question_semantic_node_ids) - covered_node_ids
    weak_edge_count = 0
    medium_edge_count = 0
    derived_edge_count = 0
    missing_edge_pairs: list[dict[str, Any]] = []

    for left, right in zip(path.node_ids, path.node_ids[1:]):
        edge = state.edges.get(_edge_key(left, right))
        if edge is None:
            weak_edge_count += 1
            missing_edge_pair = {
                "source": left,
                "target": right,
                "source_text": state.nodes[left].text if left in state.nodes else left,
                "target_text": state.nodes[right].text if right in state.nodes else right,
            }
            missing_edge_pairs.append(missing_edge_pair)
            warning = (
                "global path rank missing edge for "
                f"{missing_edge_pair['source_text']}[{left}] -- {missing_edge_pair['target_text']}[{right}] "
                f"on {path.path_id}; counted as WEAK"
            )
            if warning not in warnings:
                warnings.append(warning)
            if warning not in state.warnings:
                state.warnings.append(warning)
            continue
        quality = _normalize_edge_quality(edge.edge_quality)
        if quality == "WEAK":
            weak_edge_count += 1
        elif quality == "MEDIUM":
            medium_edge_count += 1
        if edge.derived:
            derived_edge_count += 1

    function_node_count = sum(
        1 for node_id in path.node_ids if node_id in state.nodes and state.nodes[node_id].kind == "function"
    )
    dirty_path_count = weak_edge_count + medium_edge_count + function_node_count + derived_edge_count
    path_length = max(0, len(path.node_ids) - 1)
    fallback_penalty = max(ANSWER_ANCHOR_SOURCE_ORDER.values(), default=0) + 1
    anchor_fit_penalty = min(
        (ANSWER_ANCHOR_SOURCE_ORDER.get(source_type, fallback_penalty) for source_type in anchor.source_types),
        default=fallback_penalty,
    )
    global_rank = (
        len(missing_semantic_node_ids),
        dirty_path_count,
        path_length,
        anchor_fit_penalty,
    )
    semantic_ids = _sort_node_ids(question_semantic_node_ids, state.nodes)
    covered_semantic_ids = _sort_node_ids(question_semantic_node_ids & covered_node_ids, state.nodes)
    missing_semantic_ids = _sort_node_ids(missing_semantic_node_ids, state.nodes)
    components = {
        "semantic_node_ids": semantic_ids,
        "semantic_nodes": [state.nodes[node_id].text for node_id in semantic_ids if node_id in state.nodes],
        "covered_semantic_node_ids": covered_semantic_ids,
        "covered_semantic_nodes": [
            state.nodes[node_id].text for node_id in covered_semantic_ids if node_id in state.nodes
        ],
        "missing_semantic_node_ids": missing_semantic_ids,
        "missing_semantic_nodes": [
            state.nodes[node_id].text for node_id in missing_semantic_ids if node_id in state.nodes
        ],
        "missing_semantic_nodes_count": len(missing_semantic_node_ids),
        "weak_edge_count": weak_edge_count,
        "medium_edge_count": medium_edge_count,
        "function_node_count": function_node_count,
        "derived_edge_count": derived_edge_count,
        "dirty_path_count": dirty_path_count,
        "path_length": path_length,
        "anchor_fit_penalty": anchor_fit_penalty,
        "rank": list(global_rank),
    }
    if missing_edge_pairs:
        components["missing_edge_pairs"] = missing_edge_pairs
    return global_rank, components


def _select_global_best_path(candidates: list[_GlobalPathCandidate]) -> _GlobalPathCandidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item.global_rank,
            item.anchor_index,
            item.path_index,
            tuple(item.path.node_ids),
        ),
    )


def _select_global_path_structure(
    candidates: list[_GlobalPathCandidate],
    *,
    warnings: list[str],
) -> _GlobalPathSelection | None:
    cover_selection = _select_global_best_path_cover(candidates)
    if cover_selection is not None:
        return cover_selection

    conjunctive_selection = _select_global_conjunctive_constraint_cover(candidates)
    if conjunctive_selection is not None:
        return conjunctive_selection

    if any(candidate.path_type == "candidate_path_cover" for candidate in candidates):
        warning = "global path cover selection found no complete candidate set; falling back to single global best path"
        if warning not in warnings:
            warnings.append(warning)
        for candidate in candidates:
            if warning not in candidate.anchor_state.warnings:
                candidate.anchor_state.warnings.append(warning)

    selected = _select_global_best_path(candidates)
    if selected is None:
        return None
    return _GlobalPathSelection(
        selection_type="global_best_path",
        candidates=[selected],
        winning_candidate=selected,
    )


def _select_global_conjunctive_constraint_cover(
    candidates: list[_GlobalPathCandidate],
) -> _GlobalPathSelection | None:
    cover_candidates = [
        candidate
        for candidate in candidates
        if candidate.path_type == "conjunctive_constraint_path_cover"
        and candidate.conjunctive_cover
        and candidate.conjunctive_cover.get("cover_id")
    ]
    if not cover_candidates:
        return None

    groups: dict[tuple[int, str], list[_GlobalPathCandidate]] = {}
    for candidate in cover_candidates:
        cover_id = str(candidate.conjunctive_cover.get("cover_id"))
        groups.setdefault((candidate.anchor_index, cover_id), []).append(candidate)

    complete_groups: list[list[_GlobalPathCandidate]] = []
    for group in groups.values():
        expected_count = len(group[0].conjunctive_cover.get("member_head_ids") or [])
        if expected_count < 2:
            expected_count = len(group[0].conjunctive_cover.get("branch_entity_ids") or [])
        unique_paths = {(candidate.anchor_index, candidate.path_index) for candidate in group}
        if expected_count >= 2 and len(unique_paths) >= expected_count:
            complete_groups.append(sorted(group, key=lambda item: item.path_index))

    if not complete_groups:
        return None

    candidate_pool = [candidate for group in complete_groups for candidate in group]
    winning_seed = _select_global_best_path(candidate_pool)
    if winning_seed is None:
        return None

    winning_group = next(
        (
            group
            for group in complete_groups
            if any(candidate is winning_seed for candidate in group)
        ),
        None,
    )
    if winning_group is None:
        return None

    return _GlobalPathSelection(
        selection_type="global_conjunctive_constraint_cover",
        candidates=winning_group,
        winning_candidate=winning_seed,
    )


def _renumber_selected_global_paths(candidates: list[_GlobalPathCandidate]) -> list[TokenReasoningPath]:
    return [
        TokenReasoningPath(
            path_id=f"P{index}",
            nodes=list(candidate.path.nodes),
            node_ids=list(candidate.path.node_ids),
        )
        for index, candidate in enumerate(candidates, start=1)
    ]


def _select_global_best_path_cover(candidates: list[_GlobalPathCandidate]) -> _GlobalPathSelection | None:
    complete_covers = _complete_global_cover_groups(candidates)
    if not complete_covers:
        return None

    candidate_pool: list[_GlobalPathCandidate] = []
    seen_pool_keys: set[tuple[int, int]] = set()
    for cover in complete_covers:
        for candidate in cover["candidates"]:
            key = (candidate.anchor_index, candidate.path_index)
            if key in seen_pool_keys:
                continue
            seen_pool_keys.add(key)
            candidate_pool.append(candidate)

    winning_seed = _select_global_best_path(candidate_pool)
    if winning_seed is None:
        return None

    winning_covers = [
        cover
        for cover in complete_covers
        if cover["anchor_index"] == winning_seed.anchor_index
        and any(candidate is winning_seed for candidate in cover["candidates"])
    ] or [cover for cover in complete_covers if cover["anchor_index"] == winning_seed.anchor_index]
    winning_cover = sorted(
        winning_covers,
        key=lambda cover: (
            _candidate_set_sort_key(cover["candidate_set"], cover["candidates"]),
            len(cover["candidate_set"]),
        ),
    )[0]

    selected_candidates: list[_GlobalPathCandidate] = []
    selected_keys: set[tuple[int, int]] = set()
    for entity in winning_cover["candidate_set"]:
        entity_candidates = [
            candidate for candidate in winning_cover["candidates"] if entity in candidate.path.nodes
        ]
        unused_candidates = [
            candidate
            for candidate in entity_candidates
            if (candidate.anchor_index, candidate.path_index) not in selected_keys
        ]
        selected = _select_global_best_path(unused_candidates or entity_candidates)
        if selected is None:
            return None
        selected_candidates.append(selected)
        selected_keys.add((selected.anchor_index, selected.path_index))

    return _GlobalPathSelection(
        selection_type="global_best_path_cover",
        candidates=selected_candidates,
        winning_candidate=winning_seed,
        candidate_set=list(winning_cover["candidate_set"]),
    )


def _complete_global_cover_groups(candidates: list[_GlobalPathCandidate]) -> list[dict[str, Any]]:
    by_anchor: dict[int, list[_GlobalPathCandidate]] = {}
    for candidate in candidates:
        by_anchor.setdefault(candidate.anchor_index, []).append(candidate)

    complete: list[dict[str, Any]] = []
    for anchor_index in sorted(by_anchor):
        anchor_candidates = by_anchor[anchor_index]
        cover_candidates = [candidate for candidate in anchor_candidates if candidate.path_type == "candidate_path_cover"]
        if not cover_candidates:
            continue
        for candidate_set in _ordered_candidate_sets(cover_candidates):
            if len(candidate_set) < 2:
                continue
            if all(any(entity in candidate.path.nodes for candidate in cover_candidates) for entity in candidate_set):
                complete.append(
                    {
                        "anchor_index": anchor_index,
                        "candidate_set": list(candidate_set),
                        "candidates": cover_candidates,
                    }
                )
    return complete


def _ordered_candidate_sets(candidates: list[_GlobalPathCandidate]) -> list[list[str]]:
    ordered: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in sorted(candidates, key=lambda item: (item.anchor_index, item.path_index)):
        for candidate_set in candidate.candidate_sets:
            key = tuple(candidate_set)
            if len(key) >= 2 and key not in seen:
                seen.add(key)
                ordered.append(list(key))
    return ordered


def _candidate_set_sort_key(candidate_set: list[str], candidates: list[_GlobalPathCandidate]) -> tuple[int, ...]:
    first_positions: list[int] = []
    for entity in candidate_set:
        positions = [
            min((index for index, node in enumerate(candidate.path.nodes) if node == entity), default=10**6)
            for candidate in candidates
            if entity in candidate.path.nodes
        ]
        first_positions.append(min(positions) if positions else 10**6)
    return tuple(first_positions)


def _annotate_per_anchor_global_selection(
    per_anchor_results: list[dict[str, Any]],
    global_candidates: list[_GlobalPathCandidate],
    selected: list[_GlobalPathCandidate],
) -> None:
    by_anchor: dict[int, list[_GlobalPathCandidate]] = {}
    by_anchor_path: dict[tuple[int, int], _GlobalPathCandidate] = {}
    selected_keys = {(candidate.anchor_index, candidate.path_index) for candidate in selected}
    for candidate in global_candidates:
        by_anchor.setdefault(candidate.anchor_index, []).append(candidate)
        by_anchor_path[(candidate.anchor_index, candidate.path_index)] = candidate

    for anchor_index, anchor_result in enumerate(per_anchor_results, start=1):
        anchor_candidates = by_anchor.get(anchor_index, [])
        anchor_result["global_candidates"] = [candidate.to_debug() for candidate in anchor_candidates]
        anchor_result["contains_global_best_path"] = any(
            (candidate.anchor_index, candidate.path_index) in selected_keys for candidate in anchor_candidates
        )
        paths = anchor_result.get("paths")
        if not isinstance(paths, list):
            continue
        for path_index, path_payload in enumerate(paths, start=1):
            if not isinstance(path_payload, dict):
                continue
            candidate = by_anchor_path.get((anchor_index, path_index))
            if candidate is None:
                continue
            path_payload["global_rank"] = list(candidate.global_rank)
            path_payload["global_rank_components"] = dict(candidate.global_rank_components)
            path_payload["contains_global_best_path"] = (candidate.anchor_index, candidate.path_index) in selected_keys


def _collect_question_semantic_nodes(
    state: _WorkingState,
    explicit_entity_ids: list[str],
    query_focus: _QueryFocus,
    answer_anchor_id: str | None,
) -> set[str]:
    return _collect_question_semantic_node_ids(
        state,
        explicit_entity_ids,
        focus_node_ids=[
            query_focus.answer_anchor_id,
            query_focus.query_root_id,
            query_focus.slot_id,
            query_focus.terminal_id,
            answer_anchor_id,
            *query_focus.required_ids,
        ],
    )


def _collect_question_semantic_node_ids(
    state: _WorkingState,
    explicit_entity_ids: list[str],
    *,
    focus_node_ids: Iterable[str | None] = (),
) -> set[str]:
    semantic_nodes: set[str] = set()

    def add(node_id: str | None, *, force: bool = False) -> None:
        if not node_id or node_id not in state.nodes:
            return
        if force or _is_semantic_node_candidate(state.nodes[node_id]):
            semantic_nodes.add(node_id)

    for entity_id in explicit_entity_ids:
        add(entity_id, force=True)
    for node_id in focus_node_ids:
        add(node_id, force=_is_possessive_wh_anchor_node(state.nodes.get(node_id or "")))

    for edge in state.edges.values():
        if _normalize_edge_quality(edge.edge_quality) != "STRONG":
            continue
        if not _edge_has_core_semantic_evidence(edge):
            continue
        add(edge.source)
        add(edge.target)

    if not semantic_nodes:
        for node in _sorted_nodes(state.nodes.values()):
            if node.id != "0" and node.kind != "function":
                semantic_nodes.add(node.id)
                break
    if not semantic_nodes:
        for node in _sorted_nodes(state.nodes.values()):
            if node.id != "0":
                semantic_nodes.add(node.id)
                break
    return semantic_nodes


def _is_semantic_node_candidate(node: TokenReasoningNode) -> bool:
    if node.id == "0" or node.kind == "function":
        return False
    lower = node.text.lower()
    if lower in WH_WORDS or lower in LIGHT_VERBS or lower in PREPOSITIONS or lower in DETERMINERS:
        return False
    if _is_punctuation(node.text):
        return False
    return node.kind in {"entity", "content", "constraint", "answer"}


def _is_possessive_wh_anchor_node(node: TokenReasoningNode | None) -> bool:
    if node is None or node.id == "0":
        return False
    return node.text.lower() == "whose"


def _edge_has_core_semantic_evidence(edge: TokenReasoningEdge) -> bool:
    if edge.derived and (
        "bridge_contraction" in edge.rule
        or "possessive_marker_contraction" in edge.rule
        or "function_backbone_contraction" in edge.rule
    ):
        return True
    classes = _edge_label_classes_deep(edge)
    return bool(classes & {"CORE_ARG", "RESTRICT", "IDENTITY"})


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

    semantic_nodes = _collect_question_semantic_nodes(state, candidate_ids, query_focus, query_focus.answer_anchor_id)
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
            semantic_nodes=semantic_nodes,
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
        semantic_nodes = _collect_question_semantic_nodes(state, candidate_ids, query_focus, query_focus.answer_anchor_id)
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
                semantic_nodes=semantic_nodes,
            ).with_selection(selected=True)
            selected_records.append(record)
            paths.append(_path_from_ids(f"P{path_index}", state.nodes, branch))
        if len(paths) == len(candidate_ids):
            return paths, selected_records
    return None


def detect_conjunctive_constraint_covers(
    state: _WorkingState,
    explicit_entity_ids: list[str],
    query_focus: _QueryFocus,
    *,
    direct_candidate_sets: list[list[str]],
) -> list[_ConjunctiveConstraintCover]:
    if len(explicit_entity_ids) < 2:
        return []

    groups = _shared_predicate_role_member_groups(state)
    covers: list[_ConjunctiveConstraintCover] = []
    for group in groups:
        predicate_id = str(group["predicate_id"])
        role_family = str(group["role_family"])
        member_ids = _sort_node_ids(group["member_ids"], state.nodes)
        if len(member_ids) < 2 or predicate_id not in state.nodes:
            continue

        marker_id = _conjunctive_coordination_marker_id(state, predicate_id, member_ids, group["evidence"])
        if marker_id and state.nodes[marker_id].text.lower() != "and":
            continue
        if not marker_id and not _markerless_conjunctive_group_allowed(state, member_ids, query_focus):
            continue

        target_id = _conjunctive_target_id(state, predicate_id, role_family, member_ids, query_focus, marker_id)
        if not target_id:
            continue
        if _conjunctive_group_is_candidate_like(state, member_ids, direct_candidate_sets, marker_id):
            continue

        branch_paths: list[tuple[str, ...]] = []
        branch_entity_ids: list[str] = []
        viable = True
        member_set = set(member_ids)
        for member_id in member_ids:
            blocked_ids = {predicate_id, *(member_set - {member_id})}
            if marker_id:
                blocked_ids.add(marker_id)
            branch = _best_conjunctive_branch_path(
                state,
                member_id,
                explicit_entity_ids,
                blocked_ids=blocked_ids,
            )
            if branch is None:
                viable = False
                break
            entity_id, path_ids = branch
            branch_entity_ids.append(entity_id)
            branch_paths.append(tuple(path_ids))

        if not viable or len(set(branch_entity_ids)) < len(branch_entity_ids):
            continue

        cover_id = "cc:" + ":".join([predicate_id, role_family, target_id, *(member_ids)])
        evidence = {
            "rule": "conjunctive_constraint_cover",
            "predicate_id": predicate_id,
            "predicate": state.nodes[predicate_id].text,
            "role_family": role_family,
            "marker_id": marker_id,
            "marker": state.nodes[marker_id].text if marker_id in state.nodes else None,
            "shared_role_evidence": list(group["evidence"]),
            "branch_entity_ids": list(branch_entity_ids),
            "branch_entities": [state.nodes[node_id].text for node_id in branch_entity_ids if node_id in state.nodes],
        }
        covers.append(
            _ConjunctiveConstraintCover(
                cover_id=cover_id,
                predicate_id=predicate_id,
                target_id=target_id,
                role_family=role_family,
                marker_id=marker_id,
                member_head_ids=tuple(member_ids),
                branch_entity_ids=tuple(branch_entity_ids),
                branch_paths=tuple(branch_paths),
                evidence=evidence,
            )
        )

    covers.sort(key=lambda cover: _conjunctive_cover_sort_key(state, cover, query_focus))
    return covers


def add_conjunctive_constraint_edges(
    state: _WorkingState,
    covers: list[_ConjunctiveConstraintCover],
) -> None:
    for cover in covers:
        for member_id in cover.member_head_ids:
            if member_id == cover.target_id:
                continue
            if cover.marker_id and cover.marker_id in state.nodes:
                _ensure_conjunctive_constraint_edge(
                    state,
                    cover.target_id,
                    cover.marker_id,
                    cover,
                    relation_marker="coordination_constraint_marker",
                )
                _ensure_conjunctive_constraint_edge(
                    state,
                    cover.marker_id,
                    member_id,
                    cover,
                    relation_marker="coordination_constraint_member",
                )
            else:
                _ensure_conjunctive_constraint_edge(
                    state,
                    cover.target_id,
                    member_id,
                    cover,
                    relation_marker="conjunctive_constraint_alias",
                )


def _select_conjunctive_constraint_path_cover(
    state: _WorkingState,
    query_focus: _QueryFocus,
    covers: list[_ConjunctiveConstraintCover],
) -> tuple[list[TokenReasoningPath], list[_CandidatePath], _ConjunctiveConstraintCover] | None:
    for cover in covers:
        backbone = _conjunctive_backbone_path(state, cover, query_focus)
        if len(backbone) < 2 or cover.target_id not in backbone:
            continue

        member_to_branch = dict(zip(cover.member_head_ids, cover.branch_paths))
        member_to_entity = dict(zip(cover.member_head_ids, cover.branch_entity_ids))
        branch_paths = {member_id: list(branch_path) for member_id, branch_path in member_to_branch.items()}

        selected_paths: list[TokenReasoningPath] = []
        selected_records: list[_CandidatePath] = []
        semantic_nodes = set(backbone)
        for branch_path in branch_paths.values():
            semantic_nodes.update(branch_path)
        for path_index, member_id in enumerate(cover.member_head_ids, start=1):
            branch_path = branch_paths.get(member_id, [])
            if not branch_path:
                break
            if member_id == cover.target_id:
                full_path = _dedupe_adjacent([*backbone, *branch_path[1:]])
            else:
                connector = _conjunctive_member_connector_path(cover, member_id)
                if not connector:
                    break
                full_path = _dedupe_adjacent([*backbone, *connector[1:], *branch_path[1:]])
            if len(full_path) != len(set(full_path)):
                break
            path = _path_from_ids(f"P{path_index}", state.nodes, full_path)
            selected_paths.append(path)
            selected_records.append(
                _rank_candidate_path(
                    state.nodes,
                    state.edges,
                    full_path,
                    query_focus,
                    member_to_entity.get(member_id),
                    "conjunctive_constraint_path_cover",
                    required_ids=full_path,
                    semantic_nodes=semantic_nodes,
                ).with_selection(selected=True)
            )
        if len(selected_paths) != len(cover.member_head_ids):
            continue
        threaded_cover = _ConjunctiveConstraintCover(
            cover_id=cover.cover_id,
            predicate_id=cover.predicate_id,
            target_id=cover.target_id,
            role_family=cover.role_family,
            marker_id=cover.marker_id,
            member_head_ids=cover.member_head_ids,
            branch_entity_ids=cover.branch_entity_ids,
            branch_paths=tuple(tuple(branch_paths[member_id]) for member_id in cover.member_head_ids),
            evidence=cover.evidence,
        )
        return selected_paths, selected_records, threaded_cover
    return None


def _shared_predicate_role_member_groups(state: _WorkingState) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    marker_role_edges: list[dict[str, Any]] = []

    for edge in state.raw_edges.values():
        for item in _raw_provenance(edge):
            role = _semantic_role_family(item.get("normalized_relation") or item.get("relation"))
            if role is None:
                continue
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if head_idx is None or dep_idx is None:
                continue
            predicate_id = str(head_idx)
            member_id = str(dep_idx)
            if predicate_id not in state.nodes or member_id not in state.nodes or predicate_id == "0":
                continue
            predicate_node = state.nodes[predicate_id]
            member_node = state.nodes[member_id]
            if not _is_clause_predicate_anchor_node(state.nodes, predicate_id):
                continue
            key = (predicate_id, role)
            if _is_conjunctive_marker_node(member_node):
                marker_role_edges.append(
                    {
                        "predicate_id": predicate_id,
                        "role_family": role,
                        "marker_id": member_id,
                        "edge": _edge_provenance_summary(edge),
                    }
                )
                continue
            if not _is_conjunctive_member_head(member_node):
                continue
            group = groups.setdefault(
                key,
                {
                    "predicate_id": predicate_id,
                    "predicate": predicate_node.text,
                    "role_family": role,
                    "member_ids": [],
                    "evidence": [],
                },
            )
            if member_id not in group["member_ids"]:
                group["member_ids"].append(member_id)
            group["evidence"].append(
                {
                    "rule": "shared_predicate_role",
                    "predicate_id": predicate_id,
                    "predicate": predicate_node.text,
                    "member_id": member_id,
                    "member": member_node.text,
                    "role_family": role,
                    "edge": _edge_provenance_summary(edge),
                }
            )

    adjacency = _adjacency(state.raw_edges)
    for marker_edge in marker_role_edges:
        predicate_id = str(marker_edge["predicate_id"])
        role = str(marker_edge["role_family"])
        marker_id = str(marker_edge["marker_id"])
        key = (predicate_id, role)
        group = groups.setdefault(
            key,
            {
                "predicate_id": predicate_id,
                "predicate": state.nodes[predicate_id].text,
                "role_family": role,
                "member_ids": [],
                "evidence": [],
            },
        )
        for neighbor_id, edge in _neighbor_edges(marker_id, adjacency, state.raw_edges):
            if not _is_conjunctive_member_head(state.nodes[neighbor_id]):
                continue
            if "COORD" not in _edge_label_classes(edge):
                continue
            if neighbor_id not in group["member_ids"]:
                group["member_ids"].append(neighbor_id)
            group["evidence"].append(
                {
                    "rule": "predicate_marker_member",
                    "predicate_id": predicate_id,
                    "predicate": state.nodes[predicate_id].text,
                    "marker_id": marker_id,
                    "marker": state.nodes[marker_id].text,
                    "member_id": neighbor_id,
                    "member": state.nodes[neighbor_id].text,
                    "role_family": role,
                    "predicate_marker_edge": marker_edge["edge"],
                    "marker_member_edge": _edge_provenance_summary(edge),
                }
            )

    result: list[dict[str, Any]] = []
    for group in groups.values():
        member_ids = _sort_node_ids(group["member_ids"], state.nodes)
        if len(member_ids) < 2:
            continue
        payload = dict(group)
        payload["member_ids"] = member_ids
        result.append(payload)
    result.sort(
        key=lambda group: (
            _node_sort_key(state.nodes[str(group["predicate_id"])]),
            str(group["role_family"]),
            tuple(_node_sort_key(state.nodes[node_id]) for node_id in group["member_ids"]),
        )
    )
    return result


def _semantic_role_family(relation: object) -> str | None:
    key = _normalized_relation_key(str(relation or ""))
    if not key:
        return None
    if any(key.startswith(prefix) for prefix in ("prep_", "det_", "aux_", "coord_", "conj_", "relative_")):
        return None
    if key in {"loc", "tloc", "tmp"} or key.startswith("loc_") or key.endswith("_loc") or key.startswith("dir"):
        return "loc"
    if "twhen" in key or "temporal" in key:
        return "temporal"
    if key in {"arg2", "verb_arg2", "pat_arg", "eff_arg", "compl", "object"}:
        return "patient"
    if key.endswith("_arg2") and not key.startswith(("prep_", "det_", "aux_", "coord_", "conj_")):
        return "patient"
    if key in {"arg1", "verb_arg1", "act_arg", "auth", "agent", "subject"}:
        return "agent"
    if key.endswith("_arg1") and not key.startswith(("prep_", "det_", "aux_", "coord_", "conj_")):
        return "agent"
    return None


def _is_conjunctive_member_head(node: TokenReasoningNode) -> bool:
    if node.id == "0" or node.kind == "entity" or node.kind == "function":
        return False
    if node.text.lower() in WH_WORDS or _is_scope_node(node):
        return False
    return node.kind in {"content", "constraint", "answer"}


def _is_conjunctive_marker_node(node: TokenReasoningNode) -> bool:
    return node.text.lower() in {"and", "or"}


def _conjunctive_coordination_marker_id(
    state: _WorkingState,
    predicate_id: str,
    member_ids: list[str],
    evidence: list[dict[str, Any]],
) -> str | None:
    candidate_ids: set[str] = set()
    for item in evidence:
        marker_id = str(item.get("marker_id") or "")
        if marker_id in state.nodes and _is_conjunctive_marker_node(state.nodes[marker_id]):
            candidate_ids.add(marker_id)

    member_set = set(member_ids)
    adjacency = _adjacency(state.raw_edges)
    for marker in state.nodes.values():
        if not _is_conjunctive_marker_node(marker):
            continue
        if _surface_marker_between_nodes(state.nodes, marker.id, member_ids):
            candidate_ids.add(marker.id)
            continue
        touches_member = False
        touches_predicate = False
        for neighbor_id, edge in _neighbor_edges(marker.id, adjacency, state.raw_edges):
            if neighbor_id in member_set and "COORD" in _edge_label_classes(edge):
                touches_member = True
            if neighbor_id == predicate_id and _semantic_role_family_from_edge(edge) is not None:
                touches_predicate = True
        if touches_member and (touches_predicate or _surface_marker_between_nodes(state.nodes, marker.id, member_ids)):
            candidate_ids.add(marker.id)

    if not candidate_ids:
        return None
    return sorted(
        candidate_ids,
        key=lambda node_id: (
            0 if state.nodes[node_id].text.lower() == "and" else 1,
            _node_sort_key(state.nodes[node_id]),
        ),
    )[0]


def _surface_marker_between_nodes(
    nodes: dict[str, TokenReasoningNode],
    marker_id: str,
    member_ids: list[str],
) -> bool:
    if marker_id not in nodes or len(member_ids) < 2:
        return False
    positions = [nodes[node_id].index for node_id in member_ids if node_id in nodes]
    if len(positions) < 2:
        return False
    marker_index = nodes[marker_id].index
    return min(positions) < marker_index < max(positions)


def _semantic_role_family_from_edge(edge: TokenReasoningEdge) -> str | None:
    for item in _raw_provenance(edge):
        role = _semantic_role_family(item.get("normalized_relation") or item.get("relation"))
        if role is not None:
            return role
    return None


def _markerless_conjunctive_group_allowed(
    state: _WorkingState,
    member_ids: list[str],
    query_focus: _QueryFocus,
) -> bool:
    focus_ids = _query_focus_node_ids(query_focus)
    return bool(focus_ids.intersection(member_ids))


def _conjunctive_target_id(
    state: _WorkingState,
    predicate_id: str,
    role_family: str,
    member_ids: list[str],
    query_focus: _QueryFocus,
    marker_id: str | None,
) -> str | None:
    if marker_id and marker_id in state.nodes:
        marker_index = state.nodes[marker_id].index
        left_members = [node_id for node_id in member_ids if state.nodes[node_id].index < marker_index]
        if left_members:
            return _sort_node_ids(left_members, state.nodes)[0]
        return member_ids[0]
    focus_ids = _query_focus_node_ids(query_focus)
    for node_id in _sort_node_ids(focus_ids.intersection(member_ids), state.nodes):
        return node_id
    if role_family == "agent":
        return None
    del predicate_id
    return member_ids[0]


def _query_focus_node_ids(query_focus: _QueryFocus) -> set[str]:
    return {
        node_id
        for node_id in (
            query_focus.slot_id,
            query_focus.terminal_id,
            query_focus.answer_anchor_id,
            query_focus.query_root_id,
            *query_focus.required_ids,
        )
        if node_id
    }


def _conjunctive_group_is_candidate_like(
    state: _WorkingState,
    member_ids: list[str],
    direct_candidate_sets: list[list[str]],
    marker_id: str | None,
) -> bool:
    if marker_id and state.nodes[marker_id].text.lower() == "or":
        return True
    if not direct_candidate_sets:
        return False
    member_texts = {state.nodes[node_id].text for node_id in member_ids if node_id in state.nodes}
    for candidate_set in direct_candidate_sets:
        if member_texts and member_texts == set(candidate_set):
            return True
    return False


def _best_conjunctive_branch_path(
    state: _WorkingState,
    member_id: str,
    explicit_entity_ids: list[str],
    *,
    blocked_ids: set[str],
) -> tuple[str, list[str]] | None:
    ranked: list[tuple[tuple[Any, ...], str, list[str]]] = []
    for entity_id in _sort_node_ids(explicit_entity_ids, state.nodes):
        forbidden = set(blocked_ids)
        forbidden.update(set(explicit_entity_ids) - {entity_id})
        semantic_nodes = _conjunctive_branch_semantic_nodes(
            state,
            member_id,
            entity_id,
            blocked_ids=forbidden,
        )
        raw_paths: list[list[str]] = []
        for allow_weak in (False, True):
            raw_paths.extend(
                _bounded_k_simple_paths(
                    state.nodes,
                    state.edges,
                    source_id=member_id,
                    target_id=entity_id,
                    forbidden_nodes=forbidden,
                    required_ids={entity_id, *semantic_nodes},
                    allow_weak=allow_weak,
                    max_nodes=10,
                    top_k=24,
                )
            )
        seen: set[tuple[str, ...]] = set()
        for path in raw_paths:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            if _violates_blocked_conjunctive_path(path, blocked_ids):
                continue
            missing = len(semantic_nodes - set(path))
            search_key = _path_search_sort_key(state.nodes, state.edges, path)
            ranked.append(((missing, *search_key), entity_id, path))
    if not ranked:
        return None
    _rank, entity_id, path = sorted(ranked, key=lambda item: item[0])[0]
    return entity_id, path


def _conjunctive_branch_semantic_nodes(
    state: _WorkingState,
    member_id: str,
    entity_id: str,
    *,
    blocked_ids: set[str],
) -> set[str]:
    semantic_nodes = {member_id, entity_id}
    adjacency = _adjacency(state.edges)
    queue: list[tuple[str, int]] = [(member_id, 0)]
    visited = {member_id}
    while queue:
        current, depth = queue.pop(0)
        if depth >= 5:
            continue
        for neighbor_id, edge in _neighbor_edges(current, adjacency, state.edges):
            if neighbor_id in visited or neighbor_id in blocked_ids:
                continue
            if "COORD" in _edge_label_classes_deep(edge) or _is_scope_node(state.nodes[neighbor_id]):
                continue
            if state.nodes[neighbor_id].kind == "function":
                continue
            if not (_edge_label_classes_deep(edge) & {"CORE_ARG", "RESTRICT", "IDENTITY", "MODIFIER", "BRIDGE", "UNKNOWN"}):
                continue
            visited.add(neighbor_id)
            if _is_semantic_node_candidate(state.nodes[neighbor_id]):
                semantic_nodes.add(neighbor_id)
            if neighbor_id == entity_id:
                continue
            queue.append((neighbor_id, depth + 1))
    return semantic_nodes


def _violates_blocked_conjunctive_path(path_ids: list[str], blocked_ids: set[str]) -> bool:
    return any(node_id in blocked_ids for node_id in path_ids[1:-1])


def _conjunctive_backbone_path(
    state: _WorkingState,
    cover: _ConjunctiveConstraintCover,
    query_focus: _QueryFocus,
) -> list[str]:
    predicate_id = cover.predicate_id
    target_id = cover.target_id
    start_id = _conjunctive_wh_start_id(state, predicate_id, query_focus)
    if start_id and start_id != predicate_id:
        path = _best_existing_path_between(
            state,
            start_id,
            predicate_id,
            blocked_ids=set(cover.member_head_ids) - {target_id},
            max_nodes=5,
        )
        if not path:
            path = [start_id, predicate_id]
    else:
        path = [predicate_id]
    if target_id != predicate_id:
        suffix = _best_existing_path_between(
            state,
            predicate_id,
            target_id,
            blocked_ids=set(cover.member_head_ids) - {target_id},
            max_nodes=4,
        )
        if not suffix:
            suffix = [predicate_id, target_id]
        path = _dedupe_adjacent([*path, *suffix[1:]])
    return path


def _conjunctive_wh_start_id(
    state: _WorkingState,
    predicate_id: str,
    query_focus: _QueryFocus,
) -> str | None:
    adjacency = _adjacency(state.raw_edges)
    wh_neighbors = [
        neighbor_id
        for neighbor_id, _edge in _neighbor_edges(predicate_id, adjacency, state.raw_edges)
        if neighbor_id in state.nodes and state.nodes[neighbor_id].text.lower() in WH_WORDS
    ]
    if wh_neighbors:
        return _sort_node_ids(wh_neighbors, state.nodes)[0]
    if query_focus.terminal_id in state.nodes and state.nodes[query_focus.terminal_id].text.lower() in WH_WORDS:
        return query_focus.terminal_id
    if query_focus.answer_anchor_id in state.nodes and state.nodes[query_focus.answer_anchor_id].text.lower() in WH_WORDS:
        return query_focus.answer_anchor_id
    return None


def _best_existing_path_between(
    state: _WorkingState,
    source_id: str,
    target_id: str,
    *,
    blocked_ids: set[str],
    max_nodes: int,
) -> list[str]:
    for allow_weak in (False, True):
        paths = _bounded_k_simple_paths(
            state.nodes,
            state.edges,
            source_id=source_id,
            target_id=target_id,
            forbidden_nodes=set(blocked_ids),
            required_ids={target_id},
            allow_weak=allow_weak,
            max_nodes=max_nodes,
            top_k=8,
        )
        if paths:
            return paths[0]
    return []


def _conjunctive_member_connector_path(
    cover: _ConjunctiveConstraintCover,
    member_id: str,
) -> list[str]:
    if cover.marker_id:
        return [cover.target_id, cover.marker_id, member_id]
    return [cover.target_id, member_id]


def _ensure_conjunctive_constraint_edge(
    state: _WorkingState,
    source_id: str,
    target_id: str,
    cover: _ConjunctiveConstraintCover,
    *,
    relation_marker: str,
) -> None:
    existing = state.edges.get(_edge_key(source_id, target_id))
    if existing is not None and "conjunctive_constraint" in existing.rule:
        return
    edge_quality = "MEDIUM"
    support = EDGE_QUALITY_SCORES[edge_quality]
    provenance = {
        "rule": "conjunctive_constraint_edge",
        "relation_marker": relation_marker,
        "edge_quality": edge_quality,
        "derived": True,
        "support": support,
        "cover_id": cover.cover_id,
        "predicate_id": cover.predicate_id,
        "predicate": state.nodes[cover.predicate_id].text if cover.predicate_id in state.nodes else None,
        "target_id": cover.target_id,
        "target": state.nodes[cover.target_id].text if cover.target_id in state.nodes else None,
        "marker_id": cover.marker_id,
        "marker": state.nodes[cover.marker_id].text if cover.marker_id in state.nodes else None,
        "member_head_ids": list(cover.member_head_ids),
        "member_heads": [state.nodes[node_id].text for node_id in cover.member_head_ids if node_id in state.nodes],
        "evidence": dict(cover.evidence),
    }
    virtual = _merge_edge(
        state.edges,
        state.nodes,
        source_id,
        target_id,
        support=support,
        edge_quality=edge_quality,
        derived=True,
        rule="conjunctive_constraint_edge",
        provenance=[provenance],
    )
    state.virtual_edges.append(virtual.to_dict())


def _conjunctive_cover_sort_key(
    state: _WorkingState,
    cover: _ConjunctiveConstraintCover,
    query_focus: _QueryFocus,
) -> tuple[Any, ...]:
    focus_ids = _query_focus_node_ids(query_focus)
    return (
        0 if cover.target_id in focus_ids else 1,
        0 if cover.marker_id and state.nodes[cover.marker_id].text.lower() == "and" else 1,
        len(cover.member_head_ids),
        _node_sort_key(state.nodes[cover.predicate_id]),
        _node_sort_key(state.nodes[cover.target_id]),
        tuple(_node_sort_key(state.nodes[node_id]) for node_id in cover.member_head_ids),
    )


def _violates_conjunctive_member_bridge_via_answer_predicate(
    path_ids: Iterable[str],
    covers: list[_ConjunctiveConstraintCover],
) -> bool:
    ids = list(path_ids)
    if len(ids) < 3 or not covers:
        return False
    for cover in covers:
        members = set(cover.member_head_ids)
        for left, middle, right in zip(ids, ids[1:], ids[2:]):
            if middle == cover.predicate_id and left in members and right in members and left != right:
                return True
    return False


def _extract_actual_parallel_path_cover(
    state: _WorkingState,
    query_focus: _QueryFocus,
    constraints: list[dict[str, Any]],
    parallel_entity_sets: list[dict[str, Any]],
) -> tuple[list[TokenReasoningPath], list[_CandidatePath], list[_CandidatePath]] | None:
    all_records: list[_CandidatePath] = []
    for parallel_set in _sort_parallel_entity_sets(state.nodes, parallel_entity_sets):
        entity_ids = [node_id for node_id in parallel_set.get("entity_ids", []) if node_id in state.nodes]
        if len(entity_ids) < 2:
            continue
        if (
            parallel_set.get("kind") == "direct_entity_set"
            and query_focus.slot_id
            and not _candidate_typed_slot_instantiation_edges_available(state, query_focus.slot_id, entity_ids)
        ):
            continue
        branch_heads = {
            str(entity_id): str(head_id)
            for entity_id, head_id in dict(parallel_set.get("branch_heads") or {}).items()
            if str(head_id) in state.nodes
        }
        path_query_focus = _candidate_path_query_focus(state, query_focus, constraints, parallel_set.get("kind"))
        semantic_nodes = _collect_question_semantic_nodes(state, entity_ids, query_focus, query_focus.answer_anchor_id)
        candidate_set_blocked_edge_keys = _candidate_set_traversal_edge_keys(state, entity_ids)
        selected: list[_CandidatePath] = []
        viable = True
        for entity_id in entity_ids:
            required_ids = list(path_query_focus.required_ids)
            if entity_id in branch_heads:
                required_ids.insert(0, branch_heads[entity_id])
            branch_semantic_nodes = set(semantic_nodes) - (set(entity_ids) - {entity_id})
            blocked_edge_keys = set(candidate_set_blocked_edge_keys)
            blocked_edge_keys.update(
                _other_candidate_typed_slot_instantiation_edge_keys(
                    state,
                    query_focus.slot_id,
                    entity_ids,
                    entity_id,
                )
            )
            candidates = _enumerate_entity_focus_paths(
                state,
                entity_id,
                path_query_focus,
                forbidden_entity_ids=set(entity_ids) - {entity_id},
                required_ids=_unique_node_ids(required_ids, state.nodes),
                semantic_nodes=branch_semantic_nodes,
                blocked_edge_keys=blocked_edge_keys,
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
            selected_record = candidates[0].with_selection(selected=True)
            _warn_candidate_path_contains_other_candidate(state, selected_record, entity_ids)
            selected.append(selected_record)
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


def _candidate_path_query_focus(
    state: _WorkingState,
    query_focus: _QueryFocus,
    constraints: list[dict[str, Any]],
    parallel_set_kind: object,
) -> _QueryFocus:
    if parallel_set_kind != "direct_entity_set" or not query_focus.slot_id:
        return query_focus
    focus_id = _schema_focus_id(state, query_focus, constraints)
    if not focus_id or focus_id not in state.nodes or focus_id == query_focus.terminal_id:
        return query_focus
    required_ids = _unique_node_ids([*query_focus.required_ids, focus_id], state.nodes)
    return _QueryFocus(
        answer_anchor_id=query_focus.answer_anchor_id,
        query_root_id=query_focus.query_root_id,
        slot_id=query_focus.slot_id,
        terminal_id=focus_id,
        required_ids=tuple(required_ids),
        mode=query_focus.mode,
    )


def _candidate_typed_slot_instantiation_edges_available(
    state: _WorkingState,
    slot_id: str,
    entity_ids: list[str],
) -> bool:
    return all(
        (edge := state.edges.get(_edge_key(entity_id, slot_id))) is not None
        and "candidate_typed_slot_instantiation" in edge.rule
        for entity_id in entity_ids
    )


def _candidate_set_traversal_edge_keys(
    state: _WorkingState,
    entity_ids: list[str],
) -> set[tuple[str, str]]:
    candidate_ids = set(entity_ids)
    blocked: set[tuple[str, str]] = set()
    for key, edge in state.edges.items():
        source, target = key
        if _edge_has_candidate_set_discovery_evidence(edge):
            blocked.add(key)
            continue
        if source in candidate_ids and target in candidate_ids and _is_derived_candidate_bridge(edge):
            blocked.add(key)
    return blocked


def _other_candidate_typed_slot_instantiation_edge_keys(
    state: _WorkingState,
    slot_id: str | None,
    entity_ids: list[str],
    active_entity_id: str,
) -> set[tuple[str, str]]:
    if not slot_id:
        return set()
    blocked: set[tuple[str, str]] = set()
    for entity_id in entity_ids:
        if entity_id == active_entity_id:
            continue
        key = _edge_key(entity_id, slot_id)
        edge = state.edges.get(key)
        if edge is not None and "candidate_typed_slot_instantiation" in edge.rule:
            blocked.add(key)
    return blocked


def _warn_candidate_path_contains_other_candidate(
    state: _WorkingState,
    record: _CandidatePath,
    entity_ids: list[str],
) -> None:
    if not record.source_entity_id:
        return
    other_candidate_ids = set(entity_ids) - {record.source_entity_id}
    if not other_candidate_ids.intersection(record.node_ids):
        return
    warning = "candidate_path_contains_other_candidate"
    if warning not in state.warnings:
        state.warnings.append(warning)


def _edge_has_candidate_set_discovery_evidence(edge: TokenReasoningEdge) -> bool:
    for item in _walk_provenance_payload(edge.provenance):
        label_class = str(item.get("label_class") or "").upper()
        label_classes = {str(value).upper() for value in _iter_sequence(item.get("label_classes"))}
        if label_class == "COORD" or "COORD" in label_classes:
            return True
        relation = str(item.get("normalized_relation") or item.get("relation") or "")
        relation_key = _normalized_relation_key(relation)
        if _is_candidate_set_discovery_relation_key(relation_key):
            return True
    return False


def _is_candidate_set_discovery_relation_key(relation_key: str) -> bool:
    return (
        "coord" in relation_key
        or "conj_member" in relation_key
        or "disj_member" in relation_key
        or "apps_member" in relation_key
        or "appos" in relation_key
        or relation_key.startswith("app_")
        or relation_key == "app"
    )


def _is_derived_candidate_bridge(edge: TokenReasoningEdge) -> bool:
    return edge.derived and "bridge_contraction" in edge.rule


def _select_single_main_path(
    state: _WorkingState,
    explicit_entity_ids: list[str],
    query_focus: _QueryFocus,
    *,
    conjunctive_covers: list[_ConjunctiveConstraintCover] | None = None,
) -> tuple[TokenReasoningPath | None, str, _CandidatePath | None, list[_CandidatePath]]:
    all_records: list[_CandidatePath] = []
    entity_best: list[_CandidatePath] = []
    semantic_nodes = _collect_question_semantic_nodes(state, explicit_entity_ids, query_focus, query_focus.answer_anchor_id)
    conjunctive_covers = conjunctive_covers or []
    for entity_id in explicit_entity_ids:
        candidates = _enumerate_entity_focus_paths(
            state,
            entity_id,
            query_focus,
            forbidden_entity_ids=set(explicit_entity_ids) - {entity_id},
            required_ids=list(query_focus.required_ids),
            semantic_nodes=semantic_nodes,
        )
        allowed_candidates: list[_CandidatePath] = []
        for candidate in candidates:
            if _violates_conjunctive_member_bridge_via_answer_predicate(candidate.node_ids, conjunctive_covers):
                all_records.append(
                    candidate.with_selection(
                        selected=False,
                        rejected_reason="conjunctive_member_bridge_via_answer_predicate",
                    )
                )
                continue
            all_records.append(candidate)
            allowed_candidates.append(candidate)
        if allowed_candidates:
            entity_best.append(allowed_candidates[0])

    if not entity_best:
        return None, "empty", None, all_records

    selected = sorted(entity_best, key=lambda record: record.rank)[0]
    path_type = "single_main_path"
    if selected.rank_components.get("missing_semantic_nodes_count", 0) or selected.search_pass != "strong":
        path_type = "fallback_main_path"
    return _path_from_ids("P1", state.nodes, list(selected.node_ids)), path_type, selected, all_records


def _enumerate_entity_focus_paths(
    state: _WorkingState,
    entity_id: str,
    query_focus: _QueryFocus,
    *,
    forbidden_entity_ids: set[str],
    required_ids: list[str],
    semantic_nodes: set[str],
    blocked_edge_keys: set[tuple[str, str]] | None = None,
) -> list[_CandidatePath]:
    if entity_id not in state.nodes:
        return []
    target_id = query_focus.terminal_id
    if not target_id or target_id not in state.nodes:
        return []
    candidates: list[_CandidatePath] = []
    seen_paths: set[tuple[str, ...]] = set()
    effective_forbidden = set(forbidden_entity_ids) - set(semantic_nodes)
    blocked_edge_keys = blocked_edge_keys or set()
    for search_pass, allow_weak in (("strong", False), ("weak", True)):
        raw_paths = _bounded_k_simple_paths(
            state.nodes,
            state.edges,
            source_id=entity_id,
            target_id=target_id,
            forbidden_nodes=effective_forbidden,
            required_ids=set(required_ids),
            allow_weak=allow_weak,
            max_nodes=12,
            top_k=12,
            blocked_edge_keys=blocked_edge_keys,
        )
        for path in raw_paths:
            if _path_should_start_from_wh_anchor(state.nodes, path, target_id):
                path = list(reversed(path))
            key = tuple(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            candidates.append(
                _rank_candidate_path(
                    state.nodes,
                    state.edges,
                    path,
                    query_focus,
                    entity_id,
                    search_pass,
                    required_ids=required_ids,
                    semantic_nodes=semantic_nodes,
                )
            )
    candidates.sort(key=lambda record: record.rank)
    return candidates[:12]


def _path_should_start_from_wh_anchor(
    nodes: dict[str, TokenReasoningNode],
    path: list[str],
    target_id: str,
) -> bool:
    if len(path) < 2 or not path or path[-1] != target_id or target_id not in nodes:
        return False
    return nodes[target_id].text.lower() == "whose"


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
    blocked_edge_keys: set[tuple[str, str]] | None = None,
) -> list[list[str]]:
    if source_id == target_id:
        return [[source_id]]
    adjacency = _adjacency(edges)
    blocked_edge_keys = blocked_edge_keys or set()
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
            if _edge_key(current, neighbor_id) in blocked_edge_keys:
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
) -> tuple[Any, ...]:
    weak_edge_count = 0
    medium_edge_count = 0
    derived_edge_count = 0
    strong_edge_count = 0
    consensus_count = 0
    for left, right in zip(path, path[1:]):
        edge = edges.get(_edge_key(left, right))
        if edge is None:
            weak_edge_count += 1_000_000
            continue
        quality = _normalize_edge_quality(edge.edge_quality)
        if quality == "WEAK":
            weak_edge_count += 1
        elif quality == "MEDIUM":
            medium_edge_count += 1
        elif quality == "STRONG":
            strong_edge_count += 1
        if edge.derived:
            derived_edge_count += 1
        consensus_count += edge.consensus_count
    return (weak_edge_count, medium_edge_count, derived_edge_count, len(path) - 1, -strong_edge_count, -consensus_count, _path_index_tuple(path, nodes))


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
    semantic_nodes: set[str],
) -> _CandidatePath:
    del required_ids
    path_set = set(path_ids)
    ordered_semantic_nodes = _sort_node_ids([node_id for node_id in semantic_nodes if node_id in nodes], nodes)
    covered_semantic_nodes = [node_id for node_id in ordered_semantic_nodes if node_id in path_set]
    missing_semantic_nodes = [node_id for node_id in ordered_semantic_nodes if node_id not in path_set]
    strong_edge_count = 0
    medium_edge_count = 0
    weak_edge_count = 0
    derived_edge_count = 0
    consensus_count = 0
    for left, right in zip(path_ids, path_ids[1:]):
        edge = edges.get(_edge_key(left, right))
        if edge is None:
            weak_edge_count += 1
            continue
        quality = _normalize_edge_quality(edge.edge_quality)
        if quality == "STRONG":
            strong_edge_count += 1
        elif quality == "MEDIUM":
            medium_edge_count += 1
        else:
            weak_edge_count += 1
        if edge.derived:
            derived_edge_count += 1
        consensus_count += edge.consensus_count

    function_node_count = sum(
        1
        for node_id in path_ids
        if nodes[node_id].kind == "function"
    )
    rank = (
        len(missing_semantic_nodes),
        weak_edge_count,
        medium_edge_count,
        function_node_count,
        derived_edge_count,
        max(len(path_ids) - 1, 0),
        -strong_edge_count,
        -consensus_count,
    )
    components = {
        "missing_semantic_nodes_count": len(missing_semantic_nodes),
        "semantic_nodes": [nodes[node_id].text for node_id in ordered_semantic_nodes],
        "semantic_node_ids": ordered_semantic_nodes,
        "covered_semantic_nodes": [nodes[node_id].text for node_id in covered_semantic_nodes],
        "covered_semantic_node_ids": covered_semantic_nodes,
        "missing_semantic_nodes": [nodes[node_id].text for node_id in missing_semantic_nodes],
        "missing_semantic_node_ids": missing_semantic_nodes,
        "strong_edge_count": strong_edge_count,
        "medium_edge_count": medium_edge_count,
        "weak_edge_count": weak_edge_count,
        "function_node_count": function_node_count,
        "derived_edge_count": derived_edge_count,
        "path_length": max(len(path_ids) - 1, 0),
        "consensus_count": consensus_count,
        "rank": list(rank),
    }
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
    del edge
    return 0.0


def _edge_label_classes_deep(edge: TokenReasoningEdge) -> set[str]:
    classes = set(_edge_label_classes(edge))
    classes.update(_label_class_values_from_payload(edge.provenance))
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
                    "evidence": {"rule": "content_coordination_edge", "edge": _edge_provenance_summary(edge)},
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
            source_edges.append(_edge_provenance_summary(edge))
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
            semantic_nodes={focus_id},
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
    edge_quality = "MEDIUM"
    support = EDGE_QUALITY_SCORES[edge_quality]
    provenance = {
        "rule": "candidate_slot_substitution",
        "edge_quality": edge_quality,
        "derived": True,
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
        "schema_edge": _edge_provenance_summary(schema_edge) if schema_edge else None,
    }
    virtual = _merge_edge(
        state.edges,
        state.nodes,
        candidate_id,
        successor_id,
        support=support,
        edge_quality=edge_quality,
        derived=True,
        rule="candidate_slot_substitution",
        provenance=[provenance],
    )
    state.virtual_edges.append(virtual.to_dict())


def _ensure_candidate_typed_slot_instantiation_edge(
    state: _WorkingState,
    candidate_id: str,
    slot_id: str,
    candidate_set: list[str],
) -> None:
    existing = state.edges.get(_edge_key(candidate_id, slot_id))
    if existing is not None and "candidate_typed_slot_instantiation" in existing.rule:
        return
    edge_quality = "MEDIUM"
    support = EDGE_QUALITY_SCORES[edge_quality]
    provenance = {
        "rule": "candidate_typed_slot_instantiation",
        "edge_quality": edge_quality,
        "derived": True,
        "virtual": True,
        "bidirectional": True,
        "support": support,
        "candidate_id": candidate_id,
        "candidate": state.nodes[candidate_id].text,
        "typed_slot_id": slot_id,
        "typed_slot": state.nodes[slot_id].text,
        "reason": "candidate fills the typed WH answer slot",
        "typed_wh_evidence": _typed_wh_slot_evidence(state, slot_id),
        "candidate_set": list(candidate_set),
        "candidate_set_entity_ids": _candidate_text_set_to_ids(state.nodes, candidate_set),
        "candidate_set_evidence": _candidate_set_coordination_evidence(state, candidate_set),
    }
    virtual = _merge_edge(
        state.edges,
        state.nodes,
        candidate_id,
        slot_id,
        support=support,
        edge_quality=edge_quality,
        derived=True,
        rule="candidate_typed_slot_instantiation",
        provenance=[provenance],
    )
    payload = virtual.to_dict()
    payload.update(
        {
            "source": candidate_id,
            "target": slot_id,
            "source_text": state.nodes[candidate_id].text,
            "target_text": state.nodes[slot_id].text,
            "virtual": True,
            "bidirectional": True,
        }
    )
    state.virtual_edges.append(payload)


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
            raw_edges.append(_edge_provenance_summary(edge))

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
    edge_quality = "MEDIUM"
    support = EDGE_QUALITY_SCORES[edge_quality]
    provenance = {
        "rule": "candidate_bare_wh_substitution",
        "edge_quality": edge_quality,
        "derived": True,
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
        "schema_edge": _edge_provenance_summary(schema_edge) if schema_edge else None,
        "wh_predicate_edge": _edge_provenance_summary(wh_predicate_edge) if wh_predicate_edge else None,
        "support": support,
    }
    virtual = _merge_edge(
        state.edges,
        state.nodes,
        candidate_id,
        successor_id,
        support=support,
        edge_quality=edge_quality,
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
            evidence.append(_edge_provenance_summary(edge))
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
                    "edge_quality": "MEDIUM",
                    "derived": True,
                    "candidate": state.nodes[candidate_id].text,
                    "schema_path": [state.nodes[node_id].text for node_id in schema_path],
                }
                virtual = _merge_edge(
                    state.edges,
                    state.nodes,
                    branch[0],
                    branch[1],
                    support=EDGE_QUALITY_SCORES["MEDIUM"],
                    edge_quality="MEDIUM",
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
    original_question: str,
    normalized_question: str,
    normalization_changed: bool,
    normalization_note: str,
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
        "original_question": original_question,
        "normalized_question": normalized_question,
        "normalization_changed": normalization_changed,
        "normalization_note": normalization_note,
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


def _unique_warnings(warnings: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        text = str(warning)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_edge_quality(edge_quality: str | None) -> str:
    value = str(edge_quality or "WEAK").upper()
    return value if value in EDGE_QUALITY_SCORES else "WEAK"


def _infer_edge_quality(
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
    rule: str,
    provenance: list[dict[str, Any]],
) -> str:
    explicit = _highest_quality(
        str(item.get("edge_quality"))
        for item in provenance
        if isinstance(item, dict) and item.get("edge_quality")
    )
    if explicit:
        return explicit
    if (
        "candidate_expansion" in rule
        or "candidate_slot_substitution" in rule
        or "candidate_bare_wh_substitution" in rule
        or "candidate_typed_slot_instantiation" in rule
    ):
        return "MEDIUM"
    if "bridge_contraction" in rule:
        if _is_high_salience_node(nodes[source_id], include_order_constraints=True) and _is_high_salience_node(nodes[target_id], include_order_constraints=True):
            return "STRONG" if _bridge_contraction_has_strong_evidence(provenance) else "MEDIUM"
        return "WEAK"
    if "possessive_marker_contraction" in rule:
        return "STRONG"
    if "function_backbone_contraction" in rule:
        return "MEDIUM"
    labels = _label_class_values_from_payload(provenance)
    if labels:
        return _highest_quality(LABEL_CLASS_EDGE_QUALITY.get(label, "WEAK") for label in labels) or "WEAK"
    return "WEAK"


def _highest_quality(values: Iterable[str | None]) -> str | None:
    best: str | None = None
    for value in values:
        quality = _normalize_edge_quality(value)
        if best is None or EDGE_QUALITY_RANK[quality] > EDGE_QUALITY_RANK[best]:
            best = quality
    return best


_PROVENANCE_WALK_MAX_DEPTH = 4
_PROVENANCE_NESTED_KEYS = (
    "source_edges",
    "provenance",
    "evidence",
    "raw_edges",
    "edge",
    "schema_edge",
    "wh_predicate_edge",
)


def _iter_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _walk_provenance_payload(payload: Any, *, max_depth: int = _PROVENANCE_WALK_MAX_DEPTH) -> Iterable[dict[str, Any]]:
    visited: set[int] = set()

    def visit(item: Any, depth: int) -> Iterable[dict[str, Any]]:
        if depth > max_depth:
            return
        if isinstance(item, dict):
            marker = id(item)
            if marker in visited:
                return
            visited.add(marker)
            yield item
            for key in _PROVENANCE_NESTED_KEYS:
                if key not in item:
                    continue
                nested = item.get(key)
                for child in _iter_sequence(nested):
                    yield from visit(child, depth + 1)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                yield from visit(nested, depth)

    yield from visit(payload, 0)


def _label_class_values_from_payload(payload: Any) -> list[str]:
    values: list[str] = []
    for item in _walk_provenance_payload(payload):
        label_class = item.get("label_class")
        if label_class:
            values.append(str(label_class))
        for value in _iter_sequence(item.get("label_classes")):
            if value:
                values.append(str(value))
    return values


def _bridge_contraction_has_strong_evidence(provenance: list[dict[str, Any]]) -> bool:
    for item in provenance:
        bridge = str(item.get("bridge") or "").lower()
        if bridge in STRONG_BRIDGE_TOKENS:
            return True
        source_edges = item.get("source_edges") or []
        relation_keys: set[str] = set()
        for source_edge in source_edges:
            relation_keys.update(_relation_keys_from_payload(source_edge))
        if any(key.startswith("prep_arg") for key in relation_keys):
            return True
        if {"arg1", "arg2"} <= relation_keys:
            return True
    return False


def _relation_keys_from_payload(payload: Any) -> set[str]:
    relations: set[str] = set()
    for item in _walk_provenance_payload(payload):
        for key in ("normalized_relation", "relation"):
            relation = item.get(key)
            if relation:
                relations.add(_normalized_relation_key(str(relation)))
        for key in ("normalized_relations", "relations"):
            for relation in _iter_sequence(item.get(key)):
                if relation:
                    relations.add(_normalized_relation_key(str(relation)))
    return relations


def _consensus_count_from_provenance(provenance: list[dict[str, Any]]) -> int:
    formalisms: set[str] = set()
    explicit_counts: list[int] = []
    for item in _walk_provenance_payload(provenance):
        formalism = item.get("formalism")
        if formalism:
            formalisms.add(str(formalism))
        for value in _iter_sequence(item.get("formalisms")):
            if value:
                formalisms.add(str(value))
        try:
            explicit_counts.append(int(item.get("consensus_count") or 0))
        except (TypeError, ValueError):
            continue
    return max([len(formalisms), *explicit_counts, 0])


def _edge_relation_values_from_payload(payload: Any) -> set[str]:
    values: set[str] = set()
    for item in _walk_provenance_payload(payload):
        for key in ("normalized_relation", "relation"):
            relation = item.get(key)
            if relation:
                values.add(_normalize_relation(str(relation)))
        for key in ("normalized_relations", "relations"):
            for relation in _iter_sequence(item.get(key)):
                if relation:
                    values.add(_normalize_relation(str(relation)))
    return values


def _formalism_values_from_payload(payload: Any) -> set[str]:
    values: set[str] = set()
    for item in _walk_provenance_payload(payload):
        formalism = item.get("formalism")
        if formalism:
            values.add(str(formalism))
        for value in _iter_sequence(item.get("formalisms")):
            if value:
                values.add(str(value))
    return values


def _edge_provenance_summary(edge: TokenReasoningEdge) -> dict[str, Any]:
    label_classes = sorted(set(_label_class_values_from_payload(edge.provenance)))
    relation_values = sorted(_edge_relation_values_from_payload(edge.provenance))
    relation_keys = sorted(_relation_keys_from_payload(edge.provenance))
    formalisms = sorted(_formalism_values_from_payload(edge.provenance))
    consensus_count = max(edge.consensus_count, _consensus_count_from_provenance(edge.provenance))
    summary: dict[str, Any] = {
        "source_id": edge.source,
        "target_id": edge.target,
        "source": edge.source_text,
        "target": edge.target_text,
        "label_class": label_classes[0] if len(label_classes) == 1 else None,
        "label_classes": label_classes,
        "edge_quality": edge.edge_quality,
        "consensus_count": consensus_count,
        "derived": edge.derived,
        "rule": edge.rule,
        "formalism": formalisms[0] if len(formalisms) == 1 else None,
        "formalisms": formalisms,
        "relation": relation_values[0] if len(relation_values) == 1 else None,
        "relations": relation_values,
        "normalized_relation": relation_keys[0] if len(relation_keys) == 1 else None,
        "normalized_relations": relation_keys,
        "support": edge.support,
    }
    return summary


def _edge_provenance_summaries(edges: Iterable[TokenReasoningEdge | None]) -> list[dict[str, Any]]:
    return [_edge_provenance_summary(edge) for edge in edges if edge is not None]


def _merge_edge(
    edge_map: dict[tuple[str, str], TokenReasoningEdge],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
    *,
    support: float,
    edge_quality: str | None = None,
    derived: bool,
    rule: str,
    provenance: list[dict[str, Any]],
) -> TokenReasoningEdge:
    edge_quality = _normalize_edge_quality(edge_quality or _infer_edge_quality(nodes, source_id, target_id, rule, provenance))
    if source_id == target_id:
        return TokenReasoningEdge(
            source_id,
            target_id,
            nodes[source_id].text,
            nodes[target_id].text,
            support=EDGE_QUALITY_SCORES[edge_quality],
            edge_quality=edge_quality,
            consensus_count=_consensus_count_from_provenance(provenance),
            derived=derived,
            rule=rule,
            provenance=list(provenance),
        )
    key = _edge_key(source_id, target_id)
    source, target = key
    if key not in edge_map:
        edge_map[key] = TokenReasoningEdge(
            source=source,
            target=target,
            source_text=nodes[source].text,
            target_text=nodes[target].text,
            support=EDGE_QUALITY_SCORES[edge_quality],
            edge_quality=edge_quality,
            consensus_count=0,
            derived=derived,
            rule=rule if derived else "raw_evidence",
            provenance=[],
        )
    edge = edge_map[key]
    if EDGE_QUALITY_RANK[edge_quality] > EDGE_QUALITY_RANK.get(edge.edge_quality, 0):
        edge.edge_quality = edge_quality
    edge.support = EDGE_QUALITY_SCORES[_normalize_edge_quality(edge.edge_quality)]
    edge.derived = edge.derived or derived
    if derived:
        edge.rule = _combine_rules(edge.rule, rule)
    edge.provenance.extend(provenance)
    edge.consensus_count = _consensus_count_from_provenance(edge.provenance)
    for item in provenance:
        if isinstance(item, dict):
            item.setdefault("edge_quality", edge_quality)
            item.setdefault("consensus_count", edge.consensus_count)
            item.setdefault("derived", derived)
            item.setdefault("rule", rule)
    return edge


def _copy_edge(edge: TokenReasoningEdge) -> TokenReasoningEdge:
    return TokenReasoningEdge(
        source=edge.source,
        target=edge.target,
        source_text=edge.source_text,
        target_text=edge.target_text,
        support=edge.support,
        edge_quality=edge.edge_quality,
        consensus_count=edge.consensus_count,
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
    quality = _normalize_edge_quality(edge.edge_quality)
    quality_cost = {"STRONG": 1.0, "MEDIUM": 2.0, "WEAK": 4.0}[quality]
    cost = quality_cost
    if edge.derived:
        cost += 0.25
    cost += 0.01 * abs(source.index - target.index)
    if source.kind == "function" or target.kind == "function":
        cost += 0.20
    cost -= min(edge.consensus_count, 3) * 0.02
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
                source_edges = _edge_provenance_summaries([left_edge, right_edge])
                edge_quality = "MEDIUM"
                support = EDGE_QUALITY_SCORES[edge_quality]
                _merge_edge(
                    edges,
                    nodes,
                    left_id,
                    right_id,
                    support=support,
                    edge_quality=edge_quality,
                    derived=True,
                    rule="function_backbone_contraction",
                    provenance=[
                        {
                            "rule": "function_backbone_contraction",
                            "edge_quality": edge_quality,
                            "derived": True,
                            "support": support,
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
                edge_quality="WEAK",
                consensus_count=0,
            )
        final_edges.append(
            TokenReasoningEdge(
                source=source,
                target=target,
                source_text=nodes[source].text,
                target_text=nodes[target].text,
                support=base.support,
                edge_quality=base.edge_quality,
                consensus_count=base.consensus_count,
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
