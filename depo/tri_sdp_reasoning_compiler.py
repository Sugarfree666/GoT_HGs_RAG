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

ENTITY_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d[\d,]*(?:\.\d+)?|\d{1,4}(?:[-/]\d{1,2}){1,2})%?$")

DETERMINERS = {"a", "an", "the"}
WH_WORDS = {"what", "which", "who", "whom", "whose", "where", "when"}
WH_ANCHOR_WORDS = {
    "who",
    "whom",
    "whose",
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
}
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
FUNCTION_WORDS = (
    DETERMINERS
    | WH_WORDS
    | RELATIVE_PRONOUNS
    | LIGHT_VERBS
    | PREPOSITIONS
    | SCOPE_WORDS
)
POSSESSIVE_MARKER_TOKENS = {"'", "’", "'s", "’s", "s"}
POSSESSIVE_OWNER_RELATIONS = {"poss_arg2"}
POSSESSIVE_POSSESSED_RELATIONS = {"poss_arg1", "adj_arg1", "noun_arg1", "modifier"}
ORDER_CUES = {
    "first",
    "earliest",
    "latest",
    "last",
    "older",
    "oldest",
    "younger",
    "youngest",
}
APPROX_CUES = {"approximately", "about", "around", "roughly"}

EDGE_COST_ONE_RELATIONS = {
    "verb_arg1",
    "verb_arg2",
    "verb_arg3",
    "noun_arg1",
    "noun_arg2",
    "adj_arg1",
}

EDGE_COST_TWO_RELATIONS = {
    "conj_arg1",
    "conj_arg2",
    "relative_arg1",
    "relative_arg2",
    "comp_arg1",
    "comp_mod",
    "verb_mod",
}

EDGE_COST_ONE_RULES = {
    "pas_preposition_contraction",
    "pas_possessive_contraction",
}

EDGE_COST_TWO_RULES = {
    "pas_coordination_candidate_attachment",
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
            "provenance": [
                dict(item) if isinstance(item, dict) else item
                for item in self.provenance
            ],
            "edge_cost": _edge_cost(self),
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
            "candidate_sets": [
                list(candidate_set) for candidate_set in self.candidate_sets
            ],
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
    syntax_heads: dict[str, int] = field(default_factory=dict)
    syntax_head_source: str = ""


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
    """Compile HanLP PAS SDP evidence into entity-branch token paths.

    Step 4 no longer searches over answer-anchor candidates.  Explicit masked
    entities are fixed branch starts; each branch keeps the reachable boundary
    path with the highest Semantic Path Score (SP).
    """

    state = build_evidence_graph(hanlp_sdp_result)
    explicit_entity_ids = _resolve_explicit_entity_ids(state.nodes, explicit_entities)
    _mark_anchors(state.nodes, explicit_entity_ids)

    add_pas_preposition_contraction_edges(state)
    add_pas_possessive_contraction_edges(state)
    add_pas_coordination_candidate_attachment_edges(state, explicit_entity_ids)

    branch_selection = _select_entity_branch_best_paths(state, explicit_entity_ids)
    paths = branch_selection["paths"]
    path_type = "entity_branch_best_paths" if paths else "no_entity_branch_path"
    constraints: list[dict[str, Any]] = []
    candidate_sets: list[list[str]] = []

    final_node_ids, final_pairs = _graph_from_selected_paths(paths)
    final_nodes = _final_nodes(state.nodes, final_node_ids)
    active_entity_ids = _active_entity_ids(state.nodes, paths)
    final_edges = _final_edges(
        state.nodes, state.edges, final_pairs, paths, active_entity_ids
    )

    entity_anchors = [
        state.nodes[node_id].text
        for node_id in active_entity_ids
        if node_id in state.nodes
    ]
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
        normalized_question=normalized_question
        or masked_question
        or hanlp_sdp_result.text,
        normalization_changed=bool(normalization_changed),
        normalization_note=normalization_note or "",
        hanlp_sdp_result=hanlp_sdp_result,
        explicit_entities=explicit_entities,
        state=state,
        entity_ids=explicit_entity_ids,
        terminals=branch_selection["boundary_ids"],
        final_nodes=final_nodes,
        final_edges=final_edges,
        paths=paths,
        selected_paths=paths,
        selection_mode=path_type,
    )
    debug_payload["step4_path_extraction"] = "entity_branch_best_paths"
    debug_payload["semantic_boundary_nodes"] = list(
        global_selection["semantic_boundary_nodes"]
    )
    debug_payload["semantic_node_ids"] = list(branch_selection["semantic_node_ids"])
    debug_payload["semantic_nodes"] = [
        state.nodes[node_id].text
        for node_id in branch_selection["semantic_node_ids"]
        if node_id in state.nodes
    ]
    debug_payload["unsearchable_pas_edges"] = list(
        branch_selection["unsearchable_pas_edges"]
    )
    debug_payload["entity_branch_results"] = list(
        global_selection["entity_branch_results"]
    )
    debug_payload["warnings"] = result_warnings
    debug_file = None
    if debug:
        debug_file = write_debug_json(
            debug_payload, question_id=question_id, debug_dir=debug_dir
        )

    return TokenReasoningStructureResult(
        nodes=final_nodes,
        edges=final_edges,
        paths=paths,
        path_type=path_type,
        anchor_path_results=[],
        global_selection=global_selection,
        answer_anchor=None,
        answer_anchor_id=None,
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
    if any(
        marker in compact
        for marker in ("coord", "conj_member", "disj_member", "_and_c", "_or_c")
    ):
        return "COORD"
    if any(
        marker in compact
        for marker in ("prep_arg", "bv", "det", "aux", "root", "punct", "case", "cop")
    ):
        return "BRIDGE"
    if any(marker in compact for marker in ("rstr", "descr", "relative_arg")):
        return "RESTRICT"
    if (
        any(marker in compact for marker in ("compound", "flat", "app"))
        or compact in {"id"}
        or compact.endswith("_id")
    ):
        return "IDENTITY"
    if any(
        marker in compact
        for marker in ("adj_arg", "noun_arg", "nummod", "numeric", "amod", "modifier")
    ):
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
    warnings: list[str] = list(hanlp_sdp_result.warnings)
    syntax_heads = _normalize_syntax_heads(
        getattr(hanlp_sdp_result, "syntax_heads", {}) or {}, nodes
    )

    for raw_edge in hanlp_sdp_result.edges:
        if not _is_pas_formalism(raw_edge.formalism):
            warning = f"Step4 PAS-only graph ignored non-PAS edge from {raw_edge.formalism}: {raw_edge.display()}"
            if warning not in warnings:
                warnings.append(warning)
            continue
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

    if not raw_edges:
        warnings.append(
            "Step4 PAS-only graph received no sdp/pas edges; DM/PSD fallback is disabled"
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
        syntax_heads=syntax_heads,
        syntax_head_source=getattr(hanlp_sdp_result, "syntax_head_source", "") or "",
    )


def _is_pas_formalism(formalism: str) -> bool:
    normalized = str(formalism or "").lower()
    return (
        normalized == "sdp/pas" or normalized.endswith("/pas") and "sdp" in normalized
    )


def _normalize_syntax_heads(
    syntax_heads: dict[Any, Any],
    nodes: dict[str, TokenReasoningNode],
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for dep_id, head_id in dict(syntax_heads or {}).items():
        dep = str(dep_id)
        if dep not in nodes or dep == "0":
            continue
        try:
            head = int(head_id)
        except (TypeError, ValueError):
            continue
        if head < 0:
            continue
        if head != 0 and str(head) not in nodes:
            continue
        normalized[dep] = head
    return normalized


def _select_entity_branch_best_paths(
    state: _WorkingState,
    explicit_entity_ids: list[str],
) -> dict[str, Any]:
    boundary_degree_graph = _semantic_boundary_degree_graph(state.nodes, state.edges)
    boundary_ids = _semantic_boundary_node_ids(state.nodes, boundary_degree_graph)
    semantic_node_ids = _semantic_degree_node_ids(state.nodes, boundary_degree_graph)
    search_graph = _semantic_path_search_graph(state.nodes, state.edges)
    unsearchable_edges = _unsearchable_pas_edges(state)

    if not explicit_entity_ids:
        state.warnings.append(
            "entity branch path extraction found no explicit entity starts"
        )
    if not boundary_ids:
        state.warnings.append(
            "entity branch path extraction found no semantic boundary nodes"
        )

    branch_results: list[dict[str, Any]] = []
    selected_paths: list[TokenReasoningPath] = []
    selected_payloads: list[dict[str, Any]] = []
    selected_node_paths: set[tuple[str, ...]] = set()
    semantic_node_id_set = set(semantic_node_ids)

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
            path_cost, edge_costs = _path_cost_details(state, path_ids)
            if path_cost is None:
                continue
            candidates.append(
                {
                    "entity_id": entity_id,
                    "entity": state.nodes[entity_id].text,
                    "boundary_id": boundary_id,
                    "boundary": state.nodes[boundary_id].text,
                    "node_ids": list(path_ids),
                    "nodes": [
                        state.nodes[node_id].text
                        for node_id in path_ids
                        if node_id in state.nodes
                    ],
                    "dijkstra_cost": dijkstra_cost,
                    "path_cost": path_cost,
                    "edge_costs": edge_costs,
                }
            )

        branch_semantic_ids: set[str] = set()
        for candidate in candidates:
            for node_id in candidate["node_ids"]:
                if node_id != entity_id and node_id in semantic_node_id_set:
                    branch_semantic_ids.add(node_id)

        for candidate in candidates:
            sp_score, sp_components = _semantic_path_score(
                state=state,
                entity_id=entity_id,
                path_ids=list(candidate["node_ids"]),
                branch_semantic_ids=branch_semantic_ids,
            )
            candidate["sp_score"] = sp_score
            candidate["sp_components"] = sp_components

        selected = (
            dict(max(candidates, key=lambda item: item["sp_score"]))
            if candidates
            else None
        )
        if selected is None:
            state.warnings.append(
                f"entity branch path extraction found no reachable semantic boundary for {state.nodes[entity_id].text}[{entity_id}]"
            )
        else:
            path_key = tuple(selected["node_ids"])
            if path_key not in selected_node_paths:
                selected_node_paths.add(path_key)
                path_id = f"P{len(selected_paths) + 1}"
                selected_path = _path_from_ids(
                    path_id, state.nodes, list(selected["node_ids"])
                )
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
                "candidates": [
                    _entity_branch_candidate_payload(candidate)
                    for candidate in candidates
                ],
                "selected": (
                    _entity_branch_candidate_payload(selected)
                    if selected is not None
                    else None
                ),
            }
        )

    return {
        "paths": selected_paths,
        "selected_paths": selected_payloads,
        "boundary_ids": boundary_ids,
        "semantic_node_ids": semantic_node_ids,
        "boundary_degree_graph": {
            node_id: _sort_node_ids(neighbors, state.nodes)
            for node_id, neighbors in sorted(
                boundary_degree_graph.items(),
                key=lambda item: _node_sort_key(state.nodes[item[0]]),
            )
        },
        "unsearchable_pas_edges": unsearchable_edges,
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
        if not _semantic_degree_node_allowed(
            source
        ) or not _semantic_degree_node_allowed(target):
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
) -> dict[str, list[tuple[str, int, tuple[str, str]]]]:
    graph: dict[str, list[tuple[str, int, tuple[str, str]]]] = {}
    for key in _sorted_edge_keys(edges, nodes):
        source_id, target_id = key
        if source_id not in nodes or target_id not in nodes:
            continue
        source = nodes[source_id]
        target = nodes[target_id]
        if not _semantic_search_node_allowed(
            source
        ) or not _semantic_search_node_allowed(target):
            continue
        edge = edges[key]
        if _edge_is_pure_coordination(edge):
            continue
        cost = _edge_cost(edge)
        if cost is None:
            continue
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
    if (
        lower in DETERMINERS
        or lower in PREPOSITIONS
        or lower in LIGHT_VERBS
        or lower in RELATIVE_PRONOUNS
    ):
        return False
    if (
        node.kind == "function"
        and lower not in WH_ANCHOR_WORDS
        and node.kind != "entity"
    ):
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
    graph: dict[str, list[tuple[str, int, tuple[str, str]]]],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
    *,
    blocked_internal_ids: set[str],
) -> tuple[list[str], int | float]:
    if source_id == target_id:
        return [], math.inf
    if source_id not in graph or target_id not in graph:
        return [], math.inf

    import heapq

    start_key = _path_index_tuple([source_id], nodes)
    heap: list[tuple[float, int, tuple[int, ...], str, list[str]]] = [
        (0.0, 0, start_key, source_id, [source_id])
    ]
    best: dict[str, tuple[float, int, tuple[int, ...]]] = {
        source_id: (0.0, 0, start_key)
    }

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
                heapq.heappush(
                    heap, (next_cost, next_edge_count, next_key, neighbor_id, next_path)
                )
    return [], math.inf


def _semantic_path_score(
    *,
    state: _WorkingState,
    entity_id: str,
    path_ids: list[str],
    branch_semantic_ids: set[str],
) -> tuple[float, dict[str, Any]]:
    path_node_ids_without_entity = [
        node_id
        for node_id in path_ids
        if node_id != entity_id and node_id in state.nodes
    ]
    covered_semantic_ids = _sort_node_ids(
        [
            node_id
            for node_id in path_node_ids_without_entity
            if node_id in branch_semantic_ids
        ],
        state.nodes,
    )
    branch_semantic_node_ids = _sort_node_ids(
        [node_id for node_id in branch_semantic_ids if node_id in state.nodes],
        state.nodes,
    )
    denominator = len(branch_semantic_node_ids) + len(path_node_ids_without_entity)
    sp_score = 2.0 * len(covered_semantic_ids) / denominator if denominator > 0 else 0.0
    components: dict[str, Any] = {
        "branch_semantic_node_ids": branch_semantic_node_ids,
        "branch_semantic_nodes": [
            state.nodes[node_id].text
            for node_id in branch_semantic_node_ids
            if node_id in state.nodes
        ],
        "covered_semantic_node_ids": covered_semantic_ids,
        "covered_semantic_nodes": [
            state.nodes[node_id].text
            for node_id in covered_semantic_ids
            if node_id in state.nodes
        ],
        "branch_semantic_nodes_count": len(branch_semantic_node_ids),
        "covered_semantic_nodes_count": len(covered_semantic_ids),
        "path_node_ids_without_entity": list(path_node_ids_without_entity),
        "path_nodes_without_entity": [
            state.nodes[node_id].text
            for node_id in path_node_ids_without_entity
            if node_id in state.nodes
        ],
        "path_nodes_without_entity_count": len(path_node_ids_without_entity),
    }
    return sp_score, components


def _path_cost_details(
    state: _WorkingState,
    path_ids: list[str],
) -> tuple[int | None, list[dict[str, Any]]]:
    total = 0
    details: list[dict[str, Any]] = []
    for left, right in zip(path_ids, path_ids[1:]):
        edge = state.edges.get(_edge_key(left, right))
        if edge is None:
            return None, []
        edge_cost = _edge_cost(edge)
        if edge_cost is None:
            return None, []
        total += edge_cost
        details.append(
            {
                "source_id": left,
                "source": state.nodes[left].text if left in state.nodes else left,
                "target_id": right,
                "target": state.nodes[right].text if right in state.nodes else right,
                "edge_key": list(_edge_key(left, right)),
                "edge_cost": edge_cost,
                "rules": sorted(_edge_rule_values(edge)),
                "relations": sorted(_edge_pas_relation_keys(edge)),
                "derived": edge.derived,
            }
        )
    return total, details


def _unsearchable_pas_edges(state: _WorkingState) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in _sorted_edge_keys(state.edges, state.nodes):
        edge = state.edges[key]
        if _edge_cost(edge) is not None:
            continue
        relations = sorted(_edge_pas_relation_keys(edge))
        rules = sorted(_edge_rule_values(edge))
        if not relations and not rules:
            continue
        result.append(
            {
                "source_id": edge.source,
                "source": edge.source_text,
                "target_id": edge.target,
                "target": edge.target_text,
                "relations": relations,
                "rules": rules,
                "edge_cost": None,
            }
        )
    return result


def _entity_branch_candidate_payload(
    candidate: dict[str, Any] | None
) -> dict[str, Any] | None:
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
        "path_cost": candidate.get("path_cost"),
        "edge_costs": list(candidate.get("edge_costs") or []),
        "sp_score": candidate.get("sp_score", 0.0),
        "sp_components": dict(candidate.get("sp_components") or {}),
    }
    if "path_id" in candidate:
        payload["path_id"] = candidate["path_id"]
    if "path" in candidate:
        payload["path"] = candidate["path"]
    return payload


def _entity_branch_selected_path_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return _entity_branch_candidate_payload(candidate) or {}


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
                "degree": len(
                    branch_selection["boundary_degree_graph"].get(node_id, [])
                ),
            }
            for node_id in boundary_ids
            if node_id in state.nodes
        ],
        "semantic_node_ids": list(semantic_node_ids),
        "semantic_nodes": [
            state.nodes[node_id].text
            for node_id in semantic_node_ids
            if node_id in state.nodes
        ],
        "unsearchable_pas_edges": list(branch_selection["unsearchable_pas_edges"]),
        "entity_branch_results": list(branch_selection["entity_branch_results"]),
        "paths": list(branch_selection["selected_paths"]),
        "selected_paths": list(branch_selection["selected_paths"]),
        "warnings": list(state.warnings),
    }


def add_pas_preposition_contraction_edges(state: _WorkingState) -> None:
    touched: set[tuple[str, str]] = set()
    for preposition in _sorted_nodes(state.nodes.values()):
        if preposition.text.lower() not in PREPOSITIONS:
            continue
        arg1_edges, arg2_edges = _pas_preposition_role_edges(preposition.id, state)
        if not arg1_edges or not arg2_edges:
            continue
        for arg1_id, arg1_edge in arg1_edges:
            if not _is_high_salience_node(
                state.nodes[arg1_id], include_order_constraints=False
            ):
                continue
            for arg2_id, arg2_edge in arg2_edges:
                if arg1_id == arg2_id:
                    continue
                if not _is_high_salience_node(
                    state.nodes[arg2_id], include_order_constraints=False
                ):
                    continue
                edge_quality = "WEAK"
                support = EDGE_QUALITY_SCORES[edge_quality]
                provenance = {
                    "rule": "pas_preposition_contraction",
                    "edge_quality": edge_quality,
                    "derived": True,
                    "formalism": "sdp/pas",
                    "preposition": preposition.text,
                    "preposition_id": preposition.id,
                    "arg1_id": arg1_id,
                    "arg1": state.nodes[arg1_id].text,
                    "arg2_id": arg2_id,
                    "arg2": state.nodes[arg2_id].text,
                    "collapsed_path": [
                        state.nodes[arg1_id].text,
                        preposition.text,
                        state.nodes[arg2_id].text,
                    ],
                    "source_edges": _edge_provenance_summaries([arg1_edge, arg2_edge]),
                    "support": support,
                }
                _merge_edge(
                    state.edges,
                    state.nodes,
                    arg1_id,
                    arg2_id,
                    support=support,
                    edge_quality=edge_quality,
                    derived=True,
                    rule="pas_preposition_contraction",
                    provenance=[provenance],
                )
                touched.add(_edge_key(arg1_id, arg2_id))
    _append_virtual_edges_for_keys(state, touched)


def add_pas_possessive_contraction_edges(state: _WorkingState) -> None:
    """Collapse explicit PAS possessive markers without treating all "s" as possessive."""

    touched: set[tuple[str, str]] = set()
    for marker in _sorted_nodes(state.nodes.values()):
        if not _is_contextual_possessive_marker(
            marker.id, state.nodes, state.raw_edges
        ):
            continue
        owners, possessed = _possessive_marker_role_edges(marker.id, state)
        for owner_id, owner_edge in owners:
            if not _is_high_salience_node(
                state.nodes[owner_id], include_order_constraints=False
            ):
                continue
            for possessed_id, possessed_edge in possessed:
                if owner_id == possessed_id:
                    continue
                if not _is_high_salience_node(
                    state.nodes[possessed_id], include_order_constraints=False
                ):
                    continue
                edge_quality = "STRONG"
                support = EDGE_QUALITY_SCORES[edge_quality]
                provenance = {
                    "rule": "pas_possessive_contraction",
                    "edge_quality": edge_quality,
                    "derived": True,
                    "formalism": "sdp/pas",
                    "marker": marker.text,
                    "marker_id": marker.id,
                    "owner_id": owner_id,
                    "owner": state.nodes[owner_id].text,
                    "possessed_id": possessed_id,
                    "possessed": state.nodes[possessed_id].text,
                    "collapsed_path": [
                        state.nodes[owner_id].text,
                        marker.text,
                        state.nodes[possessed_id].text,
                    ],
                    "source_edges": _edge_provenance_summaries(
                        [owner_edge, possessed_edge]
                    ),
                    "support": support,
                }
                _merge_edge(
                    state.edges,
                    state.nodes,
                    owner_id,
                    possessed_id,
                    support=support,
                    edge_quality=edge_quality,
                    derived=True,
                    rule="pas_possessive_contraction",
                    provenance=[provenance],
                )
                touched.add(_edge_key(owner_id, possessed_id))
    _append_virtual_edges_for_keys(state, touched)


def add_pas_coordination_candidate_attachment_edges(
    state: _WorkingState,
    explicit_entity_ids: list[str],
) -> None:
    if len(explicit_entity_ids) < 2:
        return
    explicit = set(explicit_entity_ids)
    touched: set[tuple[str, str]] = set()
    for group in _pas_coordination_candidate_groups(state, explicit):
        member_ids = group["member_ids"]
        attachment = _find_syntactic_coordination_attachment(
            state,
            member_ids,
            group["connector_id"],
            state.syntax_heads,
        )
        if attachment is None:
            continue
        shared_id, syntax_evidence = attachment
        for member_id in member_ids:
            if _edge_key(member_id, shared_id) in state.edges:
                continue
            edge_quality = "MEDIUM"
            support = EDGE_QUALITY_SCORES[edge_quality]
            provenance = {
                "rule": "pas_coordination_candidate_attachment",
                "edge_quality": edge_quality,
                "derived": True,
                "formalism": "sdp/pas",
                "connector_id": group["connector_id"],
                "connector": state.nodes[group["connector_id"]].text,
                "member_id": member_id,
                "member": state.nodes[member_id].text,
                "candidate_member_ids": list(member_ids),
                "candidate_members": [
                    state.nodes[node_id].text for node_id in member_ids
                ],
                "syntactic_attachment_id": shared_id,
                "syntactic_attachment": state.nodes[shared_id].text,
                "basis": "syntactic_coordination_head",
                "syntax_head_source": state.syntax_head_source,
                "syntax_head_chain": syntax_evidence["syntax_head_chain"],
                "syntax_head_chain_text": syntax_evidence["syntax_head_chain_text"],
                "coordination_edges": group["evidence"],
                "support": support,
            }
            _merge_edge(
                state.edges,
                state.nodes,
                member_id,
                shared_id,
                support=support,
                edge_quality=edge_quality,
                derived=True,
                rule="pas_coordination_candidate_attachment",
                provenance=[provenance],
            )
            touched.add(_edge_key(member_id, shared_id))
    _append_virtual_edges_for_keys(state, touched)


def _pas_preposition_role_edges(
    preposition_id: str,
    state: _WorkingState,
) -> tuple[list[tuple[str, TokenReasoningEdge]], list[tuple[str, TokenReasoningEdge]]]:
    arg1: dict[str, TokenReasoningEdge] = {}
    arg2: dict[str, TokenReasoningEdge] = {}
    for key, edge in state.raw_edges.items():
        if preposition_id not in key:
            continue
        for item in _raw_provenance(edge):
            if not _is_pas_formalism(str(item.get("formalism") or "")):
                continue
            relation = _normalized_relation_key(
                str(item.get("normalized_relation") or item.get("relation") or "")
            )
            head_idx = _coerce_provenance_index(item.get("head_idx"))
            dep_idx = _coerce_provenance_index(item.get("dep_idx"))
            if str(head_idx) != preposition_id:
                continue
            related_id = str(dep_idx)
            if related_id not in state.nodes:
                continue
            if relation == "prep_arg1":
                arg1[related_id] = edge
            elif relation == "prep_arg2":
                arg2[related_id] = edge
    return (
        sorted(arg1.items(), key=lambda item: _node_sort_key(state.nodes[item[0]])),
        sorted(arg2.items(), key=lambda item: _node_sort_key(state.nodes[item[0]])),
    )


def _pas_coordination_candidate_groups(
    state: _WorkingState,
    explicit_entity_ids: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for key, edge in state.raw_edges.items():
        for item in _raw_provenance(edge):
            if not _is_pas_formalism(str(item.get("formalism") or "")):
                continue
            relation = _normalized_relation_key(
                str(item.get("normalized_relation") or item.get("relation") or "")
            )
            if not _is_coordination_relation_key(relation):
                continue
            head_id = str(_coerce_provenance_index(item.get("head_idx")))
            dep_id = str(_coerce_provenance_index(item.get("dep_idx")))
            connector_id: str | None = None
            member_id: str | None = None
            if (
                head_id in state.nodes
                and _is_pas_candidate_connector(state.nodes[head_id])
                and dep_id in explicit_entity_ids
            ):
                connector_id = head_id
                member_id = dep_id
            elif (
                dep_id in state.nodes
                and _is_pas_candidate_connector(state.nodes[dep_id])
                and head_id in explicit_entity_ids
            ):
                connector_id = dep_id
                member_id = head_id
            if connector_id is None or member_id is None:
                continue
            group = grouped.setdefault(
                connector_id,
                {
                    "connector_id": connector_id,
                    "member_ids": set(),
                    "evidence": [],
                },
            )
            group["member_ids"].add(member_id)
            group["evidence"].append(_edge_provenance_summary(edge))

    results: list[dict[str, Any]] = []
    for connector_id, group in grouped.items():
        member_ids = _sort_node_ids(group["member_ids"], state.nodes)
        if len(member_ids) < 2:
            continue
        results.append(
            {
                "connector_id": connector_id,
                "member_ids": member_ids,
                "evidence": group["evidence"],
            }
        )
    return sorted(
        results, key=lambda item: _node_sort_key(state.nodes[item["connector_id"]])
    )


def _find_syntactic_coordination_attachment(
    state: _WorkingState,
    member_ids: list[str],
    connector_id: str,
    syntax_heads: dict[str, int],
) -> tuple[str, dict[str, Any]] | None:
    connector = state.nodes[connector_id]
    if not syntax_heads:
        state.warnings.append(
            "pas coordination candidate attachment skipped for "
            f"{connector.text}[{connector_id}] because syntactic dependency heads are missing"
        )
        return None

    group_nodes = set(member_ids)
    group_nodes.add(connector_id)
    attachments: dict[str, str] = {}
    chains: dict[str, list[str]] = {}
    chain_texts: dict[str, list[str]] = {}

    for member_id in member_ids:
        resolved = _syntactic_coordination_member_attachment(
            state,
            member_id,
            group_nodes,
            syntax_heads,
        )
        if resolved is None:
            return None
        attachment_id, chain_ids = resolved
        attachments[member_id] = attachment_id
        chains[member_id] = chain_ids
        chain_texts[member_id] = [
            _syntax_chain_node_label(state, node_id) for node_id in chain_ids
        ]

    unique_attachments = set(attachments.values())
    if len(unique_attachments) != 1:
        rendered = ", ".join(
            f"{state.nodes[member_id].text}[{member_id}]->{_syntax_chain_node_label(state, attachment_id)}"
            for member_id, attachment_id in sorted(
                attachments.items(),
                key=lambda item: _node_sort_key(state.nodes[item[0]]),
            )
        )
        state.warnings.append(
            "pas coordination candidate attachment skipped for "
            f"{connector.text}[{connector_id}] because members resolve to different syntactic attachments: {rendered}"
        )
        return None

    attachment_id = next(iter(unique_attachments))
    if not _valid_syntactic_coordination_attachment(state, attachment_id):
        state.warnings.append(
            "pas coordination candidate attachment skipped for "
            f"{connector.text}[{connector_id}] because syntactic attachment "
            f"{_syntax_chain_node_label(state, attachment_id)} is invalid"
        )
        return None

    return attachment_id, {
        "syntax_head_chain": chains,
        "syntax_head_chain_text": chain_texts,
    }


def _syntactic_coordination_member_attachment(
    state: _WorkingState,
    member_id: str,
    group_nodes: set[str],
    syntax_heads: dict[str, int],
) -> tuple[str, list[str]] | None:
    connector_ids = [
        node_id
        for node_id in group_nodes
        if node_id in state.nodes and _is_pas_candidate_connector(state.nodes[node_id])
    ]
    connector_id = connector_ids[0] if connector_ids else next(iter(group_nodes))
    current = member_id
    chain = [current]
    seen: set[str] = set()
    while True:
        if current in seen:
            state.warnings.append(
                "pas coordination candidate attachment skipped for "
                f"{state.nodes[connector_id].text}[{connector_id}] because syntactic parent chain loops at "
                f"{_syntax_chain_node_label(state, current)}"
            )
            return None
        seen.add(current)
        if current not in syntax_heads:
            state.warnings.append(
                "pas coordination candidate attachment skipped for "
                f"{state.nodes[connector_id].text}[{connector_id}] because syntactic head is missing for "
                f"{_syntax_chain_node_label(state, current)}"
            )
            return None
        head_id = str(syntax_heads[current])
        chain.append(head_id)
        if head_id in group_nodes:
            current = head_id
            continue
        return head_id, chain


def _is_coordination_relation_key(relation: str) -> bool:
    return "coord" in relation or relation in {"conj_member", "disj_member"}


def _is_pas_candidate_connector(node: TokenReasoningNode) -> bool:
    return node.text.lower() in {"and", "or"}


def _valid_syntactic_coordination_attachment(
    state: _WorkingState, attachment_id: str
) -> bool:
    if attachment_id == "0" or attachment_id not in state.nodes:
        return False
    node = state.nodes[attachment_id]
    if _is_punctuation(node.text):
        return False
    if _is_pas_candidate_connector(node):
        return False
    return True


def _syntax_chain_node_label(state: _WorkingState, node_id: str) -> str:
    if node_id == "0":
        return "ROOT[0]"
    if node_id in state.nodes:
        return f"{state.nodes[node_id].text}[{node_id}]"
    return f"?[{node_id}]"


def _append_virtual_edges_for_keys(
    state: _WorkingState, keys: set[tuple[str, str]]
) -> None:
    for key in sorted(
        keys,
        key=lambda item: (
            _node_sort_key(state.nodes[item[0]]),
            _node_sort_key(state.nodes[item[1]]),
        ),
    ):
        edge = state.edges.get(key)
        if edge is not None:
            state.virtual_edges.append(edge.to_dict())


def _path_cost_from_edge_map(
    edges: dict[tuple[str, str], TokenReasoningEdge],
    path: list[str],
) -> int | None:
    total = 0
    for left, right in zip(path, path[1:]):
        edge = edges.get(_edge_key(left, right))
        if edge is None:
            return None
        edge_cost = _edge_cost(edge)
        if edge_cost is None:
            return None
        total += edge_cost
    return total


def _edge_label_classes_deep(edge: TokenReasoningEdge) -> set[str]:
    classes = set(_edge_label_classes(edge))
    classes.update(_label_class_values_from_payload(edge.provenance))
    return classes


def _graph_from_selected_paths(
    paths: list[TokenReasoningPath],
) -> tuple[set[str], set[tuple[str, str]]]:
    node_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for path in paths:
        node_ids.update(path.node_ids)
        for left, right in zip(path.node_ids, path.node_ids[1:]):
            pairs.add(_edge_key(left, right))
    return node_ids, pairs


def _active_entity_ids(
    nodes: dict[str, TokenReasoningNode], paths: list[TokenReasoningPath]
) -> list[str]:
    active: list[str] = []
    for path in paths:
        if (
            path.node_ids
            and path.node_ids[0] in nodes
            and nodes[path.node_ids[0]].kind == "entity"
        ):
            active.append(path.node_ids[0])
    return _sort_node_ids(active, nodes)


def write_debug_json(
    payload: dict[str, Any], *, question_id: str | None, debug_dir: str | Path | None
) -> str:
    directory = (
        Path(debug_dir) if debug_dir is not None else Path("debug") / "hanlp_sdp"
    )
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_safe_filename(question_id) if question_id else 'q1'}_tri_sdp_reasoning.json"
    )
    path = directory / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(path)


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
    entity_ids: list[str],
    terminals: list[str],
    final_nodes: list[TokenReasoningNode],
    final_edges: list[TokenReasoningEdge],
    paths: list[TokenReasoningPath],
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
        "syntax_heads": dict(state.syntax_heads),
        "syntax_head_source": state.syntax_head_source,
        "normalized_evidence_edges": list(state.normalized_edges),
        "aggregated_edges": [
            edge.to_dict()
            for edge in _sorted_edges(state.raw_edges.values(), state.nodes)
        ],
        "repaired_evidence_edges": [
            edge.to_dict() for edge in _sorted_edges(state.edges.values(), state.nodes)
        ],
        "virtual_edges": list(state.virtual_edges),
        "entity_anchors": [
            {"id": node_id, "text": state.nodes[node_id].text}
            for node_id in entity_ids
            if node_id in state.nodes
        ],
        "terminals": [
            state.nodes[node_id].text for node_id in terminals if node_id in state.nodes
        ],
        "final_nodes": [node.to_dict() for node in final_nodes],
        "final_edges": [edge.to_dict() for edge in final_edges],
        "paths": [path.to_dict() for path in paths],
        "selected_paths": [path.to_dict() for path in (selected_paths or paths)],
        "selection_mode": selection_mode,
        "warnings": list(state.warnings),
    }


def _build_token_nodes(
    hanlp_sdp_result: HanLPSDPResult,
) -> dict[str, TokenReasoningNode]:
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


def _ensure_edge_nodes(
    nodes: dict[str, TokenReasoningNode], edge: HanLPSDPEdge
) -> None:
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
    entity_order = {
        entity: position for position, entity in enumerate(explicit_entities)
    }
    matched = [
        node.id
        for node in nodes.values()
        if node.text in entity_order and ENTITY_RE.fullmatch(node.text)
    ]
    return sorted(
        set(matched),
        key=lambda node_id: (
            entity_order.get(nodes[node_id].text, 9999),
            _node_sort_key(nodes[node_id]),
        ),
    )


def _mark_anchors(nodes: dict[str, TokenReasoningNode], entity_ids: list[str]) -> None:
    for node_id in entity_ids:
        if node_id in nodes:
            nodes[node_id].is_anchor = True


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
    if rule in EDGE_COST_TWO_RULES:
        return "MEDIUM"
    if rule in EDGE_COST_ONE_RULES:
        return "STRONG"
    labels = _label_class_values_from_payload(provenance)
    if labels:
        return (
            _highest_quality(
                LABEL_CLASS_EDGE_QUALITY.get(label, "WEAK") for label in labels
            )
            or "WEAK"
        )
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


def _walk_provenance_payload(
    payload: Any, *, max_depth: int = _PROVENANCE_WALK_MAX_DEPTH
) -> Iterable[dict[str, Any]]:
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
    consensus_count = max(
        edge.consensus_count, _consensus_count_from_provenance(edge.provenance)
    )
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
        "edge_cost": _edge_cost(edge),
    }
    return summary


def _edge_provenance_summaries(
    edges: Iterable[TokenReasoningEdge | None],
) -> list[dict[str, Any]]:
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
    edge_quality = _normalize_edge_quality(
        edge_quality
        or _infer_edge_quality(nodes, source_id, target_id, rule, provenance)
    )
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
    return (
        _normalize_relation(relation)
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def _raw_provenance(edge: TokenReasoningEdge) -> list[dict[str, Any]]:
    return [
        item
        for item in edge.provenance
        if isinstance(item, dict) and "head_idx" in item and "dep_idx" in item
    ]


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
    return any(
        _is_possessive_relation(item.get("normalized_relation") or item.get("relation"))
        for item in incident
    )


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
            relation = _normalized_relation_key(
                str(item.get("normalized_relation") or item.get("relation") or "")
            )
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
    owner_items = sorted(
        owners.items(), key=lambda item: _node_sort_key(state.nodes[item[0]])
    )
    possessed_items = sorted(
        possessed.items(), key=lambda item: _node_sort_key(state.nodes[item[0]])
    )
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


def _is_punctuation(text: str) -> bool:
    return all(not char.isalnum() for char in text)


def _is_high_salience_node(
    node: TokenReasoningNode, *, include_order_constraints: bool
) -> bool:
    if node.kind in {"entity", "content", "answer"}:
        return True
    if node.kind == "constraint":
        return include_order_constraints or NUMERIC_RE.fullmatch(node.text) is not None
    return False


def _edge_key(source_id: str, target_id: str) -> tuple[str, str]:
    return tuple(sorted((source_id, target_id), key=lambda item: int(item) if str(item).lstrip("-").isdigit() else 10**9))  # type: ignore[return-value]


def _node_sort_key(node: TokenReasoningNode) -> tuple[int, str]:
    return (node.index, node.text)


def _sort_node_ids(
    node_ids: Iterable[str], nodes: dict[str, TokenReasoningNode]
) -> list[str]:
    return sorted(
        dict.fromkeys(node_ids), key=lambda node_id: _node_sort_key(nodes[node_id])
    )


def _sorted_nodes(nodes: Iterable[TokenReasoningNode]) -> list[TokenReasoningNode]:
    return sorted(nodes, key=_node_sort_key)


def _sorted_edge_keys(
    edges: dict[tuple[str, str], TokenReasoningEdge],
    nodes: dict[str, TokenReasoningNode],
) -> list[tuple[str, str]]:
    return sorted(
        edges,
        key=lambda key: (_node_sort_key(nodes[key[0]]), _node_sort_key(nodes[key[1]])),
    )


def _sorted_edges(
    edges: Iterable[TokenReasoningEdge],
    nodes: dict[str, TokenReasoningNode],
) -> list[TokenReasoningEdge]:
    return sorted(
        edges,
        key=lambda edge: (
            _node_sort_key(nodes[edge.source]),
            _node_sort_key(nodes[edge.target]),
        ),
    )


def _edge_label_classes(edge: TokenReasoningEdge) -> set[str]:
    return {
        str(item.get("label_class"))
        for item in edge.provenance
        if isinstance(item, dict) and item.get("label_class")
    }


def _edge_cost(edge: TokenReasoningEdge) -> int | None:
    rules = _edge_rule_values(edge)
    if rules & EDGE_COST_ONE_RULES:
        return 1
    if rules & EDGE_COST_TWO_RULES:
        return 2

    finite_costs: list[int] = []
    for relation in _edge_pas_relation_keys(edge):
        if relation in EDGE_COST_ONE_RELATIONS:
            finite_costs.append(1)
        elif relation in EDGE_COST_TWO_RELATIONS:
            finite_costs.append(2)
    return min(finite_costs) if finite_costs else None


def _edge_rule_values(edge: TokenReasoningEdge) -> set[str]:
    values: set[str] = set()
    for rule in _combine_rule_values(edge.rule):
        if rule:
            values.add(rule)
    for item in _walk_provenance_payload(edge.provenance):
        for rule in _combine_rule_values(str(item.get("rule") or "")):
            if rule:
                values.add(rule)
    return values


def _edge_pas_relation_keys(edge: TokenReasoningEdge) -> set[str]:
    relations: set[str] = set()
    for item in _walk_provenance_payload(edge.provenance):
        if not _payload_has_pas_formalism(item):
            continue
        for key in ("normalized_relation", "relation"):
            relation = item.get(key)
            if relation:
                relations.add(_normalized_relation_key(str(relation)))
        for key in ("normalized_relations", "relations"):
            for relation in _iter_sequence(item.get(key)):
                if relation:
                    relations.add(_normalized_relation_key(str(relation)))
    return relations


def _payload_has_pas_formalism(item: dict[str, Any]) -> bool:
    formalism = item.get("formalism")
    if formalism and _is_pas_formalism(str(formalism)):
        return True
    return any(
        _is_pas_formalism(str(value))
        for value in _iter_sequence(item.get("formalisms"))
    )


def _combine_rule_values(rule: str) -> set[str]:
    return {part for part in str(rule or "").split("+") if part}


def _path_index_tuple(
    path: list[str], nodes: dict[str, TokenReasoningNode]
) -> tuple[int, ...]:
    return tuple(nodes[node_id].index for node_id in path if node_id in nodes)


def _plain_adjacency(pairs: set[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for source, target in pairs:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return adjacency


def _path_from_ids(
    path_id: str, nodes: dict[str, TokenReasoningNode], node_ids: list[str]
) -> TokenReasoningPath:
    return TokenReasoningPath(
        path_id=path_id,
        nodes=[nodes[node_id].text for node_id in node_ids if node_id in nodes],
        node_ids=[node_id for node_id in node_ids if node_id in nodes],
    )


def _node_ids_from_pairs(pairs: set[tuple[str, str]], terminals: list[str]) -> set[str]:
    node_ids = set(terminals)
    for source, target in pairs:
        node_ids.add(source)
        node_ids.add(target)
    return node_ids


def _final_nodes(
    nodes: dict[str, TokenReasoningNode], node_ids: Iterable[str]
) -> list[TokenReasoningNode]:
    return [
        TokenReasoningNode(
            id=node.id,
            text=node.text,
            index=node.index,
            kind=node.kind,
            is_anchor=node.is_anchor,
        )
        for node in _sorted_nodes(
            nodes[node_id]
            for node_id in set(node_ids)
            if node_id in nodes and node_id != "0"
        )
    ]


def _final_edges(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
    final_pairs: set[tuple[str, str]],
    paths: list[TokenReasoningPath],
    entity_ids: list[str],
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
    distances = _graph_distances(_plain_adjacency(final_pairs), root) if root else {}

    final_edges: list[TokenReasoningEdge] = []
    for key in sorted(
        final_pairs,
        key=lambda item: _oriented_edge_sort_key(item, orientations, distances, nodes),
    ):
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


def _graph_distances(
    adjacency: dict[str, set[str]], root: str | None
) -> dict[str, int]:
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
