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
    "restriction_closure": 0.25,
    "descriptor_lifting": 0.35,
    "candidate_expansion": 0.10,
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
ORDER_CUES = {"first", "earliest", "latest", "last", "older", "oldest", "younger", "youngest"}
APPROX_CUES = {"approximately", "about", "around", "roughly"}


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


def compile_token_reasoning_structure(
    hanlp_sdp_result: HanLPSDPResult,
    explicit_entities: list[str],
    *,
    masked_question: str | None = None,
    question_id: str | None = None,
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> TokenReasoningStructureResult:
    """Compile three HanLP SDP graph views into a deterministic token graph.

    The compiler is intentionally symbolic and deterministic: it consumes DM,
    PAS, and PSD edge evidence, adds generic virtual edges by graph operations,
    extracts a small undirected backbone, and emits a graph plus a path cover.
    It does not call an LLM and it does not introduce semantic relation labels.
    """

    state = build_evidence_graph(hanlp_sdp_result)
    explicit_entity_ids = _resolve_explicit_entity_ids(state.nodes, explicit_entities)

    answer_anchor_id = detect_answer_anchor(state.nodes, state.raw_edges, state.warnings)
    _mark_anchors(state.nodes, explicit_entity_ids, answer_anchor_id)

    constraints = detect_constraints(state.nodes, state.raw_edges, answer_anchor_id)
    candidate_sets = detect_candidate_sets(state.nodes, state.raw_edges, explicit_entity_ids)

    add_bridge_contraction_edges(state)
    add_restriction_closure_edges(state)
    add_descriptor_lifting_edges(state, explicit_entity_ids, answer_anchor_id)

    if candidate_sets:
        final_node_ids, final_pairs, paths, path_type = _extract_candidate_path_cover(
            state,
            candidate_sets,
            answer_anchor_id,
            constraints,
        )
        backbone_before = _graph_snapshot(final_node_ids, final_pairs, state.nodes, state.edges)
        backbone_after = backbone_before
    else:
        terminals = _select_terminals(state.nodes, state.edges, explicit_entity_ids, answer_anchor_id, constraints)
        backbone_pairs = extract_steiner_backbone(state.nodes, state.edges, terminals)
        backbone_before = _graph_snapshot(_node_ids_from_pairs(backbone_pairs, terminals), backbone_pairs, state.nodes, state.edges)
        pruned_pairs = prune_backbone(state.nodes, state.edges, backbone_pairs, terminals, answer_anchor_id)
        final_node_ids = _node_ids_from_pairs(pruned_pairs, terminals)
        final_pairs = pruned_pairs
        paths, path_type = linearize_paths(state.nodes, final_pairs, explicit_entity_ids, answer_anchor_id, terminals)
        backbone_after = _graph_snapshot(final_node_ids, final_pairs, state.nodes, state.edges)

    final_nodes = _final_nodes(state.nodes, final_node_ids)
    final_edges = _final_edges(state.nodes, state.edges, final_pairs, paths, explicit_entity_ids, answer_anchor_id)

    if not final_nodes and explicit_entity_ids:
        final_nodes = _final_nodes(state.nodes, explicit_entity_ids)
        paths = [
            TokenReasoningPath(path_id=f"P{index}", nodes=[state.nodes[node_id].text], node_ids=[node_id])
            for index, node_id in enumerate(explicit_entity_ids, start=1)
        ]
        path_type = "empty"

    answer_anchor = state.nodes[answer_anchor_id].text if answer_anchor_id in state.nodes else None
    entity_anchors = [state.nodes[node_id].text for node_id in explicit_entity_ids if node_id in state.nodes]
    terminals_for_debug = _select_terminals(state.nodes, state.edges, explicit_entity_ids, answer_anchor_id, constraints)
    debug_payload = _build_debug_payload(
        question_id=question_id,
        masked_question=masked_question or hanlp_sdp_result.text,
        hanlp_sdp_result=hanlp_sdp_result,
        explicit_entities=explicit_entities,
        state=state,
        answer_anchor_id=answer_anchor_id,
        entity_ids=explicit_entity_ids,
        constraints=constraints,
        candidate_sets=candidate_sets,
        terminals=terminals_for_debug,
        backbone_before=backbone_before,
        backbone_after=backbone_after,
        final_nodes=final_nodes,
        final_edges=final_edges,
        paths=paths,
    )
    debug_file = None
    if debug:
        debug_file = write_debug_json(debug_payload, question_id=question_id, debug_dir=debug_dir)

    return TokenReasoningStructureResult(
        nodes=final_nodes,
        edges=final_edges,
        paths=paths,
        path_type=path_type,
        answer_anchor=answer_anchor,
        answer_anchor_id=answer_anchor_id,
        entity_anchors=entity_anchors,
        constraints=constraints,
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
    typed_wh = _find_typed_wh_slot(nodes, raw_edges)
    if typed_wh:
        warnings.append("answer anchor selected by typed-wh slot")
        return typed_wh

    query_root = _find_query_root(nodes, raw_edges)
    if query_root:
        projection = _collect_root_projection_candidates(nodes, raw_edges, query_root)
        if projection:
            warnings.append("answer anchor selected by root projection")
            return projection
        modifier_projection = _find_modifier_projection_candidate(nodes, raw_edges, query_root)
        if modifier_projection:
            warnings.append("answer anchor selected by modifier projection")
            return modifier_projection

    wh_focus = _find_wh_fallback_anchor(nodes, raw_edges)
    if wh_focus:
        warnings.append("answer anchor selected by wh fallback")
        return wh_focus

    root_candidate = _root_candidate(nodes, raw_edges)
    if root_candidate:
        warnings.append("answer anchor fallback: root/high-salience content token")
        return root_candidate
    warnings.append("answer anchor fallback failed")
    return None


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
                {"positive": 0.0, "negative": 0.0, "formalisms": set(), "positive_count": 0},
            )
            if polarity == "forward":
                candidate["positive"] += support
                candidate["positive_count"] += 1
                if formalism:
                    candidate["formalisms"].add(formalism)
            elif polarity == "subject":
                candidate["negative"] += support

    scored: list[tuple[float, int, int, str]] = []
    for candidate_id, data in candidates.items():
        positive = float(data["positive"])
        if positive <= 0.0 or int(data["positive_count"]) <= 0:
            continue
        total = positive - float(data["negative"])
        if total <= 0.0:
            continue
        formalisms = data["formalisms"]
        node = nodes[candidate_id]
        scored.append((-total, -len(formalisms), node.index, candidate_id))
    return sorted(scored)[0][3] if scored else None


def _find_modifier_projection_candidate(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: dict[tuple[str, str], TokenReasoningEdge],
    query_root_id: str,
) -> str | None:
    if query_root_id not in nodes:
        return None
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
                    best_by_candidate[node_id] = {"score": score, "formalisms": formalisms}
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
    return sorted(scored)[0][3] if scored else None


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


def write_debug_json(payload: dict[str, Any], *, question_id: str | None, debug_dir: str | Path | None) -> str:
    directory = Path(debug_dir) if debug_dir is not None else Path("debug") / "hanlp_sdp"
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_filename(question_id) if question_id else 'q1'}_tri_sdp_reasoning.json"
    path = directory / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


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
