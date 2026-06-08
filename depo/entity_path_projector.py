from __future__ import annotations

import re
from typing import Any

import networkx as nx

from models import (
    EntityOriginPath,
    EntityStartNode,
    MaskReplacement,
    RestoredGraphNodeCandidate,
    SelectedEntityPath,
    SemanticASTEdge,
    SemanticASTNode,
    SemanticASTResult,
)


ENTITY_SEMANTIC_TYPES = {
    "album",
    "book",
    "city",
    "company",
    "country",
    "creativework",
    "event",
    "film",
    "game",
    "institution",
    "location",
    "movie",
    "organization",
    "organisation",
    "person",
    "place",
    "product",
    "region",
    "series",
    "song",
    "university",
    "work",
}

TYPE_VARIABLE_SURFACES = {
    "actor",
    "age",
    "author",
    "ceo",
    "city",
    "company",
    "country",
    "date",
    "director",
    "film",
    "movie",
    "nationality",
    "population",
    "region",
    "spouse",
    "university",
}

FUNCTION_SURFACES = {
    "?",
    "a",
    "an",
    "and",
    "are",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "same",
    "share",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}

TERMINAL_GLUE_TOKENS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "to",
    "with",
    "about",
    "as",
    "into",
    "over",
    "under",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "and",
    "or",
    "but",
    "?",
    ".",
    ",",
    ";",
    ":",
    "!",
    "'",
    '"',
    "``",
    "''",
}

TERMINAL_GLUE_DEP_LABELS = {
    "det",
    "case",
    "aux",
    "aux:pass",
    "punct",
    "cop",
    "cc",
    "mark",
}

WH_TOKENS = {"who", "what", "when", "where", "which", "whom", "whose"}
_TERMINAL_STRIP_CHARS = " \t\r\n.,;:!?\"'`“”‘’()[]{}"

ANSWER_CUES = {
    "age",
    "city",
    "country",
    "date",
    "nationality",
    "population",
    "university",
    "what",
    "when",
    "where",
    "which",
    "who",
}

ROLE_SLOT_CUES = {
    "actor",
    "author",
    "ceo",
    "company",
    "director",
    "founder",
    "mother",
    "nationality",
    "spouse",
    "university",
    "wife",
}

COMPARISON_CUE_SURFACES = {
    "after",
    "before",
    "both",
    "different",
    "differ",
    "earlier",
    "earliest",
    "fewer",
    "fewest",
    "first",
    "greater",
    "highest",
    "larger",
    "largest",
    "last",
    "later",
    "latest",
    "less",
    "lowest",
    "more",
    "most",
    "older",
    "same",
    "share",
    "shared",
    "smaller",
    "smallest",
    "younger",
}

IMPLICIT_COMPARISON_ATTRIBUTE_SURFACES = {
    "age",
    "birth date",
    "date of birth",
    "date_of_birth",
}

IMPLICIT_VALUE_SLOT_SURFACES = IMPLICIT_COMPARISON_ATTRIBUTE_SURFACES | {
    "birth_date",
    "birthplace",
    "date of death",
    "death date",
    "death_date",
    "cause of death",
    "cause_of_death",
    "death cause",
    "death reason",
    "death_cause",
    "death_reason",
    "location",
    "manner",
    "place of birth",
    "reason",
    "release date",
    "release_date",
}

def build_entity_start_nodes_from_explicit_entities(
    dependency_graph: nx.Graph,
    restored_graph_node_candidates: list[RestoredGraphNodeCandidate],
    replacement: MaskReplacement,
) -> list[EntityStartNode]:
    """Map Step 2/3 explicit entity placeholders to dependency graph start nodes."""

    by_placeholder = {
        str(candidate.placeholder or candidate.graph_text): candidate
        for candidate in restored_graph_node_candidates
        if candidate.placeholder or candidate.graph_text
    }
    starts: list[EntityStartNode] = []
    seen_placeholders: set[str] = set()
    missing_placeholders: list[str] = []

    mappings = sorted(
        replacement.mask_mappings,
        key=lambda item: (
            item.original_char_span[0] if item.original_char_span else 10**9,
            item.masked_char_span[0] if item.masked_char_span else 10**9,
            item.placeholder,
        ),
    )
    for mapping in mappings:
        if str(mapping.kind_hint or "").strip().lower() != "entity":
            continue
        if mapping.placeholder in seen_placeholders:
            continue
        seen_placeholders.add(mapping.placeholder)
        candidate = by_placeholder.get(mapping.placeholder)
        if candidate is None:
            candidate = _candidate_from_graph_placeholder(dependency_graph, mapping.placeholder)
        if candidate is None or str(candidate.node_id) not in {str(node_id) for node_id in dependency_graph.nodes}:
            missing_placeholders.append(mapping.placeholder)
            continue
        node_id = str(candidate.node_id)
        token_ids = list(candidate.source_token_indices or [candidate.token_index])
        starts.append(
            EntityStartNode(
                entity_id=f"e{len(starts) + 1}",
                text=mapping.original_text or candidate.text or candidate.display_text,
                graph_node_ids=[node_id],
                token_ids=[int(item) for item in token_ids if item is not None],
                kind_hint="entity",
                semantic_type_hint=mapping.semantic_type_hint or candidate.semantic_type_hint,
            )
        )

    if missing_placeholders:
        raise ValueError(
            "Explicit entity placeholder(s) were not found in the dependency graph: "
            + ", ".join(missing_placeholders)
        )

    for index, start in enumerate(starts, start=1):
        start.entity_id = f"e{index}"
    return starts


def extract_entity_start_nodes(
    dependency_graph: nx.Graph,
    restored_graph_node_candidates: list[RestoredGraphNodeCandidate],
    replacement: MaskReplacement,
) -> list[EntityStartNode]:
    """Legacy wrapper; Step 6 no longer re-detects entities or uses POS fallback."""

    return build_entity_start_nodes_from_explicit_entities(
        dependency_graph=dependency_graph,
        restored_graph_node_candidates=restored_graph_node_candidates,
        replacement=replacement,
    )


def _candidate_from_graph_placeholder(
    dependency_graph: nx.Graph,
    placeholder: str,
) -> RestoredGraphNodeCandidate | None:
    for node_id, attrs in dependency_graph.nodes(data=True):
        text = str(attrs.get("word") or attrs.get("text") or "").strip()
        if text != placeholder:
            continue
        token_index = int(attrs.get("order") or node_id)
        return RestoredGraphNodeCandidate(
            node_id=str(node_id),
            token_index=token_index,
            graph_text=placeholder,
            placeholder=placeholder,
            restored_text=placeholder,
            display_text=placeholder,
            is_mask_placeholder=True,
            kind_hint="entity_candidate",
            source_token_indices=[token_index],
            text=placeholder,
        )
    return None


def enumerate_entity_origin_paths(
    dependency_graph: nx.Graph,
    entity_starts: list[EntityStartNode],
    max_path_len: int = 7,
    max_paths_per_entity: int = 80,
) -> list[EntityOriginPath]:
    """Enumerate bounded simple dependency paths from each entity start."""

    if max_path_len < 2:
        return []
    result: list[EntityOriginPath] = []
    entity_node_ids = {
        entity.entity_id: {str(node_id) for node_id in entity.graph_node_ids}
        for entity in entity_starts
    }
    for entity in entity_starts:
        raw_paths: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        other_entity_node_ids = {
            node_id
            for other_entity_id, node_ids in entity_node_ids.items()
            if other_entity_id != entity.entity_id
            for node_id in node_ids
        }
        for graph_node_id in entity.graph_node_ids:
            if graph_node_id not in dependency_graph:
                continue
            _walk_simple_paths(
                dependency_graph=dependency_graph,
                current=graph_node_id,
                path=[graph_node_id],
                seen=seen,
                raw_paths=raw_paths,
                max_path_len=max_path_len,
                raw_limit=max_paths_per_entity * 30,
            )

        ordered_paths = sorted(
            raw_paths,
            key=lambda path: _entity_path_sort_key(
                dependency_graph,
                path,
                other_entity_node_ids=other_entity_node_ids,
            ),
        )[:max_paths_per_entity]
        for path_index, path in enumerate(ordered_paths, start=1):
            node_ids = [str(item) for item in path]
            result.append(
                EntityOriginPath(
                    path_id=f"{entity.entity_id}_p{path_index}",
                    entity_id=entity.entity_id,
                    entity_text=entity.text,
                    nodes=[_node_text(dependency_graph, node_id) for node_id in node_ids],
                    node_ids=node_ids,
                    length=len(node_ids),
                    evidence=_dependency_edge_evidence(dependency_graph, node_ids),
                )
            )
    return result


def prune_terminal_glue_paths(
    entity_origin_paths: list[EntityOriginPath],
    dependency_graph: nx.Graph | None = None,
    entity_start_nodes: list[EntityStartNode] | None = None,
    *,
    min_keep_per_entity: int = 3,
    keep_wh_terminals: bool = True,
) -> tuple[list[EntityOriginPath], dict[str, Any]]:
    """Remove paths whose terminal node is glue/function syntax.

    Paths may contain glue/function tokens internally. This pruning only removes
    paths whose terminal node is a glue/function token.
    """

    grouped: dict[str, list[EntityOriginPath]] = {}
    entity_order: list[str] = []
    for path in entity_origin_paths:
        grouped.setdefault(path.entity_id, []).append(path)
        if path.entity_id not in entity_order:
            entity_order.append(path.entity_id)

    if entity_start_nodes:
        ordered_from_entities = [entity.entity_id for entity in entity_start_nodes]
        entity_order = [
            *[entity_id for entity_id in ordered_from_entities if entity_id in grouped],
            *[entity_id for entity_id in entity_order if entity_id not in ordered_from_entities],
        ]

    pruned_paths: list[EntityOriginPath] = []
    stats_by_entity: dict[str, dict[str, Any]] = {}
    total_raw = 0
    total_kept = 0

    for entity_id in entity_order:
        raw_paths = grouped.get(entity_id, [])
        total_raw += len(raw_paths)
        kept_for_entity: list[EntityOriginPath] = []
        pruned_reasons: dict[str, str] = {}

        for path in raw_paths:
            reason = _terminal_glue_prune_reason(
                path,
                dependency_graph=dependency_graph,
                keep_wh_terminals=keep_wh_terminals,
            )
            if reason:
                pruned_reasons[path.path_id] = reason
                continue
            kept_for_entity.append(path)

        fallback_used = False
        if raw_paths and not kept_for_entity:
            fallback_used = True
            fallback_count = min(max(min_keep_per_entity, 1), len(raw_paths))
            kept_for_entity = sorted(raw_paths, key=_terminal_glue_fallback_sort_key)[:fallback_count]

        kept_ids = {path.path_id for path in kept_for_entity}
        dropped_paths = [path for path in raw_paths if path.path_id not in kept_ids]
        pruned_paths.extend(kept_for_entity)
        total_kept += len(kept_for_entity)

        examples = [
            {
                "path_id": path.path_id,
                "terminal": _path_terminal_text(path),
                "reason": pruned_reasons.get(path.path_id)
                or _terminal_glue_prune_reason(
                    path,
                    dependency_graph=dependency_graph,
                    keep_wh_terminals=keep_wh_terminals,
                )
                or "terminal_glue_path_pruned",
                "path_text": " -> ".join(path.nodes),
            }
            for path in dropped_paths[:5]
        ]
        raw_count = len(raw_paths)
        kept_count = len(kept_for_entity)
        pruned_count = raw_count - kept_count
        stats_by_entity[entity_id] = {
            "raw": raw_count,
            "kept": kept_count,
            "pruned": pruned_count,
            "pruned_ratio": (pruned_count / raw_count) if raw_count else 0.0,
            "fallback_used": fallback_used,
            "pruned_examples": examples,
        }

    total_pruned = total_raw - total_kept
    stats = {
        "total_raw_paths": total_raw,
        "total_kept_paths": total_kept,
        "total_pruned_paths": total_pruned,
        "total_pruned_ratio": (total_pruned / total_raw) if total_raw else 0.0,
        "by_entity": stats_by_entity,
    }
    return pruned_paths, stats


def undirected_graph_edge_payloads(dependency_graph: nx.Graph) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for source, target, attrs in sorted(
        dependency_graph.edges(data=True),
        key=lambda item: (
            int(dependency_graph.nodes[item[0]].get("order", 10**9)),
            int(dependency_graph.nodes[item[1]].get("order", 10**9)),
            str(item[0]),
            str(item[1]),
        ),
    ):
        payloads.append(
            {
                "source": str(source),
                "target": str(target),
                "source_text": _node_text(dependency_graph, str(source)),
                "target_text": _node_text(dependency_graph, str(target)),
                "relations": list(attrs.get("relations", [])),
                "directed_edges": list(attrs.get("directed_edges", [])),
            }
        )
    return payloads


def validate_selected_entity_paths(
    selected_paths: list[SelectedEntityPath],
    entity_starts: list[EntityStartNode],
    entity_origin_paths: list[EntityOriginPath],
) -> None:
    entity_ids = [entity.entity_id for entity in entity_starts]
    path_by_id = {path.path_id: path for path in entity_origin_paths}
    entity_by_id = {entity.entity_id: entity for entity in entity_starts}
    if len(selected_paths) != len(entity_starts):
        raise ValueError(
            f"Selected entity path count mismatch: got {len(selected_paths)}, expected {len(entity_starts)}."
        )

    seen: set[str] = set()
    for selected in selected_paths:
        if selected.entity_id not in entity_ids:
            raise ValueError(f"Selected path references unknown entity_id={selected.entity_id!r}.")
        if selected.entity_id in seen:
            raise ValueError(f"Entity {selected.entity_id!r} was selected more than once.")
        seen.add(selected.entity_id)
        path = path_by_id.get(selected.path_id)
        if path is None:
            raise ValueError(f"Selected path_id={selected.path_id!r} does not exist.")
        if path.entity_id != selected.entity_id:
            raise ValueError(
                f"Selected path_id={selected.path_id!r} belongs to entity_id={path.entity_id!r}, "
                f"not {selected.entity_id!r}."
            )
        entity = entity_by_id[selected.entity_id]
        if path.node_ids and path.node_ids[0] not in set(entity.graph_node_ids) and _norm(path.entity_text) != _norm(entity.text):
            raise ValueError(
                f"Selected path_id={selected.path_id!r} does not start from entity {entity.entity_id!r}."
            )
        other_entity_node_ids = _other_entity_node_ids(selected.entity_id, entity_starts)
        if _path_has_other_entity_intermediate(path, other_entity_node_ids) and _has_non_crossing_alternative(
            selected_entity_id=selected.entity_id,
            selected_path=path,
            entity_origin_paths=entity_origin_paths,
            other_entity_node_ids=other_entity_node_ids,
        ):
            raise ValueError(
                f"Selected path_id={selected.path_id!r} passes through another entity start as an intermediate node. "
                "For parallel/common-answer questions, choose a path from this entity directly toward the answer slot "
                "or compared attribute, not through a different entity start."
            )

    missing = [entity_id for entity_id in entity_ids if entity_id not in seen]
    if missing:
        raise ValueError("Missing selected path for entity/entities: " + ", ".join(missing))


def parse_path_pruned_ast_payload(
    payload: dict[str, Any],
    *,
    selected_paths: list[EntityOriginPath],
) -> SemanticASTResult:
    warnings: list[str] = []
    raw_nodes = payload.get("nodes", [])
    raw_edges = payload.get("edges", [])
    selected_path_ids = {path.path_id for path in selected_paths}

    nodes: list[SemanticASTNode] = []
    seen_nodes: set[str] = set()
    if isinstance(raw_nodes, list):
        for item in raw_nodes:
            if not isinstance(item, dict):
                continue
            node_id = _safe_id(str(item.get("id", "") or item.get("node_id", "")).strip())
            label = str(item.get("label", item.get("text", "")) or "").strip()
            if not node_id or not label or node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            source_path_ids = _str_list(item.get("source_path_ids", item.get("source_paths", [])))
            source_path_ids = [path_id for path_id in source_path_ids if path_id in selected_path_ids]
            source_node_ids = _str_list(item.get("source_node_ids", item.get("source_graph_nodes", [])))
            if not source_node_ids and _is_implicit_value_slot_label(label) and source_path_ids:
                source_node_ids = _infer_value_slot_source_node_ids(label, source_path_ids, selected_paths)
            token_ids = [int(value) for value in source_node_ids if value.isdigit()]
            nodes.append(
                SemanticASTNode(
                    id=node_id,
                    label=label,
                    kind=_node_kind(str(item.get("kind", "") or "")),
                    semantic_type=str(item.get("semantic_type", "") or "") or None,
                    source="path_pruned_ast",
                    source_graph_nodes=source_node_ids,
                    source_token_indices=token_ids,
                    grounding_text=str(item.get("grounding_text", label) or label),
                    cue_text=str(item.get("cue_text", "") or ""),
                    branch_of=str(item.get("branch_of", "") or "") or None,
                    expected_value_slot=str(item.get("expected_value_slot", "") or "") or None,
                    relation_hint=str(item.get("relation_hint", "") or "") or None,
                )
            )

    edges: list[SemanticASTEdge] = []
    if isinstance(raw_edges, list):
        for item in raw_edges:
            if not isinstance(item, dict):
                continue
            source = _safe_id(str(item.get("source", "") or "").strip())
            target = _safe_id(str(item.get("target", "") or "").strip())
            if not source or not target:
                continue
            support_path_id = str(item.get("support_path_id", "") or "").strip()
            support_node_ids = _str_list(item.get("support_node_ids", []))
            relations = _str_list(item.get("support_dependency_relations", item.get("relations", [])))
            relation = str(item.get("relation", item.get("relation_hint", "")) or "").strip()
            edges.append(
                SemanticASTEdge(
                    source=source,
                    target=target,
                    edge_type=str(item.get("edge_type", "attribute") or "attribute"),
                    relation_hint=relation,
                    support_path=[support_path_id, *support_node_ids] if support_path_id else support_node_ids,
                    support_dependency_relations=relations,
                )
            )

    return SemanticASTResult(
        status=str(payload.get("status", "ok") or "ok"),
        nodes=nodes,
        edges=edges,
        warnings=warnings,
        raw_payload=payload,
    )


def localize_path_pruned_ast_branches(
    *,
    semantic_ast: SemanticASTResult,
    selected_paths: list[EntityOriginPath],
) -> SemanticASTResult:
    """Keep each selected entity-origin path as an independent AST branch.

    The LLM may merge shared suffixes such as person_1 -> worked <- person_2 and
    worked -> screenplay. The atomic DAG must ask per-path lookup questions, so
    any non-entity node reachable from multiple selected entity roots is cloned
    per entity branch.
    """

    if len(selected_paths) <= 1 or not semantic_ast.nodes or not semantic_ast.edges:
        return semantic_ast

    node_by_id = semantic_ast.node_by_id()
    root_by_entity = _ast_root_by_entity_id(semantic_ast, selected_paths)
    if len(root_by_entity) <= 1:
        return semantic_ast

    successors: dict[str, list[str]] = {}
    for edge in semantic_ast.edges:
        successors.setdefault(edge.source, []).append(edge.target)

    reachable_by_node: dict[str, set[str]] = {node.id: set() for node in semantic_ast.nodes}
    for entity_id, root_id in root_by_entity.items():
        stack = [root_id]
        visited: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id not in node_by_id:
                continue
            reachable_by_node.setdefault(node_id, set()).add(entity_id)
            for target in reversed(successors.get(node_id, [])):
                stack.append(target)

    clone_nodes = {
        node_id
        for node_id, branches in reachable_by_node.items()
        if len(branches) > 1 and node_by_id.get(node_id) is not None and node_by_id[node_id].kind != "entity"
    }
    if not clone_nodes:
        return semantic_ast

    localized_nodes: list[SemanticASTNode] = []
    for node in semantic_ast.nodes:
        branches = sorted(reachable_by_node.get(node.id, set()), key=_entity_id_sort_key)
        if node.id not in clone_nodes:
            localized_nodes.append(node)
            continue
        for entity_id in branches:
            localized_nodes.append(_clone_ast_node_for_branch(node, entity_id, selected_paths))

    localized_edges: list[SemanticASTEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in semantic_ast.edges:
        source_branches = reachable_by_node.get(edge.source, set())
        target_branches = reachable_by_node.get(edge.target, set())
        edge_branches = sorted(source_branches & target_branches, key=_entity_id_sort_key)
        if not edge_branches:
            edge_branches = [""]
        for entity_id in edge_branches:
            source = _branch_node_id(edge.source, entity_id) if edge.source in clone_nodes and entity_id else edge.source
            target = _branch_node_id(edge.target, entity_id) if edge.target in clone_nodes and entity_id else edge.target
            key = (source, target, edge.edge_type)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            localized_edges.append(
                SemanticASTEdge(
                    source=source,
                    target=target,
                    edge_type=edge.edge_type,
                    relation_hint=edge.relation_hint,
                    support_path=_localized_support_path(edge.support_path, entity_id, selected_paths),
                    support_dependency_relations=list(edge.support_dependency_relations),
                )
            )

    return SemanticASTResult(
        status=semantic_ast.status,
        nodes=localized_nodes,
        edges=localized_edges,
        warnings=list(semantic_ast.warnings),
        raw_payload=semantic_ast.raw_payload,
        coreference_links=list(semantic_ast.coreference_links),
        canonical_node_map=dict(semantic_ast.canonical_node_map),
        validation_warnings=list(semantic_ast.validation_warnings),
        detected_cue_frame=dict(semantic_ast.detected_cue_frame),
        retry_count=semantic_ast.retry_count,
        fallback_repair_actions=list(semantic_ast.fallback_repair_actions),
    )


def _append_entity_start(
    starts: list[EntityStartNode],
    seen_keys: set[str],
    candidate: RestoredGraphNodeCandidate,
    dependency_graph: nx.Graph,
) -> None:
    if _candidate_is_excluded(candidate):
        return
    text = str(candidate.text or candidate.display_text or candidate.restored_text or candidate.graph_text).strip()
    if not text:
        return
    key = _norm(text)
    if key in seen_keys:
        for start in starts:
            if _norm(start.text) == key and str(candidate.node_id) not in start.graph_node_ids:
                start.graph_node_ids.append(str(candidate.node_id))
                if candidate.token_index not in start.token_ids:
                    start.token_ids.append(candidate.token_index)
        return
    node_id = str(candidate.node_id)
    if node_id not in dependency_graph:
        return
    seen_keys.add(key)
    starts.append(
        EntityStartNode(
            entity_id=f"e{len(starts) + 1}",
            text=text,
            graph_node_ids=[node_id],
            token_ids=list(candidate.source_token_indices or [candidate.token_index]),
            kind_hint="entity",
            semantic_type_hint=candidate.semantic_type_hint,
        )
    )


def _candidate_is_entity_start(candidate: RestoredGraphNodeCandidate) -> bool:
    if _candidate_is_excluded(candidate):
        return False
    semantic_type = _normalized_semantic_type(candidate.semantic_type_hint)
    if candidate.is_mask_placeholder:
        return _mask_mapping_is_entity(candidate.kind_hint, candidate.semantic_type_hint)
    if semantic_type in ENTITY_SEMANTIC_TYPES and _norm(candidate.display_text) not in TYPE_VARIABLE_SURFACES:
        return True
    return False


def _candidate_is_excluded(candidate: RestoredGraphNodeCandidate) -> bool:
    text = _norm(candidate.text or candidate.display_text or candidate.restored_text or candidate.graph_text)
    kind = str(candidate.kind_hint or "").lower()
    if text in FUNCTION_SURFACES:
        return True
    if kind in {"context", "cue_candidate"}:
        return True
    if "type_variable" in kind and not candidate.is_mask_placeholder:
        return True
    if text in TYPE_VARIABLE_SURFACES and not candidate.is_mask_placeholder:
        return True
    return False


def _mask_mapping_is_entity(kind_hint: str, semantic_type_hint: str | None) -> bool:
    kind = str(kind_hint or "").lower()
    if "type_variable" in kind or kind == "context":
        return False
    semantic_type = _normalized_semantic_type(semantic_type_hint)
    return kind in {"entity", "entity_candidate"} or semantic_type in ENTITY_SEMANTIC_TYPES or not kind


def _fallback_surface_is_entity(text: str, pos: str) -> bool:
    normalized = _norm(text)
    if not normalized or normalized in FUNCTION_SURFACES or normalized in TYPE_VARIABLE_SURFACES:
        return False
    if pos in {"NNP", "NNPS", "PROPN"}:
        return True
    return bool(re.search(r"[A-Z]", text[:1])) and normalized not in FUNCTION_SURFACES


def _entity_sort_key(entity: EntityStartNode, dependency_graph: nx.Graph) -> tuple[int, str]:
    order_values = [
        int(dependency_graph.nodes[node_id].get("order", 10**9))
        for node_id in entity.graph_node_ids
        if node_id in dependency_graph
    ]
    return (min(order_values) if order_values else 10**9, entity.text)


def _walk_simple_paths(
    *,
    dependency_graph: nx.Graph,
    current: str,
    path: list[str],
    seen: set[tuple[str, ...]],
    raw_paths: list[list[str]],
    max_path_len: int,
    raw_limit: int,
) -> None:
    if len(path) >= 2:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            raw_paths.append(list(path))
    if len(path) >= max_path_len or len(raw_paths) >= raw_limit:
        return
    neighbors = sorted(
        [str(neighbor) for neighbor in dependency_graph.neighbors(current)],
        key=lambda node: (int(dependency_graph.nodes[node].get("order", 10**9)), node),
    )
    for neighbor in neighbors:
        if neighbor in path:
            continue
        _walk_simple_paths(
            dependency_graph=dependency_graph,
            current=neighbor,
            path=[*path, neighbor],
            seen=seen,
            raw_paths=raw_paths,
            max_path_len=max_path_len,
            raw_limit=raw_limit,
        )
        if len(raw_paths) >= raw_limit:
            return


def _entity_path_sort_key(
    dependency_graph: nx.Graph,
    path: list[str],
    *,
    other_entity_node_ids: set[str] | None = None,
) -> tuple[int, int, int, int, int, int, tuple[int, ...], tuple[str, ...]]:
    texts = [_node_text(dependency_graph, node_id) for node_id in path]
    content_count = sum(1 for text in texts if _is_content_text(text))
    answer_count = sum(1 for text in texts if _norm(text) in ANSWER_CUES)
    role_count = sum(1 for text in texts if _norm(text) in ROLE_SLOT_CUES)
    other_entity_node_ids = other_entity_node_ids or set()
    other_entity_intermediate_count = sum(1 for node_id in path[1:-1] if str(node_id) in other_entity_node_ids)
    other_entity_terminal_count = 1 if path[-1:] and str(path[-1]) in other_entity_node_ids else 0
    orders = tuple(int(dependency_graph.nodes[node_id].get("order", 10**9)) for node_id in path)
    return (
        other_entity_intermediate_count,
        other_entity_terminal_count,
        -content_count,
        -answer_count,
        -role_count,
        len(path),
        orders,
        tuple(path),
    )


def _other_entity_node_ids(entity_id: str, entity_starts: list[EntityStartNode]) -> set[str]:
    return {
        str(node_id)
        for entity in entity_starts
        if entity.entity_id != entity_id
        for node_id in entity.graph_node_ids
    }


def _path_has_other_entity_intermediate(path: EntityOriginPath, other_entity_node_ids: set[str]) -> bool:
    return any(node_id in other_entity_node_ids for node_id in path.node_ids[1:-1])


def _has_non_crossing_alternative(
    *,
    selected_entity_id: str,
    selected_path: EntityOriginPath,
    entity_origin_paths: list[EntityOriginPath],
    other_entity_node_ids: set[str],
) -> bool:
    selected_useful_surfaces = _path_useful_surfaces(selected_path)
    for path in entity_origin_paths:
        if path.entity_id != selected_entity_id or path.path_id == selected_path.path_id:
            continue
        if _path_has_other_entity_intermediate(path, other_entity_node_ids):
            continue
        useful_surfaces = _path_useful_surfaces(path)
        if useful_surfaces & selected_useful_surfaces:
            return True
    return False


def _path_useful_surfaces(path: EntityOriginPath) -> set[str]:
    return {
        normalized
        for text in path.nodes[1:]
        for normalized in [_norm(text)]
        if normalized in ANSWER_CUES or normalized in ROLE_SLOT_CUES or _is_content_text(text)
    }


def _is_content_text(text: str) -> bool:
    normalized = _norm(text)
    return bool(normalized) and normalized not in FUNCTION_SURFACES and not re.fullmatch(r"\W+", text)


def _terminal_glue_prune_reason(
    path: EntityOriginPath,
    *,
    dependency_graph: nx.Graph | None,
    keep_wh_terminals: bool,
) -> str | None:
    terminal = _path_terminal_text(path)
    raw_normalized, stripped_normalized = _terminal_token_forms(terminal)
    if keep_wh_terminals and (raw_normalized in WH_TOKENS or stripped_normalized in WH_TOKENS):
        return None
    if raw_normalized in TERMINAL_GLUE_TOKENS or stripped_normalized in TERMINAL_GLUE_TOKENS:
        return "terminal_glue_token"
    if _terminal_incoming_dependency_label_is_glue(path, dependency_graph=dependency_graph):
        return "terminal_glue_dependency_label"
    return None


def _path_terminal_text(path: EntityOriginPath) -> str:
    if path.nodes:
        return str(path.nodes[-1])
    return ""


def _terminal_token_forms(text: str) -> tuple[str, str]:
    raw = str(text or "").strip().lower()
    stripped = raw.strip(_TERMINAL_STRIP_CHARS)
    return raw, stripped


def _terminal_incoming_dependency_label_is_glue(
    path: EntityOriginPath,
    *,
    dependency_graph: nx.Graph | None,
) -> bool:
    if len(path.node_ids) < 2:
        return False
    terminal_node_id = str(path.node_ids[-1])
    edge_evidence = path.evidence[-1] if path.evidence else None
    if edge_evidence is None and dependency_graph is not None:
        previous = str(path.node_ids[-2])
        if dependency_graph.has_edge(previous, terminal_node_id):
            attrs = dependency_graph.edges[previous, terminal_node_id]
            edge_evidence = {
                "relations": list(attrs.get("relations", [])),
                "directed_edges": list(attrs.get("directed_edges", [])),
            }
    if not isinstance(edge_evidence, dict):
        return False

    for directed_edge in edge_evidence.get("directed_edges", []) or []:
        if not isinstance(directed_edge, dict):
            continue
        relation = _dependency_label_for_pruning(directed_edge)
        if not _dependency_label_is_terminal_glue(relation):
            continue
        dependent_ids = {
            str(value)
            for value in (
                directed_edge.get("dependent"),
                directed_edge.get("dependent_index"),
                directed_edge.get("target_index"),
            )
            if value is not None
        }
        if terminal_node_id in dependent_ids:
            return True
    return False


def _dependency_label_for_pruning(directed_edge: dict[str, Any]) -> str:
    return str(
        directed_edge.get("dependency_label")
        or directed_edge.get("relation")
        or ""
    ).strip()


def _dependency_label_is_terminal_glue(label: str) -> bool:
    normalized = str(label or "").strip()
    if not normalized:
        return False
    return normalized in TERMINAL_GLUE_DEP_LABELS or normalized.split(":", 1)[0] in TERMINAL_GLUE_DEP_LABELS


def _terminal_glue_fallback_sort_key(path: EntityOriginPath) -> tuple[int, int, int, int, str]:
    terminal = _path_terminal_text(path)
    raw_normalized, stripped_normalized = _terminal_token_forms(terminal)
    is_punctuation = 1 if re.fullmatch(r"\W+", str(terminal or "").strip()) else 0
    is_glue = 1 if raw_normalized in TERMINAL_GLUE_TOKENS or stripped_normalized in TERMINAL_GLUE_TOKENS else 0
    is_function = 1 if raw_normalized in FUNCTION_SURFACES or stripped_normalized in FUNCTION_SURFACES else 0
    return (is_punctuation, is_glue, is_function, path.length, path.path_id)


def _dependency_edge_evidence(dependency_graph: nx.Graph, node_ids: list[str]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for source, target in zip(node_ids, node_ids[1:]):
        attrs = dependency_graph.edges[source, target]
        evidence.append(
            {
                "source": source,
                "target": target,
                "source_text": _node_text(dependency_graph, source),
                "target_text": _node_text(dependency_graph, target),
                "relations": list(attrs.get("relations", [])),
                "directed_edges": list(attrs.get("directed_edges", [])),
            }
        )
    return evidence


def _is_implicit_comparison_attribute_label(label: str) -> bool:
    return _norm(label) in IMPLICIT_COMPARISON_ATTRIBUTE_SURFACES


def _is_implicit_value_slot_label(label: str) -> bool:
    return _norm_node_label(label) in IMPLICIT_VALUE_SLOT_SURFACES


def _infer_value_slot_source_node_ids(
    label: str,
    source_path_ids: list[str],
    selected_paths: list[EntityOriginPath],
) -> list[str]:
    normalized_label = _norm_node_label(label)
    cue_surfaces: set[str]
    if normalized_label in {"death_reason", "death reason", "death_cause", "death cause", "cause_of_death", "cause of death"}:
        cue_surfaces = {"cause", "death", "die", "died", "dies", "reason", "why"}
    elif normalized_label in {"reason", "cause"}:
        cue_surfaces = {"cause", "reason", "why"}
    elif normalized_label in {"death_date", "death date", "date of death"}:
        cue_surfaces = {"die", "died", "dies", "when"}
    elif normalized_label in {"birth_date", "birth date", "date of birth"}:
        cue_surfaces = {"born", "when"}
    elif normalized_label in {"birthplace", "place of birth"}:
        cue_surfaces = {"born", "where"}
    elif normalized_label == "location":
        cue_surfaces = {"located", "where"}
    elif normalized_label in {"release_date", "release date"}:
        cue_surfaces = {"earlier", "first", "later", "latest", "release", "released", "releases", "when"}
    else:
        cue_surfaces = set(COMPARISON_CUE_SURFACES)
    return _infer_cue_source_node_ids(source_path_ids, selected_paths, cue_surfaces)


def _infer_comparison_cue_source_node_ids(
    source_path_ids: list[str],
    selected_paths: list[EntityOriginPath],
) -> list[str]:
    return _infer_cue_source_node_ids(source_path_ids, selected_paths, set(COMPARISON_CUE_SURFACES))


def _infer_cue_source_node_ids(
    source_path_ids: list[str],
    selected_paths: list[EntityOriginPath],
    cue_surfaces: set[str],
) -> list[str]:
    path_by_id = {path.path_id: path for path in selected_paths}
    inferred: list[str] = []
    for path_id in source_path_ids:
        path = path_by_id.get(path_id)
        if path is None:
            continue
        for node_id, text in zip(path.node_ids, path.nodes, strict=True):
            if _norm_node_label(text) in cue_surfaces and node_id not in inferred:
                inferred.append(node_id)
    return inferred


def _ast_root_by_entity_id(
    semantic_ast: SemanticASTResult,
    selected_paths: list[EntityOriginPath],
) -> dict[str, str]:
    node_by_id = semantic_ast.node_by_id()
    roots: dict[str, str] = {}
    for path in selected_paths:
        start_node_ids = {path.node_ids[0]} if path.node_ids else set()
        best_id = ""
        for node in semantic_ast.nodes:
            if node.kind != "entity":
                continue
            if start_node_ids and any(source_id in start_node_ids for source_id in node.source_graph_nodes):
                best_id = node.id
                break
            if _norm(node.label) == _norm(path.entity_text):
                best_id = node.id
                break
        if best_id and best_id in node_by_id:
            roots[path.entity_id] = best_id
    return roots


def _clone_ast_node_for_branch(
    node: SemanticASTNode,
    entity_id: str,
    selected_paths: list[EntityOriginPath],
) -> SemanticASTNode:
    branch_node_ids = _selected_path_node_ids(entity_id, selected_paths)
    source_graph_nodes = [node_id for node_id in node.source_graph_nodes if node_id in branch_node_ids]
    if not source_graph_nodes:
        source_graph_nodes = list(node.source_graph_nodes)
    return SemanticASTNode(
        id=_branch_node_id(node.id, entity_id),
        label=node.label,
        kind=node.kind,
        semantic_type=node.semantic_type,
        source=node.source,
        source_graph_nodes=source_graph_nodes,
        source_token_indices=[int(value) for value in source_graph_nodes if value.isdigit()],
        grounding_text=node.grounding_text,
        cue_text=node.cue_text,
        branch_of=entity_id,
        expected_value_slot=node.expected_value_slot,
        relation_hint=node.relation_hint,
    )


def _localized_support_path(
    support_path: list[str],
    entity_id: str,
    selected_paths: list[EntityOriginPath],
) -> list[str]:
    if not entity_id:
        return list(support_path)
    path = next((item for item in selected_paths if item.entity_id == entity_id), None)
    if path is None:
        return list(support_path)
    if not support_path:
        return [path.path_id]
    if support_path[0] in {item.path_id for item in selected_paths}:
        return [path.path_id, *support_path[1:]]
    return [path.path_id, *support_path]


def _selected_path_node_ids(entity_id: str, selected_paths: list[EntityOriginPath]) -> set[str]:
    for path in selected_paths:
        if path.entity_id == entity_id:
            return set(path.node_ids)
    return set()


def _branch_node_id(node_id: str, entity_id: str) -> str:
    return _safe_id(f"{node_id}_{entity_id}")


def _entity_id_sort_key(entity_id: str) -> tuple[int, str]:
    match = re.search(r"\d+", entity_id)
    return (int(match.group(0)) if match else 10**9, entity_id)


def _node_text(dependency_graph: nx.Graph, node_id: str) -> str:
    attrs = dependency_graph.nodes[node_id]
    return str(attrs.get("text") or attrs.get("word") or node_id)


def _normalized_semantic_type(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _token_ids_from_node_ids(node_ids: list[str]) -> list[int]:
    result: list[int] = []
    for node_id in node_ids:
        try:
            result.append(int(node_id))
        except (TypeError, ValueError):
            continue
    return result


def _node_kind(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"entity", "named_entity"}:
        return "entity"
    if normalized in {"implicit_type_variable", "implicit"}:
        return "implicit_type_variable"
    if normalized in {"value", "value_slot", "slot"}:
        return "value_slot"
    return "type_variable"


def _safe_id(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", text):
        return text
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return "_".join(words) if words else ""


def _str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _norm_node_label(value: str) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    text = re.sub(r"[^\w\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_possessive_surface(value: str) -> str:
    normalized = _norm_node_label(value)
    return re.sub(r"(?:'s|s')$", "", normalized).strip()
