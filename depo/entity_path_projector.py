from __future__ import annotations

import re
from typing import Any

import networkx as nx

from models import (
    EntityOriginPath,
    EntityStartNode,
    MaskReplacement,
    RestoredGraphNodeCandidate,
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
        if candidate is None or str(candidate.node_id) not in {str(node_id) for node_id in dependency_graph.nodes}:
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
        collapsed_placeholders = {str(item) for item in attrs.get("collapsed_placeholders", []) if str(item)}
        collapsed_node_ids = {str(item) for item in attrs.get("collapsed_node_ids", []) if str(item)}
        source_tokens = attrs.get("source_tokens") or []
        token_placeholders = {
            str(token.get("graph_text") or token.get("word") or "")
            for token in source_tokens
            if isinstance(token, dict)
        }
        if (
            text != placeholder
            and placeholder not in collapsed_placeholders
            and placeholder not in collapsed_node_ids
            and placeholder not in token_placeholders
        ):
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
            source_token_indices=list(attrs.get("source_token_indices") or [token_index]),
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
