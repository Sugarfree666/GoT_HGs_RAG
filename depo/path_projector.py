from __future__ import annotations

import re
from typing import Any

import networkx as nx

from models import CandidateNode, CandidatePath, DependencyParse, Requirement, RestoredGraphNodeCandidate

DEFAULT_MAX_BRIDGE_HOPS = 4
DEFAULT_MAX_PATH_LEN = 5


def build_undirected_dependency_graph(
    dependency_parse: DependencyParse,
    restored_graph_node_candidates: list[RestoredGraphNodeCandidate] | None = None,
) -> nx.Graph:
    """Convert a CoreNLP dependency parse into an unweighted undirected graph.

    Edge metadata preserves the original directed dependency information for
    debugging and later relation-label prompting. No dependency weights are
    used in this graph.
    """

    restored_by_id = {
        str(candidate.node_id): candidate
        for candidate in restored_graph_node_candidates or []
    }
    graph = nx.Graph()
    for token in dependency_parse.tokens:
        node_id = str(token.index)
        restored = restored_by_id.get(node_id)
        text = restored.display_text if restored is not None else token.word
        graph.add_node(
            node_id,
            token_index=token.index,
            word=token.word,
            graph_text=token.word,
            text=text,
            label=f"{text}[{token.index}]",
            order=token.index,
            pos=token.pos,
            lemma=token.lemma,
            character_offset_begin=token.character_offset_begin,
            character_offset_end=token.character_offset_end,
        )

    for edge in dependency_parse.edges:
        source = str(edge.source_index)
        target = str(edge.target_index)
        if source not in graph or target not in graph:
            continue
        directed_edge = {
            "governor": source,
            "dependent": target,
            "governor_text": graph.nodes[source].get("text", edge.source),
            "dependent_text": graph.nodes[target].get("text", edge.target),
            "governor_graph_text": edge.source,
            "dependent_graph_text": edge.target,
            "governor_index": edge.source_index,
            "dependent_index": edge.target_index,
            "dependency_label": edge.relation,
        }
        if graph.has_edge(source, target):
            relations = list(graph.edges[source, target].get("relations", []))
            directed_edges = list(graph.edges[source, target].get("directed_edges", []))
        else:
            relations = []
            directed_edges = []
        if edge.relation not in relations:
            relations.append(edge.relation)
        directed_edges.append(directed_edge)
        graph.add_edge(
            source,
            target,
            relation="|".join(relations),
            relations=relations,
            directed_edges=directed_edges,
            original_governor=source,
            original_dependent=target,
            dependency_label=edge.relation,
            token_text={
                source: graph.nodes[source].get("text", edge.source),
                target: graph.nodes[target].get("text", edge.target),
            },
            token_index={
                source: edge.source_index,
                target: edge.target_index,
            },
        )
    return graph


def ground_candidate_nodes(
    candidate_nodes: list[CandidateNode],
    dependency_graph: nx.Graph,
) -> list[CandidateNode]:
    """Fill missing candidate graph_node_ids by exact text/id matching."""

    by_norm_text: dict[str, list[str]] = {}
    by_norm_graph_text: dict[str, list[str]] = {}
    for node_id, attrs in dependency_graph.nodes(data=True):
        by_norm_text.setdefault(_norm(str(attrs.get("text", node_id))), []).append(str(node_id))
        by_norm_graph_text.setdefault(_norm(str(attrs.get("graph_text", node_id))), []).append(str(node_id))
        by_norm_text.setdefault(_norm(str(node_id)), []).append(str(node_id))

    grounded: list[CandidateNode] = []
    for candidate in candidate_nodes:
        graph_node_ids = [str(item) for item in candidate.graph_node_ids if str(item) in dependency_graph]
        for token_id in candidate.token_ids:
            token_node_id = str(token_id)
            if token_node_id in dependency_graph and token_node_id not in graph_node_ids:
                graph_node_ids.append(token_node_id)
        if not graph_node_ids:
            normalized = _norm(candidate.text)
            graph_node_ids.extend(by_norm_text.get(normalized, []))
            for node_id in by_norm_graph_text.get(normalized, []):
                if node_id not in graph_node_ids:
                    graph_node_ids.append(node_id)
        if not graph_node_ids and candidate.id in dependency_graph:
            graph_node_ids.append(candidate.id)
        candidate.graph_node_ids = _dedupe_preserve_order(graph_node_ids)
        candidate.token_ids = [
            int(node_id)
            for node_id in candidate.graph_node_ids
            if str(node_id).isdigit()
        ]
        grounded.append(candidate)
    return grounded


def build_candidate_projected_graph(
    dependency_graph: nx.Graph,
    candidate_nodes: list[CandidateNode],
    max_bridge_hops: int = DEFAULT_MAX_BRIDGE_HOPS,
) -> nx.Graph:
    """Project dependency graph paths onto candidate nodes only.

    Two candidates are connected when a dependency path of at most
    max_bridge_hops edges exists and the internal dependency nodes do not pass
    through another candidate node.
    """

    candidates = ground_candidate_nodes(candidate_nodes, dependency_graph)
    projected = nx.Graph()
    for order, candidate in enumerate(candidates):
        projected.add_node(
            candidate.id,
            text=candidate.text,
            kind=candidate.kind,
            confidence=candidate.confidence,
            graph_node_ids=list(candidate.graph_node_ids),
            token_ids=list(candidate.token_ids),
            order=order,
        )

    candidate_by_graph_node: dict[str, set[str]] = {}
    for candidate in candidates:
        for graph_node_id in candidate.graph_node_ids:
            candidate_by_graph_node.setdefault(str(graph_node_id), set()).add(candidate.id)

    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            evidence_path = _best_bridge_path(
                dependency_graph=dependency_graph,
                left=left,
                right=right,
                candidate_by_graph_node=candidate_by_graph_node,
                max_bridge_hops=max_bridge_hops,
            )
            if not evidence_path:
                continue
            projected.add_edge(
                left.id,
                right.id,
                evidence_path=_evidence_path_payload(dependency_graph, evidence_path),
                evidence_node_ids=[str(item) for item in evidence_path],
                evidence_text_path=[
                    str(dependency_graph.nodes[item].get("text", item))
                    for item in evidence_path
                ],
                dependency_edges=_dependency_edge_payloads(dependency_graph, evidence_path),
            )
    return projected


def enumerate_candidate_paths(
    projected_graph: nx.Graph,
    max_path_len: int = DEFAULT_MAX_PATH_LEN,
) -> list[CandidatePath]:
    """Enumerate de-duplicated simple paths over the candidate-projected graph."""

    if max_path_len < 1:
        return []
    seen: set[tuple[str, ...]] = set()
    paths: list[list[str]] = []
    ordered_nodes = sorted(projected_graph.nodes, key=lambda node: _candidate_order(projected_graph, node))

    def walk(path: list[str]) -> None:
        if 2 <= len(path) <= max_path_len:
            key = _canonical_path_key(path)
            if key not in seen:
                seen.add(key)
                paths.append(list(path))
        if len(path) >= max_path_len:
            return
        current = path[-1]
        for neighbor in sorted(projected_graph.neighbors(current), key=lambda node: _candidate_order(projected_graph, node)):
            if neighbor in path:
                continue
            walk([*path, neighbor])

    for node in ordered_nodes:
        walk([node])

    paths.sort(key=lambda path: (len(path), [_candidate_order(projected_graph, node) for node in path], path))
    result: list[CandidatePath] = []
    for index, path in enumerate(paths, start=1):
        result.append(
            CandidatePath(
                path_id=f"p{index}",
                nodes=[str(projected_graph.nodes[node].get("text", node)) for node in path],
                node_ids=[str(node) for node in path],
                candidate_for=[],
                evidence=_candidate_path_evidence(projected_graph, path),
            )
        )
    return result


def filter_candidate_paths(
    candidate_paths: list[CandidatePath],
    requirements: list[Requirement],
) -> list[CandidatePath]:
    """Apply the initial deterministic candidate path filters.

    Rule 1: reverse duplicates collapse to one canonical path.
    Rule 2: the path must contain at least one requirement root or target.
    """

    seen: set[tuple[str, ...]] = set()
    filtered: list[CandidatePath] = []
    for path in candidate_paths:
        key = _canonical_path_key(path.node_ids)
        if key in seen:
            continue
        seen.add(key)
        candidate_for = [
            requirement.id
            for requirement in requirements
            if _path_mentions_requirement(path, requirement)
        ]
        if not candidate_for:
            continue
        filtered.append(
            CandidatePath(
                path_id=path.path_id,
                nodes=list(path.nodes),
                node_ids=list(path.node_ids),
                candidate_for=candidate_for,
                evidence=list(path.evidence),
            )
        )
    return filtered


def format_projected_graph_edges(projected_graph: nx.Graph) -> list[str]:
    """Format projected edges with evidence paths for debug output."""

    lines: list[str] = []
    for source, target, attrs in sorted(
        projected_graph.edges(data=True),
        key=lambda item: (_candidate_order(projected_graph, item[0]), _candidate_order(projected_graph, item[1])),
    ):
        source_text = projected_graph.nodes[source].get("text", source)
        target_text = projected_graph.nodes[target].get("text", target)
        evidence = " -- ".join(str(item) for item in attrs.get("evidence_text_path", []))
        lines.append(f"  - {source_text}[{source}] -- {target_text}[{target}] evidence=({evidence})")
    return lines


def _best_bridge_path(
    dependency_graph: nx.Graph,
    left: CandidateNode,
    right: CandidateNode,
    candidate_by_graph_node: dict[str, set[str]],
    max_bridge_hops: int,
) -> list[str]:
    best: list[str] = []
    for left_graph_node in left.graph_node_ids:
        for right_graph_node in right.graph_node_ids:
            if left_graph_node not in dependency_graph or right_graph_node not in dependency_graph:
                continue
            for path in nx.all_simple_paths(
                dependency_graph,
                source=left_graph_node,
                target=right_graph_node,
                cutoff=max_bridge_hops,
            ):
                path = [str(item) for item in path]
                if not _bridge_path_is_valid(path, left.id, right.id, candidate_by_graph_node):
                    continue
                if not best or _dependency_path_sort_key(dependency_graph, path) < _dependency_path_sort_key(dependency_graph, best):
                    best = path
    return best


def _bridge_path_is_valid(
    path: list[str],
    left_candidate_id: str,
    right_candidate_id: str,
    candidate_by_graph_node: dict[str, set[str]],
) -> bool:
    for graph_node_id in path[1:-1]:
        owners = candidate_by_graph_node.get(str(graph_node_id), set())
        if owners - {left_candidate_id, right_candidate_id}:
            return False
    return True


def _dependency_path_sort_key(dependency_graph: nx.Graph, path: list[str]) -> tuple[int, list[int], list[str]]:
    orders = [
        int(dependency_graph.nodes[node].get("order", 10**9))
        for node in path
    ]
    return (len(path), orders, path)


def _evidence_path_payload(dependency_graph: nx.Graph, path: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": str(node),
            "text": str(dependency_graph.nodes[node].get("text", node)),
            "graph_text": str(dependency_graph.nodes[node].get("graph_text", node)),
            "token_index": dependency_graph.nodes[node].get("token_index"),
            "pos": dependency_graph.nodes[node].get("pos"),
        }
        for node in path
    ]


def _dependency_edge_payloads(dependency_graph: nx.Graph, path: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for source, target in zip(path, path[1:]):
        attrs = dependency_graph.edges[source, target]
        payloads.append(
            {
                "source": str(source),
                "target": str(target),
                "source_text": str(dependency_graph.nodes[source].get("text", source)),
                "target_text": str(dependency_graph.nodes[target].get("text", target)),
                "relations": list(attrs.get("relations", [])),
                "directed_edges": list(attrs.get("directed_edges", [])),
            }
        )
    return payloads


def _candidate_path_evidence(projected_graph: nx.Graph, path: list[str]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for source, target in zip(path, path[1:]):
        attrs = projected_graph.edges[source, target]
        evidence.append(
            {
                "source": str(source),
                "target": str(target),
                "source_text": str(projected_graph.nodes[source].get("text", source)),
                "target_text": str(projected_graph.nodes[target].get("text", target)),
                "evidence_path": list(attrs.get("evidence_path", [])),
                "evidence_node_ids": list(attrs.get("evidence_node_ids", [])),
                "evidence_text_path": list(attrs.get("evidence_text_path", [])),
                "dependency_edges": list(attrs.get("dependency_edges", [])),
            }
        )
    return evidence


def _path_mentions_requirement(path: CandidatePath, requirement: Requirement) -> bool:
    path_values = {_norm(value) for value in [*path.nodes, *path.node_ids]}
    root = _norm(requirement.root)
    target = _norm(requirement.target)
    return bool(root and root in path_values) or bool(target and target in path_values)


def _canonical_path_key(path: list[str]) -> tuple[str, ...]:
    forward = tuple(str(item) for item in path)
    backward = tuple(reversed(forward))
    return min(forward, backward)


def _candidate_order(projected_graph: nx.Graph, node: str) -> tuple[int, str]:
    return (int(projected_graph.nodes[node].get("order", 10**9)), str(node))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
