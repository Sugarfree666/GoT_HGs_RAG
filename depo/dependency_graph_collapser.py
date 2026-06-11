from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

import networkx as nx


COLLAPSIBLE_RELS = {"compound", "case", "det", "advmod"}
PLACEHOLDER_RE = re.compile(
    r"^(?:Person|Film|Movie|Song|Location|Entity|Organization|Organisation|Event|Book|Work|City|Country|Region|Game)[A-Z]$"
)


@dataclass
class CollapseDecision:
    relation: str
    head: str
    child: str
    head_text_before: str
    child_text: str
    head_text_after: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DependencyGraphCollapser:
    """Collapse simple syntax modifiers into their dependency heads.

    This intentionally handles only COLLAPSIBLE_RELS. The goal is a cleaner
    dependency graph without adding semantic interpretation.
    """

    def __init__(self, collapsible_rels: set[str] | None = None) -> None:
        self.collapsible_rels = set(collapsible_rels or COLLAPSIBLE_RELS)
        self.decisions: list[CollapseDecision] = []

    def collapse(self, graph: nx.Graph) -> nx.Graph:
        collapsed = graph.copy()
        for node_id in list(collapsed.nodes):
            self._initialize_node(collapsed, str(node_id))

        changed = True
        while changed:
            changed = False
            candidates = self._collapse_candidates(collapsed)
            if not candidates:
                break
            selected = self._select_next_candidate(collapsed, candidates)
            if selected is None:
                break
            head, child, relation = selected
            self._collapse_child_into_head(collapsed, head, child, relation)
            changed = True

        collapsed.graph["dependency_collapsing_enabled"] = True
        collapsed.graph["collapse_relations"] = sorted(self.collapsible_rels)
        collapsed.graph["collapse_decisions"] = [decision.to_dict() for decision in self.decisions]
        collapsed.graph["raw_node_count"] = graph.number_of_nodes()
        collapsed.graph["raw_edge_count"] = graph.number_of_edges()
        collapsed.graph["collapsed_node_count"] = collapsed.number_of_nodes()
        collapsed.graph["collapsed_edge_count"] = collapsed.number_of_edges()
        return collapsed

    def _initialize_node(self, graph: nx.Graph, node_id: str) -> None:
        attrs = graph.nodes[node_id]
        token_index = _token_index(node_id, attrs)
        token = {
            "node_id": str(node_id),
            "token_index": token_index,
            "text": str(attrs.get("text") or attrs.get("word") or node_id),
            "word": str(attrs.get("word") or attrs.get("text") or node_id),
            "graph_text": str(attrs.get("graph_text") or attrs.get("word") or attrs.get("text") or node_id),
        }
        attrs.setdefault("source_tokens", [token])
        attrs.setdefault("source_token_indices", [token_index])
        attrs.setdefault("collapsed_node_ids", [str(node_id)])
        attrs.setdefault("collapsed_relations", [])
        placeholders = [value for value in {token["graph_text"], token["word"]} if PLACEHOLDER_RE.fullmatch(value)]
        attrs.setdefault("collapsed_placeholders", placeholders)

    def _collapse_candidates(self, graph: nx.Graph) -> list[tuple[str, str, str]]:
        candidates: list[tuple[str, str, str]] = []
        for source, target, attrs in graph.edges(data=True):
            directed_edges = attrs.get("directed_edges") or []
            if directed_edges:
                for edge in directed_edges:
                    relation = str(edge.get("dependency_label") or edge.get("relation") or "")
                    if relation not in self.collapsible_rels:
                        continue
                    head = str(edge.get("governor") or edge.get("source_index") or source)
                    child = str(edge.get("dependent") or edge.get("target_index") or target)
                    if head in graph and child in graph and head != child:
                        candidates.append((head, child, relation))
                continue
            for relation in _relations(attrs):
                if relation in self.collapsible_rels and source in graph and target in graph:
                    candidates.append((str(source), str(target), relation))
        return sorted(candidates, key=lambda item: (_node_order(graph, item[0]), _node_order(graph, item[1]), item[2]))

    def _select_next_candidate(
        self,
        graph: nx.Graph,
        candidates: list[tuple[str, str, str]],
    ) -> tuple[str, str, str] | None:
        for head, child, relation in candidates:
            if not self._has_collapsible_child(graph, child):
                return head, child, relation
        return candidates[0] if candidates else None

    def _has_collapsible_child(self, graph: nx.Graph, node_id: str) -> bool:
        for head, child, _relation in self._collapse_candidates(graph):
            if head == node_id and child in graph:
                return True
        return False

    def _collapse_child_into_head(self, graph: nx.Graph, head: str, child: str, relation: str) -> None:
        if head not in graph or child not in graph or head == child:
            return
        head_text_before = str(graph.nodes[head].get("text") or head)
        child_text = str(graph.nodes[child].get("text") or child)

        for neighbor in list(graph.neighbors(child)):
            neighbor = str(neighbor)
            if neighbor == head or neighbor not in graph:
                continue
            edge_attrs = dict(graph.edges[child, neighbor])
            self._add_or_merge_edge(
                graph,
                head,
                neighbor,
                edge_attrs,
                collapsed_via={
                    "relation": relation,
                    "head": head,
                    "child": child,
                    "child_text": child_text,
                },
            )

        self._merge_node_attrs(graph, head, child, relation)
        head_text_after = str(graph.nodes[head].get("text") or head)
        self.decisions.append(
            CollapseDecision(
                relation=relation,
                head=head,
                child=child,
                head_text_before=head_text_before,
                child_text=child_text,
                head_text_after=head_text_after,
            )
        )
        graph.remove_node(child)

    def _merge_node_attrs(self, graph: nx.Graph, head: str, child: str, relation: str) -> None:
        head_attrs = graph.nodes[head]
        child_attrs = graph.nodes[child]
        tokens = _dedupe_tokens(
            [
                *list(head_attrs.get("source_tokens") or []),
                *list(child_attrs.get("source_tokens") or []),
            ]
        )
        tokens = sorted(tokens, key=lambda item: (int(item.get("token_index", 10**9)), str(item.get("node_id", ""))))
        head_attrs["source_tokens"] = tokens
        head_attrs["source_token_indices"] = [int(item["token_index"]) for item in tokens if _is_int_like(item.get("token_index"))]
        head_attrs["collapsed_node_ids"] = _dedupe_strings(
            [
                *list(head_attrs.get("collapsed_node_ids") or [head]),
                *list(child_attrs.get("collapsed_node_ids") or [child]),
            ]
        )
        head_attrs["collapsed_relations"] = _dedupe_strings(
            [
                *list(head_attrs.get("collapsed_relations") or []),
                relation,
                *list(child_attrs.get("collapsed_relations") or []),
            ]
        )
        head_attrs["collapsed_placeholders"] = _dedupe_strings(
            [
                *list(head_attrs.get("collapsed_placeholders") or []),
                *list(child_attrs.get("collapsed_placeholders") or []),
                *[
                    str(item.get("graph_text") or item.get("word") or "")
                    for item in tokens
                    if PLACEHOLDER_RE.fullmatch(str(item.get("graph_text") or item.get("word") or ""))
                ],
            ]
        )
        head_attrs["text"] = " ".join(str(item.get("text") or item.get("word") or "") for item in tokens).strip()
        head_attrs["word"] = " ".join(str(item.get("word") or item.get("text") or "") for item in tokens).strip()
        head_attrs["graph_text"] = " ".join(str(item.get("graph_text") or item.get("word") or "") for item in tokens).strip()
        head_attrs["label"] = f"{head_attrs['text']}[{head_attrs.get('order', head)}]"

    def _add_or_merge_edge(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        attrs: dict[str, Any],
        *,
        collapsed_via: dict[str, Any],
    ) -> None:
        if source == target or source not in graph or target not in graph:
            return
        existing = dict(graph.edges[source, target]) if graph.has_edge(source, target) else {}
        relations = _dedupe_strings([*_relations(existing), *_relations(attrs)])
        directed_edges = [
            *list(existing.get("directed_edges") or []),
            *list(attrs.get("directed_edges") or []),
        ]
        collapsed_via_items = [
            *list(existing.get("collapsed_via") or []),
            *list(attrs.get("collapsed_via") or []),
            collapsed_via,
        ]
        graph.add_edge(
            source,
            target,
            relation="|".join(relations),
            relations=relations,
            directed_edges=directed_edges,
            collapsed_via=collapsed_via_items,
            dependency_label="|".join(relations),
        )


def collapse_dependency_graph(graph: nx.Graph) -> nx.Graph:
    return DependencyGraphCollapser().collapse(graph)


def _relations(attrs: dict[str, Any]) -> list[str]:
    raw = attrs.get("relations")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    relation = str(attrs.get("relation") or "")
    return [item for item in relation.split("|") if item]


def _token_index(node_id: str, attrs: dict[str, Any]) -> int:
    for key in ("order", "token_index"):
        if _is_int_like(attrs.get(key)):
            return int(attrs[key])
    if _is_int_like(node_id):
        return int(node_id)
    return 10**9


def _node_order(graph: nx.Graph, node_id: str) -> int:
    return _token_index(node_id, graph.nodes[node_id])


def _dedupe_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for token in tokens:
        key = (str(token.get("node_id") or ""), str(token.get("token_index") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(token))
    return result


def _dedupe_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True
