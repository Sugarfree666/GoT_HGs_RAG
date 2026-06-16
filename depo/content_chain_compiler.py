from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import heapq
import re
from typing import Iterable

from models import HanLPSDPEdge, HanLPSDPResult


@dataclass
class ContentChainResult:
    chains: dict[str, list[str]]
    path_type: str


@dataclass(frozen=True, order=True)
class TokenNode:
    index: int
    text: str


@dataclass
class _CompiledGraph:
    nodes_by_index: dict[int, TokenNode] = field(default_factory=dict)
    graph: dict[TokenNode, set[TokenNode]] = field(default_factory=lambda: defaultdict(set))
    raw_adjacent: dict[TokenNode, list[tuple[TokenNode, HanLPSDPEdge]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    modifiers_by_head: dict[TokenNode, set[TokenNode]] = field(default_factory=lambda: defaultdict(set))
    answer_nodes: list[TokenNode] = field(default_factory=list)
    root_candidates: list[TokenNode] = field(default_factory=list)


_PLACEHOLDER_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
_PUNCT_RE = re.compile(r"^[^\w]+$", re.UNICODE)

_FUNCTION_TOKENS = {
    "a",
    "an",
    "the",
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
    "has",
    "have",
    "had",
    "having",
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
    "who",
    "which",
    "that",
    "where",
    "when",
    "whom",
    "whose",
    "root",
}

_PREPOSITION_TOKENS = {
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

_BE_TOKENS = {"is", "am", "are", "was", "were", "be", "been", "being"}
_SCOPE_TOKENS = {"among", "between", "and", "or", "than"}
_ORDER_CUE_TOKENS = {"first", "earliest", "latest", "last"}
_GLUE_RELATION_PARTS = {"root", "bv", "det", "aux", "punct"}
_MODIFIER_RELATION_PARTS = {"compound", "rstr", "adj", "amod", "noun_arg1"}
_COORD_RELATION_PARTS = {"_and", "_or", "coord", "conj.member", "conj"}


def compile_content_chains(
    hanlp_sdp_result: HanLPSDPResult,
    explicit_entities: list[str],
) -> ContentChainResult:
    entity_texts = [entity for entity in explicit_entities if entity]
    compiled = _compile_graph(hanlp_sdp_result)

    if not entity_texts:
        return ContentChainResult(chains={}, path_type="no_entities")

    focus = _detect_focus(compiled, entity_texts)
    candidate_set = _detect_candidate_set(compiled, entity_texts)

    if candidate_set and focus is not None:
        chains = _compile_candidate_chains(compiled, candidate_set, focus)
        return ContentChainResult(chains=chains, path_type="candidate_selection")

    chains: dict[str, list[str]] = {}
    for entity in entity_texts:
        entity_node = _find_node_by_text(compiled, entity)
        if entity_node is None or focus is None:
            chains[entity] = [entity]
            continue
        path = _shortest_path(compiled.graph, entity_node, focus)
        if path is None:
            chains[entity] = [entity]
            continue
        chains[entity] = _render_path(_insert_modifiers(path, compiled), replacement_start=entity)
    return ContentChainResult(chains=chains, path_type="value_answer")


def _compile_graph(hanlp_result: HanLPSDPResult) -> _CompiledGraph:
    compiled = _CompiledGraph()
    for index, text in enumerate(hanlp_result.tokens or [], start=1):
        compiled.nodes_by_index[index] = TokenNode(index, str(text))

    for edge in hanlp_result.edges or []:
        head = _node_for_edge_endpoint(compiled, edge.head_idx, edge.head)
        dep = _node_for_edge_endpoint(compiled, edge.dep_idx, edge.dep)
        if head is None or dep is None:
            continue
        compiled.raw_adjacent[head].append((dep, edge))
        compiled.raw_adjacent[dep].append((head, edge))
        if _is_root_edge(edge):
            root_candidate = dep if head.index == 0 else head
            if _is_content_node(root_candidate):
                compiled.root_candidates.append(root_candidate)

    for node in compiled.nodes_by_index.values():
        if _is_answer_node(node):
            compiled.answer_nodes.append(node)

    for edge in hanlp_result.edges or []:
        head = _node_for_edge_endpoint(compiled, edge.head_idx, edge.head)
        dep = _node_for_edge_endpoint(compiled, edge.dep_idx, edge.dep)
        if head is None or dep is None or _is_glue_edge(edge) or _is_coordination_edge(edge):
            continue
        if _is_graph_node(head) and _is_graph_node(dep):
            _add_graph_edge(compiled.graph, head, dep)
            _record_modifier(compiled, head, dep, edge)

    for function_node in list(compiled.nodes_by_index.values()):
        if not _is_contractible_function_node(function_node):
            continue
        neighbors = [
            neighbor
            for neighbor, edge in compiled.raw_adjacent.get(function_node, [])
            if _is_graph_node(neighbor) and not _is_glue_edge(edge) and not _is_coordination_edge(edge)
        ]
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1 :]:
                _add_graph_edge(compiled.graph, left, right)

    return compiled


def _node_for_edge_endpoint(compiled: _CompiledGraph, index: int, text: str) -> TokenNode | None:
    if index < 0:
        return None
    if index == 0:
        return TokenNode(0, "ROOT")
    node = compiled.nodes_by_index.get(index)
    if node is not None:
        return node
    node = TokenNode(index, str(text))
    compiled.nodes_by_index[index] = node
    return node


def _add_graph_edge(graph: dict[TokenNode, set[TokenNode]], left: TokenNode, right: TokenNode) -> None:
    if left == right:
        return
    graph[left].add(right)
    graph[right].add(left)


def _record_modifier(compiled: _CompiledGraph, head: TokenNode, dep: TokenNode, edge: HanLPSDPEdge) -> None:
    if not _is_modifier_edge(edge):
        return
    modifier, modified = head, dep
    if _is_order_cue_node(modifier) or not _is_content_node(modifier) or not _is_content_node(modified):
        return
    if _is_placeholder_node(modifier) or _is_placeholder_node(modified):
        return
    compiled.modifiers_by_head[modified].add(modifier)


def _detect_focus(compiled: _CompiledGraph, explicit_entities: list[str]) -> TokenNode | None:
    order_focus = _focus_from_order_cue(compiled)
    if order_focus is not None:
        return order_focus

    scope_focus = _focus_from_scope(compiled, explicit_entities)
    if scope_focus is not None:
        return scope_focus

    preposition_focus = _focus_from_answer_bridge(compiled, _PREPOSITION_TOKENS)
    if preposition_focus is not None:
        return preposition_focus

    copula_focus = _focus_from_answer_bridge(compiled, _BE_TOKENS)
    if copula_focus is not None:
        return copula_focus

    nearest_answer_focus = _nearest_content_to_answer(compiled)
    if nearest_answer_focus is not None:
        return nearest_answer_focus

    if compiled.root_candidates:
        return sorted(set(compiled.root_candidates), key=_node_sort_key)[0]

    content_nodes = [node for node in compiled.nodes_by_index.values() if _is_content_node(node)]
    if not content_nodes:
        return None
    return sorted(content_nodes, key=_node_sort_key)[0]


def _focus_from_order_cue(compiled: _CompiledGraph) -> TokenNode | None:
    candidates: list[TokenNode] = []
    for node in compiled.nodes_by_index.values():
        if not _is_order_cue_node(node):
            continue
        for neighbor, edge in compiled.raw_adjacent.get(node, []):
            if _is_content_node(neighbor) and not _is_placeholder_node(neighbor) and not _is_glue_edge(edge):
                candidates.append(neighbor)
    return sorted(set(candidates), key=_node_sort_key)[0] if candidates else None


def _focus_from_scope(compiled: _CompiledGraph, explicit_entities: list[str]) -> TokenNode | None:
    explicit_set = set(explicit_entities)
    prioritized: list[TokenNode] = []
    fallback: list[TokenNode] = []
    for node in compiled.nodes_by_index.values():
        if not _is_scope_node(node):
            continue
        for neighbor, edge in compiled.raw_adjacent.get(node, []):
            if not _is_content_node(neighbor) or neighbor.text in explicit_set or _is_placeholder_node(neighbor):
                continue
            if _relation_has(edge.relation, "arg1") or _relation_has(edge.relation, "prep_arg1"):
                prioritized.append(neighbor)
            else:
                fallback.append(neighbor)
    candidates = prioritized or fallback
    return sorted(set(candidates), key=_node_sort_key)[0] if candidates else None


def _focus_from_answer_bridge(compiled: _CompiledGraph, bridge_tokens: set[str]) -> TokenNode | None:
    candidates: list[TokenNode] = []
    for node in compiled.nodes_by_index.values():
        if _normalized_text(node) not in bridge_tokens:
            continue
        neighbors = [neighbor for neighbor, edge in compiled.raw_adjacent.get(node, []) if not _is_glue_edge(edge)]
        has_answer = any(_is_answer_node(neighbor) for neighbor in neighbors)
        if not has_answer:
            continue
        candidates.extend(
            neighbor
            for neighbor in neighbors
            if _is_content_node(neighbor) and not _is_answer_node(neighbor) and not _is_placeholder_node(neighbor)
        )
    return sorted(set(candidates), key=_node_sort_key)[0] if candidates else None


def _nearest_content_to_answer(compiled: _CompiledGraph) -> TokenNode | None:
    candidates: list[tuple[tuple[int, int, tuple[int, ...]], TokenNode]] = []
    for answer in compiled.answer_nodes:
        for node in compiled.graph:
            if not _is_content_node(node) or _is_placeholder_node(node):
                continue
            path = _shortest_path(compiled.graph, answer, node)
            if path is None:
                continue
            candidates.append(((len(path), _path_order_score(path), tuple(item.index for item in path)), node))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _detect_candidate_set(compiled: _CompiledGraph, explicit_entities: list[str]) -> list[str]:
    if len(explicit_entities) < 2:
        return []

    explicit_nodes = [node for node in compiled.nodes_by_index.values() if node.text in set(explicit_entities)]
    explicit_by_text = {node.text: node for node in explicit_nodes}
    found: set[str] = set()

    for edge_neighbors in compiled.raw_adjacent.values():
        for neighbor, edge in edge_neighbors:
            if _is_coordination_edge(edge):
                head = _node_for_known_endpoint(compiled, edge.head_idx)
                dep = _node_for_known_endpoint(compiled, edge.dep_idx)
                if head is not None and dep is not None and head.text in explicit_by_text and dep.text in explicit_by_text:
                    found.add(head.text)
                    found.add(dep.text)

    for node in compiled.nodes_by_index.values():
        if _is_scope_node(node):
            found.update(_explicit_entities_reachable_from_scope(compiled, node, explicit_by_text))
        if _normalized_text(node) in {"and", "or"}:
            linked = [
                neighbor.text
                for neighbor, edge in compiled.raw_adjacent.get(node, [])
                if neighbor.text in explicit_by_text and (_is_coordination_edge(edge) or _is_scope_node(node))
            ]
            if len(set(linked)) >= 2:
                found.update(linked)

    if len(found) < 2:
        return []
    return [entity for entity in explicit_entities if entity in found]


def _node_for_known_endpoint(compiled: _CompiledGraph, index: int) -> TokenNode | None:
    if index == 0:
        return TokenNode(0, "ROOT")
    return compiled.nodes_by_index.get(index)


def _explicit_entities_reachable_from_scope(
    compiled: _CompiledGraph,
    scope_node: TokenNode,
    explicit_by_text: dict[str, TokenNode],
) -> set[str]:
    found: set[str] = set()
    queue: deque[tuple[TokenNode, int]] = deque([(scope_node, 0)])
    seen = {scope_node}
    while queue:
        node, depth = queue.popleft()
        if depth >= 3:
            continue
        for neighbor, _edge in compiled.raw_adjacent.get(node, []):
            if neighbor in seen:
                continue
            if neighbor.text in explicit_by_text:
                found.add(neighbor.text)
                seen.add(neighbor)
                continue
            if _is_scope_node(neighbor):
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
    return found


def _compile_candidate_chains(
    compiled: _CompiledGraph,
    candidate_set: list[str],
    focus: TokenNode,
) -> dict[str, list[str]]:
    schema_root = _detect_schema_root(compiled, focus)
    chains: dict[str, list[str]] = {}
    if schema_root is None:
        for entity in candidate_set:
            chains[entity] = _fallback_entity_focus_chain(entity, focus)
        return chains

    schema_path = _shortest_path(compiled.graph, schema_root, focus)
    if schema_path is None:
        for entity in candidate_set:
            chains[entity] = _fallback_entity_focus_chain(entity, focus)
        return chains

    schema_path = _insert_modifiers(schema_path, compiled)
    for entity in candidate_set:
        projected = list(schema_path)
        if projected:
            projected[0] = TokenNode(projected[0].index, entity)
        chains[entity] = _render_path(projected, replacement_start=entity)
        if len(chains[entity]) == 1 and chains[entity][0] == entity and focus.text != entity:
            chains[entity] = _fallback_entity_focus_chain(entity, focus)
    return chains


def _detect_schema_root(compiled: _CompiledGraph, focus: TokenNode) -> TokenNode | None:
    answer_type_candidates = _answer_type_candidates(compiled, focus)
    if answer_type_candidates:
        return sorted(answer_type_candidates, key=lambda item: item[0])[0][1]

    for answer in compiled.answer_nodes:
        if _shortest_path(compiled.graph, answer, focus) is not None:
            return answer
    return None


def _answer_type_candidates(
    compiled: _CompiledGraph,
    focus: TokenNode,
) -> list[tuple[tuple[int, int, tuple[int, ...]], TokenNode]]:
    candidates: list[tuple[tuple[int, int, tuple[int, ...]], TokenNode]] = []
    for node in compiled.nodes_by_index.values():
        if _normalized_text(node) not in _BE_TOKENS:
            continue
        neighbors = [neighbor for neighbor, edge in compiled.raw_adjacent.get(node, []) if not _is_glue_edge(edge)]
        if not any(_is_answer_node(neighbor) for neighbor in neighbors):
            continue
        for neighbor in neighbors:
            if neighbor == focus or not _is_content_node(neighbor) or _is_placeholder_node(neighbor):
                continue
            path = _shortest_path(compiled.graph, neighbor, focus)
            if path is None:
                continue
            candidates.append(((len(path), _path_order_score(path), tuple(item.index for item in path)), neighbor))
    return candidates


def _fallback_entity_focus_chain(entity: str, focus: TokenNode) -> list[str]:
    if _is_answer_node(focus) or _is_function_node(focus) or _is_scope_node(focus):
        return [entity]
    if focus.text == entity:
        return [entity]
    return [entity, focus.text]


def _shortest_path(
    graph: dict[TokenNode, set[TokenNode]],
    start: TokenNode,
    target: TokenNode,
) -> list[TokenNode] | None:
    if start == target:
        return [start]
    if start not in graph or target not in graph:
        return None

    heap: list[tuple[int, int, tuple[int, ...], TokenNode, list[TokenNode]]] = []
    initial_path = [start]
    heapq.heappush(heap, (0, _path_order_score(initial_path), (start.index,), start, initial_path))
    best: dict[TokenNode, tuple[int, int, tuple[int, ...]]] = {}

    while heap:
        distance, order_score, index_tuple, node, path = heapq.heappop(heap)
        state_key = (distance, order_score, index_tuple)
        if node in best and best[node] <= state_key:
            continue
        best[node] = state_key
        if node == target:
            return path
        for neighbor in sorted(graph.get(node, set()), key=_node_sort_key):
            if neighbor in path:
                continue
            next_path = [*path, neighbor]
            next_distance = distance + 1
            next_order = _path_order_score(next_path)
            next_tuple = tuple(item.index for item in next_path)
            heapq.heappush(heap, (next_distance, next_order, next_tuple, neighbor, next_path))
    return None


def _insert_modifiers(path: list[TokenNode], compiled: _CompiledGraph) -> list[TokenNode]:
    if not path:
        return []
    path_set = set(path)
    expanded: list[TokenNode] = []
    for node in path:
        modifiers = [
            modifier
            for modifier in compiled.modifiers_by_head.get(node, set())
            if modifier not in path_set and not _is_order_cue_node(modifier)
        ]
        for modifier in sorted(modifiers, key=_node_sort_key):
            if expanded and expanded[-1] == modifier:
                continue
            expanded.append(modifier)
        expanded.append(node)
    return expanded


def _render_path(path: Iterable[TokenNode], replacement_start: str | None = None) -> list[str]:
    rendered: list[str] = []
    for index, node in enumerate(path):
        text = replacement_start if index == 0 and replacement_start else node.text
        output_node = TokenNode(node.index, text)
        if _is_answer_node(output_node) or _is_function_node(output_node) or _is_scope_node(output_node):
            continue
        if _is_order_cue_node(output_node):
            continue
        if rendered and rendered[-1] == text:
            continue
        rendered.append(text)
    return rendered or ([replacement_start] if replacement_start else [])


def _find_node_by_text(compiled: _CompiledGraph, text: str) -> TokenNode | None:
    candidates = [node for node in compiled.nodes_by_index.values() if node.text == text]
    return sorted(candidates, key=_node_sort_key)[0] if candidates else None


def _path_order_score(path: list[TokenNode]) -> int:
    if len(path) < 2:
        return 0
    backward_steps = sum(1 for left, right in zip(path, path[1:]) if right.index < left.index)
    distance = sum(abs(right.index - left.index) for left, right in zip(path, path[1:]))
    return backward_steps * 1000 + distance


def _is_graph_node(node: TokenNode) -> bool:
    return _is_answer_node(node) or _is_content_node(node)


def _is_content_node(node: TokenNode) -> bool:
    return not _is_answer_node(node) and not _is_function_node(node) and not _is_scope_node(node)


def _is_answer_node(node: TokenNode) -> bool:
    return _normalized_text(node) == "answer"


def _is_placeholder_node(node: TokenNode) -> bool:
    return bool(_PLACEHOLDER_RE.fullmatch(node.text))


def _is_function_node(node: TokenNode) -> bool:
    text = _normalized_text(node)
    return node.index == 0 or text in _FUNCTION_TOKENS or bool(_PUNCT_RE.fullmatch(node.text))


def _is_contractible_function_node(node: TokenNode) -> bool:
    return _is_function_node(node) and not _is_scope_node(node) and node.index != 0


def _is_scope_node(node: TokenNode) -> bool:
    return _normalized_text(node) in _SCOPE_TOKENS


def _is_order_cue_node(node: TokenNode) -> bool:
    return _normalized_text(node) in _ORDER_CUE_TOKENS


def _is_glue_edge(edge: HanLPSDPEdge) -> bool:
    relation = _normalized_relation(edge.relation)
    return any(part == relation or part in relation.split(".") for part in _GLUE_RELATION_PARTS)


def _is_root_edge(edge: HanLPSDPEdge) -> bool:
    return _relation_has(edge.relation, "root") or edge.head_idx == 0


def _is_modifier_edge(edge: HanLPSDPEdge) -> bool:
    relation = _normalized_relation(edge.relation)
    return any(part in relation for part in _MODIFIER_RELATION_PARTS)


def _is_coordination_edge(edge: HanLPSDPEdge) -> bool:
    relation = _normalized_relation(edge.relation)
    return any(part in relation for part in _COORD_RELATION_PARTS)


def _relation_has(relation: str, part: str) -> bool:
    return part in _normalized_relation(relation)


def _normalized_text(node: TokenNode) -> str:
    return node.text.strip().lower()


def _normalized_relation(relation: str) -> str:
    return str(relation or "").strip().lower()


def _node_sort_key(node: TokenNode) -> tuple[int, str]:
    return (node.index if node.index >= 0 else 10**9, node.text)
