from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from math import inf, isinf

from models import HanLPSDPEdge, HanLPSDPResult


# 节点分类
ENTITY_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d[\d,]*(?:\.\d+)?|\d{1,4}(?:[-/]\d{1,2}){1,2})%?$"
)
DETERMINERS = {"a", "an", "the"}
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
SEMANTIC_SCOPE_WORDS = SCOPE_WORDS | {"both", "either", "neither"}
FUNCTION_WORDS = (
    DETERMINERS
    | RELATIVE_PRONOUNS
    | LIGHT_VERBS
    | PREPOSITIONS
    | SCOPE_WORDS
)
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

# PAS 增强
POSSESSIVE_MARKERS = {"'", "’", "'s", "’s", "s"}
POSSESSIVE_OWNER_RELATIONS = {"poss_arg2"}
POSSESSIVE_POSSESSED_RELATIONS = {
    "poss_arg1",
    "adj_arg1",
    "noun_arg1",
    "modifier",
}

# Equation (2): core=1, structural=2, general=3, noise=+∞。
CORE_RELATION_PREFIXES = (
    "verb_arg",
    "noun_arg",
    "adj_arg",
    "act_arg",
    "pat_arg",
    "eff_arg",
)
CORE_RELATIONS = {"loc", "twhen"}
STRUCTURAL_RELATION_PREFIXES = ("conj_arg", "relative_arg", "comp_")
STRUCTURAL_RELATIONS = {"comparison", "modifier"}
INVALID_RELATION_MARKERS = ("root", "punct", "quote", "paren", "bracket")
AUGMENTED_CORE_RULES = {
    "pas_preposition_contraction",
    "pas_possessive_contraction",
}
STRUCTURAL_RULES = {"pas_coordination_candidate_attachment"}


@dataclass
class TokenReasoningNode:
    id: str
    text: str
    kind: str


@dataclass
class TokenReasoningEdge:
    relations: set[str] = field(default_factory=set)
    rules: set[str] = field(default_factory=set)


@dataclass
class _WorkingState:
    nodes: dict[str, TokenReasoningNode]
    pas_edges: list[HanLPSDPEdge]
    graph: dict[str, dict[str, TokenReasoningEdge]]
    syntax_heads: dict[str, int]


def compile_token_reasoning_structure(
    hanlp_sdp_result: HanLPSDPResult,
    explicit_entities: list[str],
) -> list[list[str]]:
    """Compile PAS evidence into one best semantic path per explicit entity."""
    state = build_evidence_graph(hanlp_sdp_result)
    entity_order = {
        entity: position for position, entity in enumerate(explicit_entities)
    }
    entity_ids = sorted(
        {
            node.id
            for node in state.nodes.values()
            if node.text in entity_order
        },
        key=lambda node_id: (entity_order[state.nodes[node_id].text], int(node_id)),
    )

    add_pas_preposition_contraction_edges(state)
    add_pas_possessive_contraction_edges(state)
    add_pas_coordination_candidate_attachment_edges(state, entity_ids)
    return _select_entity_best_paths(state.nodes, state.graph, entity_ids)


def classify_node(text: str) -> str:
    """Classify a PAS token as entity, content, constraint, or function."""
    token = text.strip()
    lower = token.lower()
    if not token or _is_punctuation(token):
        return "function"
    if ENTITY_RE.fullmatch(token):
        return "entity"
    if NUMERIC_RE.fullmatch(token) or lower in ORDER_CUES | APPROX_CUES:
        return "constraint"
    return "function" if lower in FUNCTION_WORDS else "content"


def build_evidence_graph(hanlp_sdp_result: HanLPSDPResult) -> _WorkingState:
    """Build the single undirected PAS graph used by all Step 4 operations."""
    nodes = _build_token_nodes(hanlp_sdp_result)
    graph: dict[str, dict[str, TokenReasoningEdge]] = {}
    for edge in hanlp_sdp_result.edges:
        head_id, dep_id = str(edge.head_idx), str(edge.dep_idx)
        if head_id in nodes and dep_id in nodes:
            _add_edge(graph, head_id, dep_id, relation=edge.relation)

    state = _WorkingState(
        nodes=nodes,
        pas_edges=hanlp_sdp_result.edges,
        graph=graph,
        syntax_heads={
            str(dep_id): head_id
            for dep_id, head_id in hanlp_sdp_result.syntax_heads.items()
            if str(dep_id) in nodes and str(dep_id) != "0"
        },
    )
    _mark_possessive_markers(state)
    return state


def add_pas_preposition_contraction_edges(state: _WorkingState) -> None:
    """Contract preposition ARG1--preposition--ARG2 into a core edge."""
    contracted: set[str] = set()
    for preposition in state.nodes.values():
        if preposition.text.lower() not in PREPOSITIONS:
            continue
        arg1_ids, arg2_ids = _preposition_arguments(preposition.id, state)
        for arg1_id in sorted(arg1_ids, key=int):
            for arg2_id in sorted(arg2_ids, key=int):
                if (
                    arg1_id != arg2_id
                    and _is_high_salience_node(state.nodes[arg1_id])
                    and _is_high_salience_node(state.nodes[arg2_id])
                ):
                    _add_edge(
                        state.graph,
                        arg1_id,
                        arg2_id,
                        rule="pas_preposition_contraction",
                    )
                    contracted.add(preposition.id)
    _remove_graph_nodes(state.graph, contracted)


def add_pas_possessive_contraction_edges(state: _WorkingState) -> None:
    """Contract single-token and split English possessive markers."""
    contracted: set[str] = set()
    nodes = sorted(state.nodes.values(), key=lambda node: int(node.id))

    for marker in nodes:
        if marker.text.lower() not in POSSESSIVE_MARKERS:
            continue
        owners, possessed = _possessive_arguments(marker.id, state)
        if _add_possessive_edges(state, owners, possessed):
            contracted.add(marker.id)

    for marker, suffix in zip(nodes, nodes[1:]):
        if marker.text not in {"'", "’"} or suffix.text.lower() != "s":
            continue
        marker_owners, marker_possessed = _possessive_arguments(marker.id, state)
        suffix_owners, suffix_possessed = _possessive_arguments(suffix.id, state)
        if _add_possessive_edges(
            state,
            marker_owners | suffix_owners,
            marker_possessed | suffix_possessed,
        ):
            contracted.update({marker.id, suffix.id})

    _remove_graph_nodes(state.graph, contracted)


def _add_possessive_edges(
    state: _WorkingState,
    owner_ids: set[str],
    possessed_ids: set[str],
) -> bool:
    added = False
    for owner_id in sorted(owner_ids, key=int):
        for possessed_id in sorted(possessed_ids, key=int):
            if (
                owner_id != possessed_id
                and _is_high_salience_node(state.nodes[owner_id])
                and _is_high_salience_node(state.nodes[possessed_id])
            ):
                _add_edge(
                    state.graph,
                    owner_id,
                    possessed_id,
                    rule="pas_possessive_contraction",
                )
                added = True
    return added


def add_pas_coordination_candidate_attachment_edges(
    state: _WorkingState,
    entity_ids: list[str],
) -> None:
    """Attach coordinated explicit entities to their shared syntactic head."""
    if len(entity_ids) < 2:
        return
    for connector_id, member_ids in _coordination_groups(state, set(entity_ids)):
        attachment_id = _shared_syntactic_attachment(
            state, connector_id, member_ids
        )
        if attachment_id is None:
            continue
        for member_id in member_ids:
            if attachment_id not in state.graph.get(member_id, {}):
                _add_edge(
                    state.graph,
                    member_id,
                    attachment_id,
                    rule="pas_coordination_candidate_attachment",
                )


def _select_entity_best_paths(
    nodes: dict[str, TokenReasoningNode],
    graph: dict[str, dict[str, TokenReasoningEdge]],
    entity_ids: list[str],
) -> list[list[str]]:
    semantic_ids = {
        node_id for node_id in graph if _is_semantic_node(nodes[node_id])
    }
    boundary_ids = sorted(
        (
            node_id
            for node_id in semantic_ids
            if _semantic_degree(node_id, nodes, graph) == 1
        ),
        key=int,
    )
    selected: list[list[str]] = []
    for entity_id in entity_ids:
        candidates = [
            path
            for boundary_id in boundary_ids
            if boundary_id != entity_id
            for path in [
                _shortest_path(
                    graph,
                    nodes,
                    entity_id,
                    boundary_id,
                    blocked_ids=set(entity_ids) - {entity_id},
                )
            ]
            if path
        ]
        if not candidates:
            continue
        branch_ids = {
            node_id
            for path in candidates
            for node_id in path
            if node_id != entity_id and node_id in semantic_ids
        }
        best_path = max(
            candidates,
            key=lambda path: _sp_score(entity_id, path, branch_ids),
        )
        selected.append([nodes[node_id].text for node_id in best_path])
    return selected


def _shortest_path(
    graph: dict[str, dict[str, TokenReasoningEdge]],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
    *,
    blocked_ids: set[str],
) -> list[str]:
    """Find the least-cost path, then break ties by length and token order."""
    if source_id not in graph or target_id not in graph:
        return []

    start_key = _path_key([source_id])
    heap: list[tuple[float, int, tuple[int, ...], str, list[str]]] = [
        (0, 0, start_key, source_id, [source_id])
    ]
    best: dict[str, tuple[float, int, tuple[int, ...]]] = {
        source_id: (0, 0, start_key)
    }
    while heap:
        cost, edge_count, path_key, node_id, path = heapq.heappop(heap)
        if best[node_id] != (cost, edge_count, path_key):
            continue
        if node_id == target_id:
            return path
        neighbors = [
            (neighbor_id, _edge_cost(edge))
            for neighbor_id, edge in graph[node_id].items()
            if _search_node_allowed(nodes[neighbor_id])
            and not isinf(_edge_cost(edge))
        ]
        for neighbor_id, edge_cost in sorted(
            neighbors, key=lambda item: (item[1], int(item[0]))
        ):
            if neighbor_id in path or neighbor_id in blocked_ids:
                continue
            next_path = [*path, neighbor_id]
            candidate = (
                cost + edge_cost,
                edge_count + 1,
                _path_key(next_path),
            )
            if neighbor_id not in best or candidate < best[neighbor_id]:
                best[neighbor_id] = candidate
                heapq.heappush(heap, (*candidate, neighbor_id, next_path))
    return []


def _sp_score(entity_id: str, path: list[str], branch_ids: set[str]) -> float:
    """Compute 2|C_m(P)| / (|S_m| + |V(P) minus {m}|)."""
    path_without_entity = [node_id for node_id in path if node_id != entity_id]
    denominator = len(branch_ids) + len(path_without_entity)
    return (
        2 * sum(node_id in branch_ids for node_id in path_without_entity)
        / denominator
        if denominator
        else 0.0
    )


def _preposition_arguments(
    preposition_id: str,
    state: _WorkingState,
) -> tuple[set[str], set[str]]:
    arguments = {"prep_arg1": set(), "prep_arg2": set()}
    for edge in state.pas_edges:
        dep_id = str(edge.dep_idx)
        relation = _relation_key(edge.relation)
        if (
            str(edge.head_idx) == preposition_id
            and dep_id in state.nodes
            and relation in arguments
        ):
            arguments[relation].add(dep_id)
    return arguments["prep_arg1"], arguments["prep_arg2"]


def _possessive_arguments(
    marker_id: str,
    state: _WorkingState,
) -> tuple[set[str], set[str]]:
    owners: set[str] = set()
    possessed: set[str] = set()
    for edge in state.pas_edges:
        head_id, dep_id = str(edge.head_idx), str(edge.dep_idx)
        if marker_id not in {head_id, dep_id}:
            continue
        related_id = dep_id if head_id == marker_id else head_id
        if related_id not in state.nodes:
            continue
        relation = _relation_key(edge.relation)
        if relation in POSSESSIVE_OWNER_RELATIONS:
            owners.add(related_id)
        elif relation in POSSESSIVE_POSSESSED_RELATIONS:
            possessed.add(related_id)
    return owners, possessed


def _coordination_groups(
    state: _WorkingState,
    entity_ids: set[str],
) -> list[tuple[str, list[str]]]:
    groups: dict[str, set[str]] = {}
    for edge in state.pas_edges:
        relation = _relation_key(edge.relation)
        if "coord" not in relation and relation not in {
            "conj_member",
            "disj_member",
        }:
            continue
        head_id, dep_id = str(edge.head_idx), str(edge.dep_idx)
        if (
            head_id in state.nodes
            and state.nodes[head_id].text.lower() in {"and", "or"}
            and dep_id in entity_ids
        ):
            groups.setdefault(head_id, set()).add(dep_id)
        elif (
            dep_id in state.nodes
            and state.nodes[dep_id].text.lower() in {"and", "or"}
            and head_id in entity_ids
        ):
            groups.setdefault(dep_id, set()).add(head_id)
    return [
        (connector_id, sorted(member_ids, key=int))
        for connector_id, member_ids in sorted(
            groups.items(), key=lambda item: int(item[0])
        )
        if len(member_ids) >= 2
    ]


def _shared_syntactic_attachment(
    state: _WorkingState,
    connector_id: str,
    member_ids: list[str],
) -> str | None:
    group_ids = {*member_ids, connector_id}
    attachments = [
        _external_syntax_head(member_id, group_ids, state.syntax_heads)
        for member_id in member_ids
    ]
    if not attachments or None in attachments or len(set(attachments)) != 1:
        return None
    attachment_id = attachments[0]
    if attachment_id == "0" or attachment_id not in state.nodes:
        return None
    attachment = state.nodes[attachment_id]
    if _is_punctuation(attachment.text) or attachment.text.lower() in {"and", "or"}:
        return None
    return attachment_id


def _external_syntax_head(
    member_id: str,
    group_ids: set[str],
    syntax_heads: dict[str, int],
) -> str | None:
    current = member_id
    seen: set[str] = set()
    while current in syntax_heads and current not in seen:
        seen.add(current)
        head_id = str(syntax_heads[current])
        if head_id not in group_ids:
            return head_id
        current = head_id
    return None


def _is_pure_coordination(edge: TokenReasoningEdge) -> bool:
    markers = ("coord", "conj_member", "disj_member", "_and_c", "_or_c")
    return bool(edge.relations) and all(
        any(marker in _relation_key(relation) for marker in markers)
        for relation in edge.relations
    )


def _semantic_node_allowed(node: TokenReasoningNode) -> bool:
    """Return whether a node participates in semantic boundary degree."""
    lower = node.text.lower()
    if node.id == "0" or _is_punctuation(node.text):
        return False
    if lower in (
        SEMANTIC_SCOPE_WORDS
        | DETERMINERS
        | PREPOSITIONS
        | LIGHT_VERBS
        | RELATIVE_PRONOUNS
    ):
        return False
    return node.kind != "function"


def _search_node_allowed(node: TokenReasoningNode) -> bool:
    """Function words may bridge paths; only virtual root and punctuation may not."""
    return node.id != "0" and not _is_punctuation(node.text)


def _is_semantic_node(node: TokenReasoningNode) -> bool:
    if node.kind == "entity" or ENTITY_RE.fullmatch(node.text):
        return False
    return _semantic_node_allowed(node) and node.kind in {"content", "constraint"}


def _semantic_degree(
    node_id: str,
    nodes: dict[str, TokenReasoningNode],
    graph: dict[str, dict[str, TokenReasoningEdge]],
) -> int:
    return sum(
        1
        for neighbor_id, edge in graph.get(node_id, {}).items()
        if not _is_pure_coordination(edge)
        and _semantic_node_allowed(nodes[neighbor_id])
    )


def _edge_cost(edge: TokenReasoningEdge) -> int | float:
    """Return the Equation (2) cost; ``inf`` represents pure noise."""
    if edge.rules & AUGMENTED_CORE_RULES:
        return 1
    if edge.rules & STRUCTURAL_RULES:
        return 2
    return min((_pas_relation_cost(relation) for relation in edge.relations), default=inf)


def _pas_relation_cost(relation: str) -> int | float:
    relation = _relation_key(relation)
    if any(marker in relation for marker in INVALID_RELATION_MARKERS):
        return inf
    if relation in CORE_RELATIONS or relation.startswith(CORE_RELATION_PREFIXES):
        return 1
    if (
        relation in STRUCTURAL_RELATIONS
        or relation.startswith(STRUCTURAL_RELATION_PREFIXES)
        or relation.endswith("_mod")
    ):
        return 2
    return 3


def _build_token_nodes(result: HanLPSDPResult) -> dict[str, TokenReasoningNode]:
    nodes = {"0": TokenReasoningNode("0", "ROOT", "function")}
    for index, token in enumerate(result.tokens, start=1):
        text = str(token)
        nodes[str(index)] = TokenReasoningNode(
            id=str(index), text=text, kind=classify_node(text)
        )
    return nodes


def _mark_possessive_markers(state: _WorkingState) -> None:
    for node in state.nodes.values():
        if node.text.lower() in POSSESSIVE_MARKERS:
            owners, possessed = _possessive_arguments(node.id, state)
            if owners or possessed:
                node.kind = "function"


def _is_high_salience_node(node: TokenReasoningNode) -> bool:
    return node.kind in {"entity", "content"} or (
        node.kind == "constraint" and NUMERIC_RE.fullmatch(node.text) is not None
    )


def _remove_graph_nodes(
    graph: dict[str, dict[str, TokenReasoningEdge]],
    node_ids: set[str],
) -> None:
    for node_id in node_ids:
        for neighbor_id in list(graph.get(node_id, {})):
            graph[neighbor_id].pop(node_id, None)
        graph.pop(node_id, None)


def _add_edge(
    graph: dict[str, dict[str, TokenReasoningEdge]],
    source_id: str,
    target_id: str,
    *,
    relation: str | None = None,
    rule: str | None = None,
) -> None:
    edge = graph.setdefault(source_id, {}).get(target_id)
    if edge is None:
        edge = TokenReasoningEdge()
        graph[source_id][target_id] = edge
        graph.setdefault(target_id, {})[source_id] = edge
    if relation:
        edge.relations.add(relation)
    if rule:
        edge.rules.add(rule)


def _relation_key(relation: str) -> str:
    return (
        relation.strip()
        .lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def _path_key(path: list[str]) -> tuple[int, ...]:
    return tuple(int(node_id) for node_id in path)


def _is_punctuation(text: str) -> bool:
    return bool(text) and all(not character.isalnum() for character in text)
