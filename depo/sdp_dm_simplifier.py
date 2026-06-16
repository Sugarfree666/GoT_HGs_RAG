from __future__ import annotations

from collections import defaultdict

from models import HanLPSDPEdge, HanLPSDPResult, SimplifiedSDPDMEdge, SimplifiedSDPDMGraph


GLUE_RELATIONS = {"BV", "compound", "punct", "det", "det_ARG1"}
CORE_RELATIONS = {"ARG1", "ARG2", "ARG3", "ARG4", "_and_c", "_or_c", "conj", "root"}
STOP_TOKENS = {"the", "a", "an", ".", "?", ","}
PREPOSITION_PREDICATES = {"of", "in", "from", "about", "on", "at"}
ANSWER_TOKEN = "answer"


class SDPDMSimplifier:
    def simplify(self, hanlp_result: HanLPSDPResult) -> SimplifiedSDPDMGraph:
        dm_edges = [edge for edge in hanlp_result.edges if edge.formalism == "sdp/dm"]
        warnings: list[str] = []
        if not dm_edges:
            warnings.append("HanLP result did not contain readable sdp/dm edges.")
            return SimplifiedSDPDMGraph(nodes=[], edges=[], warnings=warnings)

        removed_edges: list[str] = []
        edges_to_remove: set[int] = set()
        simplified_edges: list[SimplifiedSDPDMEdge] = []
        answer_neighbor_keys = _answer_neighbor_keys(dm_edges)

        for edge in _collapse_preposition_arg_pairs(dm_edges, edges_to_remove, answer_neighbor_keys):
            simplified_edges.append(edge)

        for index, edge in enumerate(dm_edges):
            if index in edges_to_remove:
                removed_edges.append(_edge_display(edge))
                continue
            if _touches_any_node(edge, answer_neighbor_keys):
                removed_edges.append(_edge_display(edge))
                continue
            if _is_answer_edge(edge):
                removed_edges.append(_edge_display(edge))
                continue
            if _is_glue_edge(edge):
                removed_edges.append(_edge_display(edge))
                continue
            if _is_stop_token_edge(edge):
                removed_edges.append(_edge_display(edge))
                continue
            simplified_edges.append(_to_simplified_edge(edge))

        simplified_edges = _dedupe_edges(simplified_edges)
        nodes = sorted({edge.head for edge in simplified_edges} | {edge.dep for edge in simplified_edges})
        return SimplifiedSDPDMGraph(
            nodes=nodes,
            edges=simplified_edges,
            removed_edges=removed_edges,
            warnings=warnings,
        )


def _collapse_preposition_arg_pairs(
    edges: list[HanLPSDPEdge],
    edges_to_remove: set[int],
    answer_neighbor_keys: set[tuple[str, int | None]],
) -> list[SimplifiedSDPDMEdge]:
    by_prep: dict[tuple[str, int], dict[str, list[tuple[int, HanLPSDPEdge]]]] = defaultdict(lambda: defaultdict(list))
    for index, edge in enumerate(edges):
        if edge.head.lower() not in PREPOSITION_PREDICATES or edge.relation not in {"ARG1", "ARG2"}:
            continue
        by_prep[(edge.head.lower(), edge.head_idx)][edge.relation].append((index, edge))

    derived_edges: list[SimplifiedSDPDMEdge] = []
    for (prep, _prep_idx), grouped in by_prep.items():
        arg1_edges = grouped.get("ARG1") or []
        arg2_edges = grouped.get("ARG2") or []
        if not arg1_edges or not arg2_edges:
            continue
        grouped_edges = [*arg1_edges, *arg2_edges]
        if any(_is_answer_edge(edge) or _touches_any_node(edge, answer_neighbor_keys) for _index, edge in grouped_edges):
            for index, _edge in grouped_edges:
                edges_to_remove.add(index)
            continue
        for arg1_index, arg1 in arg1_edges:
            for arg2_index, arg2 in arg2_edges:
                edges_to_remove.add(arg1_index)
                edges_to_remove.add(arg2_index)
                provenance = [_edge_display(arg1), _edge_display(arg2)]
                derived_edges.append(
                    SimplifiedSDPDMEdge(
                        head=arg1.dep,
                        relation=prep,
                        dep=arg2.dep,
                        head_idx=arg1.dep_idx,
                        dep_idx=arg2.dep_idx,
                        source_relation=f"{arg1.relation}+{arg2.relation}",
                        source_formalism="sdp/dm",
                        derived=True,
                        rule="collapse_preposition_arg_pair",
                        provenance=provenance,
                    )
                )
    return derived_edges


def _is_glue_edge(edge: HanLPSDPEdge) -> bool:
    return edge.relation in GLUE_RELATIONS


def _is_answer_edge(edge: HanLPSDPEdge) -> bool:
    return edge.head.lower() == ANSWER_TOKEN or edge.dep.lower() == ANSWER_TOKEN


def _answer_neighbor_keys(edges: list[HanLPSDPEdge]) -> set[tuple[str, int | None]]:
    neighbors: set[tuple[str, int | None]] = set()
    for edge in edges:
        if edge.head.lower() == ANSWER_TOKEN:
            neighbors.add(_node_key(edge.dep, edge.dep_idx))
        if edge.dep.lower() == ANSWER_TOKEN:
            neighbors.add(_node_key(edge.head, edge.head_idx))
    return neighbors


def _touches_any_node(edge: HanLPSDPEdge, node_keys: set[tuple[str, int | None]]) -> bool:
    return _node_key(edge.head, edge.head_idx) in node_keys or _node_key(edge.dep, edge.dep_idx) in node_keys


def _node_key(token: str, index: int | None) -> tuple[str, int | None]:
    return (token.lower(), index)


def _is_stop_token_edge(edge: HanLPSDPEdge) -> bool:
    if edge.relation in CORE_RELATIONS:
        return False
    return edge.head.lower() in STOP_TOKENS or edge.dep.lower() in STOP_TOKENS


def _to_simplified_edge(edge: HanLPSDPEdge) -> SimplifiedSDPDMEdge:
    return SimplifiedSDPDMEdge(
        head=edge.head,
        relation=edge.relation,
        dep=edge.dep,
        head_idx=edge.head_idx,
        dep_idx=edge.dep_idx,
        source_relation=edge.relation,
        source_formalism=edge.formalism,
        derived=False,
        provenance=[_edge_display(edge)],
    )


def _dedupe_edges(edges: list[SimplifiedSDPDMEdge]) -> list[SimplifiedSDPDMEdge]:
    result: list[SimplifiedSDPDMEdge] = []
    seen: set[tuple[str, str, str, int | None, int | None, bool, str | None]] = set()
    for edge in edges:
        key = (edge.head, edge.relation, edge.dep, edge.head_idx, edge.dep_idx, edge.derived, edge.rule)
        if key in seen:
            continue
        seen.add(key)
        result.append(edge)
    return result


def _edge_display(edge: HanLPSDPEdge) -> str:
    return f"{edge.head} --{edge.relation}--> {edge.dep}"
