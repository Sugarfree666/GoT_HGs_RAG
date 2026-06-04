from __future__ import annotations

import copy
import re
from typing import Any

from models import (
    ASTSkeleton,
    CandidateNode,
    CandidatePath,
    ProblemFrame,
    Requirement,
    SelectedPath,
    SemanticASTEdge,
    SemanticASTNode,
    SemanticASTPrimaryOperator,
    SemanticASTResult,
)


def validate_selected_paths(
    selected_paths: list[SelectedPath],
    requirements: list[Requirement],
    filtered_paths: list[CandidatePath],
) -> None:
    """Validate the LLM's path choices against filtered program candidates."""

    requirement_ids = [requirement.id for requirement in requirements]
    path_by_id = {path.path_id: path for path in filtered_paths}
    if len(selected_paths) != len(requirements):
        raise ValueError(
            f"Selected path count mismatch: got {len(selected_paths)}, expected {len(requirements)}."
        )

    seen_requirements: set[str] = set()
    for selected in selected_paths:
        if selected.requirement_id not in requirement_ids:
            raise ValueError(f"Selected path references unknown requirement_id={selected.requirement_id!r}.")
        if selected.requirement_id in seen_requirements:
            raise ValueError(f"Requirement {selected.requirement_id!r} was selected more than once.")
        seen_requirements.add(selected.requirement_id)

        path = path_by_id.get(selected.path_id)
        if path is None:
            raise ValueError(f"Selected path_id={selected.path_id!r} does not exist in filtered candidate paths.")
        if selected.requirement_id not in path.candidate_for:
            raise ValueError(
                f"Selected path_id={selected.path_id!r} is not candidate_for requirement "
                f"{selected.requirement_id!r}; candidate_for={path.candidate_for}."
            )

    missing = [requirement_id for requirement_id in requirement_ids if requirement_id not in seen_requirements]
    if missing:
        raise ValueError("Missing selected path for requirement(s): " + ", ".join(missing))


def prefer_endpoint_complete_selected_paths(
    selected_paths: list[SelectedPath],
    requirements: list[Requirement],
    filtered_paths: list[CandidatePath],
) -> tuple[list[SelectedPath], list[str]]:
    """Prefer a shortest path containing both root and target when available."""

    requirement_by_id = {requirement.id: requirement for requirement in requirements}
    path_by_id = {path.path_id: path for path in filtered_paths}
    repaired: list[SelectedPath] = []
    actions: list[str] = []
    for selected in selected_paths:
        requirement = requirement_by_id.get(selected.requirement_id)
        current_path = path_by_id.get(selected.path_id)
        if requirement is None or current_path is None:
            repaired.append(selected)
            continue
        if _path_contains_endpoint(current_path, requirement.root) and _path_contains_endpoint(current_path, requirement.target):
            repaired.append(selected)
            continue
        endpoint_complete = [
            path
            for path in filtered_paths
            if selected.requirement_id in path.candidate_for
            and _path_contains_endpoint(path, requirement.root)
            and _path_contains_endpoint(path, requirement.target)
        ]
        if not endpoint_complete:
            repaired.append(selected)
            continue
        replacement = min(endpoint_complete, key=lambda path: (len(path.node_ids), path.path_id))
        if replacement.path_id != selected.path_id:
            actions.append(
                f"Replaced selected path {selected.path_id} with endpoint-complete path "
                f"{replacement.path_id} for requirement {selected.requirement_id}."
            )
        repaired.append(SelectedPath(requirement_id=selected.requirement_id, path_id=replacement.path_id))
    return repaired, actions


def selected_paths_to_ast_skeleton(
    problem_frame: ProblemFrame,
    selected_paths: list[SelectedPath],
    filtered_paths: list[CandidatePath],
    candidate_nodes: list[CandidateNode],
) -> ASTSkeleton:
    """Build a deterministic AST skeleton from LLM-selected candidate paths."""

    validate_selected_paths(selected_paths, problem_frame.requirements, filtered_paths)
    path_by_id = {path.path_id: path for path in filtered_paths}
    candidate_by_id = {candidate.id: candidate for candidate in candidate_nodes}
    requirement_by_id = {requirement.id: requirement for requirement in problem_frame.requirements}
    selected_by_requirement = {selected.requirement_id: selected for selected in selected_paths}

    nodes: list[SemanticASTNode] = []
    edges: list[SemanticASTEdge] = []
    nodes_by_id: dict[str, SemanticASTNode] = {}
    branch_terminals: dict[str, str] = {}
    requirement_paths: dict[str, list[str]] = {}
    requirement_node_ids: dict[str, list[str]] = {}
    node_surface: dict[str, str] = {}
    node_candidate_ids: dict[str, str] = {}
    surface_by_requirement: dict[str, dict[str, str]] = {}
    terminal_requirement_descriptions: dict[str, str] = {}

    multi_branch = len(problem_frame.requirements) > 1
    used_node_ids: set[str] = set()

    for requirement in problem_frame.requirements:
        selected = selected_by_requirement[requirement.id]
        path = path_by_id[selected.path_id]
        oriented_candidate_ids = _orient_path_node_ids(path, requirement, candidate_by_id)
        oriented_candidate_ids = _trim_to_requirement_span(oriented_candidate_ids, requirement, candidate_by_id)
        requirement_paths[requirement.id] = list(oriented_candidate_ids)
        branch_node_ids: list[str] = []
        branch_surface_map: dict[str, str] = {}
        terminal_candidate_id = _terminal_candidate_id(oriented_candidate_ids, requirement, candidate_by_id)

        for index, candidate_id in enumerate(oriented_candidate_ids):
            candidate = candidate_by_id.get(candidate_id)
            surface = candidate.text if candidate is not None else candidate_id
            is_root = _candidate_matches(requirement.root, candidate_id, surface)
            base_id = _safe_node_id(surface or candidate_id)
            if multi_branch and not is_root:
                node_id = f"{base_id}_{requirement.id}"
            else:
                node_id = base_id
            node_id = _unique_node_id_for_branch(node_id, used_node_ids, nodes_by_id, surface, requirement.id, is_root)
            used_node_ids.add(node_id)
            branch_node_ids.append(node_id)
            branch_surface_map[node_id] = surface
            node_surface[node_id] = surface
            node_candidate_ids[node_id] = candidate_id

            if node_id not in nodes_by_id:
                node = SemanticASTNode(
                    id=node_id,
                    label=surface,
                    kind=_semantic_node_kind(candidate),
                    semantic_type=_semantic_type_for_candidate(candidate),
                    source="selected_path",
                    source_graph_nodes=list(candidate.graph_node_ids) if candidate is not None else [],
                    source_token_indices=list(candidate.token_ids) if candidate is not None else [],
                    grounding_text=surface,
                    branch_of=None if is_root else requirement.id,
                )
                nodes_by_id[node_id] = node
                nodes.append(node)

            del index

        terminal_node_id = ""
        if terminal_candidate_id in oriented_candidate_ids:
            terminal_node_id = branch_node_ids[oriented_candidate_ids.index(terminal_candidate_id)]

        for left_id, right_id, left_node_id, right_node_id in zip(
            oriented_candidate_ids,
            oriented_candidate_ids[1:],
            branch_node_ids,
            branch_node_ids[1:],
        ):
            evidence = _path_evidence_for_pair(path, left_id, right_id)
            relation_hint = (requirement.description or "") if right_node_id == terminal_node_id else ""
            edges.append(
                SemanticASTEdge(
                    source=left_node_id,
                    target=right_node_id,
                    edge_type="attribute",
                    relation_hint=relation_hint,
                    support_path=[str(item) for item in evidence.get("evidence_text_path", [])],
                    support_dependency_relations=_relations_from_evidence(evidence),
                )
            )

        terminal_index = oriented_candidate_ids.index(terminal_candidate_id) if terminal_candidate_id in oriented_candidate_ids else len(oriented_candidate_ids) - 1
        branch_terminals[requirement.id] = branch_node_ids[terminal_index]
        if requirement.description:
            terminal_requirement_descriptions[branch_node_ids[terminal_index]] = requirement.description
        requirement_node_ids[requirement.id] = list(branch_node_ids)
        surface_by_requirement[requirement.id] = branch_surface_map

    operator_inputs = [
        branch_terminals[requirement.id]
        for requirement in problem_frame.requirements
        if requirement.id in branch_terminals
    ]
    operator = SemanticASTPrimaryOperator(
        operator=problem_frame.operator or "NONE",
        inputs=operator_inputs,
        output=problem_frame.answer_mode or "answer",
        cue_text=problem_frame.notes or "",
    )
    return ASTSkeleton(
        nodes=nodes,
        edges=edges,
        operator=operator,
        branch_terminals=branch_terminals,
        requirement_paths=requirement_paths,
        requirement_node_ids=requirement_node_ids,
        node_surface=node_surface,
        node_candidate_ids=node_candidate_ids,
        metadata={
            "selected_paths": [selected.to_dict() for selected in selected_paths],
            "requirements": [requirement.to_dict() for requirement in problem_frame.requirements],
            "surface_by_requirement": surface_by_requirement,
            "terminal_requirement_descriptions": terminal_requirement_descriptions,
            "allowed_edge_pairs": [[edge.source, edge.target] for edge in edges],
        },
    )


def labeled_ast_from_skeleton(
    ast_skeleton: ASTSkeleton,
    label_payload: dict[str, Any],
    problem_frame: ProblemFrame,
) -> SemanticASTResult:
    """Apply LLM relation labels to a fixed skeleton without structural edits."""

    warnings: list[str] = []
    edge_labels = _parse_labeled_edges(label_payload.get("edges", []), warnings)
    labeled_edges: list[SemanticASTEdge] = []
    for edge in ast_skeleton.edges:
        relation = edge_labels.get((edge.source, edge.target), "")
        if not relation:
            relation = _fallback_relation(edge, ast_skeleton)
            warnings.append(f"Missing relation label for {edge.source}->{edge.target}; used fallback.")
        relation = _preserve_terminal_requirement_context(edge, relation, ast_skeleton)
        labeled_edge = copy.deepcopy(edge)
        labeled_edge.relation_hint = relation
        labeled_edges.append(labeled_edge)

    extra_edges = [
        f"{source}->{target}"
        for source, target in edge_labels
        if (source, target) not in {(edge.source, edge.target) for edge in ast_skeleton.edges}
    ]
    if extra_edges:
        warnings.append("Ignored LLM-labeled edges outside the skeleton: " + ", ".join(extra_edges))

    operator = copy.deepcopy(ast_skeleton.operator)
    operator_payload = label_payload.get("operator")
    if isinstance(operator_payload, dict):
        confirmed_operator = str(operator_payload.get("type", operator.operator) or operator.operator).strip().upper()
        if confirmed_operator and confirmed_operator != operator.operator:
            warnings.append(
                f"Ignored LLM operator change {confirmed_operator!r}; using ProblemFrame operator {operator.operator!r}."
            )
        confirmed_inputs = operator_payload.get("inputs", [])
        if isinstance(confirmed_inputs, list):
            confirmed_input_ids = [str(item) for item in confirmed_inputs]
            if confirmed_input_ids and confirmed_input_ids != operator.inputs:
                warnings.append(
                    "Ignored LLM operator input change; using skeleton terminal inputs "
                    + ", ".join(operator.inputs)
                    + "."
                )
        output = str(operator_payload.get("output", "") or "").strip()
        if output:
            operator.output = output

    return SemanticASTResult(
        status="ok",
        primary_operator=operator,
        nodes=copy.deepcopy(ast_skeleton.nodes),
        edges=labeled_edges,
        warnings=warnings,
        raw_payload=label_payload or None,
    )


def _parse_labeled_edges(raw_edges: Any, warnings: list[str]) -> dict[tuple[str, str], str]:
    if not isinstance(raw_edges, list):
        warnings.append("Relation labeling payload did not contain a list of edges.")
        return {}
    result: dict[tuple[str, str], str] = {}
    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source", "")).strip()
        target = str(raw.get("target", "")).strip()
        relation = str(raw.get("relation", raw.get("relation_hint", "")) or "").strip()
        if not source or not target or not relation:
            continue
        result[(source, target)] = relation
    return result


def _orient_path_node_ids(
    path: CandidatePath,
    requirement: Requirement,
    candidate_by_id: dict[str, CandidateNode],
) -> list[str]:
    node_ids = list(path.node_ids)
    root_index = _first_matching_index(node_ids, requirement.root, candidate_by_id)
    target_index = _first_matching_index(node_ids, requirement.target, candidate_by_id)
    if root_index is not None and target_index is not None:
        return node_ids if root_index <= target_index else list(reversed(node_ids))
    if root_index is not None:
        return node_ids if root_index == 0 or root_index < len(node_ids) / 2 else list(reversed(node_ids))
    if target_index is not None:
        return list(reversed(node_ids)) if target_index == 0 else node_ids
    return node_ids


def _first_matching_index(
    node_ids: list[str],
    value: str,
    candidate_by_id: dict[str, CandidateNode],
) -> int | None:
    for index, candidate_id in enumerate(node_ids):
        candidate = candidate_by_id.get(candidate_id)
        surface = candidate.text if candidate is not None else candidate_id
        if _candidate_matches(value, candidate_id, surface):
            return index
    return None


def _terminal_candidate_id(
    oriented_candidate_ids: list[str],
    requirement: Requirement,
    candidate_by_id: dict[str, CandidateNode],
) -> str:
    target_index = _first_matching_index(oriented_candidate_ids, requirement.target, candidate_by_id)
    if target_index is not None:
        return oriented_candidate_ids[target_index]
    return oriented_candidate_ids[-1]


def _trim_to_requirement_span(
    oriented_candidate_ids: list[str],
    requirement: Requirement,
    candidate_by_id: dict[str, CandidateNode],
) -> list[str]:
    root_index = _first_matching_index(oriented_candidate_ids, requirement.root, candidate_by_id)
    target_index = _first_matching_index(oriented_candidate_ids, requirement.target, candidate_by_id)
    if root_index is None or target_index is None:
        return oriented_candidate_ids
    start = min(root_index, target_index)
    end = max(root_index, target_index)
    trimmed = oriented_candidate_ids[start : end + 1]
    if root_index > target_index:
        trimmed = list(reversed(trimmed))
    return trimmed


def _path_contains_endpoint(path: CandidatePath, endpoint: str) -> bool:
    normalized = _norm(endpoint)
    values = {_norm(value) for value in [*path.nodes, *path.node_ids]}
    return bool(normalized) and normalized in values


def _candidate_matches(value: str, candidate_id: str, surface: str) -> bool:
    normalized = _norm(value)
    return bool(normalized) and normalized in {_norm(candidate_id), _norm(surface)}


def _unique_node_id_for_branch(
    node_id: str,
    used_node_ids: set[str],
    nodes_by_id: dict[str, SemanticASTNode],
    surface: str,
    requirement_id: str,
    is_root: bool,
) -> str:
    existing = nodes_by_id.get(node_id)
    if existing is not None and _norm(existing.label) == _norm(surface) and is_root:
        return node_id
    if node_id not in used_node_ids and node_id not in nodes_by_id:
        return node_id
    if existing is not None and _norm(existing.label) == _norm(surface) and not is_root:
        return f"{node_id}_{requirement_id}"
    index = 2
    base = node_id
    while f"{base}_{index}" in used_node_ids or f"{base}_{index}" in nodes_by_id:
        index += 1
    return f"{base}_{index}"


def _path_evidence_for_pair(path: CandidatePath, left_id: str, right_id: str) -> dict[str, Any]:
    for evidence in path.evidence:
        source = str(evidence.get("source", ""))
        target = str(evidence.get("target", ""))
        if {source, target} == {left_id, right_id}:
            return evidence
    return {}


def _relations_from_evidence(evidence: dict[str, Any]) -> list[str]:
    relations: list[str] = []
    for dependency_edge in evidence.get("dependency_edges", []):
        if not isinstance(dependency_edge, dict):
            continue
        for relation in dependency_edge.get("relations", []):
            text = str(relation).strip()
            if text and text not in relations:
                relations.append(text)
    return relations


def _fallback_relation(edge: SemanticASTEdge, ast_skeleton: ASTSkeleton) -> str:
    if edge.relation_hint:
        return edge.relation_hint
    source = ast_skeleton.node_surface.get(edge.source, edge.source)
    target = ast_skeleton.node_surface.get(edge.target, edge.target)
    return f"{target} of {source}"


def _preserve_terminal_requirement_context(
    edge: SemanticASTEdge,
    relation: str,
    ast_skeleton: ASTSkeleton,
) -> str:
    terminal_descriptions = ast_skeleton.metadata.get("terminal_requirement_descriptions", {})
    if not isinstance(terminal_descriptions, dict):
        return relation
    description = str(terminal_descriptions.get(edge.target, "") or "").strip()
    if not description:
        return relation
    description_norm = _norm(description)
    relation_norm = _norm(relation)
    if " in " in description_norm and description_norm not in relation_norm:
        return description
    return relation


def _semantic_node_kind(candidate: CandidateNode | None) -> str:
    if candidate is None:
        return "type_variable"
    if candidate.kind in {"entity", "coref"}:
        return "entity"
    return "type_variable"


def _semantic_type_for_candidate(candidate: CandidateNode | None) -> str | None:
    if candidate is None:
        return None
    if candidate.kind == "entity":
        return "Entity"
    if candidate.kind == "role":
        return "Role"
    if candidate.kind == "slot":
        return "Value"
    if candidate.kind == "type_qualifier":
        return "Type"
    return None


def _safe_node_id(value: str) -> str:
    text = value.strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", text):
        return text
    parts = re.findall(r"[A-Za-z0-9]+", text.lower())
    if not parts:
        return "node"
    return "_".join(parts[:5])


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
