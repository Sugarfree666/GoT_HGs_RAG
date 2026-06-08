from __future__ import annotations

from itertools import product
from typing import Any

from entity_path_projector import (
    localize_path_pruned_ast_branches,
    parse_path_pruned_ast_payload,
    validate_selected_entity_paths,
)
from models import (
    AtomicQuestionDAG,
    AtomicQuestionEdge,
    AtomicQuestionNode,
    EntityOriginPath,
    EntityStartNode,
    PathSetCandidate,
    ScoredEntityPath,
    SelectedEntityPath,
    SemanticASTResult,
)
from prompts import (
    ENTITY_PATH_SCORING_SYSTEM,
    ENTITY_PATH_SELECTION_SYSTEM,
    GROUNDED_ATOMIC_DAG_SYSTEM,
    SELECTED_PATH_SEMANTIC_TRANSDUCTION_SYSTEM,
    build_grounded_atomic_dag_prompt,
    build_score_entity_paths_prompt,
    build_select_entity_paths_prompt,
    build_selected_path_semantic_transduction_prompt,
)


class EntityPathSemanticParser:
    """LLM-facing stages for the entity-origin DEPO backend."""

    def __init__(self, llm_client: Any) -> None:
        if llm_client is None:
            raise TypeError("EntityPathSemanticParser requires an llm_client.")
        self.llm_client = llm_client

    def score_entity_paths(
        self,
        *,
        original_question: str,
        restored_question: str,
        entity_start_nodes: list[EntityStartNode],
        entity_origin_paths: list[EntityOriginPath],
    ) -> tuple[list[ScoredEntityPath], dict[str, Any]]:
        payload = self.llm_client.chat_json(
            ENTITY_PATH_SCORING_SYSTEM,
            build_score_entity_paths_prompt(
                original_question=original_question,
                restored_question=restored_question,
                entity_start_nodes=[entity.to_dict() for entity in entity_start_nodes],
                entity_origin_paths_by_entity=_paths_grouped_for_prompt(
                    entity_origin_paths,
                    entity_start_nodes,
                ),
                question_intent_metadata=_lightweight_question_intent_metadata(original_question),
            ),
        )
        raw_payload = payload if isinstance(payload, dict) else {}
        scored_paths = _parse_scored_entity_paths(raw_payload.get("path_scores"), entity_origin_paths)
        return scored_paths, raw_payload

    def build_grounded_atomic_dag(
        self,
        *,
        original_question: str,
        selected_dependency_path_evidence: list[dict[str, object]] | None = None,
        **legacy_kwargs: Any,
    ) -> tuple[AtomicQuestionDAG, dict[str, Any]]:
        """Generate a grounded Atomic DAG directly from top path-set evidence."""
        if selected_dependency_path_evidence is None:
            path_set_candidates = legacy_kwargs.get("path_set_candidates")
            entity_origin_paths = legacy_kwargs.get("entity_origin_paths")
            if path_set_candidates is None or entity_origin_paths is None:
                raise TypeError(
                    "build_grounded_atomic_dag requires selected_dependency_path_evidence. "
                    "Legacy callers must provide path_set_candidates and entity_origin_paths "
                    "so compact selected dependency path evidence can be built."
                )
            selected_dependency_path_evidence = build_selected_dependency_path_evidence(
                path_set_candidates=path_set_candidates,
                entity_origin_paths=entity_origin_paths,
                max_path_sets=4,
            )
        validation_feedback: str | None = None
        last_payload: dict[str, Any] = {}
        for attempt in range(2):
            payload = self.llm_client.chat_json(
                GROUNDED_ATOMIC_DAG_SYSTEM,
                build_grounded_atomic_dag_prompt(
                    original_question=original_question,
                    selected_dependency_path_evidence=selected_dependency_path_evidence,
                    validation_feedback=validation_feedback,
                ),
            )
            raw_payload = payload if isinstance(payload, dict) else {}
            last_payload = raw_payload
            errors = validate_grounded_atomic_dag_support(
                raw_payload,
                selected_dependency_path_evidence,
            )
            if errors:
                validation_feedback = "\n".join(errors)
                if attempt == 1:
                    raise ValueError(
                        "Grounded Atomic DAG support validation failed after retry: "
                        + validation_feedback
                    )
                continue

            dag, warnings = _parse_grounded_atomic_dag_payload(
                raw_payload,
                selected_dependency_path_evidence=selected_dependency_path_evidence,
            )
            if warnings:
                raw_payload["normalization_warnings"] = warnings
            raw_payload.setdefault("selected_dependency_path_evidence", selected_dependency_path_evidence)
            return dag, raw_payload
        raise ValueError("Grounded Atomic DAG generation failed. Last payload: " + repr(last_payload))

    # Legacy methods kept for tests and compatibility. The main pipeline no
    # longer uses exactly-one path selection.
    def select_entity_paths(
        self,
        *,
        original_question: str,
        restored_question: str,
        entity_start_nodes: list[EntityStartNode],
        entity_origin_paths: list[EntityOriginPath],
    ) -> tuple[list[SelectedEntityPath], dict[str, Any]]:
        validation_feedback: str | None = None
        last_payload: dict[str, Any] = {}
        for attempt in range(2):
            payload = self.llm_client.chat_json(
                ENTITY_PATH_SELECTION_SYSTEM,
                build_select_entity_paths_prompt(
                    original_question=original_question,
                    restored_question=restored_question,
                    entity_start_nodes=[entity.to_dict() for entity in entity_start_nodes],
                    entity_origin_paths_by_entity=_paths_grouped_for_prompt(
                        entity_origin_paths,
                        entity_start_nodes,
                    ),
                    validation_feedback=validation_feedback,
                ),
            )
            last_payload = payload if isinstance(payload, dict) else {}
            selected_paths = _parse_selected_entity_paths(last_payload.get("selected_paths"))
            try:
                validate_selected_entity_paths(
                    selected_paths=selected_paths,
                    entity_starts=entity_start_nodes,
                    entity_origin_paths=entity_origin_paths,
                )
                return selected_paths, last_payload
            except ValueError as exc:
                validation_feedback = str(exc)
                if attempt == 1:
                    raise ValueError(f"LLM selected invalid entity-origin paths after retry: {exc}") from exc
        raise ValueError("LLM selected invalid entity-origin paths.")

    def build_selected_path_semantic_ast(
        self,
        *,
        original_question: str,
        restored_question: str,
        selected_entity_paths: list[SelectedEntityPath],
        entity_origin_paths: list[EntityOriginPath],
        undirected_graph_edges: list[dict[str, Any]],
    ) -> tuple[SemanticASTResult, dict[str, Any]]:
        selected_path_objects = _selected_path_objects(selected_entity_paths, entity_origin_paths)
        prompt_paths = [
            {
                **path.to_dict(),
                "selection_reason": _selection_reason(path.path_id, selected_entity_paths),
            }
            for path in selected_path_objects
        ]
        payload = self.llm_client.chat_json(
            SELECTED_PATH_SEMANTIC_TRANSDUCTION_SYSTEM,
            build_selected_path_semantic_transduction_prompt(
                original_question=original_question,
                restored_question=restored_question,
                selected_entity_paths=prompt_paths,
                undirected_graph_edges=undirected_graph_edges,
            ),
        )
        semantic_ast_payload = payload if isinstance(payload, dict) else {}
        semantic_ast = parse_path_pruned_ast_payload(
            semantic_ast_payload,
            selected_paths=selected_path_objects,
        )
        semantic_ast = localize_path_pruned_ast_branches(
            semantic_ast=semantic_ast,
            selected_paths=selected_path_objects,
        )
        semantic_ast.raw_payload = semantic_ast_payload
        semantic_ast.retry_count = 0
        return semantic_ast, semantic_ast_payload

    def build_path_pruned_ast(
        self,
        *,
        original_question: str,
        restored_question: str,
        selected_entity_paths: list[SelectedEntityPath],
        entity_origin_paths: list[EntityOriginPath],
        undirected_graph_edges: list[dict[str, Any]],
    ) -> tuple[SemanticASTResult, dict[str, Any]]:
        return self.build_selected_path_semantic_ast(
            original_question=original_question,
            restored_question=restored_question,
            selected_entity_paths=selected_entity_paths,
            entity_origin_paths=entity_origin_paths,
            undirected_graph_edges=undirected_graph_edges,
        )


def select_top_paths_by_entity(
    *,
    scored_paths: list[ScoredEntityPath],
    entity_start_nodes: list[EntityStartNode],
    entity_origin_paths: list[EntityOriginPath],
    top_k: int = 2,
    min_valid_score: float = 55.0,
) -> dict[str, list[ScoredEntityPath]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    path_by_id = {path.path_id: path for path in entity_origin_paths}
    score_by_path_id = {score.path_id: score for score in scored_paths if score.path_id in path_by_id}
    result: dict[str, list[ScoredEntityPath]] = {}
    for entity in entity_start_nodes:
        entity_paths = [path for path in entity_origin_paths if path.entity_id == entity.entity_id]
        if not entity_paths:
            raise ValueError(f"No entity-origin paths exist for entity_id={entity.entity_id!r}.")
        entity_scores = [
            score_by_path_id.get(path.path_id)
            or ScoredEntityPath(
                entity_id=entity.entity_id,
                path_id=path.path_id,
                score=0.0,
                valid=False,
                reason="missing score for path",
            )
            for path in entity_paths
        ]
        ordered = sorted(entity_scores, key=lambda item: (-item.score, item.path_id))
        selected = [
            item
            for item in ordered
            if item.valid and item.score >= min_valid_score
        ][:top_k]
        if len(selected) < min(top_k, len(ordered)):
            selected_ids = {item.path_id for item in selected}
            selected.extend(
                item
                for item in ordered
                if item.path_id not in selected_ids
            )
        result[entity.entity_id] = selected[: min(top_k, len(ordered))]
        if not result[entity.entity_id]:
            raise ValueError(f"No top path could be selected for entity_id={entity.entity_id!r}.")
    return result


def build_path_set_candidates(
    *,
    top_paths_by_entity: dict[str, list[ScoredEntityPath]],
    max_path_sets: int | None = None,
) -> list[PathSetCandidate]:
    if not top_paths_by_entity:
        return []
    entity_ids = sorted(top_paths_by_entity, key=_entity_id_sort_key)
    for entity_id in entity_ids:
        if not top_paths_by_entity[entity_id]:
            raise ValueError(f"Entity {entity_id!r} has no top paths.")

    raw_candidates: list[tuple[dict[str, str], float]] = []
    for combo in product(*(top_paths_by_entity[entity_id] for entity_id in entity_ids)):
        path_ids_by_entity = {
            entity_id: scored_path.path_id
            for entity_id, scored_path in zip(entity_ids, combo, strict=True)
        }
        mean_score = sum(scored_path.score for scored_path in combo) / len(combo)
        raw_candidates.append((path_ids_by_entity, mean_score))
    if max_path_sets is not None and len(raw_candidates) > max_path_sets:
        raw_candidates = sorted(raw_candidates, key=lambda item: item[1], reverse=True)[:max_path_sets]
    return [
        PathSetCandidate(
            path_set_id=f"ps{index}",
            path_ids_by_entity=path_ids_by_entity,
            mean_path_score=mean_score,
        )
        for index, (path_ids_by_entity, mean_score) in enumerate(raw_candidates, start=1)
    ]


def build_selected_dependency_path_evidence(
    *,
    path_set_candidates: list[PathSetCandidate],
    entity_origin_paths: list[EntityOriginPath],
    max_path_sets: int | None = 4,
) -> list[dict[str, Any]]:
    if not path_set_candidates:
        raise ValueError("No path-set candidates available for selected dependency path evidence.")

    path_by_id = {path.path_id: path for path in entity_origin_paths}
    selected_path_sets = path_set_candidates[:max_path_sets] if max_path_sets is not None else path_set_candidates
    evidence: list[dict[str, Any]] = []
    seen_path_set_ids: set[str] = set()
    for path_set in selected_path_sets:
        if path_set.path_set_id in seen_path_set_ids:
            continue
        seen_path_set_ids.add(path_set.path_set_id)
        paths_payload: list[dict[str, Any]] = []
        seen_path_ids: set[str] = set()
        for entity_id, path_id in sorted(path_set.path_ids_by_entity.items(), key=lambda item: _entity_id_sort_key(item[0])):
            if path_id in seen_path_ids:
                continue
            seen_path_ids.add(path_id)
            path = path_by_id.get(path_id)
            if path is None:
                raise ValueError(
                    f"Path-set {path_set.path_set_id!r} references missing entity-origin path {path_id!r}."
                )
            paths_payload.append(
                {
                    "entity_id": entity_id,
                    "entity_text": path.entity_text,
                    "path_id": path.path_id,
                    "path_text": " -> ".join(path.nodes),
                    "node_texts": list(path.nodes),
                    "node_ids": list(path.node_ids),
                }
            )
        evidence.append(
            {
                "path_set_id": path_set.path_set_id,
                "paths": paths_payload,
            }
        )
    if not evidence:
        raise ValueError("Selected dependency path evidence is empty after path-set de-duplication.")
    return evidence


def validate_grounded_atomic_dag_support(
    payload: dict[str, Any],
    selected_dependency_path_evidence: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    support_index = _selected_dependency_support_index(selected_dependency_path_evidence)
    raw_nodes = payload.get("nodes")
    if raw_nodes is None:
        raw_nodes = payload.get("atomic_questions") or payload.get("subquestions")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return ["Grounded Atomic DAG payload must contain a non-empty nodes list."]

    for index, raw_node in enumerate(raw_nodes, start=1):
        if not isinstance(raw_node, dict):
            errors.append(f"Node at position {index} is not a JSON object.")
            continue
        node_id = str(raw_node.get("node_id") or raw_node.get("id") or f"q{index}").strip()
        raw_support = raw_node.get("support")
        if isinstance(raw_support, dict):
            support_items = [raw_support]
        elif isinstance(raw_support, list):
            support_items = raw_support
        else:
            errors.append(f"Node {node_id} has no support list.")
            continue
        if not support_items:
            errors.append(f"Node {node_id} has an empty support list.")
            continue

        valid_support_count = 0
        for support_index_in_node, support in enumerate(support_items, start=1):
            if not isinstance(support, dict):
                errors.append(f"Node {node_id} support #{support_index_in_node} is not a JSON object.")
                continue
            path_set_id = str(support.get("path_set_id") or "").strip()
            path_id = str(support.get("path_id") or "").strip()
            key = (path_set_id, path_id)
            if key not in support_index:
                errors.append(
                    f"Node {node_id} support #{support_index_in_node} cites invalid path_set_id/path_id "
                    f"{path_set_id!r}/{path_id!r}."
                )
                continue
            node_texts = _str_list(support.get("node_texts"))
            if not node_texts:
                errors.append(f"Node {node_id} support #{support_index_in_node} has empty node_texts.")
                continue
            available = support_index[key]["normalized_node_texts"]
            missing = [
                text
                for text in node_texts
                if _normalize_support_text(text) not in available
            ]
            if missing:
                errors.append(
                    f"Node {node_id} support #{support_index_in_node} cites node_texts not present in "
                    f"{path_set_id}/{path_id}: {missing}."
                )
                continue
            valid_support_count += 1
        if valid_support_count == 0:
            errors.append(f"Node {node_id} has no valid selected dependency path support.")
    return errors


def _paths_grouped_for_prompt(
    entity_origin_paths: list[EntityOriginPath],
    entity_start_nodes: list[EntityStartNode],
) -> dict[str, list[dict[str, Any]]]:
    entity_text_by_node_id = {
        str(node_id): entity.text
        for entity in entity_start_nodes
        for node_id in entity.graph_node_ids
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in entity_origin_paths:
        other_entity_texts = [
            entity_text_by_node_id[node_id]
            for node_id in path.node_ids[1:-1]
            if node_id in entity_text_by_node_id and entity_text_by_node_id[node_id] != path.entity_text
        ]
        payload = path.to_dict()
        payload["passes_through_other_entity_start"] = bool(other_entity_texts)
        payload["intermediate_entity_start_texts"] = other_entity_texts
        grouped.setdefault(path.entity_id, []).append(payload)
    return grouped


def _parse_selected_entity_paths(raw: Any) -> list[SelectedEntityPath]:
    if not isinstance(raw, list):
        return []
    selected: list[SelectedEntityPath] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("entity_id", "") or "").strip()
        path_id = str(item.get("path_id", "") or "").strip()
        if not entity_id or not path_id:
            continue
        selected.append(
            SelectedEntityPath(
                entity_id=entity_id,
                path_id=path_id,
                reason=str(item.get("reason", "") or "").strip(),
            )
        )
    return selected


def _parse_scored_entity_paths(raw: Any, entity_origin_paths: list[EntityOriginPath]) -> list[ScoredEntityPath]:
    path_by_id = {path.path_id: path for path in entity_origin_paths}
    scored_by_path_id: dict[str, ScoredEntityPath] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            path_id = str(item.get("path_id", "") or "").strip()
            entity_id = str(item.get("entity_id", "") or "").strip()
            path = path_by_id.get(path_id)
            if path is None or path.entity_id != entity_id:
                continue
            scored_by_path_id[path_id] = ScoredEntityPath(
                entity_id=entity_id,
                path_id=path_id,
                score=_clamp_score(item.get("score")),
                valid=_bool_value(item.get("valid"), default=True),
                terminal_hint=_optional_str(item.get("terminal_hint")),
                semantic_chain_hint=_str_list(item.get("semantic_chain_hint")),
                covered_cues=_str_list(item.get("covered_cues")),
                missing_cues=_str_list(item.get("missing_cues")),
                fatal_errors=_str_list(item.get("fatal_errors")),
                reason=str(item.get("reason", "") or "").strip(),
            )
    result: list[ScoredEntityPath] = []
    for path in entity_origin_paths:
        result.append(
            scored_by_path_id.get(path.path_id)
            or ScoredEntityPath(
                entity_id=path.entity_id,
                path_id=path.path_id,
                score=0.0,
                valid=False,
                fatal_errors=["missing_from_llm_output"],
                reason="missing from LLM output",
            )
        )
    return result


def _selected_path_objects(
    selected_entity_paths: list[SelectedEntityPath],
    entity_origin_paths: list[EntityOriginPath],
) -> list[EntityOriginPath]:
    path_by_id = {path.path_id: path for path in entity_origin_paths}
    result: list[EntityOriginPath] = []
    for selected in selected_entity_paths:
        path = path_by_id.get(selected.path_id)
        if path is not None:
            result.append(path)
    return result


def _selection_reason(path_id: str, selected_entity_paths: list[SelectedEntityPath]) -> str:
    for selected in selected_entity_paths:
        if selected.path_id == path_id:
            return selected.reason
    return ""


def _parse_grounded_atomic_dag_payload(
    payload: dict[str, Any],
    *,
    selected_dependency_path_evidence: list[dict[str, Any]],
) -> tuple[AtomicQuestionDAG, list[str]]:
    raw_nodes = payload.get("nodes")
    if raw_nodes is None:
        raw_nodes = payload.get("atomic_questions") or payload.get("subquestions")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Grounded Atomic DAG payload must contain a non-empty nodes list.")

    support_index = _selected_dependency_support_index(selected_dependency_path_evidence)
    warnings: list[str] = []
    nodes: list[AtomicQuestionNode] = []
    edges: list[AtomicQuestionEdge] = []
    seen_ids: set[str] = set()
    output_by_node_id: dict[str, str] = {}

    for index, raw_node in enumerate(raw_nodes, start=1):
        if not isinstance(raw_node, dict):
            warnings.append(f"Dropped non-object node at position {index}.")
            continue
        node_id = _normalize_grounded_node_id(raw_node.get("node_id") or raw_node.get("id"), index, seen_ids, warnings)
        question = str(raw_node.get("question") or raw_node.get("subquestion") or raw_node.get("sub_question") or "").strip()
        if not question:
            warnings.append(f"Dropped node {node_id} because question is empty.")
            continue
        dependencies = _normalize_grounded_dependencies(
            raw_node.get("dependencies") if "dependencies" in raw_node else raw_node.get("depends_on"),
            seen_ids=seen_ids,
            node_id=node_id,
            warnings=warnings,
        )
        support = _normalize_grounded_support(
            raw_node.get("support"),
            support_index=support_index,
            node_id=node_id,
            warnings=warnings,
        )
        if not support:
            raise ValueError(f"Node {node_id} has no valid selected dependency path support.")
        metadata: dict[str, Any] = {
            "source": "grounded_atomic_dag",
            "support": support,
            "support_path_ids": sorted({item["path_id"] for item in support if item.get("path_id")}),
        }
        for metadata_key in ("operation", "input", "one_hop_relation", "answer_type"):
            if metadata_key in raw_node:
                metadata[metadata_key] = raw_node.get(metadata_key)
        output = str(raw_node.get("output") or f"X{len(nodes) + 1}").strip()
        node = AtomicQuestionNode(
            id=node_id,
            question=question,
            type=str(raw_node.get("operation") or raw_node.get("type") or "lookup"),
            inputs=_str_list(raw_node.get("inputs")),
            output=output,
            depends_on=dependencies,
            metadata=metadata,
            source="grounded_atomic_dag",
        )
        nodes.append(node)
        seen_ids.add(node_id)
        output_by_node_id[node_id] = output
        for dependency in dependencies:
            edges.append(AtomicQuestionEdge(source=dependency, target=node_id, variable=output_by_node_id.get(dependency, dependency)))

    if not nodes:
        raise ValueError("Grounded Atomic DAG payload produced no usable nodes.")
    dag = AtomicQuestionDAG(
        nodes=nodes,
        edges=edges,
        variable_to_question={
            node.output: node.id
            for node in nodes
            if node.output
        },
        warnings=warnings,
    )
    return dag, warnings


def _normalize_grounded_node_id(raw: Any, index: int, seen_ids: set[str], warnings: list[str]) -> str:
    node_id = str(raw or f"q{index}").strip()
    if not node_id or not node_id.startswith("q") or not node_id[1:].isdigit():
        replacement = f"q{index}"
        warnings.append(f"Renamed invalid node_id {node_id!r} to {replacement}.")
        node_id = replacement
    if node_id in seen_ids:
        replacement = f"q{index}"
        suffix = index
        while replacement in seen_ids:
            suffix += 1
            replacement = f"q{suffix}"
        warnings.append(f"Renamed duplicate node_id {node_id!r} to {replacement}.")
        node_id = replacement
    return node_id


def _normalize_grounded_dependencies(
    raw: Any,
    *,
    seen_ids: set[str],
    node_id: str,
    warnings: list[str],
) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        warnings.append(f"Ignored invalid dependencies for {node_id}: expected list or string.")
        return []
    dependencies: list[str] = []
    for item in candidates:
        dependency = str(item).strip()
        if not dependency or dependency == node_id:
            continue
        if dependency not in seen_ids:
            warnings.append(f"Ignored dependency {dependency!r} for {node_id}; it does not reference an earlier node.")
            continue
        if dependency not in dependencies:
            dependencies.append(dependency)
    return dependencies


def _normalize_grounded_support(
    raw: Any,
    *,
    support_index: dict[tuple[str, str], dict[str, Any]],
    node_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw_items = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        return []
    support: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        path_set_id = str(item.get("path_set_id") or "").strip()
        path_id = str(item.get("path_id") or "").strip()
        key = (path_set_id, path_id)
        if key not in support_index:
            warnings.append(f"Ignored invalid support path_set/path {path_set_id!r}/{path_id!r} for {node_id}.")
            continue
        node_texts = _str_list(item.get("node_texts"))
        if not node_texts:
            warnings.append(f"Ignored support with empty node_texts for {node_id}.")
            continue
        available = support_index[key]["normalized_node_texts"]
        invalid_node_texts = [
            text
            for text in node_texts
            if _normalize_support_text(text) not in available
        ]
        if invalid_node_texts:
            warnings.append(
                f"Ignored support for {node_id}; node_texts are not in selected path {path_set_id}/{path_id}: "
                f"{invalid_node_texts}."
            )
            continue
        support.append(
            {
                "path_set_id": path_set_id,
                "path_id": path_id,
                "node_texts": node_texts,
                "node_ids": _str_list(item.get("node_ids")),
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    return support


def _selected_dependency_support_index(
    selected_dependency_path_evidence: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path_set in selected_dependency_path_evidence:
        if not isinstance(path_set, dict):
            continue
        path_set_id = str(path_set.get("path_set_id") or "").strip()
        paths = path_set.get("paths")
        if not path_set_id or not isinstance(paths, list):
            continue
        for path in paths:
            if not isinstance(path, dict):
                continue
            path_id = str(path.get("path_id") or "").strip()
            node_texts = _str_list(path.get("node_texts"))
            if not path_id or not node_texts:
                continue
            index[(path_set_id, path_id)] = {
                "node_texts": node_texts,
                "normalized_node_texts": {_normalize_support_text(text) for text in node_texts},
            }
    return index


def _normalize_support_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _lightweight_question_intent_metadata(question: str) -> dict[str, object]:
    text = " ".join(str(question or "").strip().split())
    lower = text.lower()
    wh_cue = None
    if "how many" in lower or "number of" in lower:
        wh_cue = "how many"
        answer_kind = "count"
    else:
        for cue in ("why", "when", "where", "who", "which", "what", "how"):
            if cue in lower.split():
                wh_cue = cue
                break
        answer_kind = {
            "why": "reason",
            "when": "temporal",
            "where": "location",
            "who": "person_or_entity",
            "how": "manner_or_method",
        }.get(wh_cue or "", "entity_or_attribute")
    return {"wh_cue": wh_cue, "answer_kind": answer_kind}


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(100.0, score))


def _bool_value(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if not isinstance(raw, list):
        return []
    return [text for item in raw for text in [str(item).strip()] if text]


def _entity_id_sort_key(entity_id: str) -> tuple[int, str]:
    text = str(entity_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10**9, text)
