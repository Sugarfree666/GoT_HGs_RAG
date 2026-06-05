from __future__ import annotations

from typing import Any

from entity_path_projector import (
    localize_path_pruned_ast_branches,
    parse_path_pruned_ast_payload,
    validate_selected_entity_paths,
    validate_selected_path_semantic_ast,
)
from models import (
    EntityOriginPath,
    EntityStartNode,
    SelectedEntityPath,
    SemanticASTResult,
)
from prompts import (
    ENTITY_PATH_SELECTION_SYSTEM,
    SELECTED_PATH_SEMANTIC_TRANSDUCTION_SYSTEM,
    build_selected_path_semantic_transduction_prompt,
    build_select_entity_paths_prompt,
)


class EntityPathSemanticParser:
    """LLM-facing stages for the entity-origin DEPO backend."""

    def __init__(self, llm_client: Any) -> None:
        if llm_client is None:
            raise TypeError("EntityPathSemanticParser requires an llm_client.")
        self.llm_client = llm_client

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
        validation_feedback: str | None = None
        last_payload: dict[str, Any] = {}
        prompt_paths = [
            {
                **path.to_dict(),
                "selection_reason": _selection_reason(path.path_id, selected_entity_paths),
            }
            for path in selected_path_objects
        ]
        for attempt in range(2):
            payload = self.llm_client.chat_json(
                SELECTED_PATH_SEMANTIC_TRANSDUCTION_SYSTEM,
                build_selected_path_semantic_transduction_prompt(
                    original_question=original_question,
                    restored_question=restored_question,
                    selected_entity_paths=prompt_paths,
                    undirected_graph_edges=undirected_graph_edges,
                    validation_feedback=validation_feedback,
                ),
            )
            last_payload = payload if isinstance(payload, dict) else {}
            semantic_ast = parse_path_pruned_ast_payload(
                last_payload,
                selected_paths=selected_path_objects,
            )
            semantic_ast = localize_path_pruned_ast_branches(
                semantic_ast=semantic_ast,
                selected_paths=selected_path_objects,
            )
            semantic_ast.raw_payload = last_payload
            semantic_ast.retry_count = attempt
            try:
                validate_selected_path_semantic_ast(
                    semantic_ast=semantic_ast,
                    selected_paths=selected_path_objects,
                    original_question=original_question,
                )
                return semantic_ast, last_payload
            except ValueError as exc:
                validation_feedback = str(exc)
                if attempt == 1:
                    raise ValueError(
                        f"LLM produced invalid selected-path semantic transduction after retry: {exc}"
                    ) from exc
        raise ValueError("LLM produced invalid selected-path semantic transduction.")

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
