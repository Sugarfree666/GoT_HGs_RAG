from __future__ import annotations

from itertools import product
from typing import Any

from entity_path_projector import (
    localize_path_pruned_ast_branches,
    parse_path_pruned_ast_payload,
    validate_selected_entity_paths,
)
from models import (
    BestASTReview,
    CandidateSemanticAST,
    EntityOriginPath,
    EntityStartNode,
    PathSetCandidate,
    ScoredEntityPath,
    SelectedEntityPath,
    SemanticASTResult,
)
from prompts import (
    BEST_AST_SELECTION_SYSTEM,
    ENTITY_PATH_SCORING_SYSTEM,
    ENTITY_PATH_SELECTION_SYSTEM,
    SELECTED_PATH_SEMANTIC_TRANSDUCTION_SYSTEM,
    build_best_ast_selection_prompt,
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

    def build_candidate_semantic_asts(
        self,
        *,
        original_question: str,
        restored_question: str,
        path_set_candidates: list[PathSetCandidate],
        entity_origin_paths: list[EntityOriginPath],
        scored_paths: list[ScoredEntityPath],
        undirected_graph_edges: list[dict[str, Any]],
    ) -> list[CandidateSemanticAST]:
        path_by_id = {path.path_id: path for path in entity_origin_paths}
        score_by_path_id = {score.path_id: score for score in scored_paths}
        candidates: list[CandidateSemanticAST] = []
        for path_set in path_set_candidates:
            candidate_id = f"ast_{path_set.path_set_id}"
            selected_paths = [
                path_by_id[path_id]
                for _, path_id in sorted(path_set.path_ids_by_entity.items(), key=lambda item: _entity_id_sort_key(item[0]))
                if path_id in path_by_id
            ]
            candidate = CandidateSemanticAST(
                candidate_id=candidate_id,
                path_set_id=path_set.path_set_id,
                path_ids_by_entity=dict(path_set.path_ids_by_entity),
                path_score_summary=_path_score_summary(path_set, scored_paths),
            )
            if len(selected_paths) != len(path_set.path_ids_by_entity):
                missing = [
                    path_id
                    for path_id in path_set.path_ids_by_entity.values()
                    if path_id not in path_by_id
                ]
                candidate.generation_error = "Path-set references missing path_id(s): " + ", ".join(missing)
                candidates.append(candidate)
                continue

            prompt_paths = []
            for path in selected_paths:
                score = score_by_path_id.get(path.path_id)
                prompt_paths.append(
                    {
                        **path.to_dict(),
                        "path_set_id": path_set.path_set_id,
                        "path_score": score.score if score is not None else 0.0,
                        "path_score_valid": score.valid if score is not None else False,
                        "path_score_reason": score.reason if score is not None else "",
                        "terminal_hint": score.terminal_hint if score is not None else None,
                        "semantic_chain_hint": score.semantic_chain_hint if score is not None else [],
                    }
                )
            try:
                payload = self.llm_client.chat_json(
                    SELECTED_PATH_SEMANTIC_TRANSDUCTION_SYSTEM,
                    build_selected_path_semantic_transduction_prompt(
                        original_question=original_question,
                        restored_question=restored_question,
                        selected_entity_paths=prompt_paths,
                        undirected_graph_edges=undirected_graph_edges,
                    ),
                )
                candidate.raw_payload = payload if isinstance(payload, dict) else {}
                if not isinstance(payload, dict):
                    candidate.parse_error = "LLM returned non-object payload for candidate AST."
                    candidates.append(candidate)
                    continue
                semantic_ast = parse_path_pruned_ast_payload(
                    candidate.raw_payload,
                    selected_paths=selected_paths,
                )
                semantic_ast = localize_path_pruned_ast_branches(
                    semantic_ast=semantic_ast,
                    selected_paths=selected_paths,
                )
                semantic_ast.raw_payload = candidate.raw_payload
                semantic_ast.retry_count = 0
                candidate.semantic_ast = semantic_ast
            except Exception as exc:
                candidate.generation_error = str(exc)
            candidates.append(candidate)
        return candidates

    def select_best_candidate_ast(
        self,
        *,
        original_question: str,
        restored_question: str,
        entity_start_nodes: list[EntityStartNode],
        path_set_candidates: list[PathSetCandidate],
        scored_paths: list[ScoredEntityPath],
        candidate_asts: list[CandidateSemanticAST],
    ) -> tuple[SemanticASTResult, dict[str, Any]]:
        payload = self.llm_client.chat_json(
            BEST_AST_SELECTION_SYSTEM,
            build_best_ast_selection_prompt(
                original_question=original_question,
                restored_question=restored_question,
                entity_start_nodes=[entity.to_dict() for entity in entity_start_nodes],
                path_set_candidates=[candidate.to_dict() for candidate in path_set_candidates],
                path_scores=[score.to_dict() for score in scored_paths],
                candidate_asts=[_candidate_ast_prompt_payload(candidate) for candidate in candidate_asts],
                question_intent_metadata=_lightweight_question_intent_metadata(original_question),
            ),
        )
        raw_payload = payload if isinstance(payload, dict) else {}
        reviews = _parse_best_ast_reviews(raw_payload.get("ast_reviews"))
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_asts}

        best_candidate_id = str(raw_payload.get("best_candidate_id", "") or "").strip()
        chosen = candidate_by_id.get(best_candidate_id)
        if chosen is not None and chosen.semantic_ast is not None:
            raw_payload["selected_candidate_id"] = chosen.candidate_id
            return chosen.semantic_ast, raw_payload

        for review in sorted(reviews, key=lambda item: item.score, reverse=True):
            if not review.valid_for_decomposition:
                continue
            candidate = candidate_by_id.get(review.candidate_id)
            if candidate is not None and candidate.semantic_ast is not None:
                raw_payload["selected_candidate_id"] = candidate.candidate_id
                raw_payload["selection_fallback"] = "highest_valid_review"
                return candidate.semantic_ast, raw_payload

        for review in sorted(reviews, key=lambda item: item.score, reverse=True):
            candidate = candidate_by_id.get(review.candidate_id)
            if candidate is not None and candidate.semantic_ast is not None:
                raw_payload["selected_candidate_id"] = candidate.candidate_id
                raw_payload["selection_fallback"] = "highest_review_score"
                return candidate.semantic_ast, raw_payload

        for candidate in candidate_asts:
            if candidate.semantic_ast is not None:
                raw_payload["selected_candidate_id"] = candidate.candidate_id
                raw_payload["selection_fallback"] = "first_parseable_candidate"
                return candidate.semantic_ast, raw_payload

        errors = [
            f"{candidate.candidate_id}: {candidate.parse_error or candidate.generation_error or 'unparseable'}"
            for candidate in candidate_asts
        ]
        raise ValueError("No parseable candidate Semantic AST was produced. " + "; ".join(errors))

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


def _parse_best_ast_reviews(raw: Any) -> list[BestASTReview]:
    if not isinstance(raw, list):
        return []
    reviews: list[BestASTReview] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "") or "").strip()
        path_set_id = str(item.get("path_set_id", "") or "").strip()
        if not candidate_id:
            continue
        reviews.append(
            BestASTReview(
                candidate_id=candidate_id,
                path_set_id=path_set_id,
                score=_clamp_review_score(item.get("score")),
                valid_for_decomposition=_bool_value(item.get("valid_for_decomposition"), default=True),
                covers_original_question=_bool_value(item.get("covers_original_question"), default=True),
                answer_intent_compatible=_bool_value(item.get("answer_intent_compatible"), default=True),
                branch_complete=_bool_value(item.get("branch_complete"), default=True),
                atomic_questions_would_be_executable=_bool_value(item.get("atomic_questions_would_be_executable"), default=True),
                has_final_operator_question=_bool_value(item.get("has_final_operator_question"), default=False),
                fatal_errors=_str_list(item.get("fatal_errors")),
                reason=str(item.get("reason", "") or "").strip(),
            )
        )
    return reviews


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


def _path_score_summary(path_set: PathSetCandidate, scored_paths: list[ScoredEntityPath]) -> dict[str, Any]:
    score_by_path_id = {score.path_id: score for score in scored_paths}
    return {
        "mean_path_score": path_set.mean_path_score,
        "paths": {
            entity_id: score_by_path_id[path_id].to_dict()
            for entity_id, path_id in path_set.path_ids_by_entity.items()
            if path_id in score_by_path_id
        },
    }


def _candidate_ast_prompt_payload(candidate: CandidateSemanticAST) -> dict[str, Any]:
    semantic_ast_payload: dict[str, Any] | None = None
    if candidate.semantic_ast is not None:
        semantic_ast_payload = candidate.semantic_ast.to_dict()
    return {
        "candidate_id": candidate.candidate_id,
        "path_set_id": candidate.path_set_id,
        "path_ids_by_entity": dict(candidate.path_ids_by_entity),
        "semantic_ast": semantic_ast_payload,
        "raw_payload": candidate.raw_payload,
        "parse_error": candidate.parse_error,
        "generation_error": candidate.generation_error,
        "path_score_summary": candidate.path_score_summary,
    }


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


def _clamp_review_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


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
