from __future__ import annotations

import re
from typing import Any

from models import (
    ASTSkeleton,
    CandidateNode,
    CandidatePath,
    ProblemFrame,
    Requirement,
    SelectedPath,
    SemanticASTResult,
)
from path_ast_builder import labeled_ast_from_skeleton
from prompts import (
    ALLOWED_OPERATORS,
    CANDIDATE_NODES_SYSTEM,
    LABEL_AST_EDGES_SYSTEM,
    PROBLEM_FRAME_SYSTEM,
    SELECT_PATHS_SYSTEM,
    build_candidate_nodes_prompt,
    build_label_ast_edges_prompt,
    build_problem_frame_prompt,
    build_select_paths_prompt,
)


MAX_PATH_SELECTION_CANDIDATES_PER_REQUIREMENT = 40
MAX_PATH_SELECTION_CANDIDATES_TOTAL = 120


class PathBasedSemanticParser:
    """LLM-facing stages for the selected-path DEPO backend."""

    def __init__(self, llm_client: Any) -> None:
        if llm_client is None:
            raise TypeError("PathBasedSemanticParser requires an llm_client.")
        self.llm_client = llm_client

    def build_candidate_nodes_and_frame(
        self,
        question: str,
        restored_question: str,
        graph_nodes: list[dict[str, object]],
        masked_question: str | None = None,
    ) -> tuple[list[CandidateNode], ProblemFrame, dict[str, Any]]:
        candidate_payload = self.llm_client.chat_json(
            CANDIDATE_NODES_SYSTEM,
            build_candidate_nodes_prompt(
                question=question,
                restored_question=restored_question,
                graph_nodes=graph_nodes,
            ),
        )
        candidate_nodes = _parse_candidate_nodes(candidate_payload.get("candidate_nodes"))
        if not candidate_nodes:
            raise ValueError("Candidate-node LLM response contained no valid candidate_nodes.")

        frame_payload = self.llm_client.chat_json(
            PROBLEM_FRAME_SYSTEM,
            build_problem_frame_prompt(
                question=question,
                restored_question=restored_question,
                graph_nodes=graph_nodes,
                candidate_nodes=[candidate.to_dict() for candidate in candidate_nodes],
                masked_question=masked_question,
            ),
        )
        problem_frame = _parse_problem_frame(frame_payload)
        if not problem_frame.requirements:
            raise ValueError("Problem Frame LLM response contained no valid requirements.")
        problem_frame = _repair_problem_frame(
            question=restored_question or question,
            candidate_nodes=candidate_nodes,
            problem_frame=problem_frame,
        )
        return candidate_nodes, problem_frame, {
            "candidate_nodes": candidate_payload,
            "problem_frame": frame_payload,
        }

    def select_paths(
        self,
        question: str,
        problem_frame: ProblemFrame,
        filtered_candidate_paths: list[CandidatePath],
        validation_feedback: str | None = None,
    ) -> tuple[list[SelectedPath], dict[str, Any]]:
        payload = self.llm_client.chat_json(
            SELECT_PATHS_SYSTEM,
            build_select_paths_prompt(
                question=question,
                problem_frame=_problem_frame_for_path_selection(problem_frame),
                filtered_candidate_paths=[
                    _candidate_path_for_selection(path)
                    for path in _candidate_paths_for_selection_prompt(
                        filtered_candidate_paths,
                        problem_frame.requirements,
                    )
                ],
                validation_feedback=validation_feedback,
            ),
        )
        selected_paths = _parse_selected_paths(payload.get("selected_paths"))
        if not selected_paths:
            raise ValueError("Path-selection LLM response contained no valid selected_paths.")
        return selected_paths, payload

    def label_ast_edges(
        self,
        question: str,
        ast_skeleton: ASTSkeleton,
        selected_paths: list[SelectedPath],
        problem_frame: ProblemFrame,
    ) -> tuple[SemanticASTResult, dict[str, Any]]:
        payload = self.llm_client.chat_json(
            LABEL_AST_EDGES_SYSTEM,
            build_label_ast_edges_prompt(
                question=question,
                ast_skeleton=ast_skeleton.to_dict(),
                selected_paths=[selected.to_dict() for selected in selected_paths],
                problem_frame=problem_frame.to_dict(),
            ),
        )
        semantic_ast = labeled_ast_from_skeleton(
            ast_skeleton=ast_skeleton,
            label_payload=payload,
            problem_frame=problem_frame,
        )
        return semantic_ast, payload


def _parse_candidate_nodes(raw: Any) -> list[CandidateNode]:
    if not isinstance(raw, list):
        return []
    candidates: list[CandidateNode] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not candidate_id or not text or candidate_id in seen:
            continue
        kind = _candidate_kind(str(item.get("kind", "other") or "other"))
        if kind is None or _is_forbidden_candidate_endpoint(text):
            continue
        seen.add(candidate_id)
        token_ids = _int_list(item.get("token_ids", []))
        graph_node_ids = _str_list(item.get("graph_node_ids", item.get("node_ids", [])))
        if not graph_node_ids and token_ids:
            graph_node_ids = [str(token_id) for token_id in token_ids]
        candidates.append(
            CandidateNode(
                id=candidate_id,
                text=text,
                kind=kind,
                token_ids=token_ids,
                graph_node_ids=graph_node_ids,
                confidence=_float_value(item.get("confidence", 1.0), default=1.0),
            )
        )
    return candidates


def _problem_frame_for_path_selection(problem_frame: ProblemFrame) -> dict[str, Any]:
    """Return the LLM-visible Problem Frame without internal operator fields."""

    return {
        "answer_focus": problem_frame.answer_focus,
        "answer_mode": problem_frame.answer_mode,
        "requirements": [requirement.to_dict() for requirement in problem_frame.requirements],
    }


def _candidate_path_for_selection(path: CandidatePath) -> dict[str, Any]:
    """Return a compact LLM path-selection view without bulky dependency evidence."""

    return {
        "path_id": path.path_id,
        "nodes": list(path.nodes),
        "node_ids": list(path.node_ids),
        "candidate_for": list(path.candidate_for),
        "length": len(path.node_ids),
    }


def _candidate_paths_for_selection_prompt(
    filtered_candidate_paths: list[CandidatePath],
    requirements: list[Requirement],
) -> list[CandidatePath]:
    """Keep the LLM path-selection prompt bounded while preserving full debug paths."""

    selected: list[CandidatePath] = []
    seen: set[str] = set()
    per_requirement_limit = max(
        1,
        min(
            MAX_PATH_SELECTION_CANDIDATES_PER_REQUIREMENT,
            MAX_PATH_SELECTION_CANDIDATES_TOTAL // max(1, len(requirements)),
        ),
    )
    for requirement in requirements:
        relevant = [
            path
            for path in filtered_candidate_paths
            if requirement.id in path.candidate_for
        ]
        endpoint_complete = [
            path
            for path in relevant
            if _path_mentions_endpoint(path, requirement.root)
            and _path_mentions_endpoint(path, requirement.target)
        ]
        partial = [
            path
            for path in relevant
            if path not in endpoint_complete
        ]
        ordered = [
            *sorted(endpoint_complete, key=_path_selection_sort_key),
            *sorted(partial, key=_path_selection_sort_key),
        ]
        for path in ordered[:per_requirement_limit]:
            if path.path_id in seen:
                continue
            selected.append(path)
            seen.add(path.path_id)
            if len(selected) >= MAX_PATH_SELECTION_CANDIDATES_TOTAL:
                return selected

    if len(selected) >= MAX_PATH_SELECTION_CANDIDATES_TOTAL:
        return selected
    for path in sorted(filtered_candidate_paths, key=_path_selection_sort_key):
        if path.path_id in seen:
            continue
        selected.append(path)
        seen.add(path.path_id)
        if len(selected) >= MAX_PATH_SELECTION_CANDIDATES_TOTAL:
            break
    return selected


def _parse_problem_frame(raw: Any) -> ProblemFrame:
    if not isinstance(raw, dict):
        raise ValueError("problem_frame must be a JSON object.")
    frame = raw.get("problem_frame") if isinstance(raw.get("problem_frame"), dict) else raw
    notes = _optional_str(frame.get("notes"))
    raw_operator = _optional_str(frame.get("operator"))
    if raw_operator:
        notes = _join_notes(notes, [f"Ignored Problem Frame operator {raw_operator!r}; downstream synthesis handles operators."])
    requirements = _parse_requirements(frame.get("requirements", []))
    return ProblemFrame(
        operator="NONE",
        requirements=requirements,
        answer_mode=_optional_str(frame.get("answer_mode")),
        answer_focus=_optional_str(frame.get("answer_focus")),
        notes=notes,
    )


def _parse_requirements(raw: Any) -> list[Requirement]:
    if not isinstance(raw, list):
        return []
    requirements: list[Requirement] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        requirement_id = str(item.get("id", "")).strip()
        root = str(item.get("root", "")).strip()
        target = str(item.get("target", "")).strip()
        if not requirement_id or not root or not target or requirement_id in seen:
            continue
        seen.add(requirement_id)
        requirements.append(
            Requirement(
                id=requirement_id,
                root=root,
                target=target,
                description=_optional_str(item.get("description")),
                context=_str_list(item.get("context", [])),
            )
        )
    return requirements


def _parse_selected_paths(raw: Any) -> list[SelectedPath]:
    if not isinstance(raw, list):
        return []
    selected: list[SelectedPath] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        requirement_id = str(item.get("requirement_id", "")).strip()
        path_id = str(item.get("path_id", "")).strip()
        if not requirement_id or not path_id:
            continue
        selected.append(SelectedPath(requirement_id=requirement_id, path_id=path_id))
    return selected


def _repair_problem_frame(
    question: str,
    candidate_nodes: list[CandidateNode],
    problem_frame: ProblemFrame,
) -> ProblemFrame:
    """Normalize LLM ProblemFrame to candidate-path constraints.

    This is intentionally conservative: it does not infer a full AST. It only
    rejects unsupported operators and snaps requirement endpoints to candidate
    node text so deterministic path filtering has stable anchors.
    """

    notes: list[str] = []
    operator = problem_frame.operator
    if not _operator_supported_by_question(operator, question):
        notes.append(f"Demoted unsupported operator {operator} to NONE for a question with no matching operator cue.")
        operator = "NONE"

    requirements = [
        Requirement(
            id=requirement.id,
            root=_snap_endpoint(requirement.root, candidate_nodes),
            target=_snap_endpoint(requirement.target, candidate_nodes),
            description=requirement.description,
            context=list(requirement.context),
        )
        for requirement in problem_frame.requirements
    ]

    special_requirement = _played_by_director_requirement(question, candidate_nodes)
    if special_requirement is not None and operator == "NONE":
        notes.append("Repaired played-by-director frame to one serial lookup requirement.")
        requirements = [special_requirement]
    elif operator == "NONE":
        deduped = _dedupe_requirements(requirements)
        if len(deduped) < len(requirements):
            notes.append("Collapsed duplicate NONE-operator requirements.")
        requirements = deduped

    answer_mode = problem_frame.answer_mode

    return ProblemFrame(
        operator=operator,
        requirements=requirements,
        answer_mode=answer_mode,
        answer_focus=problem_frame.answer_focus,
        notes=_join_notes(problem_frame.notes, notes),
    )


def _operator_supported_by_question(operator: str, question: str) -> bool:
    if operator == "NONE":
        return True
    text = _norm(question)
    cue_patterns = {
        "COMPARE_SAME": [r"\bsame\b", r"\bshare\b", r"\bboth\b"],
        "COMPARE_DIFF": [r"\bdifferent\b", r"\bdiffer\b"],
        "COMPARE_GREATER": [r"\bolder\b", r"\byounger\b", r"\bgreater\b", r"\blarger\b", r"\bmore\b"],
        "COMPARE_LESS": [r"\bolder\b", r"\byounger\b", r"\bless\b", r"\bsmaller\b", r"\bfewer\b"],
        "ARGMAX": [r"\blargest\b", r"\bhighest\b", r"\bmost\b", r"\blatest\b", r"\blast\b"],
        "ARGMIN": [r"\bsmallest\b", r"\blowest\b", r"\bfewest\b", r"\bearliest\b", r"\bfirst\b"],
        "INTERSECTION": [r"\bcommon\b", r"\bshared\b", r"\bboth\b"],
        "UNION": [r"\beither\b", r"\bor\b", r"\bany\b"],
        "DIFFERENCE": [r"\bnot\b", r"\bexcept\b", r"\bother than\b"],
        "LOGICAL_AND": [r"\band\b", r"\bboth\b"],
        "LOGICAL_OR": [r"\bor\b", r"\beither\b"],
    }
    return any(re.search(pattern, text) for pattern in cue_patterns.get(operator, []))


def _snap_endpoint(value: str, candidate_nodes: list[CandidateNode]) -> str:
    normalized = _norm(value).replace("_", " ")
    if not normalized:
        return value
    candidates = [(candidate.text, _norm(candidate.text)) for candidate in candidate_nodes]
    for text, candidate_norm in candidates:
        if normalized == candidate_norm:
            return text
    for text, candidate_norm in candidates:
        if normalized in candidate_norm or candidate_norm in normalized:
            return text
    value_words = set(re.findall(r"[a-z0-9]+", normalized))
    best_text = ""
    best_overlap = 0
    for text, candidate_norm in candidates:
        candidate_words = set(re.findall(r"[a-z0-9]+", candidate_norm))
        overlap = len(value_words & candidate_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_text = text
    if best_overlap > 0:
        return best_text
    return value


def _played_by_director_requirement(
    question: str,
    candidate_nodes: list[CandidateNode],
) -> Requirement | None:
    text = question.strip().rstrip("?")
    match = re.search(
        r"\bplayed\s+by\s+the\s+director\s+of\s+(?P<root>.+?)\s+in\s+(?P<context>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    root_phrase = match.group("root")
    context_phrase = match.group("context")
    root = _candidate_text_in_phrase(root_phrase, candidate_nodes)
    context = _candidate_text_in_phrase(context_phrase, candidate_nodes)
    target = _candidate_text_by_norm("played", candidate_nodes)
    if not root or not target:
        return None
    description = f"character played by the director of {root}"
    if context:
        description += f" in {context}"
    return Requirement(
        id="r1",
        root=root,
        target=target,
        description=description,
        context=[context] if context else [],
    )


def _candidate_text_in_phrase(phrase: str, candidate_nodes: list[CandidateNode]) -> str:
    phrase_norm = _norm(phrase)
    matches = [
        candidate.text
        for candidate in candidate_nodes
        if _norm(candidate.text) and _norm(candidate.text) in phrase_norm
    ]
    if not matches:
        return ""
    return max(matches, key=len)


def _candidate_text_by_norm(value: str, candidate_nodes: list[CandidateNode]) -> str:
    normalized = _norm(value)
    for candidate in candidate_nodes:
        if _norm(candidate.text) == normalized:
            return candidate.text
    return ""


def _dedupe_requirements(requirements: list[Requirement]) -> list[Requirement]:
    result: list[Requirement] = []
    seen: set[tuple[str, str]] = set()
    for requirement in requirements:
        key = (_norm(requirement.root), _norm(requirement.target))
        if key in seen:
            continue
        result.append(requirement)
        seen.add(key)
    return result


def _path_mentions_endpoint(path: CandidatePath, endpoint: str) -> bool:
    normalized = _norm(endpoint)
    values = {_norm(value) for value in [*path.nodes, *path.node_ids]}
    return bool(normalized) and normalized in values


def _path_selection_sort_key(path: CandidatePath) -> tuple[int, int, str]:
    return (len(path.node_ids), len(" ".join(path.nodes)), path.path_id)


def _join_notes(existing: str | None, notes: list[str]) -> str | None:
    pieces = [existing] if existing else []
    pieces.extend(notes)
    return " | ".join(piece for piece in pieces if piece) or None


def _candidate_kind(value: str) -> str | None:
    normalized = value.strip().lower()
    allowed = {
        "entity",
        "role",
        "slot",
        "type_qualifier",
        "constraint_value",
        "coref",
        "other",
    }
    if normalized == "operator_cue":
        return None
    return normalized if normalized in allowed else "other"


def _is_forbidden_candidate_endpoint(value: str) -> bool:
    normalized = _norm(value)
    if not normalized:
        return True
    forbidden = {
        "a",
        "an",
        "and",
        "at",
        "be",
        "by",
        "by whom",
        "did",
        "different",
        "do",
        "does",
        "for",
        "from",
        "from which",
        "had",
        "has",
        "have",
        "in",
        "in which",
        "is",
        "of",
        "on",
        "or",
        "same",
        "share",
        "that",
        "the",
        "these",
        "this",
        "those",
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
        "with",
    }
    predicate_noise = {
        "developed",
        "develop",
        "graduate",
        "graduated",
        "located",
        "locate",
    }
    return normalized in forbidden or normalized in predicate_noise


def _canonical_operator(operator: str) -> str:
    aliases = {
        "COMPARE_DIFFERENT": "COMPARE_DIFF",
        "AND": "LOGICAL_AND",
        "OR": "LOGICAL_OR",
        "BRIDGE": "NONE",
        "COUNT": "NONE",
        "FILTER": "NONE",
        "VERIFY": "NONE",
    }
    resolved = aliases.get(operator, operator)
    if resolved not in ALLOWED_OPERATORS:
        raise ValueError(f"ProblemFrame operator {operator!r} is not allowed.")
    return resolved


def _str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for item in raw:
        value = str(item).strip()
        if value:
            result.append(value)
    return result


def _int_list(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for item in raw:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _float_value(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _optional_str(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
