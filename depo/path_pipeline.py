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
    CANDIDATE_NODES_AND_FRAME_SYSTEM,
    LABEL_AST_EDGES_SYSTEM,
    SELECT_PATHS_SYSTEM,
    build_candidate_nodes_and_frame_prompt,
    build_label_ast_edges_prompt,
    build_select_paths_prompt,
)


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
    ) -> tuple[list[CandidateNode], ProblemFrame, dict[str, Any]]:
        payload = self.llm_client.chat_json(
            CANDIDATE_NODES_AND_FRAME_SYSTEM,
            build_candidate_nodes_and_frame_prompt(
                question=question,
                restored_question=restored_question,
                graph_nodes=graph_nodes,
            ),
        )
        candidate_nodes = _parse_candidate_nodes(payload.get("candidate_nodes"))
        problem_frame = _parse_problem_frame(payload.get("problem_frame"))
        if not candidate_nodes:
            raise ValueError("Candidate-node LLM response contained no valid candidate_nodes.")
        if not problem_frame.requirements:
            raise ValueError("Candidate-node LLM response contained no valid problem_frame.requirements.")
        problem_frame = _repair_problem_frame(
            question=restored_question or question,
            candidate_nodes=candidate_nodes,
            problem_frame=problem_frame,
        )
        return candidate_nodes, problem_frame, payload

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
                problem_frame=problem_frame.to_dict(),
                filtered_candidate_paths=[path.to_dict() for path in filtered_candidate_paths],
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
        seen.add(candidate_id)
        kind = _candidate_kind(str(item.get("kind", "other") or "other"))
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


def _parse_problem_frame(raw: Any) -> ProblemFrame:
    if not isinstance(raw, dict):
        raise ValueError("problem_frame must be a JSON object.")
    operator = _canonical_operator(str(raw.get("operator", "NONE") or "NONE").strip().upper())
    requirements = _parse_requirements(raw.get("requirements", []))
    return ProblemFrame(
        operator=operator,
        requirements=requirements,
        answer_mode=_optional_str(raw.get("answer_mode")),
        answer_focus=_optional_str(raw.get("answer_focus")),
        notes=_optional_str(raw.get("notes")),
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
        if len(requirements) > 1:
            notes.append("Collapsed multi-requirement NONE frame to the first serial lookup requirement.")
            requirements = requirements[:1]

    answer_mode = problem_frame.answer_mode
    if operator == "NONE" and answer_mode == "boolean":
        answer_mode = None
        notes.append("Cleared boolean answer_mode after operator demotion.")

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


def _join_notes(existing: str | None, notes: list[str]) -> str | None:
    pieces = [existing] if existing else []
    pieces.extend(notes)
    return " | ".join(piece for piece in pieces if piece) or None


def _candidate_kind(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {
        "entity",
        "role",
        "slot",
        "type_qualifier",
        "operator_cue",
        "constraint_value",
        "coref",
        "other",
    }
    return normalized if normalized in allowed else "other"


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
