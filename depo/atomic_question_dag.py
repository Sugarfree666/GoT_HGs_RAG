from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from prompts import (
    ATOMIC_QUESTION_DAG_NO_PATH_SYSTEM,
    ATOMIC_QUESTION_DAG_SYSTEM,
    build_atomic_question_dag_no_path_prompt,
    build_atomic_question_dag_prompt,
)

if TYPE_CHECKING:
    from llm_client import LLMClient


PLACEHOLDER_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
ANSWER_REF_RE = re.compile(
    "(?<!\\w)(q[1-9][0-9]*)(?:_answer|(?:'|\\u2019)s\\s+answer)\\b",
    re.IGNORECASE,
)
BRACED_QUESTION_REF_RE = re.compile(r"\{\{q[1-9][0-9]*\}\}", re.IGNORECASE)


@dataclass(frozen=True)
class RestoredTokenPath:
    path_id: str
    nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"path_id": self.path_id, "nodes": list(self.nodes)}


@dataclass(frozen=True)
class AtomicQuestionNode:
    id: str
    question: str
    depends_on: tuple[str, ...]
    operation: str = "lookup"
    support_step_ids: tuple[str, ...] = ()
    output_type: str = ""
    semantic_edge_ids: tuple[str, ...] = ()
    support: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "depends_on": list(self.depends_on),
            "operation": self.operation,
            "support_step_ids": list(self.support_step_ids),
            "output_type": self.output_type,
            "semantic_edge_ids": list(self.semantic_edge_ids),
        }


@dataclass(frozen=True)
class AtomicQuestionEdge:
    source: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AtomicQuestionDAGResult:
    nodes: list[AtomicQuestionNode]
    edges: list[AtomicQuestionEdge]
    leaf_node_ids: list[str]
    valid: bool
    validation_errors: list[str]
    raw_payload: dict[str, Any] | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "leaf_node_ids": list(self.leaf_node_ids),
            "valid": self.valid,
            "validation_errors": list(self.validation_errors),
            "warnings": list(self.warnings),
            "raw_payload": self.raw_payload,
        }


class PathAlignedAtomicDAGGenerator:
    """Run Step5 as token path -> semantic reasoning path -> atomic question DAG."""

    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def generate(
        self,
        *,
        original_question: str,
        explicit_entities: list[str],
        global_best_paths: list[list[str]],
    ) -> AtomicQuestionDAGResult:
        explicit_entity_texts = [str(entity) for entity in explicit_entities]
        restored_global_best_paths = [[str(node) for node in path] for path in global_best_paths]
        preflight_errors = _preflight_errors(
            explicit_entities=explicit_entity_texts,
            global_best_paths=restored_global_best_paths,
        )
        if preflight_errors:
            return _invalid_result(preflight_errors, raw_payload=None)

        user_prompt = build_atomic_question_dag_prompt(
            original_question=original_question,
            explicit_entities=explicit_entity_texts,
            global_best_paths=restored_global_best_paths,
        )
        raw_payload = self.llm_client.chat_json(ATOMIC_QUESTION_DAG_SYSTEM, user_prompt)
        return validate_atomic_question_dag(
            raw_payload,
            original_question=original_question,
            explicit_entities=explicit_entity_texts,
            global_best_paths=restored_global_best_paths,
        )


class NoPathAtomicDAGGenerator:
    """Generate and validate an original-question-only action-trace DAG."""

    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def generate(self, *, original_question: str) -> AtomicQuestionDAGResult:
        user_prompt = build_atomic_question_dag_no_path_prompt(original_question=original_question)
        raw_payload = self.llm_client.chat_json(ATOMIC_QUESTION_DAG_NO_PATH_SYSTEM, user_prompt)
        sanitized_payload, warnings = _sanitize_no_path_action_trace(raw_payload)
        result = _validate_action_trace_atomic_question_dag(sanitized_payload)
        result.warnings.extend(warnings)
        return result


def restore_entity_paths(step4_paths: Any, mask_mappings: Any) -> list[RestoredTokenPath]:
    """Restore ENTITY placeholders in Step4 paths.

    Kept for tests/debugging. Step5 now consumes only the single restored global best path.
    """

    restored: list[RestoredTokenPath] = []
    for path in step4_paths:
        path_id = str(getattr(path, "path_id", "") or "")
        restored.append(
            RestoredTokenPath(
                path_id=path_id,
                nodes=tuple(_restore_path_nodes(getattr(path, "nodes", []), mask_mappings)),
            )
        )
    return restored


def restore_global_best_path(step4_global_selection: Any, mask_mappings: Any) -> list[str]:
    if isinstance(step4_global_selection, dict):
        raw_nodes = step4_global_selection.get("nodes") or []
    elif isinstance(step4_global_selection, (list, tuple)):
        raw_nodes = step4_global_selection
    else:
        raw_nodes = getattr(step4_global_selection, "nodes", [])
    return _restore_path_nodes(raw_nodes, mask_mappings)


def restore_global_best_paths(step4_paths: Any, mask_mappings: Any) -> list[list[str]]:
    restored_paths: list[list[str]] = []
    for path in step4_paths or []:
        if isinstance(path, dict):
            raw_nodes = path.get("nodes") or []
        elif isinstance(path, (list, tuple)):
            raw_nodes = path
        else:
            raw_nodes = getattr(path, "nodes", [])
        restored_paths.append(_restore_path_nodes(raw_nodes, mask_mappings))
    return restored_paths


def validate_atomic_question_dag(
    raw_payload: Any,
    *,
    original_question: str | None = None,
    explicit_entities: list[str] | None = None,
    global_best_paths: list[list[str]] | None = None,
) -> AtomicQuestionDAGResult:
    del explicit_entities
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(raw_payload, dict):
        return _invalid_result(["raw_payload must be an object."], raw_payload=None)

    question_plan = raw_payload.get("question_plan")
    raw_paths = raw_payload.get("semantic_reasoning_paths")
    raw_questions = raw_payload.get("atomic_questions")
    final_answer_type = ""

    if not isinstance(question_plan, dict):
        errors.append("question_plan must be an object.")
    else:
        final_answer_intent = str(question_plan.get("final_answer_intent") or "").strip()
        final_answer_type = str(question_plan.get("final_answer_type") or "").strip()
        if not final_answer_intent:
            errors.append("question_plan.final_answer_intent must be a non-empty string.")
        elif _contains_placeholder(final_answer_intent):
            errors.append("question_plan.final_answer_intent contains unresolved ENTITY placeholder.")
        if not final_answer_type:
            errors.append("question_plan.final_answer_type must be a non-empty string.")
        elif final_answer_type not in OUTPUT_TYPES:
            errors.append(
                f"question_plan.final_answer_type must be one of {sorted(OUTPUT_TYPES)}, got {final_answer_type!r}."
            )
        constraints = _string_list(
            question_plan.get("must_preserve_constraints"),
            "question_plan.must_preserve_constraints",
            errors,
        )
        for constraint in constraints:
            if _contains_placeholder(constraint):
                errors.append("question_plan.must_preserve_constraints contains unresolved ENTITY placeholder.")
                break

    if not isinstance(raw_paths, list) or not raw_paths:
        errors.append("semantic_reasoning_paths must be a non-empty list.")
    if not isinstance(raw_questions, list) or not raw_questions:
        errors.append("atomic_questions must be a non-empty list.")

    step_ids: set[str] = set()
    if isinstance(raw_paths, list):
        step_ids = _validate_semantic_reasoning_paths(
            raw_paths,
            errors,
            warnings,
            original_question=original_question,
            global_best_paths=global_best_paths,
        )

    parsed_nodes: list[AtomicQuestionNode] = []
    if isinstance(raw_questions, list):
        parsed_nodes = _parse_atomic_questions(
            raw_questions,
            support_step_ids=step_ids,
            final_answer_type=final_answer_type,
            errors=errors,
            warnings=warnings,
        )

    edges = _edges_from_nodes(parsed_nodes)
    leaf_node_ids = _leaf_node_ids(parsed_nodes, edges)
    _validate_final_answer_leaf_type(parsed_nodes, leaf_node_ids, final_answer_type, errors)
    return AtomicQuestionDAGResult(
        nodes=parsed_nodes,
        edges=edges,
        leaf_node_ids=leaf_node_ids,
        valid=not errors,
        validation_errors=errors,
        raw_payload=raw_payload,
        warnings=warnings,
    )


OUTPUT_TYPES = {"entity", "person", "place", "organization", "work", "event", "date", "number", "boolean", "value", "set", "unknown"}
STEP_TYPES = {"lookup", "constraint", "compare", "select", "verify", "intersect", "aggregate"}
EVIDENCE_STATUSES = {"path_grounded", "question_only_required", "operator"}
ATOMIC_OPERATIONS = {"lookup", "compare", "select", "verify", "intersect", "aggregate"}
VAGUE_OPERATION_LABELS = ("related to", "associated with", "connected to", "about", "path to")
WH_OUTPUT_TOKENS = {"who", "what", "which", "where", "when", "why", "whom", "whose"}


def _validate_semantic_reasoning_paths(
    raw_paths: list[Any],
    errors: list[str],
    warnings: list[str],
    *,
    original_question: str | None,
    global_best_paths: list[list[str]] | None,
) -> set[str]:
    all_step_ids: set[str] = set()

    for path_index, raw_path in enumerate(raw_paths, start=1):
        prefix = f"semantic_reasoning_paths[{path_index - 1}]"
        if not isinstance(raw_path, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        expected_branch_id = f"p{path_index}"
        branch_id = str(raw_path.get("branch_id") or "").strip()
        if branch_id != expected_branch_id:
            errors.append(f"{prefix}.branch_id must be {expected_branch_id!r}, got {branch_id!r}.")

        source_token_path = _string_list(raw_path.get("source_token_path"), f"{prefix}.source_token_path", errors)
        if not source_token_path:
            errors.append(f"{prefix}.source_token_path must be a non-empty list of strings.")
        if global_best_paths is not None and path_index <= len(global_best_paths):
            expected_source_path = [str(token) for token in global_best_paths[path_index - 1]]
            if source_token_path and source_token_path != expected_source_path:
                errors.append(f"{prefix}.source_token_path must match global_best_paths[{path_index - 1}] exactly.")
        for token in source_token_path:
            if _contains_placeholder(token):
                errors.append(f"{prefix}.source_token_path contains unresolved ENTITY placeholder.")
                break
        source_token_set = set(source_token_path)

        if "semantic_nodes" in raw_path or "semantic_edges" in raw_path:
            errors.append(f"{prefix}: Step5 V2 uses reasoning_steps, not semantic_nodes/semantic_edges.")

        raw_steps = raw_path.get("reasoning_steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            errors.append(f"{prefix}.reasoning_steps must be a non-empty list.")
            raw_steps = []

        step_outputs: list[str] = []
        local_step_ids: set[str] = set()
        for step_index, raw_step in enumerate(raw_steps, start=1):
            step_prefix = f"{prefix}.reasoning_steps[{step_index - 1}]"
            if not isinstance(raw_step, dict):
                errors.append(f"{step_prefix} must be an object.")
                continue

            expected_step_id = f"{expected_branch_id}_s{step_index}"
            step_id = str(raw_step.get("id") or "").strip()
            if step_id != expected_step_id:
                errors.append(f"{step_prefix}.id must be {expected_step_id!r}, got {step_id!r}.")
            elif step_id in all_step_ids:
                errors.append(f"duplicate reasoning step id: {step_id}.")
            if step_id:
                local_step_ids.add(step_id)
                all_step_ids.add(step_id)

            evidence_status = str(raw_step.get("evidence_status") or "").strip()
            path_evidence = _string_list(raw_step.get("path_evidence"), f"{step_prefix}.path_evidence", errors)
            for token in path_evidence:
                if _contains_placeholder(token):
                    errors.append(f"{step_prefix}.path_evidence contains unresolved ENTITY placeholder.")
                    break
                if token not in source_token_set:
                    errors.append(f"{step_prefix}.path_evidence contains token not copied from source_token_path: {token!r}.")
            if not path_evidence and evidence_status not in {"question_only_required", "operator"}:
                errors.append(f"{step_prefix}.path_evidence may be empty only for question_only_required or operator steps.")

            question_evidence = _string_list(raw_step.get("question_evidence"), f"{step_prefix}.question_evidence", errors)
            for evidence in question_evidence:
                if _contains_placeholder(evidence):
                    errors.append(f"{step_prefix}.question_evidence contains unresolved ENTITY placeholder.")
                    break

            known_inputs = _string_list(raw_step.get("known_inputs"), f"{step_prefix}.known_inputs", errors)
            for known_input in known_inputs:
                if _contains_placeholder(known_input):
                    errors.append(f"{step_prefix}.known_inputs contains unresolved ENTITY placeholder.")
                    break

            operation = str(raw_step.get("operation") or "").strip()
            if not operation:
                errors.append(f"{step_prefix}.operation must be a non-empty string.")
            elif _contains_placeholder(operation):
                errors.append(f"{step_prefix}.operation contains unresolved ENTITY placeholder.")
            elif _has_vague_operation(operation, original_question):
                errors.append(f"{step_prefix}.operation is too vague for Step5 V2: {operation!r}.")

            output = str(raw_step.get("output") or "").strip()
            if not output:
                errors.append(f"{step_prefix}.output must be a non-empty string.")
            elif _contains_placeholder(output):
                errors.append(f"{step_prefix}.output contains unresolved ENTITY placeholder.")
            else:
                step_outputs.append(output)

            output_type = str(raw_step.get("output_type") or "").strip()
            if not output_type:
                errors.append(f"{step_prefix}.output_type must be a non-empty string.")
            elif output_type not in OUTPUT_TYPES:
                errors.append(f"{step_prefix}.output_type must be one of {sorted(OUTPUT_TYPES)}, got {output_type!r}.")

            step_type = str(raw_step.get("step_type") or "").strip()
            if step_type not in STEP_TYPES:
                errors.append(f"{step_prefix}.step_type must be one of {sorted(STEP_TYPES)}, got {step_type!r}.")

            if evidence_status not in EVIDENCE_STATUSES:
                errors.append(
                    f"{step_prefix}.evidence_status must be one of {sorted(EVIDENCE_STATUSES)}, got {evidence_status!r}."
                )

        _check_likely_token_path_relabeling(prefix, source_token_path, step_outputs, errors, warnings)

    return all_step_ids


def _parse_atomic_questions(
    raw_questions: list[Any],
    *,
    support_step_ids: set[str],
    final_answer_type: str,
    errors: list[str],
    warnings: list[str],
) -> list[AtomicQuestionNode]:
    parsed_nodes: list[AtomicQuestionNode] = []
    seen_ids: set[str] = set()

    for index, raw_question in enumerate(raw_questions, start=1):
        prefix = f"atomic_questions[{index - 1}]"
        expected_id = f"q{index}"
        if not isinstance(raw_question, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        node_id = str(raw_question.get("id") or "").strip()
        previous_ids = set(seen_ids)
        if node_id != expected_id:
            errors.append(f"atomic question id must be {expected_id}, got {node_id!r}.")
        if node_id in seen_ids:
            errors.append(f"duplicate atomic question id: {node_id}.")
        if node_id:
            seen_ids.add(node_id)

        question = str(raw_question.get("question") or "").strip()
        if not _is_single_question(question):
            errors.append(f"{node_id or expected_id}: question must be a non-empty single question.")
        if BRACED_QUESTION_REF_RE.search(question):
            errors.append(f"{node_id or expected_id}: question must use qN's answer references, not braced references.")
        if _contains_placeholder(question):
            errors.append(f"{node_id or expected_id}: question contains unresolved ENTITY placeholder.")

        depends_on = _coerce_depends_on(raw_question.get("depends_on"), node_id or expected_id, previous_ids, errors)
        missing_refs = [dependency for dependency in depends_on if not _question_mentions_dependency(question, dependency)]
        if missing_refs:
            warnings.append(
                f"{node_id or expected_id}: depends_on entries are not mentioned as qN's answer in question: {missing_refs}."
            )

        operation = str(raw_question.get("operation") or "").strip()
        if operation not in ATOMIC_OPERATIONS:
            errors.append(f"{node_id or expected_id}: operation must be one of {sorted(ATOMIC_OPERATIONS)}, got {operation!r}.")

        raw_support_step_ids = _string_list(
            raw_question.get("support_step_ids"),
            f"{prefix}.support_step_ids",
            errors,
        )
        if operation == "lookup" and not raw_support_step_ids:
            errors.append(f"{node_id or expected_id}: lookup questions must include at least one support_step_id.")
        for step_id in raw_support_step_ids:
            if _contains_placeholder(step_id):
                errors.append(f"{node_id or expected_id}: support_step_ids contains unresolved ENTITY placeholder.")
            if step_id not in support_step_ids:
                errors.append(f"{node_id or expected_id}: unknown support_step_id {step_id!r}.")

        output_type = str(raw_question.get("output_type") or "").strip()
        if not output_type:
            errors.append(f"{node_id or expected_id}: output_type must be a non-empty string.")
        elif output_type not in OUTPUT_TYPES:
            errors.append(f"{node_id or expected_id}: output_type must be one of {sorted(OUTPUT_TYPES)}, got {output_type!r}.")

        raw_semantic_edge_ids = _string_list(
            raw_question.get("semantic_edge_ids", []),
            f"{prefix}.semantic_edge_ids",
            errors,
        )
        for edge_id in raw_semantic_edge_ids:
            if _contains_placeholder(edge_id):
                errors.append(f"{node_id or expected_id}: semantic_edge_ids contains unresolved ENTITY placeholder.")

        parsed_nodes.append(
            AtomicQuestionNode(
                id=node_id,
                question=question,
                depends_on=tuple(depends_on),
                operation=operation or "lookup",
                support_step_ids=tuple(raw_support_step_ids),
                output_type=output_type or final_answer_type,
                semantic_edge_ids=tuple(raw_semantic_edge_ids),
            )
        )

    return parsed_nodes


def _has_vague_operation(operation: str, original_question: str | None) -> bool:
    operation_lc = operation.casefold()
    question_lc = (original_question or "").casefold()
    for label in VAGUE_OPERATION_LABELS:
        if label in operation_lc and label not in question_lc:
            return True
    return False


def _check_likely_token_path_relabeling(
    prefix: str,
    source_token_path: list[str],
    step_outputs: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    del warnings
    source_norm = [_normalize_for_relabeling(token) for token in source_token_path]
    output_norm = [_normalize_for_relabeling(output) for output in step_outputs]
    if any(output in WH_OUTPUT_TOKENS for output in output_norm):
        errors.append(f"{prefix}: reasoning step output uses a wh-token from the token path; likely token-path relabeling.")

    matched_positions: list[int] = []
    for output in output_norm:
        if not output:
            continue
        try:
            matched_positions.append(source_norm.index(output))
        except ValueError:
            continue
    if len(matched_positions) >= 3 and matched_positions == sorted(matched_positions):
        errors.append(f"{prefix}: likely token-path relabeling; reasoning step outputs copy source_token_path tokens in order.")


def _normalize_for_relabeling(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _validate_final_answer_leaf_type(
    parsed_nodes: list[AtomicQuestionNode],
    leaf_node_ids: list[str],
    final_answer_type: str,
    errors: list[str],
) -> None:
    if not parsed_nodes:
        return
    if not leaf_node_ids:
        errors.append("Atomic Question DAG must have at least one leaf node.")
        return
    if len(leaf_node_ids) != 1 or final_answer_type in {"", "unknown", "value"}:
        return
    node_by_id = {node.id: node for node in parsed_nodes}
    leaf = node_by_id.get(leaf_node_ids[0])
    if leaf is None or leaf.output_type in {"", "unknown", "value"}:
        return
    if leaf.output_type != final_answer_type:
        errors.append(
            f"final leaf output_type {leaf.output_type!r} does not match question_plan.final_answer_type {final_answer_type!r}."
        )


def _coerce_depends_on(value: Any, node_id: str, previous_ids: set[str], errors: list[str]) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{node_id}: depends_on must be a list.")
        return []
    dependencies: list[str] = []
    seen: set[str] = set()
    for item in value:
        dependency = str(item or "").strip().lower()
        if not re.fullmatch(r"q[1-9][0-9]*", dependency):
            errors.append(f"{node_id}: depends_on contains invalid dependency id {dependency!r}.")
            continue
        if dependency not in previous_ids:
            errors.append(f"{node_id}: depends_on references non-previous node {dependency!r}.")
            continue
        if dependency in seen:
            errors.append(f"{node_id}: duplicate depends_on entry {dependency!r}.")
            continue
        seen.add(dependency)
        dependencies.append(dependency)
    return dependencies


def _question_mentions_dependency(question: str, dependency: str) -> bool:
    return dependency in _question_refs([question])


def _string_list(value: Any, field_name: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list of strings.")
        return []
    items: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{field_name}[{item_index}] must be a string.")
            continue
        items.append(item.strip())
    return items


def _validate_action_trace_atomic_question_dag(raw_payload: Any) -> AtomicQuestionDAGResult:
    errors: list[str] = []
    raw_actions = raw_payload.get("actions") if isinstance(raw_payload, dict) else None
    if not isinstance(raw_actions, list) or not raw_actions:
        return _invalid_result(
            ["actions must be a non-empty list."],
            raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
        )

    parsed_nodes: list[AtomicQuestionNode] = []
    seen_ids: set[str] = set()

    for index, raw_action in enumerate(raw_actions, start=1):
        if not isinstance(raw_action, dict):
            errors.append(f"actions[{index - 1}] must be an object.")
            continue

        expected_id = f"q{index}"
        node_id = str(raw_action.get("id") or "").strip()
        previous_ids = set(seen_ids)
        if node_id != expected_id:
            errors.append(f"action id must be {expected_id}, got {node_id!r}.")
        if node_id in seen_ids:
            errors.append(f"duplicate action id: {node_id}.")
        if node_id:
            seen_ids.add(node_id)

        consume = _coerce_consume(raw_action.get("consume"), index, errors)
        produce = str(raw_action.get("produce") or "").strip()
        expected_produce = f"{expected_id}_answer"
        if produce != expected_produce:
            errors.append(f"{node_id or expected_id}: produce must be {expected_produce!r}, got {produce!r}.")

        question = str(raw_action.get("question") or "").strip()
        if not _is_single_question(question):
            errors.append(f"{node_id or expected_id}: question must be a non-empty single question.")
        if BRACED_QUESTION_REF_RE.search(question):
            errors.append(f"{node_id or expected_id}: question must use qN's answer references, not braced references.")
        if _contains_placeholder(question):
            errors.append(f"{node_id or expected_id}: question contains unresolved ENTITY placeholder.")
        for consume_item in consume:
            if _contains_placeholder(consume_item):
                errors.append(f"{node_id or expected_id}: consume contains unresolved ENTITY placeholder.")
                break

        refs = _question_refs([question])
        invalid_refs = [ref for ref in refs if ref not in previous_ids]
        for ref in invalid_refs:
            errors.append(f"{node_id or expected_id}: qN reference points to non-previous node {ref!r}.")
        depends_on = tuple(ref for ref in refs if ref in previous_ids)

        parsed_nodes.append(
            AtomicQuestionNode(
                id=node_id,
                question=question,
                depends_on=depends_on,
            )
        )

    edges = _edges_from_nodes(parsed_nodes)
    leaf_node_ids = _leaf_node_ids(parsed_nodes, edges)
    return AtomicQuestionDAGResult(
        nodes=parsed_nodes,
        edges=edges,
        leaf_node_ids=leaf_node_ids,
        valid=not errors,
        validation_errors=errors,
        raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
    )


def validate_action_trace_atomic_question_dag(raw_payload: Any) -> AtomicQuestionDAGResult:
    return _validate_action_trace_atomic_question_dag(raw_payload)


def _sanitize_no_path_action_trace(raw_payload: Any) -> tuple[Any, list[str]]:
    warnings: list[str] = []
    if not isinstance(raw_payload, dict):
        return raw_payload, warnings
    raw_actions = raw_payload.get("actions")
    if not isinstance(raw_actions, list):
        return dict(raw_payload), warnings

    sanitized_payload = dict(raw_payload)
    sanitized_actions: list[Any] = []
    for index, raw_action in enumerate(raw_actions, start=1):
        if not isinstance(raw_action, dict):
            sanitized_actions.append(raw_action)
            continue
        action = dict(raw_action)
        if "consume" not in action:
            warnings.append(f"q{index}: no-path mode inserted empty consume [].")
        elif action.get("consume") != []:
            warnings.append(f"q{index}: no-path mode ignored non-empty consume and replaced it with [].")
        action["consume"] = []
        sanitized_actions.append(action)

    sanitized_payload["actions"] = sanitized_actions
    return sanitized_payload, warnings


def invalid_atomic_question_dag(errors: list[str]) -> AtomicQuestionDAGResult:
    return _invalid_result(errors, raw_payload=None)


def _placeholder_mapping(mask_mappings: Any) -> dict[str, str]:
    return {
        str(item.placeholder): str(item.original_text)
        for item in mask_mappings
        if getattr(item, "placeholder", None) and getattr(item, "original_text", None)
    }


def _restore_path_nodes(raw_nodes: Any, mask_mappings: Any) -> list[str]:
    mapping = _placeholder_mapping(mask_mappings)
    restored_nodes: list[str] = []
    for raw_node in raw_nodes or []:
        node = str(raw_node)
        if PLACEHOLDER_RE.fullmatch(node):
            if node not in mapping:
                raise ValueError(f"Unresolved entity placeholder in Step4 path: {node}")
            restored_nodes.append(mapping[node])
        else:
            restored_nodes.append(node)
    return restored_nodes


def _preflight_errors(*, explicit_entities: list[str], global_best_paths: list[list[str]]) -> list[str]:
    errors: list[str] = []
    if not isinstance(explicit_entities, list):
        errors.append("Step5 explicit_entities must be a list.")
    for entity in explicit_entities:
        if _contains_placeholder(entity):
            errors.append(f"Step5 explicit_entities contains unresolved ENTITY placeholder: {entity}.")
    if not global_best_paths:
        errors.append("Step5 requires at least one non-empty global_best_paths entry.")
        return errors
    for path_index, path in enumerate(global_best_paths, start=1):
        if not isinstance(path, list) or not path:
            errors.append(f"Step5 global_best_paths[{path_index - 1}] must be a non-empty list.")
            continue
        for node in path:
            if _contains_placeholder(node):
                errors.append(f"Step5 global_best_paths[{path_index - 1}] contains unresolved ENTITY placeholder: {node}.")
    return errors


def _coerce_consume(value: Any, action_index: int, errors: list[str]) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"actions[{action_index - 1}]: consume must be a list.")
        return []
    consume: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"actions[{action_index - 1}].consume[{item_index}] must be a string.")
            continue
        consume.append(item.strip())
    return consume


def _edges_from_nodes(nodes: list[AtomicQuestionNode]) -> list[AtomicQuestionEdge]:
    edges: list[AtomicQuestionEdge] = []
    for node in nodes:
        for dependency in node.depends_on:
            edges.append(AtomicQuestionEdge(source=dependency, target=node.id))
    return edges


def _leaf_node_ids(nodes: list[AtomicQuestionNode], edges: list[AtomicQuestionEdge]) -> list[str]:
    parents = {edge.source for edge in edges}
    return [node.id for node in nodes if node.id not in parents]


def _question_refs(values: list[str]) -> tuple[str, ...]:
    refs: set[str] = set()
    for value in values:
        for match in ANSWER_REF_RE.finditer(value):
            refs.add(str(match.group(1)).lower())
    return tuple(sorted(refs, key=lambda item: int(item[1:])))


def _is_single_question(question: str) -> bool:
    if not question or not question.endswith("?"):
        return False
    return question.count("?") == 1


def _contains_placeholder(text: str) -> bool:
    return any(PLACEHOLDER_RE.fullmatch(token) for token in re.findall(r"\bENTITY[A-Z0-9]*\b", text))


def _invalid_result(errors: list[str], raw_payload: dict[str, Any] | None) -> AtomicQuestionDAGResult:
    return AtomicQuestionDAGResult(
        nodes=[],
        edges=[],
        leaf_node_ids=[],
        valid=False,
        validation_errors=list(errors),
        raw_payload=raw_payload,
    )


def prompt_input_payload(
    original_question: str,
    explicit_entities: list[str],
    global_best_paths: list[list[str]],
) -> dict[str, Any]:
    return json.loads(
        build_atomic_question_dag_prompt(
            original_question=original_question,
            explicit_entities=explicit_entities,
            global_best_paths=global_best_paths,
        )
    )
