from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
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
    support: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "depends_on": list(self.depends_on),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "leaf_node_ids": list(self.leaf_node_ids),
            "valid": self.valid,
            "validation_errors": list(self.validation_errors),
            "raw_payload": self.raw_payload,
        }


class PathAlignedAtomicDAGGenerator:
    """Generate and validate an action-trace-backed atomic question DAG."""

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
        return validate_atomic_question_dag(raw_payload)


class NoPathAtomicDAGGenerator:
    """Generate and validate an original-question-only action-trace DAG."""

    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def generate(self, *, original_question: str) -> AtomicQuestionDAGResult:
        user_prompt = build_atomic_question_dag_no_path_prompt(original_question=original_question)
        raw_payload = self.llm_client.chat_json(ATOMIC_QUESTION_DAG_NO_PATH_SYSTEM, user_prompt)
        return validate_atomic_question_dag(raw_payload)


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


def validate_atomic_question_dag(raw_payload: dict[str, Any]) -> AtomicQuestionDAGResult:
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

        refs = _question_refs([question, *consume])
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
    if errors:
        return AtomicQuestionDAGResult(
            nodes=parsed_nodes,
            edges=edges,
            leaf_node_ids=leaf_node_ids,
            valid=False,
            validation_errors=errors,
            raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
        )
    return AtomicQuestionDAGResult(
        nodes=parsed_nodes,
        edges=edges,
        leaf_node_ids=leaf_node_ids,
        valid=True,
        validation_errors=[],
        raw_payload=raw_payload,
    )


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
