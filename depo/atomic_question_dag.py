"""Step 5: convert restored Step4 paths into an atomic-question DAG."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from prompts import ATOMIC_QUESTION_DAG_SYSTEM, build_atomic_question_dag_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient


PLACEHOLDER_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
ANSWER_REF_RE = re.compile(
    r"(?<!\w)(q[1-9][0-9]*)(?:_answer|(?:'|\u2019)s\s+answer)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AtomicQuestionNode:
    id: str
    question: str
    depends_on: tuple[str, ...]
    operation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "depends_on": list(self.depends_on),
            "operation": self.operation,
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


class QuestionStructureAtomicDAGGenerator:
    """Generate and validate a Step5 DAG from the original question and paths."""

    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def generate(
        self,
        *,
        original_question: str,
        question_entities: list[str],
        question_structure: list[list[str]],
    ) -> AtomicQuestionDAGResult:
        question_structure = _sanitize_question_structure(question_structure)
        user_prompt = build_atomic_question_dag_prompt(
            original_question=original_question,
            question_entities=question_entities,
            question_structure=question_structure,
        )
        return validate_atomic_question_dag(
            self.llm_client.chat_json(ATOMIC_QUESTION_DAG_SYSTEM, user_prompt)
        )


def restore_global_best_paths(step4_paths: Any, mask_mappings: Any) -> list[list[str]]:
    """Restore entity placeholders in the selected Step4 paths."""

    restored_paths: list[list[str]] = []
    for path in step4_paths:
        nodes = path.get("nodes", []) if isinstance(path, dict) else path.nodes
        restored_paths.append(
            [_restore_placeholder(str(node), mask_mappings) for node in nodes]
        )
    return restored_paths


def validate_atomic_question_dag(raw_payload: Any) -> AtomicQuestionDAGResult:
    """Validate the stable DAG fields used by the downstream executor."""

    if not isinstance(raw_payload, dict):
        return _invalid_result(["raw_payload must be an object."], None)

    raw_questions = raw_payload.get("atomic_questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        return _invalid_result(["atomic_questions must be a non-empty list."], raw_payload)

    nodes, errors = _parse_nodes(raw_questions)
    if not errors:
        errors = _structure_errors(nodes)
    if errors:
        return _invalid_result(errors, raw_payload)

    edges = [
        AtomicQuestionEdge(source=dependency, target=node.id)
        for node in nodes
        for dependency in node.depends_on
    ]
    parent_ids = {edge.source for edge in edges}
    return AtomicQuestionDAGResult(
        nodes=nodes,
        edges=edges,
        leaf_node_ids=[node.id for node in nodes if node.id not in parent_ids],
        valid=True,
        validation_errors=[],
        raw_payload=raw_payload,
    )


def _parse_nodes(
    raw_questions: list[Any],
) -> tuple[list[AtomicQuestionNode], list[str]]:
    nodes: list[AtomicQuestionNode] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for index, raw_question in enumerate(raw_questions):
        prefix = f"atomic_questions[{index}]"
        if not isinstance(raw_question, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        node_id = raw_question.get("id")
        question = raw_question.get("question")
        depends_on = raw_question.get("depends_on", [])
        if not isinstance(node_id, str) or not node_id.strip():
            errors.append(f"{prefix}.id must be a non-empty string.")
            continue
        node_id = node_id.strip()
        if node_id in seen_ids:
            errors.append(f"{prefix}.id duplicates node id {node_id!r}.")
            continue
        seen_ids.add(node_id)
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{prefix}.question must be a non-empty string.")
            continue
        if not isinstance(depends_on, list) or not all(
            isinstance(dependency, str) and dependency.strip()
            for dependency in depends_on
        ):
            errors.append(f"{prefix}.depends_on must be a list of node ids.")
            continue
        dependencies = tuple(dependency.strip() for dependency in depends_on)
        if len(dependencies) != len(set(dependencies)):
            errors.append(f"{prefix}.depends_on duplicates a node id.")
            continue
        nodes.append(
            AtomicQuestionNode(
                id=node_id,
                question=question.strip(),
                depends_on=dependencies,
                operation=str(raw_question.get("operation") or "lookup").strip(),
            )
        )
    return nodes, errors


def _structure_errors(nodes: list[AtomicQuestionNode]) -> list[str]:
    positions = {node.id: index for index, node in enumerate(nodes)}
    errors: list[str] = []
    for index, node in enumerate(nodes):
        if _contains_placeholder(node.question):
            errors.append(f"{node.id}: question contains unresolved ENTITY placeholder.")
        for dependency in node.depends_on:
            dependency_index = positions.get(dependency)
            if dependency_index is None:
                errors.append(
                    f"{node.id}: depends_on references unknown node {dependency!r}."
                )
            elif dependency == node.id:
                errors.append(f"{node.id}: depends_on references itself.")
            elif dependency_index >= index:
                errors.append(
                    f"{node.id}: depends_on references a later node {dependency!r}."
                )
        for reference in _question_refs(node.question):
            reference_index = positions.get(reference)
            if reference_index is None:
                errors.append(
                    f"{node.id}: question references unknown answer {reference!r}."
                )
            elif reference == node.id:
                errors.append(f"{node.id}: question references its own answer.")
            elif reference_index >= index:
                errors.append(
                    f"{node.id}: question references a later answer {reference!r}."
                )
    return errors


def _question_refs(question: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {match.group(1).lower() for match in ANSWER_REF_RE.finditer(question)},
            key=lambda node_id: int(node_id[1:]),
        )
    )


def _restore_placeholder(token: str, mask_mappings: Any) -> str:
    mapping = {
        item.placeholder: item.original_text
        for item in mask_mappings
    }

    def replace(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        if placeholder not in mapping:
            raise ValueError(f"Unresolved entity placeholder in Step4 path: {placeholder}")
        return mapping[placeholder]

    return re.sub(r"\bENTITY[A-Z0-9]*\b", replace, token)


def _sanitize_question_structure(
    question_structure: list[list[str]],
) -> list[list[str]]:
    return [
        [str(node).strip() for node in branch if str(node).strip()]
        for branch in question_structure
        if branch
    ]


def _contains_placeholder(text: str) -> bool:
    return any(
        PLACEHOLDER_RE.fullmatch(token)
        for token in re.findall(r"\bENTITY[A-Z0-9]*\b", text)
    )


def _invalid_result(
    errors: list[str], raw_payload: dict[str, Any] | None
) -> AtomicQuestionDAGResult:
    return AtomicQuestionDAGResult(
        nodes=[],
        edges=[],
        leaf_node_ids=[],
        valid=False,
        validation_errors=errors,
        raw_payload=raw_payload,
    )


def prompt_input_text(
    original_question: str,
    question_entities: list[str],
    question_structure: list[list[str]],
) -> str:
    return build_atomic_question_dag_prompt(
        original_question=original_question,
        question_entities=question_entities,
        question_structure=question_structure,
    )
