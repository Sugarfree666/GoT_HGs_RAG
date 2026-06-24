from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from prompts import ATOMIC_QUESTION_DAG_SYSTEM, build_atomic_question_dag_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient


PLACEHOLDER_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
QUESTION_REF_RE = re.compile(r"(?<!\w)(q[1-9][0-9]*)'s answer")


@dataclass(frozen=True)
class RestoredTokenPath:
    path_id: str
    nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"path_id": self.path_id, "nodes": list(self.nodes)}


@dataclass(frozen=True)
class PathSpanSupport:
    path_id: str
    start_index: int
    end_index: int
    nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "nodes": list(self.nodes),
        }


@dataclass(frozen=True)
class AtomicQuestionNode:
    id: str
    question: str
    depends_on: tuple[str, ...]
    support: PathSpanSupport | None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "question": self.question,
            "depends_on": list(self.depends_on),
        }
        if self.support is not None:
            payload["support"] = self.support.to_dict()
        return payload


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
    """Generate and validate a complete atomic question DAG for answering the original question."""

    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def generate(
        self,
        *,
        original_question: str,
        paths: list[RestoredTokenPath],
    ) -> AtomicQuestionDAGResult:
        preflight_errors = _preflight_errors(paths)
        if preflight_errors:
            return _invalid_result(preflight_errors, raw_payload=None)

        user_prompt = build_atomic_question_dag_prompt(
            original_question=original_question,
            paths=[path.to_dict() for path in paths],
        )
        raw_payload = self.llm_client.chat_json(ATOMIC_QUESTION_DAG_SYSTEM, user_prompt)
        return validate_atomic_question_dag(raw_payload, paths)


def restore_entity_paths(step4_paths: Any, mask_mappings: Any) -> list[RestoredTokenPath]:
    mapping = {
        str(item.placeholder): str(item.original_text)
        for item in mask_mappings
        if getattr(item, "placeholder", None) and getattr(item, "original_text", None)
    }
    restored: list[RestoredTokenPath] = []
    for path in step4_paths:
        path_id = str(getattr(path, "path_id", "") or "")
        nodes = [str(node) for node in getattr(path, "nodes", [])]
        restored_nodes: list[str] = []
        for node in nodes:
            if PLACEHOLDER_RE.fullmatch(node):
                if node not in mapping:
                    raise ValueError(f"Unresolved entity placeholder in Step4 path: {node}")
                restored_nodes.append(mapping[node])
            else:
                restored_nodes.append(node)
        restored.append(RestoredTokenPath(path_id=path_id, nodes=tuple(restored_nodes)))
    return restored


def validate_atomic_question_dag(
    raw_payload: dict[str, Any],
    paths: list[RestoredTokenPath],
) -> AtomicQuestionDAGResult:
    errors: list[str] = []
    del paths
    raw_nodes = raw_payload.get("nodes") if isinstance(raw_payload, dict) else None
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return _invalid_result(["nodes must be a non-empty list."], raw_payload=raw_payload if isinstance(raw_payload, dict) else None)

    parsed_nodes: list[AtomicQuestionNode] = []
    seen_ids: set[str] = set()

    for index, raw_node in enumerate(raw_nodes, start=1):
        if not isinstance(raw_node, dict):
            errors.append(f"nodes[{index - 1}] must be an object.")
            continue
        node_id = str(raw_node.get("id") or "").strip()
        expected_id = f"q{index}"
        if node_id != expected_id:
            errors.append(f"node id must be {expected_id}, got {node_id!r}.")
        if node_id in seen_ids:
            errors.append(f"duplicate node id: {node_id}.")
        seen_ids.add(node_id)

        question = str(raw_node.get("question") or "").strip()
        if not _is_single_question(question):
            errors.append(f"{node_id or expected_id}: question must be a non-empty single question.")
        if re.search(r"\{\{q[1-9][0-9]*\}\}", question):
            errors.append(f"{node_id or expected_id}: question must use qN's answer references, not braced references.")
        if _contains_placeholder(question):
            errors.append(f"{node_id or expected_id}: question contains unresolved ENTITY placeholder.")

        depends_on_raw = raw_node.get("depends_on", [])
        depends_on = _coerce_depends_on(depends_on_raw)
        if depends_on is None:
            errors.append(f"{node_id or expected_id}: depends_on must be a list of node ids.")
            depends_on = []
        if len(set(depends_on)) != len(depends_on):
            errors.append(f"{node_id or expected_id}: depends_on contains duplicate node ids.")
        for dependency in depends_on:
            if dependency not in seen_ids or dependency == node_id:
                errors.append(f"{node_id or expected_id}: depends_on references non-previous node {dependency!r}.")

        question_refs = _question_refs(question)
        missing_dependencies = [ref for ref in question_refs if ref not in depends_on]
        if missing_dependencies:
            errors.append(f"{node_id or expected_id}: question references {sorted(question_refs)} but depends_on is {sorted(depends_on)}.")

        parsed_nodes.append(
            AtomicQuestionNode(
                id=node_id,
                question=question,
                depends_on=tuple(depends_on),
                support=None,
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
            raw_payload=raw_payload,
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


def _preflight_errors(paths: list[RestoredTokenPath]) -> list[str]:
    errors: list[str] = []
    if not paths:
        errors.append("Step5 requires at least one restored path.")
    seen: set[str] = set()
    for path in paths:
        if not path.path_id:
            errors.append("Restored path is missing path_id.")
        if path.path_id in seen:
            errors.append(f"Duplicate restored path_id: {path.path_id}.")
        seen.add(path.path_id)
        if not path.nodes:
            label = path.path_id or "(missing)"
            errors.append(f"Restored path {label} has no nodes.")
        for node in path.nodes:
            if _contains_placeholder(node):
                errors.append(f"Restored path {path.path_id} contains unresolved ENTITY placeholder: {node}.")
    return errors


def _parse_support(
    raw: Any,
    path_by_id: dict[str, RestoredTokenPath],
    node_id: str,
    errors: list[str],
) -> PathSpanSupport | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        errors.append(f"{node_id}: support must be an object.")
        return None
    path_id = str(raw.get("path_id") or "").strip()
    path = path_by_id.get(path_id)
    if path is None:
        errors.append(f"{node_id}: support.path_id does not exist: {path_id!r}.")
        return None
    start = _coerce_int(raw.get("start_index"))
    end = _coerce_int(raw.get("end_index"))
    if start is None or end is None:
        errors.append(f"{node_id}: support start_index/end_index must be integers.")
        return None
    if start < 0 or end < 0 or start >= len(path.nodes) or end >= len(path.nodes):
        errors.append(f"{node_id}: support indexes out of range for path {path_id}.")
        return None
    if start > end:
        errors.append(f"{node_id}: support start_index must be <= end_index.")
        return None
    return PathSpanSupport(
        path_id=path_id,
        start_index=start,
        end_index=end,
        nodes=tuple(path.nodes[start : end + 1]),
    )


def _edges_from_nodes(nodes: list[AtomicQuestionNode]) -> list[AtomicQuestionEdge]:
    edges: list[AtomicQuestionEdge] = []
    for node in nodes:
        for dependency in node.depends_on:
            edges.append(AtomicQuestionEdge(source=dependency, target=node.id))
    return edges


def _leaf_node_ids(nodes: list[AtomicQuestionNode], edges: list[AtomicQuestionEdge]) -> list[str]:
    parents = {edge.source for edge in edges}
    return [node.id for node in nodes if node.id not in parents]


def _coerce_depends_on(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item).strip() for item in value if str(item).strip()]


def _question_refs(question: str) -> list[str]:
    refs: list[str] = []
    for match in QUESTION_REF_RE.finditer(question):
        refs.append(str(match.group(1)))
    return sorted(set(refs), key=lambda item: int(item[1:]))


def _is_single_question(question: str) -> bool:
    if not question or not question.endswith("?"):
        return False
    return question.count("?") == 1


def _contains_placeholder(text: str) -> bool:
    return any(PLACEHOLDER_RE.fullmatch(token) for token in re.findall(r"\bENTITY[A-Z0-9]*\b", text))


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _invalid_result(errors: list[str], raw_payload: dict[str, Any] | None) -> AtomicQuestionDAGResult:
    return AtomicQuestionDAGResult(
        nodes=[],
        edges=[],
        leaf_node_ids=[],
        valid=False,
        validation_errors=list(errors),
        raw_payload=raw_payload,
    )


def prompt_input_payload(original_question: str, paths: list[RestoredTokenPath]) -> dict[str, Any]:
    return json.loads(
        build_atomic_question_dag_prompt(
            original_question=original_question,
            paths=[path.to_dict() for path in paths],
        )
    )
