from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any


BARE_VARIABLE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:X\d+(?:_[A-Za-z0-9_]+)?|V\d+|VAR_[A-Za-z0-9_]+)(?![A-Za-z0-9_])"
)


def contains_bare_variable(text: str) -> bool:
    return bool(BARE_VARIABLE_RE.search(str(text or "")))


def validate_atomic_dag_surface(dag: Any) -> list[str]:
    payload = _to_plain_payload(dag)
    nodes = _nodes_from_payload(payload)
    messages: list[str] = []
    node_ids = {_node_id(node) for node in nodes if _node_id(node)}

    for index, node in enumerate(nodes, start=1):
        node_id = _node_id(node) or f"<node {index}>"
        question = str(node.get("question", "") or "").strip()
        if not question:
            messages.append(f"{node_id}: question is empty.")
        elif contains_bare_variable(question):
            messages.append(f"{node_id}: question exposes an internal variable: {question}")

        dependencies = _dependencies(node)
        for dependency in dependencies:
            if dependency not in node_ids:
                messages.append(f"{node_id}: dependency does not exist: {dependency}")

        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        candidates = metadata.get("candidates", [])
        if candidates is None:
            candidates = []
        if not isinstance(candidates, list):
            messages.append(f"{node_id}: metadata.candidates must be a list.")
            continue
        for candidate_index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                messages.append(f"{node_id}: candidate {candidate_index} must be an object.")
                continue
            label = str(candidate.get("label", "") or "").strip()
            source_node_id = str(candidate.get("source_node_id", "") or "").strip()
            if not label:
                messages.append(f"{node_id}: candidate {candidate_index} label is empty.")
            elif contains_bare_variable(label):
                messages.append(f"{node_id}: candidate {candidate_index} label exposes an internal variable: {label}")
            if not source_node_id:
                messages.append(f"{node_id}: candidate {candidate_index} source_node_id is empty.")
            elif source_node_id not in dependencies:
                messages.append(
                    f"{node_id}: candidate {candidate_index} source_node_id is not one of dependencies: {source_node_id}"
                )
    return messages


def _to_plain_payload(dag: Any) -> Any:
    if hasattr(dag, "to_dict") and callable(dag.to_dict):
        return dag.to_dict()
    if is_dataclass(dag):
        return asdict(dag)
    return dag


def _nodes_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        nodes = payload.get("nodes", [])
    elif isinstance(payload, list):
        nodes = payload
    else:
        nodes = []
    return [node for node in nodes if isinstance(node, dict)]


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("node_id") or node.get("id") or "").strip()


def _dependencies(node: dict[str, Any]) -> list[str]:
    raw = node.get("dependencies")
    if raw is None:
        raw = node.get("depends_on", [])
    if not isinstance(raw, list):
        raw = [raw]
    return [str(item).strip() for item in raw if str(item).strip()]
