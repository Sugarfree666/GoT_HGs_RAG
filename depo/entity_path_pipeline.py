from __future__ import annotations

import re
from itertools import product
from typing import Any

from models import (
    AtomicQuestionDAG,
    AtomicQuestionEdge,
    AtomicQuestionNode,
    EntityOriginPath,
    EntityStartNode,
    AtomicEvidence,
    PathSetCandidate,
    ScoredEntityPath,
    SemanticReasoningEdge,
    SemanticReasoningNode,
    SemanticReasoningPath,
    SemanticReasoningPathResult,
)
from prompts import (
    ATOMIC_DAG_FROM_SEMANTIC_REASONING_PATH_SYSTEM,
    ENTITY_PATH_SCORING_SYSTEM,
    SEMANTIC_REASONING_PATH_SYSTEM,
    build_atomic_dag_from_semantic_reasoning_path_prompt,
    build_score_entity_paths_prompt,
    build_semantic_reasoning_path_prompt,
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
        semantic_reasoning_paths: SemanticReasoningPathResult | dict[str, Any] | None = None,
    ) -> tuple[AtomicQuestionDAG, dict[str, Any]]:
        """Compile Semantic Reasoning Paths into the Step 10 Atomic DAG."""
        if semantic_reasoning_paths is None:
            raise TypeError("build_grounded_atomic_dag requires semantic_reasoning_paths.")
        return self._build_atomic_dag_from_semantic_reasoning_paths(
            original_question=original_question,
            semantic_reasoning_paths=semantic_reasoning_paths,
        )

    def build_semantic_reasoning_paths(
        self,
        *,
        original_question: str,
        selected_dependency_path_evidence: list[dict[str, object]],
    ) -> tuple[SemanticReasoningPathResult, dict[str, Any]]:
        atomic_evidences = extract_atomic_evidences(selected_dependency_path_evidence)
        atomic_evidence_payload = [atom.to_dict() for atom in atomic_evidences]
        validation_feedback: str | None = None
        last_payload: dict[str, Any] = {}
        for attempt in range(2):
            payload = self.llm_client.chat_json(
                SEMANTIC_REASONING_PATH_SYSTEM,
                build_semantic_reasoning_path_prompt(
                    original_question=original_question,
                    atomic_evidences=atomic_evidence_payload,
                    validation_feedback=validation_feedback,
                ),
            )
            raw_payload = payload if isinstance(payload, dict) else {}
            last_payload = raw_payload
            try:
                result = _parse_semantic_reasoning_path_payload(
                    raw_payload,
                    selected_dependency_path_evidence=selected_dependency_path_evidence,
                    evidence_atoms=atomic_evidences,
                )
            except ValueError as exc:
                validation_feedback = str(exc)
                if attempt == 1:
                    raise ValueError(
                        "Semantic Reasoning Path induction failed after retry: "
                        + validation_feedback
                    ) from exc
                continue
            issues = _semantic_reasoning_path_support_issues(
                result,
                selected_dependency_path_evidence,
                atomic_evidences,
            )
            if issues:
                validation_feedback = "\n".join(issues)
                if attempt == 1:
                    raise ValueError(
                        "Semantic Reasoning Path validation failed after retry: "
                        + validation_feedback
                    )
                continue
            raw_payload.setdefault("selected_dependency_path_evidence", selected_dependency_path_evidence)
            raw_payload.setdefault("semantic_reasoning_paths", [path.to_dict() for path in result.paths])
            raw_payload["atomic_evidences"] = atomic_evidence_payload
            raw_payload["evidence_atoms"] = atomic_evidence_payload
            raw_payload["step9_llm_input_contains_raw_dependency_paths"] = False
            result.raw_payload = raw_payload
            return result, raw_payload
        raise ValueError("Semantic Reasoning Path induction failed. Last payload: " + repr(last_payload))

    def _build_atomic_dag_from_semantic_reasoning_paths(
        self,
        *,
        original_question: str,
        semantic_reasoning_paths: SemanticReasoningPathResult | dict[str, Any],
    ) -> tuple[AtomicQuestionDAG, dict[str, Any]]:
        semantic_payload = (
            semantic_reasoning_paths.to_dict()
            if isinstance(semantic_reasoning_paths, SemanticReasoningPathResult)
            else semantic_reasoning_paths
        )
        validation_feedback: str | None = None
        last_payload: dict[str, Any] = {}
        final_support_warnings: list[str] = []
        for attempt in range(2):
            payload = self.llm_client.chat_json(
                ATOMIC_DAG_FROM_SEMANTIC_REASONING_PATH_SYSTEM,
                build_atomic_dag_from_semantic_reasoning_path_prompt(
                    original_question=original_question,
                    semantic_reasoning_paths=semantic_payload,
                    validation_feedback=validation_feedback,
                ),
            )
            raw_payload = payload if isinstance(payload, dict) else {}
            last_payload = raw_payload
            hard_errors: list[str] = []
            support_warnings: list[str] = []
            hard_errors.extend(_atomic_question_variable_placeholder_errors(raw_payload))
            hard_errors.extend(_atomic_dag_required_semantic_fields_errors(raw_payload))
            hard_errors.extend(_semantic_edge_source_errors(raw_payload, semantic_payload))
            final_support_warnings = support_warnings
            if hard_errors:
                validation_feedback = "\n".join(hard_errors)
                if attempt == 1:
                    raise ValueError(
                        "Atomic DAG compilation from Semantic Reasoning Path failed after retry: "
                        + validation_feedback
                    )
                continue
            if support_warnings:
                validation_feedback = "\n".join(support_warnings)
                if attempt == 0:
                    continue
            dag, warnings = _parse_grounded_atomic_dag_payload(
                raw_payload,
                semantic_payload=semantic_payload,
            )
            all_warnings = [*final_support_warnings, *warnings]
            if all_warnings:
                raw_payload["normalization_warnings"] = all_warnings
            raw_payload.setdefault("semantic_reasoning_paths", semantic_payload)
            return dag, raw_payload
        raise ValueError("Atomic DAG compilation failed. Last payload: " + repr(last_payload))

FORBIDDEN_SEMANTIC_NODE_LABELS = {
    "?",
    "a",
    "an",
    "by",
    "compare",
    "common answer",
    "compared to",
    "did",
    "different",
    "do",
    "does",
    "earlier",
    "first",
    "for",
    "from",
    "in",
    "intersection",
    "is",
    "later",
    "of",
    "older",
    "produced first",
    "ranking",
    "released first",
    "same",
    "share",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "yes/no",
    "younger",
}

FORBIDDEN_SEMANTIC_RELATIONS = {
    "compared to",
    "or compared to",
    "produced first",
    "released first",
}


def _parse_semantic_reasoning_path_payload(
    payload: dict[str, Any],
    *,
    selected_dependency_path_evidence: list[dict[str, Any]],
    evidence_atoms: list[AtomicEvidence],
) -> SemanticReasoningPathResult:
    payload = _coerce_flat_semantic_reasoning_payload(
        payload,
        evidence_atoms=evidence_atoms,
        selected_dependency_path_evidence=selected_dependency_path_evidence,
    )
    raw_paths = payload.get("semantic_reasoning_paths")
    if raw_paths is None:
        raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("Semantic Reasoning Path payload must contain a non-empty semantic_reasoning_paths list.")

    selected_path_ids = _selected_dependency_path_ids(selected_dependency_path_evidence)
    selected_path_set_ids = _selected_dependency_path_set_ids(selected_dependency_path_evidence)
    paths: list[SemanticReasoningPath] = []
    for index, raw_path in enumerate(raw_paths, start=1):
        if not isinstance(raw_path, dict):
            raise ValueError(f"semantic_reasoning_paths[{index}] must be a JSON object.")
        branch_id = str(raw_path.get("branch_id") or f"b{index}").strip()
        entity_id = str(raw_path.get("entity_id") or "").strip()
        nodes = _parse_semantic_reasoning_nodes(raw_path.get("nodes"), branch_id)
        edges = _parse_semantic_reasoning_edges(raw_path.get("edges"), branch_id, evidence_atoms=evidence_atoms)
        inferred_atom = _first_atom_for_edges(edges, evidence_atoms)
        if not entity_id and inferred_atom is not None:
            entity_id = inferred_atom.entity_id
        source_path_id = str(raw_path.get("source_path_id") or raw_path.get("path_id") or "").strip()
        if not source_path_id and inferred_atom is not None:
            source_path_id = inferred_atom.source_path_id
        if not branch_id or not entity_id or not source_path_id:
            raise ValueError(f"Semantic reasoning path #{index} missing branch_id/entity_id/source_path_id.")
        if source_path_id not in selected_path_ids:
            raise ValueError(f"Semantic reasoning path {branch_id} references unselected source_path_id={source_path_id!r}.")
        if len(nodes) < 2:
            raise ValueError(f"Semantic reasoning path {branch_id} must contain at least 2 nodes.")
        if not edges:
            raise ValueError(f"Semantic reasoning path {branch_id} must contain at least 1 edge.")
        node_ids = {node.node_id for node in nodes}
        label_by_id = {node.node_id: node.label for node in nodes}
        for node in nodes:
            if _forbidden_semantic_node_label(node.label):
                raise ValueError(f"Semantic reasoning path {branch_id} contains forbidden semantic node label={node.label!r}.")
        for edge in edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"Semantic reasoning edge {edge.edge_id} references missing source/target node.")
            if edge.source == edge.target:
                raise ValueError(f"Semantic reasoning edge {edge.edge_id} has identical source and target.")
            if not edge.relation:
                raise ValueError(f"Semantic reasoning edge {edge.edge_id} has empty relation.")
            if _forbidden_semantic_relation(edge.relation):
                raise ValueError(f"Semantic reasoning edge {edge.edge_id} has forbidden dependency-cue relation={edge.relation!r}.")
            if _forbidden_semantic_node_label(label_by_id.get(edge.source, "")) or _forbidden_semantic_node_label(label_by_id.get(edge.target, "")):
                raise ValueError(f"Semantic reasoning edge {edge.edge_id} connects through a forbidden dependency cue node.")
            if not edge.is_one_hop:
                raise ValueError(f"Semantic reasoning edge {edge.edge_id} must be one-hop.")
            if not edge.support:
                raise ValueError(f"Semantic reasoning edge {edge.edge_id} has empty support.")
            for support in edge.support:
                atom_ids = _str_list(support.get("atom_ids") or support.get("supported_by"))
                if not atom_ids:
                    raise ValueError(f"Semantic reasoning edge {edge.edge_id} support has empty supported_by atom ids.")
                unknown_atom_ids = [atom_id for atom_id in atom_ids if atom_id not in {atom.id for atom in evidence_atoms}]
                if unknown_atom_ids:
                    raise ValueError(f"Semantic reasoning edge {edge.edge_id} cites unknown evidence atoms: {unknown_atom_ids}.")

        terminal_node_id = str(raw_path.get("terminal_node_id") or "").strip() or (edges[-1].target if edges else None)
        if terminal_node_id and terminal_node_id not in node_ids:
            raise ValueError(f"Semantic reasoning path {branch_id} terminal_node_id references a missing node.")
        paths.append(
            SemanticReasoningPath(
                branch_id=branch_id,
                entity_id=entity_id,
                source_path_id=source_path_id,
                nodes=nodes,
                edges=edges,
                terminal_node_id=terminal_node_id,
                score=_clamp_score(raw_path.get("score")),
                warnings=_str_list(raw_path.get("warnings")),
            )
        )

    score_breakdown = {
        str(key): float(value)
        for key, value in (payload.get("score_breakdown") or {}).items()
        if isinstance(value, (int, float))
    } if isinstance(payload.get("score_breakdown"), dict) else {}
    selected_ids = _str_list(payload.get("selected_path_set_ids")) or sorted(selected_path_set_ids)
    return SemanticReasoningPathResult(
        paths=paths,
        selected_path_set_ids=selected_ids,
        operator_intent=payload.get("operator_intent") if isinstance(payload.get("operator_intent"), dict) else {},
        score=_clamp_score(payload.get("score")),
        score_breakdown=score_breakdown,
        warnings=_str_list(payload.get("warnings")),
        raw_payload=payload,
    )


def _coerce_flat_semantic_reasoning_payload(
    payload: dict[str, Any],
    *,
    evidence_atoms: list[AtomicEvidence],
    selected_dependency_path_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(payload.get("semantic_reasoning_paths"), list) or isinstance(payload.get("paths"), list):
        return payload
    raw_edges = payload.get("semantic_reasoning_path")
    if not isinstance(raw_edges, list) or not raw_edges:
        return payload

    atom_by_id = {atom.id: atom for atom in evidence_atoms}
    default_path = _first_selected_path_payload(selected_dependency_path_evidence)
    branch_id = "b1"
    entity_id = str(default_path.get("entity_id") or "e1")
    source_path_id = str(default_path.get("path_id") or "")
    nodes: list[dict[str, Any]] = []
    node_id_by_label: dict[str, str] = {}
    edges: list[dict[str, Any]] = []

    def node_id_for(label: str, *, preferred_kind: str = "semantic_object") -> str:
        normalized = str(label or "").strip()
        if normalized in node_id_by_label:
            return node_id_by_label[normalized]
        node_id = f"{branch_id}_n{len(nodes) + 1}"
        node_id_by_label[normalized] = node_id
        kind = preferred_kind
        lower = normalized.lower()
        if len(nodes) == 0:
            kind = "entity"
        elif lower in {"answer"} or lower.endswith("_date") or lower.endswith("_place") or "date" in lower:
            kind = "value_slot"
        nodes.append({"node_id": node_id, "label": normalized, "kind": kind})
        return node_id

    for index, raw_edge in enumerate(raw_edges, start=1):
        if not isinstance(raw_edge, dict):
            continue
        supported_by = _str_list(raw_edge.get("supported_by"))
        first_atom = atom_by_id.get(supported_by[0]) if supported_by else None
        if first_atom is not None:
            entity_id = entity_id or first_atom.entity_id
            source_path_id = source_path_id or first_atom.source_path_id
        source_label = str(raw_edge.get("source") or "").strip()
        target_label = str(raw_edge.get("target") or "").strip()
        relation = str(raw_edge.get("semantic_relation") or raw_edge.get("relation") or "").strip()
        if not source_label or not target_label or not relation:
            continue
        source_id = node_id_for(source_label)
        target_id = node_id_for(target_label)
        edges.append(
            {
                "edge_id": str(raw_edge.get("edge_id") or raw_edge.get("id") or f"{branch_id}_e{index}"),
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "answer_type": raw_edge.get("answer_type"),
                "is_one_hop": True,
                "supported_by": supported_by,
                "support": [{"supported_by": supported_by, "reason": str(raw_edge.get("reason") or "")}],
                "atomic_question_template": raw_edge.get("atomic_question_template"),
            }
        )

    if not edges:
        return payload
    coerced = dict(payload)
    coerced["semantic_reasoning_paths"] = [
        {
            "branch_id": branch_id,
            "entity_id": entity_id,
            "source_path_id": source_path_id,
            "nodes": nodes,
            "edges": edges,
            "terminal_node_id": edges[-1]["target"],
            "score": payload.get("score", 0),
            "warnings": _str_list(payload.get("warnings")),
        }
    ]
    return coerced


def _first_selected_path_payload(selected_dependency_path_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    for path_set in selected_dependency_path_evidence:
        if not isinstance(path_set, dict):
            continue
        for path in path_set.get("paths", []) or []:
            if isinstance(path, dict):
                return path
    return {}


def _first_atom_for_edges(edges: list[SemanticReasoningEdge], evidence_atoms: list[AtomicEvidence]) -> AtomicEvidence | None:
    atom_by_id = {atom.id: atom for atom in evidence_atoms}
    for edge in edges:
        for support in edge.support:
            if not isinstance(support, dict):
                continue
            for atom_id in _str_list(support.get("atom_ids") or support.get("supported_by")):
                atom = atom_by_id.get(atom_id)
                if atom is not None:
                    return atom
    return None


def _parse_semantic_reasoning_nodes(raw: Any, branch_id: str) -> list[SemanticReasoningNode]:
    if not isinstance(raw, list):
        raise ValueError(f"Semantic reasoning path {branch_id} nodes must be a list.")
    nodes: list[SemanticReasoningNode] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Semantic reasoning path {branch_id} node #{index} must be an object.")
        node_id = _normalize_semantic_reasoning_node_id(item.get("node_id") or item.get("id"), branch_id, index, seen)
        label = str(item.get("label") or item.get("text") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not label or not kind:
            raise ValueError(f"Semantic reasoning node {node_id} missing label/kind.")
        nodes.append(
            SemanticReasoningNode(
                node_id=node_id,
                label=label,
                kind=kind,
                semantic_type=_optional_str(item.get("semantic_type")),
                source_path_id=_optional_str(item.get("source_path_id")),
                source_node_texts=_str_list(item.get("source_node_texts")),
                source_node_ids=_str_list(item.get("source_node_ids")),
            )
        )
    return nodes


def _parse_semantic_reasoning_edges(
    raw: Any,
    branch_id: str,
    *,
    evidence_atoms: list[AtomicEvidence],
) -> list[SemanticReasoningEdge]:
    if not isinstance(raw, list):
        raise ValueError(f"Semantic reasoning path {branch_id} edges must be a list.")
    edges: list[SemanticReasoningEdge] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Semantic reasoning path {branch_id} edge #{index} must be an object.")
        edge_id = str(item.get("edge_id") or item.get("id") or f"{branch_id}_e{index}").strip()
        if not edge_id:
            edge_id = f"{branch_id}_e{index}"
        if edge_id in seen:
            edge_id = f"{branch_id}_e{index}"
        seen.add(edge_id)
        support = _normalize_semantic_reasoning_support(
            item.get("support"),
            evidence_atoms=evidence_atoms,
            edge_supported_by=item.get("supported_by"),
        )
        edges.append(
            SemanticReasoningEdge(
                edge_id=edge_id,
                source=str(item.get("source") or "").strip(),
                target=str(item.get("target") or "").strip(),
                relation=str(item.get("relation") or item.get("semantic_relation") or "").strip(),
                answer_type=_optional_str(item.get("answer_type")),
                is_one_hop=_bool_value(item.get("is_one_hop"), default=True),
                support=support,
                atomic_question_template=_optional_str(item.get("atomic_question_template")),
            )
        )
    return edges


def _normalize_semantic_reasoning_node_id(raw: Any, branch_id: str, index: int, seen: set[str]) -> str:
    node_id = str(raw or f"{branch_id}_n{index}").strip()
    if not node_id:
        node_id = f"{branch_id}_n{index}"
    if node_id in seen:
        node_id = f"{branch_id}_n{index}"
    seen.add(node_id)
    return node_id


def _normalize_semantic_reasoning_support(
    raw: Any,
    *,
    evidence_atoms: list[AtomicEvidence],
    edge_supported_by: Any = None,
) -> list[dict[str, Any]]:
    atom_by_id = {atom.id: atom for atom in evidence_atoms}
    supported_by = _str_list(edge_supported_by)
    if supported_by and raw is None:
        return [_support_from_atom_ids(supported_by, atom_by_id)]

    if isinstance(raw, dict):
        raw_items = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        return []
    result: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        atom_ids = _str_list(item.get("atom_ids") or item.get("supported_by"))
        if not atom_ids and supported_by:
            atom_ids = supported_by
        path_set_id = str(item.get("path_set_id") or "").strip()
        path_id = str(item.get("path_id") or "").strip()
        node_texts = _str_list(item.get("node_texts"))
        node_ids = _str_list(item.get("node_ids"))
        if not atom_ids:
            atom_ids = _matching_atom_ids_for_support(
                evidence_atoms,
                path_set_id=path_set_id,
                path_id=path_id,
                node_texts=node_texts,
            )
        if atom_ids and (not path_set_id or not path_id or not node_texts):
            known_atoms = [atom_by_id[atom_id] for atom_id in atom_ids if atom_id in atom_by_id]
            if known_atoms:
                first_atom = known_atoms[0]
                path_set_id = path_set_id or first_atom.path_set_id
                path_id = path_id or first_atom.source_path_id
                node_texts = node_texts or _unique_preserve(
                    text
                    for atom in known_atoms
                    for text in atom.node_texts
                )
                node_ids = node_ids or _unique_preserve(
                    node_id
                    for atom in known_atoms
                    for node_id in atom.node_ids
                )
        result.append(
            {
                "path_set_id": path_set_id,
                "path_id": path_id,
                "node_texts": node_texts,
                "node_ids": node_ids,
                "atom_ids": atom_ids,
                "supported_by": atom_ids,
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    return result


def _support_from_atom_ids(atom_ids: list[str], atom_by_id: dict[str, AtomicEvidence]) -> dict[str, Any]:
    atom_ids = _unique_preserve(atom_ids)
    known_atoms = [atom_by_id[atom_id] for atom_id in atom_ids if atom_id in atom_by_id]
    first_atom = known_atoms[0] if known_atoms else None
    return {
        "path_set_id": first_atom.path_set_id if first_atom is not None else "",
        "path_id": first_atom.source_path_id if first_atom is not None else "",
        "node_texts": _unique_preserve(
            text
            for atom in known_atoms
            for text in atom.node_texts
        ),
        "node_ids": _unique_preserve(
            node_id
            for atom in known_atoms
            for node_id in atom.node_ids
        ),
        "atom_ids": atom_ids,
        "supported_by": atom_ids,
        "reason": "",
    }


def _unique_preserve(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _matching_atom_ids_for_support(
    evidence_atoms: list[AtomicEvidence],
    *,
    path_set_id: str,
    path_id: str,
    node_texts: list[str],
) -> list[str]:
    if not path_id and not node_texts:
        return []
    normalized_support_texts = {_normalize_support_text(text) for text in node_texts}
    normalized_support_texts.discard("")
    matches: list[str] = []
    for atom in evidence_atoms:
        if path_set_id and atom.path_set_id != path_set_id:
            continue
        if path_id and atom.source_path_id != path_id:
            continue
        atom_texts = {_normalize_support_text(text) for text in atom.node_texts}
        atom_texts.discard("")
        if normalized_support_texts and normalized_support_texts.issubset(atom_texts):
            matches.append(atom.id)
    if matches:
        return matches[:3]
    for atom in evidence_atoms:
        if path_id and atom.source_path_id != path_id:
            continue
        atom_texts = {_normalize_support_text(text) for text in atom.node_texts}
        if normalized_support_texts & atom_texts:
            matches.append(atom.id)
    return matches[:3]


def _semantic_reasoning_path_support_issues(
    result: SemanticReasoningPathResult,
    selected_dependency_path_evidence: list[dict[str, Any]],
    evidence_atoms: list[AtomicEvidence],
) -> list[str]:
    """Validate that semantic edges are grounded by evidence atom ids."""
    support_index = _selected_dependency_support_index(selected_dependency_path_evidence)
    atom_ids = {atom.id for atom in evidence_atoms}
    issues: list[str] = []
    for path in result.paths:
        for edge in path.edges:
            edge_atom_ids: set[str] = set()
            for support_index_in_edge, support in enumerate(edge.support, start=1):
                support_atom_ids = set(_str_list(support.get("atom_ids") or support.get("supported_by")))
                if not support_atom_ids:
                    issues.append(f"Semantic edge {edge.edge_id} support #{support_index_in_edge} has empty supported_by.")
                    continue
                unknown = sorted(support_atom_ids - atom_ids)
                if unknown:
                    issues.append(
                        f"Semantic edge {edge.edge_id} support #{support_index_in_edge} cites unknown evidence atoms {unknown}."
                    )
                    continue
                edge_atom_ids.update(support_atom_ids)
                path_set_id = str(support.get("path_set_id") or "").strip()
                path_id = str(support.get("path_id") or "").strip()
                if path_set_id or path_id:
                    key = (path_set_id, path_id)
                    if key not in support_index:
                        issues.append(
                            f"Semantic edge {edge.edge_id} support #{support_index_in_edge} cites invalid "
                            f"path_set_id/path_id {path_set_id!r}/{path_id!r}."
                        )
            if not edge_atom_ids:
                issues.append(f"Semantic edge {edge.edge_id} has no valid evidence atom support.")
    return issues


def _semantic_edge_source_errors(payload: dict[str, Any], semantic_payload: dict[str, Any]) -> list[str]:
    expected_edge_ids = {
        str(edge.get("edge_id") or "").strip()
        for path in semantic_payload.get("paths", semantic_payload.get("semantic_reasoning_paths", []))
        if isinstance(path, dict)
        for edge in path.get("edges", [])
        if isinstance(edge, dict)
    }
    expected_edge_ids.discard("")
    if not expected_edge_ids:
        return []
    raw_nodes = payload.get("nodes")
    if raw_nodes is None:
        raw_nodes = payload.get("atomic_questions") or payload.get("subquestions")
    if not isinstance(raw_nodes, list):
        return []
    errors: list[str] = []
    used_edge_ids: set[str] = set()
    for index, node in enumerate(raw_nodes, start=1):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or node.get("id") or f"q{index}").strip()
        edge_id = str(node.get("source_semantic_edge_id") or "").strip()
        if not edge_id:
            errors.append(f"Atomic node {node_id} is missing source_semantic_edge_id.")
            continue
        if edge_id not in expected_edge_ids:
            errors.append(f"Atomic node {node_id} references unknown source_semantic_edge_id={edge_id!r}.")
            continue
        used_edge_ids.add(edge_id)
    missing = sorted(expected_edge_ids - used_edge_ids)
    if missing:
        errors.append(
            f"Atomic DAG skipped semantic reasoning edges: {missing}. "
            "Every semantic edge must be covered by at least one atomic DAG node."
        )
    return errors


def _atomic_dag_required_semantic_fields_errors(payload: dict[str, Any]) -> list[str]:
    raw_nodes = payload.get("nodes")
    if raw_nodes is None:
        raw_nodes = payload.get("atomic_questions") or payload.get("subquestions")
    if not isinstance(raw_nodes, list):
        return []
    errors: list[str] = []
    required_fields = (
        "node_id",
        "question",
        "operation",
        "one_hop_relation",
        "answer_type",
        "dependencies",
        "source_semantic_path_id",
        "source_semantic_edge_id",
    )
    for index, node in enumerate(raw_nodes, start=1):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or node.get("id") or f"q{index}").strip()
        missing = [
            field
            for field in required_fields
            if field not in node or node.get(field) in (None, "")
        ]
        if missing:
            errors.append(f"Atomic node {node_id} is missing required semantic-DAG fields: {missing}.")
        if str(node.get("operation") or "").strip() != "lookup":
            errors.append(f"Atomic node {node_id} must have operation='lookup'.")
    return errors


def _selected_dependency_path_ids(selected_dependency_path_evidence: list[dict[str, Any]]) -> set[str]:
    return {
        str(path.get("path_id") or "").strip()
        for path_set in selected_dependency_path_evidence
        if isinstance(path_set, dict)
        for path in path_set.get("paths", [])
        if isinstance(path, dict) and str(path.get("path_id") or "").strip()
    }


def _selected_dependency_path_set_ids(selected_dependency_path_evidence: list[dict[str, Any]]) -> set[str]:
    return {
        str(path_set.get("path_set_id") or "").strip()
        for path_set in selected_dependency_path_evidence
        if isinstance(path_set, dict) and str(path_set.get("path_set_id") or "").strip()
    }


def _forbidden_semantic_node_label(label: str) -> bool:
    normalized = re.sub(r"[^a-z0-9/ ]+", " ", str(label or "").lower())
    normalized = " ".join(normalized.split())
    return normalized in FORBIDDEN_SEMANTIC_NODE_LABELS


def _forbidden_semantic_relation(relation: str) -> bool:
    normalized = re.sub(r"[^a-z0-9/ ]+", " ", str(relation or "").lower())
    normalized = " ".join(normalized.split())
    return normalized in FORBIDDEN_SEMANTIC_RELATIONS


def select_best_path_by_entity(
    *,
    scored_paths: list[ScoredEntityPath],
    entity_start_nodes: list[EntityStartNode],
    entity_origin_paths: list[EntityOriginPath],
    min_valid_score: float = 55.0,
) -> dict[str, ScoredEntityPath]:
    """Select exactly one highest-scoring path for each explicit entity."""
    path_by_id = {path.path_id: path for path in entity_origin_paths}
    score_by_path_id = {score.path_id: score for score in scored_paths if score.path_id in path_by_id}
    result: dict[str, ScoredEntityPath] = {}
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
        eligible = [
            item
            for item in ordered
            if item.valid and item.score >= min_valid_score
        ]
        selected = eligible[0] if eligible else ordered[0]
        result[entity.entity_id] = selected
        if not result[entity.entity_id].path_id:
            raise ValueError(f"No top path could be selected for entity_id={entity.entity_id!r}.")
    return result


def build_single_path_set_candidate(
    *,
    best_paths_by_entity: dict[str, ScoredEntityPath],
) -> list[PathSetCandidate]:
    if not best_paths_by_entity:
        return []
    path_ids_by_entity: dict[str, str] = {}
    scores: list[float] = []
    for entity_id in sorted(best_paths_by_entity, key=_entity_id_sort_key):
        scored_path = best_paths_by_entity[entity_id]
        if not scored_path.path_id:
            raise ValueError(f"Entity {entity_id!r} has no selected best path.")
        path_ids_by_entity[entity_id] = scored_path.path_id
        scores.append(scored_path.score)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return [
        PathSetCandidate(
            path_set_id="ps1",
            path_ids_by_entity=path_ids_by_entity,
            mean_path_score=mean_score,
        )
    ]


def build_path_set_candidates(
    *,
    paths_by_entity: dict[str, list[ScoredEntityPath]],
    max_path_sets: int | None = None,
) -> list[PathSetCandidate]:
    if not paths_by_entity:
        return []
    entity_ids = sorted(paths_by_entity, key=_entity_id_sort_key)
    for entity_id in entity_ids:
        if not paths_by_entity[entity_id]:
            raise ValueError(f"Entity {entity_id!r} has no selected paths.")

    raw_candidates: list[tuple[dict[str, str], float]] = []
    for combo in product(*(paths_by_entity[entity_id] for entity_id in entity_ids)):
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


def extract_atomic_evidences(selected_dependency_path_evidence: list[dict[str, Any]]) -> list[AtomicEvidence]:
    """Extract adjacent local path-edge evidences from selected dependency paths.

    Atomic evidences are not heuristic entity-predicate combinations. Each atom
    corresponds to one adjacent edge in a selected path, so Step 9 can ground
    each semantic edge in a concrete local dependency fragment without seeing
    or copying the full dependency path.
    """

    atoms: list[AtomicEvidence] = []
    seen: set[tuple[Any, ...]] = set()
    for path_set in selected_dependency_path_evidence:
        if not isinstance(path_set, dict):
            continue
        path_set_id = str(path_set.get("path_set_id") or "").strip()
        for path in path_set.get("paths", []) or []:
            if not isinstance(path, dict):
                continue
            node_texts = _str_list(path.get("node_texts"))
            if len(node_texts) < 2:
                continue
            node_ids = _str_list(path.get("node_ids"))
            source_path_id = str(path.get("path_id") or "").strip()
            entity_id = str(path.get("entity_id") or "").strip()
            entity_text = str(path.get("entity_text") or "").strip()

            for position, left in enumerate(node_texts[:-1]):
                right = node_texts[position + 1]
                left_node_id = node_ids[position] if position < len(node_ids) else ""
                right_node_id = node_ids[position + 1] if position + 1 < len(node_ids) else ""
                atom = AtomicEvidence(
                    id="",
                    kind="path_edge",
                    text=f"{left} ---- {right}",
                    left=left,
                    right=right,
                    source_path_id=source_path_id,
                    source_path_set_id=path_set_id,
                    metadata={
                        "entity_id": entity_id,
                        "entity_text": entity_text,
                        "node_texts": [left, right],
                        "node_ids": [node_id for node_id in [left_node_id, right_node_id] if node_id],
                        "position": position,
                    },
                )
                key = _atomic_evidence_key(atom)
                if key in seen:
                    continue
                seen.add(key)
                atom.id = f"atom_{len(atoms) + 1}"
                atoms.append(atom)
    return atoms


def _atomic_evidence_key(atom: AtomicEvidence) -> tuple[Any, ...]:
    return (
        atom.kind,
        _normalize_support_text(atom.text),
        atom.source_path_set_id or "",
        atom.source_path_id or "",
        tuple(atom.node_ids),
        atom.metadata.get("position"),
    )


def _atomic_question_variable_placeholder_errors(payload: dict[str, Any]) -> list[str]:
    raw_nodes = payload.get("nodes")
    if raw_nodes is None:
        raw_nodes = payload.get("atomic_questions") or payload.get("subquestions")
    if not isinstance(raw_nodes, list):
        return []

    errors: list[str] = []
    for index, raw_node in enumerate(raw_nodes, start=1):
        if not isinstance(raw_node, dict):
            continue
        node_id = str(raw_node.get("node_id") or raw_node.get("id") or f"q{index}").strip()
        question = str(raw_node.get("question") or raw_node.get("subquestion") or raw_node.get("sub_question") or "").strip()
        if _contains_atomic_variable_placeholder(question):
            errors.append(
                f"Atomic node {node_id} question contains a forbidden dependency placeholder. "
                "Step 10 questions must be self-contained and retrieval-friendly."
            )
    return errors


def _contains_atomic_variable_placeholder(question: str) -> bool:
    text = str(question or "")
    patterns = (
        r"\{\s*q\d+\.answer\s*\}",
        r"\bq\d+\s*['\u2019]s\s+answer\b",
        r"\bq\d+\s+answer\b",
        r"\b(?:the\s+)?answer\s+(?:of|to)\s+q\d+\b",
        r"\b(?:the\s+)?result\s+of\s+q\d+\b",
        r"\bprevious\s+(?:atomic\s+)?answer\b",
        r"\bthe\s+previous\s+answer\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


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


def _parse_grounded_atomic_dag_payload(
    payload: dict[str, Any],
    *,
    semantic_payload: dict[str, Any],
) -> tuple[AtomicQuestionDAG, list[str]]:
    raw_nodes = payload.get("nodes")
    if raw_nodes is None:
        raw_nodes = payload.get("atomic_questions") or payload.get("subquestions")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Grounded Atomic DAG payload must contain a non-empty nodes list.")

    semantic_edge_index = _semantic_edge_index(semantic_payload)
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
        support = _normalize_semantic_grounded_support(
            raw_node.get("support"),
            raw_node=raw_node,
            semantic_edge_index=semantic_edge_index,
            node_id=node_id,
            warnings=warnings,
        )
        metadata: dict[str, Any] = {
            "source": "grounded_atomic_dag",
            "support": support,
        }
        support_path_ids = sorted({item["path_id"] for item in support if item.get("path_id")})
        if support_path_ids:
            metadata["support_path_ids"] = support_path_ids
        for metadata_key in ("operation", "input", "one_hop_relation", "answer_type"):
            if metadata_key in raw_node:
                metadata[metadata_key] = raw_node.get(metadata_key)
        for metadata_key in ("source_semantic_path_id", "source_semantic_edge_id"):
            if metadata_key in raw_node:
                metadata[metadata_key] = raw_node.get(metadata_key)
        if "source_semantic_edge_id" in metadata:
            metadata["semantic_reasoning_path_source"] = True
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


def _normalize_semantic_grounded_support(
    raw: Any,
    *,
    raw_node: dict[str, Any],
    semantic_edge_index: dict[str, dict[str, Any]],
    node_id: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    edge_id = str(raw_node.get("source_semantic_edge_id") or "").strip()
    edge_payload = semantic_edge_index.get(edge_id)
    semantic_path_id = str(raw_node.get("source_semantic_path_id") or "").strip()
    if edge_payload:
        semantic_path_id = semantic_path_id or str(edge_payload.get("semantic_path_id") or "").strip()
        if not semantic_path_id:
            semantic_path_id = str(edge_payload.get("branch_id") or "").strip()

    raw_items: list[Any]
    if isinstance(raw, dict):
        raw_items = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raw_items = []

    support: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        item_edge_id = str(item.get("semantic_edge_id") or item.get("source_semantic_edge_id") or "").strip()
        item_path_id = str(item.get("semantic_path_id") or item.get("source_semantic_path_id") or "").strip()
        if item_edge_id and item_edge_id != edge_id:
            warnings.append(
                f"Ignored semantic support for {node_id}: semantic_edge_id {item_edge_id!r} "
                f"does not match source_semantic_edge_id {edge_id!r}."
            )
            continue
        normalized_item = {
            "semantic_path_id": item_path_id or semantic_path_id,
            "semantic_edge_id": item_edge_id or edge_id,
        }
        reason = str(item.get("reason") or "").strip()
        if reason:
            normalized_item["reason"] = reason
        support.append(normalized_item)

    if support:
        return support

    if edge_payload:
        return [
            {
                "semantic_path_id": semantic_path_id,
                "semantic_edge_id": edge_id,
            }
        ]

    warnings.append(f"Node {node_id} has no semantic edge support because source_semantic_edge_id={edge_id!r} is unknown.")
    return []


def _semantic_edge_index(semantic_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(semantic_payload, dict):
        return {}
    raw_paths = semantic_payload.get("paths")
    if raw_paths is None:
        raw_paths = semantic_payload.get("semantic_reasoning_paths")
    if not isinstance(raw_paths, list):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for path in raw_paths:
        if not isinstance(path, dict):
            continue
        branch_id = str(path.get("branch_id") or path.get("path_id") or "").strip()
        raw_edges = path.get("edges")
        if not isinstance(raw_edges, list):
            continue
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            edge_id = str(edge.get("edge_id") or edge.get("id") or "").strip()
            if not edge_id:
                continue
            payload = dict(edge)
            payload["semantic_path_id"] = branch_id
            payload["branch_id"] = branch_id
            index[edge_id] = payload
    return index


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
