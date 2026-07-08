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
    semantic_edge_ids: tuple[str, ...] = ()
    output_node_id: str = ""
    output_type: str = ""
    support_step_ids: tuple[str, ...] = ()
    support: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "depends_on": list(self.depends_on),
            "operation": self.operation,
            "semantic_edge_ids": list(self.semantic_edge_ids),
            "output_node_id": self.output_node_id,
            "output_type": self.output_type,
            "support_step_ids": list(self.support_step_ids),
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
    """Run Step5 as original question + topic entities + Step4 path hints -> atomic question DAG."""

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
    del global_best_paths
    if not isinstance(raw_payload, dict):
        return _invalid_result(["raw_payload must be an object."], raw_payload=None)

    raw_questions = _extract_atomic_questions(raw_payload)
    if not isinstance(raw_questions, list):
        return _invalid_result(["atomic_questions must be a non-empty list."], raw_payload=raw_payload)
    if not raw_questions:
        return _invalid_result(["atomic_questions must be a non-empty list."], raw_payload=raw_payload)

    parsed_nodes = _coerce_atomic_question_nodes(raw_questions)
    if not parsed_nodes:
        return _invalid_result(["atomic_questions did not contain any parseable question objects."], raw_payload=raw_payload)

    edges = _edges_from_nodes(parsed_nodes)
    leaf_node_ids = _leaf_node_ids(parsed_nodes, edges)
    warnings = _atomic_question_dag_quality_warnings(
        parsed_nodes,
        original_question=original_question,
        explicit_entities=explicit_entities,
    )
    return AtomicQuestionDAGResult(
        nodes=parsed_nodes,
        edges=edges,
        leaf_node_ids=leaf_node_ids,
        valid=True,
        validation_errors=[],
        raw_payload=raw_payload,
        warnings=warnings,
    )


def _extract_atomic_questions(raw_payload: dict[str, Any]) -> Any:
    """Read Step5 questions from the direct DAG envelope, with legacy fallback."""

    if isinstance(raw_payload.get("atomic_questions"), list):
        return raw_payload.get("atomic_questions")

    raw_dag = raw_payload.get("atomic_question_dag")
    if isinstance(raw_dag, dict):
        if isinstance(raw_dag.get("atomic_questions"), list):
            return raw_dag.get("atomic_questions")
        if isinstance(raw_dag.get("nodes"), list):
            return raw_dag.get("nodes")
    if isinstance(raw_dag, list):
        return raw_dag
    return None


def _coerce_atomic_question_nodes(raw_questions: Any) -> list[AtomicQuestionNode]:
    """Normalize Step5 DAG nodes without enforcing semantic path support."""

    if not isinstance(raw_questions, list):
        return []

    nodes: list[AtomicQuestionNode] = []
    for index, raw_question in enumerate(raw_questions, start=1):
        if not isinstance(raw_question, dict):
            continue
        node_id = str(raw_question.get("id") or f"q{index}").strip() or f"q{index}"
        question = str(raw_question.get("question") or "").strip()
        nodes.append(
            AtomicQuestionNode(
                id=node_id,
                question=question,
                depends_on=_coerce_string_tuple(raw_question.get("depends_on")),
                operation=str(raw_question.get("operation") or "lookup").strip() or "lookup",
                semantic_edge_ids=_coerce_string_tuple(raw_question.get("semantic_edge_ids")),
                output_node_id=str(raw_question.get("output_node_id") or "").strip(),
                output_type=str(raw_question.get("output_type") or "unknown").strip() or "unknown",
                support_step_ids=_coerce_string_tuple(raw_question.get("support_step_ids")),
            )
        )
    return nodes


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(text for item in value if (text := str(item).strip()))


def _atomic_question_dag_quality_warnings(
    nodes: list[AtomicQuestionNode],
    *,
    original_question: str | None,
    explicit_entities: list[str] | None,
) -> list[str]:
    warnings: list[str] = []
    question_text = original_question or ""
    question_lc = question_text.casefold()

    warnings.extend(_dependency_binding_warnings(nodes))
    warnings.extend(_placeholder_warnings(nodes))
    warnings.extend(_possessive_wh_warnings(nodes, question_lc))
    warnings.extend(_lifespan_comparison_warnings(nodes, question_lc))
    warnings.extend(_appositive_identity_warnings(nodes, question_text, explicit_entities or []))

    return _dedupe_preserve_order(warnings)


def _dependency_binding_warnings(nodes: list[AtomicQuestionNode]) -> list[str]:
    warnings: list[str] = []
    previous_ids: set[str] = set()
    node_ids = {node.id for node in nodes}
    for node in nodes:
        refs = set(_question_refs([node.question]))
        deps = set(node.depends_on)
        for dependency in node.depends_on:
            if dependency not in node_ids:
                warnings.append(f"{node.id}: depends_on references unknown node {dependency!r}.")
            elif dependency not in previous_ids:
                warnings.append(f"{node.id}: depends_on references non-previous node {dependency!r}.")
            elif dependency not in refs:
                warnings.append(f"{node.id}: depends_on includes {dependency!r}, but the question text does not mention {dependency}'s answer.")
        for ref in refs:
            if ref == node.id:
                warnings.append(f"{node.id}: question refers to its own answer.")
            elif ref not in previous_ids:
                warnings.append(f"{node.id}: question mentions {ref}'s answer before that answer exists.")
            elif ref not in deps:
                warnings.append(f"{node.id}: question mentions {ref}'s answer but depends_on does not include {ref!r}.")
        previous_ids.add(node.id)
    return warnings


def _placeholder_warnings(nodes: list[AtomicQuestionNode]) -> list[str]:
    warnings: list[str] = []
    for node in nodes:
        if _contains_placeholder(node.question):
            warnings.append(f"{node.id}: question contains unresolved ENTITY placeholder.")
    return warnings


def _possessive_wh_warnings(nodes: list[AtomicQuestionNode], original_question_lc: str) -> list[str]:
    if "whose " not in original_question_lc:
        return []
    kinship_terms = (
        "sister",
        "brother",
        "father",
        "mother",
        "child",
        "son",
        "daughter",
        "spouse",
        "wife",
        "husband",
        "parent",
        "grandfather",
        "grandmother",
    )
    pattern = re.compile(
        r"\bwho\s+is\s+(?:the\s+)?(?:" + "|".join(re.escape(term) for term in kinship_terms) + r")\s+of\s+q[1-9][0-9]*(?:'|\u2019)s\s+answer\b",
        re.IGNORECASE,
    )
    return [
        f"{node.id}: possessive-WH question likely reversed; ask whose relation qN's answer has instead."
        for node in nodes
        if pattern.search(node.question)
    ]


def _lifespan_comparison_warnings(nodes: list[AtomicQuestionNode], original_question_lc: str) -> list[str]:
    if not re.search(r"\bliv(?:e|ed|es|ing)\s+longer\b", original_question_lc):
        return []
    questions = [node.question.casefold() for node in nodes]
    birth_count = sum(1 for question in questions if re.search(r"\b(born|birth date|date of birth)\b", question))
    death_count = sum(1 for question in questions if re.search(r"\b(die|died|death date|date of death)\b", question))
    lifespan_count = sum(1 for question in questions if "how long" in question and re.search(r"\bliv(?:e|ed)\b", question))
    if (birth_count >= 2 and death_count >= 2) or lifespan_count >= 2:
        return []
    return ["lived-longer comparison needs lifespan evidence for each branch, not only birth dates."]


def _appositive_identity_warnings(
    nodes: list[AtomicQuestionNode],
    original_question: str,
    explicit_entities: list[str],
) -> list[str]:
    del explicit_entities
    warnings: list[str] = []
    node_text = " ".join(node.question for node in nodes)
    for mention, base in _disambiguated_mentions(original_question):
        if mention in node_text:
            continue
        if base and base in node_text:
            warnings.append(f"Entity mention {mention!r} appears to be shortened to {base!r}; preserve disambiguating parenthetical/appositive text.")
    return warnings


def _disambiguated_mentions(question: str) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    token = r"[A-Z][A-Za-z0-9.'’-]*"
    name = rf"{token}(?:\s+{token}){{0,5}}"

    for match in re.finditer(rf"\b({name}\s*\([^()]+\))", question):
        mention = match.group(1).strip()
        base = re.sub(r"\s*\([^()]+\)\s*$", "", mention).strip()
        if base and mention:
            mentions.append((mention, base))

    appositive_titles = (
        "Duke",
        "Duchess",
        "Count",
        "Countess",
        "Prince",
        "Princess",
        "King",
        "Queen",
        "Emperor",
        "Empress",
        "Baron",
        "Baronet",
        "Lord",
        "Lady",
        "Earl",
        "Bishop",
        "Pope",
        "Saint",
    )
    title_pattern = "|".join(re.escape(title) for title in appositive_titles)
    for match in re.finditer(rf"\b({name}),\s+((?:{title_pattern})\b[^?]*)", question):
        base = match.group(1).strip()
        tail = match.group(2).strip()
        tail = re.split(r"\s+(?:or|and)\s+", tail, maxsplit=1)[0].strip()
        tail = tail.rstrip(" ?")
        mention = f"{base}, {tail}".strip()
        if base and tail:
            mentions.append((mention, base))
    return _dedupe_preserve_order(mentions)


def _dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    deduped: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


OUTPUT_TYPES = {"entity", "person", "place", "organization", "work", "event", "date", "number", "boolean", "value", "set", "unknown"}
SEMANTIC_NODE_KINDS = {"entity", "intermediate_variable", "value_slot", "constraint", "operator", "answer_slot"}
SEMANTIC_NODE_ORIGINS = {"explicit_entity", "path_evidence", "question_required", "derived_variable", "operator"}
SEMANTIC_EDGE_TYPES = {"lookup", "constraint", "compare", "select", "verify", "intersect", "aggregate"}
SEMANTIC_EDGE_EVIDENCE_STATUSES = {"path_grounded", "question_required", "mixed", "operator"}
FOLDED_TOKEN_REASONS = {"folded_into_relation", "grammatical", "wh_answer_intent", "noisy_parser_artifact", "duplicate"}
ATOMIC_OPERATIONS = {"lookup", "compare", "select", "verify", "intersect", "aggregate"}
VAGUE_OPERATION_LABELS = ("related to", "associated with", "connected to", "about", "path to")
WH_NODE_LABELS = {"who", "what", "which", "where", "when", "why", "whom", "whose"}
GRAMMATICAL_NODE_LABELS = {
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "did",
    "does",
    "have",
    "has",
    "had",
    "the",
    "a",
    "an",
    "of",
    "in",
    "by",
    "from",
    "to",
    "with",
    "for",
    "at",
    "and",
    "or",
}


def _validate_semantic_reasoning_paths(
    raw_paths: list[Any],
    errors: list[str],
    warnings: list[str],
    *,
    original_question: str | None,
    global_best_paths: list[list[str]] | None,
) -> tuple[set[str], set[str], dict[str, str], dict[str, str]]:
    all_node_ids: set[str] = set()
    all_edge_ids: set[str] = set()
    edge_targets: dict[str, str] = {}
    edge_types: dict[str, str] = {}

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

        if "reasoning_steps" in raw_path:
            errors.append(f"{prefix}: Step5 V3 uses semantic_nodes/semantic_edges, not reasoning_steps.")

        raw_nodes = raw_path.get("semantic_nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            errors.append(f"{prefix}.semantic_nodes must be a non-empty list.")
            raw_nodes = []
        raw_edges = raw_path.get("semantic_edges")
        if not isinstance(raw_edges, list):
            errors.append(f"{prefix}.semantic_edges must be a list.")
            raw_edges = []

        local_node_ids: set[str] = set()
        node_labels: dict[str, str] = {}
        for node_index, raw_node in enumerate(raw_nodes, start=1):
            node_prefix = f"{prefix}.semantic_nodes[{node_index - 1}]"
            if not isinstance(raw_node, dict):
                errors.append(f"{node_prefix} must be an object.")
                continue

            expected_node_id = f"{expected_branch_id}_n{node_index}"
            node_id = str(raw_node.get("id") or "").strip()
            if node_id != expected_node_id:
                errors.append(f"{node_prefix}.id must be {expected_node_id!r}, got {node_id!r}.")
            elif node_id in all_node_ids:
                errors.append(f"duplicate semantic node id: {node_id}.")
            if node_id:
                local_node_ids.add(node_id)
                all_node_ids.add(node_id)

            label = str(raw_node.get("label") or "").strip()
            if not label:
                errors.append(f"{node_prefix}.label must be a non-empty string.")
            elif _contains_placeholder(label):
                errors.append(f"{node_prefix}.label contains unresolved ENTITY placeholder.")
            else:
                node_labels[node_id] = label

            kind = str(raw_node.get("kind") or "").strip()
            if kind not in SEMANTIC_NODE_KINDS:
                errors.append(f"{node_prefix}.kind must be one of {sorted(SEMANTIC_NODE_KINDS)}, got {kind!r}.")

            output_type = str(raw_node.get("output_type") or "").strip()
            if output_type not in OUTPUT_TYPES:
                errors.append(f"{node_prefix}.output_type must be one of {sorted(OUTPUT_TYPES)}, got {output_type!r}.")

            origin = str(raw_node.get("origin") or "").strip()
            if origin not in SEMANTIC_NODE_ORIGINS:
                errors.append(f"{node_prefix}.origin must be one of {sorted(SEMANTIC_NODE_ORIGINS)}, got {origin!r}.")

            path_evidence = _string_list(raw_node.get("path_evidence"), f"{node_prefix}.path_evidence", errors)
            for token in path_evidence:
                if _contains_placeholder(token):
                    errors.append(f"{node_prefix}.path_evidence contains unresolved ENTITY placeholder.")
                    break
                if token not in source_token_set:
                    warnings.append(
                        f"{node_prefix}.path_evidence contains evidence not copied from source_token_path: {token!r}."
                    )
            if not path_evidence and origin not in {"question_required", "operator"}:
                warnings.append(
                    f"{node_prefix}.path_evidence is empty for origin {origin!r}; treating evidence grounding as underspecified."
                )

            question_evidence = _string_list(raw_node.get("question_evidence"), f"{node_prefix}.question_evidence", errors)
            for evidence in question_evidence:
                if _contains_placeholder(evidence):
                    errors.append(f"{node_prefix}.question_evidence contains unresolved ENTITY placeholder.")
                    break

            _validate_semantic_node_label(
                node_prefix,
                label,
                kind,
                question_evidence,
                errors,
            )

        terminal_node_id = str(raw_path.get("terminal_node_id") or "").strip()
        if terminal_node_id not in local_node_ids:
            errors.append(f"{prefix}.terminal_node_id must refer to a semantic node in the same path.")

        for edge_index, raw_edge in enumerate(raw_edges, start=1):
            edge_prefix = f"{prefix}.semantic_edges[{edge_index - 1}]"
            if not isinstance(raw_edge, dict):
                errors.append(f"{edge_prefix} must be an object.")
                continue

            expected_edge_id = f"{expected_branch_id}_e{edge_index}"
            edge_id = str(raw_edge.get("id") or "").strip()
            if edge_id != expected_edge_id:
                errors.append(f"{edge_prefix}.id must be {expected_edge_id!r}, got {edge_id!r}.")
            elif edge_id in all_edge_ids:
                errors.append(f"duplicate semantic edge id: {edge_id}.")
            if edge_id:
                all_edge_ids.add(edge_id)

            source = str(raw_edge.get("source") or "").strip()
            target = str(raw_edge.get("target") or "").strip()
            if source not in local_node_ids:
                errors.append(f"{edge_prefix}.source must refer to a semantic node in the same path.")
            if target not in local_node_ids:
                errors.append(f"{edge_prefix}.target must refer to a semantic node in the same path.")
            if edge_id:
                edge_targets[edge_id] = target

            condition_node_ids = _string_list(raw_edge.get("condition_node_ids"), f"{edge_prefix}.condition_node_ids", errors)
            for condition_node_id in condition_node_ids:
                if condition_node_id not in local_node_ids:
                    errors.append(f"{edge_prefix}.condition_node_ids contains unknown semantic node id {condition_node_id!r}.")

            relation = str(raw_edge.get("relation") or "").strip()
            if not relation:
                errors.append(f"{edge_prefix}.relation must be a non-empty string.")
            elif _contains_placeholder(relation):
                errors.append(f"{edge_prefix}.relation contains unresolved ENTITY placeholder.")
            elif _has_vague_operation(relation, original_question):
                errors.append(f"{edge_prefix}.relation is too vague for Step5 V3: {relation!r}.")

            edge_type = str(raw_edge.get("edge_type") or "").strip()
            if edge_type not in SEMANTIC_EDGE_TYPES:
                errors.append(f"{edge_prefix}.edge_type must be one of {sorted(SEMANTIC_EDGE_TYPES)}, got {edge_type!r}.")
            if edge_id:
                edge_types[edge_id] = edge_type

            evidence_status = str(raw_edge.get("evidence_status") or "").strip()
            if evidence_status not in SEMANTIC_EDGE_EVIDENCE_STATUSES:
                errors.append(
                    f"{edge_prefix}.evidence_status must be one of {sorted(SEMANTIC_EDGE_EVIDENCE_STATUSES)}, got {evidence_status!r}."
                )

            support_tokens = _string_list(raw_edge.get("support_tokens"), f"{edge_prefix}.support_tokens", errors)
            for token in support_tokens:
                if _contains_placeholder(token):
                    errors.append(f"{edge_prefix}.support_tokens contains unresolved ENTITY placeholder.")
                    break
                if token not in source_token_set:
                    warnings.append(
                        f"{edge_prefix}.support_tokens contains evidence not copied from source_token_path: {token!r}."
                    )
            if not support_tokens and evidence_status not in {"question_required", "operator"}:
                warnings.append(
                    f"{edge_prefix}.support_tokens is empty for evidence_status {evidence_status!r}; treating evidence grounding as underspecified."
                )

            question_evidence = _string_list(raw_edge.get("question_evidence"), f"{edge_prefix}.question_evidence", errors)
            for evidence in question_evidence:
                if _contains_placeholder(evidence):
                    errors.append(f"{edge_prefix}.question_evidence contains unresolved ENTITY placeholder.")
                    break

            atomic_question_hint = str(raw_edge.get("atomic_question_hint") or "").strip()
            if edge_type == "lookup" and not _is_single_question(atomic_question_hint):
                errors.append(f"{edge_prefix}.atomic_question_hint must be a non-empty single question for lookup edges.")
            elif _contains_placeholder(atomic_question_hint):
                errors.append(f"{edge_prefix}.atomic_question_hint contains unresolved ENTITY placeholder.")

            _validate_adjacent_token_edge_relabeling(
                edge_prefix,
                source,
                target,
                node_labels,
                source_token_path,
                relation,
                errors,
            )

        folded_or_discarded_tokens = raw_path.get("folded_or_discarded_tokens")
        if not isinstance(folded_or_discarded_tokens, list):
            errors.append(f"{prefix}.folded_or_discarded_tokens must be a list.")
            folded_or_discarded_tokens = []
        for folded_index, raw_folded in enumerate(folded_or_discarded_tokens):
            folded_prefix = f"{prefix}.folded_or_discarded_tokens[{folded_index}]"
            if not isinstance(raw_folded, dict):
                errors.append(f"{folded_prefix} must be an object.")
                continue
            token = str(raw_folded.get("token") or "").strip()
            reason = str(raw_folded.get("reason") or "").strip()
            if token not in source_token_set:
                errors.append(f"{folded_prefix}.token must be copied from source_token_path: {token!r}.")
            if _contains_placeholder(token):
                errors.append(f"{folded_prefix}.token contains unresolved ENTITY placeholder.")
            if reason not in FOLDED_TOKEN_REASONS:
                errors.append(f"{folded_prefix}.reason must be one of {sorted(FOLDED_TOKEN_REASONS)}, got {reason!r}.")

        _check_likely_token_path_relabeling(prefix, source_token_path, list(node_labels.values()), errors, warnings)

    return all_edge_ids, all_node_ids, edge_targets, edge_types


def _parse_atomic_questions(
    raw_questions: list[Any],
    *,
    semantic_edge_ids: set[str],
    semantic_node_ids: set[str],
    semantic_edge_targets: dict[str, str],
    semantic_edge_types: dict[str, str],
    final_answer_type: str,
    final_answer_intent: str,
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
        if _looks_like_possessive_wh_reversal(question, final_answer_intent):
            errors.append(f"{node_id or expected_id}: question reverses possessive-WH semantics.")

        depends_on = _coerce_depends_on(raw_question.get("depends_on"), node_id or expected_id, previous_ids, errors)
        refs = _question_refs([question])
        for ref in refs:
            if ref not in previous_ids:
                errors.append(f"{node_id or expected_id}: question references non-previous node {ref!r}.")
            elif ref not in depends_on:
                errors.append(f"{node_id or expected_id}: question mentions {ref}'s answer but depends_on does not include {ref!r}.")
        missing_refs = [dependency for dependency in depends_on if not _question_mentions_dependency(question, dependency)]
        if missing_refs:
            warnings.append(
                f"{node_id or expected_id}: depends_on entries are not mentioned as qN's answer in question: {missing_refs}."
            )

        operation = str(raw_question.get("operation") or "").strip()
        if operation not in ATOMIC_OPERATIONS:
            errors.append(f"{node_id or expected_id}: operation must be one of {sorted(ATOMIC_OPERATIONS)}, got {operation!r}.")

        if "support_step_ids" in raw_question and "semantic_edge_ids" not in raw_question:
            errors.append(f"{node_id or expected_id}: Step5 V3 uses semantic_edge_ids, not support_step_ids.")

        output_type = str(raw_question.get("output_type") or "").strip()
        if not output_type:
            errors.append(f"{node_id or expected_id}: output_type must be a non-empty string.")
        elif output_type not in OUTPUT_TYPES:
            errors.append(f"{node_id or expected_id}: output_type must be one of {sorted(OUTPUT_TYPES)}, got {output_type!r}.")

        raw_semantic_edge_ids = _string_list(
            raw_question.get("semantic_edge_ids"),
            f"{prefix}.semantic_edge_ids",
            errors,
        )
        if operation == "lookup" and not raw_semantic_edge_ids:
            errors.append(f"{node_id or expected_id}: lookup questions must include at least one semantic_edge_id.")
        if operation != "lookup" and not raw_semantic_edge_ids and not depends_on:
            errors.append(f"{node_id or expected_id}: operator questions with semantic_edge_ids=[] must combine previous answers.")
        for edge_id in raw_semantic_edge_ids:
            if _contains_placeholder(edge_id):
                errors.append(f"{node_id or expected_id}: semantic_edge_ids contains unresolved ENTITY placeholder.")
            if edge_id not in semantic_edge_ids:
                errors.append(f"{node_id or expected_id}: unknown semantic_edge_id {edge_id!r}.")
            elif operation == "lookup" and semantic_edge_types.get(edge_id) != "lookup":
                errors.append(f"{node_id or expected_id}: lookup question cites non-lookup semantic edge {edge_id!r}.")

        output_node_id = str(raw_question.get("output_node_id") or "").strip()
        if operation == "lookup" and not output_node_id:
            errors.append(f"{node_id or expected_id}: lookup questions must include output_node_id.")
        if output_node_id:
            if output_node_id not in semantic_node_ids:
                errors.append(f"{node_id or expected_id}: output_node_id must refer to an existing semantic node id.")
            lookup_targets = [
                semantic_edge_targets[edge_id]
                for edge_id in raw_semantic_edge_ids
                if edge_id in semantic_edge_targets and semantic_edge_types.get(edge_id) == "lookup"
            ]
            if operation == "lookup" and len(lookup_targets) == 1 and output_node_id != lookup_targets[0]:
                errors.append(
                    f"{node_id or expected_id}: output_node_id must refer to the target node of cited lookup edge {raw_semantic_edge_ids[0]!r}."
                )

        parsed_nodes.append(
            AtomicQuestionNode(
                id=node_id,
                question=question,
                depends_on=tuple(depends_on),
                operation=operation or "lookup",
                semantic_edge_ids=tuple(raw_semantic_edge_ids),
                output_node_id=output_node_id,
                output_type=output_type or final_answer_type,
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


def _validate_semantic_node_label(
    node_prefix: str,
    label: str,
    kind: str,
    question_evidence: list[str],
    errors: list[str],
) -> None:
    normalized = _normalize_for_relabeling(label)
    if normalized in WH_NODE_LABELS and not (kind in {"operator", "answer_slot"} and question_evidence):
        errors.append(
            f"{node_prefix}.label uses an ordinary wh-token as a semantic node; use answer_slot/operator with question_evidence only when needed."
        )
    if normalized in GRAMMATICAL_NODE_LABELS:
        errors.append(f"{node_prefix}.label is a pure grammatical token, not a semantic object.")


def _validate_adjacent_token_edge_relabeling(
    edge_prefix: str,
    source: str,
    target: str,
    node_labels: dict[str, str],
    source_token_path: list[str],
    relation: str,
    errors: list[str],
) -> None:
    source_label = _normalize_for_relabeling(node_labels.get(source, ""))
    target_label = _normalize_for_relabeling(node_labels.get(target, ""))
    source_tokens = [_normalize_for_relabeling(token) for token in source_token_path]
    if source_label not in source_tokens or target_label not in source_tokens:
        return
    if abs(source_tokens.index(source_label) - source_tokens.index(target_label)) != 1:
        return
    if _has_vague_operation(relation, None):
        errors.append(f"{edge_prefix}: edge connects adjacent copied token nodes with a generic relation label.")


def _check_likely_token_path_relabeling(
    prefix: str,
    source_token_path: list[str],
    semantic_node_labels: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    del warnings
    source_norm = [_normalize_for_relabeling(token) for token in source_token_path]
    label_norm = [_normalize_for_relabeling(label) for label in semantic_node_labels]
    copied_labels = [label for label in label_norm if label in source_norm]

    matched_positions: list[int] = []
    for label in label_norm:
        if not label:
            continue
        try:
            matched_positions.append(source_norm.index(label))
        except ValueError:
            continue
    if len(source_norm) and len(label_norm) >= len(source_norm) and len(copied_labels) / max(1, len(label_norm)) >= 0.75:
        errors.append(f"{prefix}: likely token-path relabeling; semantic node labels mostly copy source_token_path tokens.")
    elif len(matched_positions) >= 3 and matched_positions == sorted(matched_positions) and len(copied_labels) / max(1, len(label_norm)) >= 0.7:
        errors.append(f"{prefix}: likely token-path relabeling; semantic node labels enumerate source_token_path tokens in order.")


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


def _looks_like_possessive_wh_reversal(question: str, final_answer_intent: str) -> bool:
    if "whose " not in final_answer_intent.casefold():
        return False
    return bool(re.search(r"\bwho\s+is\s+the\s+\w+\s+of\s+the\s+person\s+who\b", question.casefold()))


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
        errors.append("Step5 topic_entities/explicit_entities must be a list.")
    for entity in explicit_entities:
        if _contains_placeholder(entity):
            errors.append(f"Step5 topic_entities contains unresolved ENTITY placeholder: {entity}.")
    if not global_best_paths:
        errors.append("Step5 requires at least one non-empty step4_paths/global_best_paths entry.")
        return errors
    for path_index, path in enumerate(global_best_paths, start=1):
        if not isinstance(path, list) or not path:
            errors.append(f"Step5 step4_paths[{path_index - 1}] must be a non-empty list.")
            continue
        for node in path:
            if _contains_placeholder(node):
                errors.append(f"Step5 step4_paths[{path_index - 1}] contains unresolved ENTITY placeholder: {node}.")
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
