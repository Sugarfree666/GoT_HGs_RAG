from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from models import (
    ASTSkeleton,
    ProblemFrame,
    SemanticASTEdge,
    SemanticASTNode,
    SemanticASTPrimaryOperator,
    SemanticASTResult,
)

COMPARE_OPERATORS = {
    "COMPARE_SAME",
    "COMPARE_DIFF",
    "COMPARE_GREATER",
    "COMPARE_LESS",
}

MIN_OPERATOR_ARITY = {
    "COMPARE_SAME": 2,
    "COMPARE_DIFF": 2,
    "COMPARE_GREATER": 2,
    "COMPARE_LESS": 2,
    "INTERSECTION": 2,
    "UNION": 2,
    "DIFFERENCE": 2,
    "LOGICAL_AND": 2,
    "LOGICAL_OR": 2,
}

PATH_PIPELINE_ALLOWED_OPERATORS = {
    "NONE",
    "COMPARE_SAME",
    "COMPARE_DIFF",
    "COMPARE_GREATER",
    "COMPARE_LESS",
    "ARGMAX",
    "ARGMIN",
    "INTERSECTION",
    "UNION",
    "DIFFERENCE",
    "LOGICAL_AND",
    "LOGICAL_OR",
}


@dataclass
class ValueSlotCueFrame:
    cue_text: str
    expected_value_slot: str
    slot_label: str
    expected_operators: list[str]
    relation_hint: str
    grounding_text: str = ""
    acceptable_value_slots: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.grounding_text:
            self.grounding_text = self.cue_text
        if not self.acceptable_value_slots:
            self.acceptable_value_slots = [self.expected_value_slot]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_value_slot_cue_frame(original_question: str) -> ValueSlotCueFrame | None:
    text = _norm(original_question)

    temporal_patterns = [
        (r"\breleased\s+(?:first|earliest|earlier)\b", "release_date", "release date", ["ARGMIN", "COMPARE_LESS"], "release date of {label}"),
        (r"\breleased\s+(?:later|latest|last)\b", "release_date", "release date", ["ARGMAX", "COMPARE_GREATER"], "release date of {label}"),
        (r"\bborn\s+earlier\b", "birth_date", "birth date", ["COMPARE_LESS", "ARGMIN"], "birth date of {label}"),
        (r"\bborn\s+later\b", "birth_date", "birth date", ["COMPARE_GREATER", "ARGMAX"], "birth date of {label}"),
        (r"\bdied\s+earlier\b", "death_date", "death date", ["COMPARE_LESS", "ARGMIN"], "death date of {label}"),
        (r"\bdied\s+later\b", "death_date", "death date", ["COMPARE_GREATER", "ARGMAX"], "death date of {label}"),
        (r"\bfounded\s+earlier\b", "founding_date", "founding date", ["COMPARE_LESS", "ARGMIN"], "founding date of {label}"),
        (r"\bfounded\s+later\b", "founding_date", "founding date", ["COMPARE_GREATER", "ARGMAX"], "founding date of {label}"),
        (r"\bpublished\s+(?:first|earliest|earlier)\b", "publication_date", "publication date", ["ARGMIN", "COMPARE_LESS"], "publication date of {label}"),
        (r"\bpublished\s+(?:later|latest|last)\b", "publication_date", "publication date", ["ARGMAX", "COMPARE_GREATER"], "publication date of {label}"),
        (r"\blaunched\s+(?:first|earliest|earlier)\b", "launch_date", "launch date", ["ARGMIN", "COMPARE_LESS"], "launch date of {label}"),
        (r"\blaunched\s+(?:later|latest|last)\b", "launch_date", "launch date", ["ARGMAX", "COMPARE_GREATER"], "launch date of {label}"),
    ]
    for pattern, slot, label, operators, relation in temporal_patterns:
        match = re.search(pattern, text)
        if match:
            return ValueSlotCueFrame(
                cue_text=match.group(0),
                expected_value_slot=slot,
                slot_label=label,
                expected_operators=operators,
                relation_hint=relation,
            )

    older_match = re.search(r"\bolder\b", text)
    if older_match:
        return ValueSlotCueFrame(
            cue_text=older_match.group(0),
            expected_value_slot="age",
            slot_label="age",
            expected_operators=["COMPARE_GREATER", "COMPARE_LESS"],
            relation_hint="age of {label}",
            acceptable_value_slots=["age", "birth_date"],
        )
    younger_match = re.search(r"\byounger\b", text)
    if younger_match:
        return ValueSlotCueFrame(
            cue_text=younger_match.group(0),
            expected_value_slot="age",
            slot_label="age",
            expected_operators=["COMPARE_LESS", "COMPARE_GREATER"],
            relation_hint="age of {label}",
            acceptable_value_slots=["age", "birth_date"],
        )

    same_diff_match = re.search(
        r"\b(?P<cue>same|different)\s+(?P<slot>nationality|director|author|country)\b",
        text,
    )
    if same_diff_match:
        slot = same_diff_match.group("slot")
        cue = same_diff_match.group("cue")
        return ValueSlotCueFrame(
            cue_text=same_diff_match.group(0),
            expected_value_slot=slot,
            slot_label=slot.replace("_", " "),
            expected_operators=["COMPARE_SAME"] if cue == "same" else ["COMPARE_DIFF"],
            relation_hint=f"{slot} of {{label}}",
        )

    superlative_patterns = [
        (r"\blargest\s+population\b", "population", "population", ["ARGMAX", "COMPARE_GREATER"], "population of {label}"),
        (r"\bsmallest\s+population\b", "population", "population", ["ARGMIN", "COMPARE_LESS"], "population of {label}"),
        (r"\bhighest\s+mountain\b", "height", "height", ["ARGMAX", "COMPARE_GREATER"], "height of {label}", ["height", "elevation"]),
        (r"\blowest\s+point\b", "elevation", "elevation", ["ARGMIN", "COMPARE_LESS"], "elevation of {label}"),
        (r"\blongest\s+river\b", "length", "length", ["ARGMAX", "COMPARE_GREATER"], "length of {label}"),
        (r"\bshortest\s+river\b", "length", "length", ["ARGMIN", "COMPARE_LESS"], "length of {label}"),
        (r"\bmost\s+awards\b", "award_count", "award count", ["ARGMAX", "COMPARE_GREATER"], "award count of {label}"),
        (r"\bfewest\s+awards\b", "award_count", "award count", ["ARGMIN", "COMPARE_LESS"], "award count of {label}"),
        (r"\bleast\s+awards\b", "award_count", "award count", ["ARGMIN", "COMPARE_LESS"], "award count of {label}"),
        (r"\bmost\s+populous\s+city\b", "population", "population", ["ARGMAX", "COMPARE_GREATER"], "population of {label}"),
    ]
    for item in superlative_patterns:
        pattern, slot, label, operators, relation = item[:5]
        acceptable = item[5] if len(item) > 5 else [slot]
        match = re.search(pattern, text)
        if match:
            return ValueSlotCueFrame(
                cue_text=match.group(0),
                expected_value_slot=slot,
                slot_label=label,
                expected_operators=operators,
                relation_hint=relation,
                acceptable_value_slots=list(acceptable),
            )

    largest_named_slot = re.search(r"\b(?P<cue>largest|smallest|highest|lowest|longest|shortest|most|least|fewest)\s+(?P<slot>[a-z][a-z0-9_-]+)\b", text)
    if largest_named_slot:
        cue = largest_named_slot.group("cue")
        slot = _slot_from_surface(largest_named_slot.group("slot"), cue)
        operators = ["ARGMAX", "COMPARE_GREATER"] if cue in {"largest", "highest", "longest", "most"} else ["ARGMIN", "COMPARE_LESS"]
        return ValueSlotCueFrame(
            cue_text=largest_named_slot.group(0),
            expected_value_slot=slot,
            slot_label=slot.replace("_", " "),
            expected_operators=operators,
            relation_hint=f"{slot.replace('_', ' ')} of {{label}}",
        )
    return None


def validate_ast_completeness(original_question: str, ast: SemanticASTResult) -> list[str]:
    warnings = validate_operator_usage(original_question, ast)
    operator = ast.primary_operator.operator
    if operator == "NONE":
        return warnings

    frame = detect_value_slot_cue_frame(original_question)
    if frame is None:
        return warnings

    if operator not in frame.expected_operators:
        warnings.append(
            f"Operator {operator} does not match confirmed value slot cue {frame.cue_text!r}; "
            f"expected one of {', '.join(frame.expected_operators)} for expected_value_slot={frame.expected_value_slot}. "
            "If this is only a chain lookup, use primary_operator=NONE."
        )

    input_ids = operator_input_node_ids(ast)
    if not input_ids:
        warnings.append(
            f"Operator {operator} is non-NONE but no operator inputs were provided."
        )
        return warnings

    for input_id in input_ids:
        if not _operator_input_satisfies_slot(ast, input_id, frame):
            node = ast.node_by_id().get(input_id)
            label = node.label if node is not None else input_id
            warnings.append(
                f"Operator {operator} uses cue {frame.cue_text!r}, which requires expected_value_slot={frame.expected_value_slot}, "
                f"but current input is {input_id} ({label}). Missing implicit variable {frame.expected_value_slot} for {input_id}."
            )
    return warnings


def validate_path_based_ast(
    semantic_ast: SemanticASTResult,
    ast_skeleton: ASTSkeleton,
    problem_frame: ProblemFrame,
) -> list[str]:
    """Validate an AST produced by the selected-path pipeline.

    The LLM may label relations but must not add, remove, merge, or shortcut
    the program-built skeleton.
    """

    warnings: list[str] = []
    skeleton_node_ids = {node.id for node in ast_skeleton.nodes}
    ast_node_ids = {node.id for node in semantic_ast.nodes}
    extra_nodes = sorted(ast_node_ids - skeleton_node_ids)
    missing_nodes = sorted(skeleton_node_ids - ast_node_ids)
    if extra_nodes:
        warnings.append("AST contains nodes outside selected paths: " + ", ".join(extra_nodes))
    if missing_nodes:
        warnings.append("AST is missing selected-path nodes: " + ", ".join(missing_nodes))

    allowed_edge_pairs = {
        (edge.source, edge.target)
        for edge in ast_skeleton.edges
    }
    ast_edge_pairs = [
        (edge.source, edge.target)
        for edge in semantic_ast.edges
        if edge.edge_type != "operator"
    ]
    for source, target in ast_edge_pairs:
        if (source, target) not in allowed_edge_pairs:
            warnings.append(
                f"AST contains shortcut or non-selected edge {source}->{target}; "
                "edges must be adjacent candidate nodes from selected paths."
            )
    missing_edges = sorted(allowed_edge_pairs - set(ast_edge_pairs))
    if missing_edges:
        warnings.append(
            "AST is missing selected-path edge(s): "
            + ", ".join(f"{source}->{target}" for source, target in missing_edges)
        )

    operator = semantic_ast.primary_operator.operator or "NONE"
    if operator not in PATH_PIPELINE_ALLOWED_OPERATORS:
        warnings.append(f"Operator {operator!r} is not in the allowed operator set.")
    if operator != (problem_frame.operator or "NONE"):
        warnings.append(
            f"Operator {operator!r} does not match ProblemFrame operator {problem_frame.operator!r}."
        )

    expected_inputs = [
        ast_skeleton.branch_terminals[requirement.id]
        for requirement in problem_frame.requirements
        if requirement.id in ast_skeleton.branch_terminals
    ]
    actual_inputs = list(semantic_ast.primary_operator.inputs)
    if actual_inputs != expected_inputs:
        warnings.append(
            "Operator inputs must be selected branch terminal nodes; "
            f"expected {expected_inputs}, got {actual_inputs}."
        )

    for requirement in problem_frame.requirements:
        if requirement.id not in ast_skeleton.branch_terminals:
            warnings.append(f"AST skeleton does not cover requirement {requirement.id!r}.")
        if not ast_skeleton.requirement_node_ids.get(requirement.id):
            warnings.append(f"AST skeleton has no branch nodes for requirement {requirement.id!r}.")

    warnings.extend(_validate_branch_specific_clones(semantic_ast, ast_skeleton, problem_frame))
    return warnings


def validate_operator_usage(original_question: str, ast: SemanticASTResult) -> list[str]:
    """Validate operator arity and reject comparison operators on serial lookup chains."""
    del original_question
    operator = ast.primary_operator.operator
    if operator == "NONE":
        return []

    warnings: list[str] = []
    input_ids = operator_input_node_ids(ast)
    min_arity = MIN_OPERATOR_ARITY.get(operator)
    if min_arity is not None and len(input_ids) < min_arity:
        warnings.append(
            f"Operator {operator} has {len(input_ids)} input(s) {input_ids}; "
            f"{operator} requires at least {min_arity} independent input values. "
            "This looks like a single-chain lookup or incomplete branch split; use primary_operator=NONE unless the final answer truly compares multiple branch results."
        )

    if operator in COMPARE_OPERATORS and _is_single_chain_ast(ast):
        warnings.append(
            f"Operator {operator} is attached to a single-chain semantic AST. "
            "Chain lookup predicates/events/attributes must be represented as semantic edges with primary_operator=NONE."
        )
    return warnings


def operator_input_node_ids(ast: SemanticASTResult) -> list[str]:
    if ast.primary_operator.inputs:
        return list(ast.primary_operator.inputs)
    operator_node_ids = {
        node.id
        for node in ast.nodes
        if node.kind == "operator" or node.label == ast.primary_operator.operator
    }
    return [
        edge.source
        for edge in ast.edges
        if edge.edge_type == "operator" and edge.target in operator_node_ids
    ]


def repair_missing_value_slots(original_question: str, semantic_ast: SemanticASTResult) -> SemanticASTResult:
    misuse_warnings = validate_operator_usage(original_question, semantic_ast)
    if misuse_warnings and semantic_ast.primary_operator.operator in COMPARE_OPERATORS:
        return _demote_operator_to_none(semantic_ast, misuse_warnings)

    frame = detect_value_slot_cue_frame(original_question)
    if frame is None:
        return semantic_ast
    if semantic_ast.primary_operator.operator == "NONE":
        return semantic_ast
    if semantic_ast.primary_operator.operator not in frame.expected_operators:
        return semantic_ast

    repaired = copy.deepcopy(semantic_ast)
    before_inputs = operator_input_node_ids(repaired)
    repaired.detected_cue_frame = frame.to_dict()
    repaired.operator_inputs_before_validation = list(before_inputs)
    actions: list[str] = []

    if not before_inputs:
        before_inputs = _infer_operator_inputs_from_edges(repaired)

    node_by_id = repaired.node_by_id()
    new_inputs: list[str] = []
    slot_index = _next_slot_index(repaired.nodes, frame.expected_value_slot)
    for input_id in before_inputs:
        if _operator_input_satisfies_slot(repaired, input_id, frame):
            new_inputs.append(input_id)
            continue
        source_node = node_by_id.get(input_id)
        if source_node is None:
            new_inputs.append(input_id)
            continue
        implicit_id = _unique_node_id(repaired.nodes, f"{frame.expected_value_slot}_{slot_index}")
        slot_index += 1
        relation_hint = _format_relation_hint(frame, source_node)
        repaired.nodes.append(
            SemanticASTNode(
                id=implicit_id,
                label=frame.slot_label,
                kind="implicit_type_variable",
                semantic_type=_semantic_type_for_slot(frame.expected_value_slot),
                source="step10_fallback_repair",
                grounding_text=frame.grounding_text,
                cue_text=frame.cue_text,
                branch_of=input_id,
                expected_value_slot=frame.expected_value_slot,
                relation_hint=relation_hint,
            )
        )
        repaired.edges.append(
            SemanticASTEdge(
                source=input_id,
                target=implicit_id,
                edge_type="attribute",
                relation_hint=relation_hint,
                support_path=[frame.cue_text],
                support_dependency_relations=[],
            )
        )
        new_inputs.append(implicit_id)
        actions.append(
            f"Inserted implicit value slot {implicit_id} ({frame.expected_value_slot}) after {input_id}."
        )

    repaired.primary_operator.inputs = _dedupe_preserve_order(new_inputs)
    _rewrite_operator_edges(repaired, before_inputs, repaired.primary_operator.inputs)
    repaired.fallback_repair_actions.extend(actions)
    if actions:
        repaired.validation_warnings.extend(validate_ast_completeness(original_question, semantic_ast))
    return repaired


def _demote_operator_to_none(ast: SemanticASTResult, reasons: list[str]) -> SemanticASTResult:
    repaired = copy.deepcopy(ast)
    old_operator = repaired.primary_operator.operator
    operator_node_ids = {
        node.id
        for node in repaired.nodes
        if node.kind == "operator" or node.label == old_operator
    }
    repaired.primary_operator = SemanticASTPrimaryOperator(operator="NONE")
    repaired.nodes = [
        node
        for node in repaired.nodes
        if node.id not in operator_node_ids and node.kind != "operator"
    ]
    repaired.edges = [
        edge
        for edge in repaired.edges
        if edge.edge_type != "operator"
        and edge.source not in operator_node_ids
        and edge.target not in operator_node_ids
    ]
    repaired.validation_warnings.extend(reason for reason in reasons if reason not in repaired.validation_warnings)
    repaired.fallback_repair_actions.append(
        f"Changed operator from {old_operator} to NONE because the AST did not provide enough independent operator inputs."
    )
    return repaired


def _operator_input_satisfies_slot(
    ast: SemanticASTResult,
    input_id: str,
    frame: ValueSlotCueFrame,
) -> bool:
    node_by_id = ast.node_by_id()
    node = node_by_id.get(input_id)
    if node is None:
        return False
    incoming = [
        edge
        for edge in ast.edges
        if edge.target == input_id
    ]
    outgoing = [
        edge
        for edge in ast.edges
        if edge.source == input_id
    ]
    text = " ".join(
        [
            input_id,
            node.label,
            node.semantic_type or "",
            node.grounding_text,
            node.cue_text,
            node.expected_value_slot or "",
            node.relation_hint or "",
            *[edge.relation_hint for edge in incoming],
            *[edge.relation_hint for edge in outgoing if edge.edge_type == "operator"],
        ]
    )
    normalized = _slot_text(text)
    return any(_slot_matches_text(slot, normalized) for slot in frame.acceptable_value_slots)


def _infer_operator_inputs_from_edges(ast: SemanticASTResult) -> list[str]:
    operator_node_ids = {
        node.id
        for node in ast.nodes
        if node.kind == "operator" or node.label == ast.primary_operator.operator
    }
    if operator_node_ids:
        inputs = [
            edge.source
            for edge in ast.edges
            if edge.edge_type == "operator" and edge.target in operator_node_ids
        ]
        if inputs:
            return _dedupe_preserve_order(inputs)
    return [
        node.id
        for node in ast.nodes
        if node.kind != "operator" and not any(edge.source == node.id for edge in ast.edges if edge.edge_type != "operator")
    ]


def _rewrite_operator_edges(ast: SemanticASTResult, old_inputs: list[str], new_inputs: list[str]) -> None:
    operator_node_ids = {
        node.id
        for node in ast.nodes
        if node.kind == "operator" or node.label == ast.primary_operator.operator
    }
    if not operator_node_ids:
        return
    old_input_set = set(old_inputs)
    ast.edges = [
        edge
        for edge in ast.edges
        if not (edge.edge_type == "operator" and edge.target in operator_node_ids and edge.source in old_input_set)
    ]
    existing = {(edge.source, edge.target, edge.edge_type) for edge in ast.edges}
    for operator_node_id in sorted(operator_node_ids):
        for input_id in new_inputs:
            key = (input_id, operator_node_id, "operator")
            if key in existing:
                continue
            ast.edges.append(
                SemanticASTEdge(
                    source=input_id,
                    target=operator_node_id,
                    edge_type="operator",
                    relation_hint=ast.primary_operator.operator,
                    support_path=[ast.primary_operator.cue_text] if ast.primary_operator.cue_text else [],
                    support_dependency_relations=[],
                )
            )
            existing.add(key)


def _is_single_chain_ast(ast: SemanticASTResult) -> bool:
    ordinary_edges = [edge for edge in ast.edges if edge.edge_type != "operator"]
    ordinary_node_ids = {
        node.id
        for node in ast.nodes
        if node.kind != "operator" and node.label != ast.primary_operator.operator
    }
    if not ordinary_edges:
        return len(ordinary_node_ids) <= 1

    involved = {edge.source for edge in ordinary_edges} | {edge.target for edge in ordinary_edges}
    if len(ordinary_edges) != max(0, len(involved) - 1):
        return False

    indegree = {node_id: 0 for node_id in involved}
    outdegree = {node_id: 0 for node_id in involved}
    undirected: dict[str, set[str]] = {node_id: set() for node_id in involved}
    for edge in ordinary_edges:
        outdegree[edge.source] = outdegree.get(edge.source, 0) + 1
        indegree[edge.target] = indegree.get(edge.target, 0) + 1
        undirected.setdefault(edge.source, set()).add(edge.target)
        undirected.setdefault(edge.target, set()).add(edge.source)

    if any(value > 1 for value in indegree.values()):
        return False
    if any(value > 1 for value in outdegree.values()):
        return False

    roots = [node_id for node_id in involved if indegree.get(node_id, 0) == 0]
    leaves = [node_id for node_id in involved if outdegree.get(node_id, 0) == 0]
    if len(roots) != 1 or len(leaves) != 1:
        return False

    seen: set[str] = set()
    stack = [roots[0]]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(sorted(undirected.get(node_id, set()) - seen))
    return seen == involved


def _validate_branch_specific_clones(
    semantic_ast: SemanticASTResult,
    ast_skeleton: ASTSkeleton,
    problem_frame: ProblemFrame,
) -> list[str]:
    if len(problem_frame.requirements) <= 1:
        return []

    warnings: list[str] = []
    node_by_id = semantic_ast.node_by_id()
    root_surfaces = {_norm(requirement.root) for requirement in problem_frame.requirements}
    surface_to_requirement_nodes: dict[str, dict[str, set[str]]] = {}
    for requirement in problem_frame.requirements:
        for node_id in ast_skeleton.requirement_node_ids.get(requirement.id, []):
            node = node_by_id.get(node_id)
            if node is None:
                continue
            surface = _norm(node.label)
            if not surface or surface in root_surfaces:
                continue
            surface_to_requirement_nodes.setdefault(surface, {}).setdefault(requirement.id, set()).add(node_id)

    for surface, by_requirement in surface_to_requirement_nodes.items():
        if len(by_requirement) <= 1:
            continue
        all_node_ids = {node_id for node_ids in by_requirement.values() for node_id in node_ids}
        if len(all_node_ids) < len(by_requirement):
            warnings.append(
                f"Shared surface node {surface!r} was merged across branches; branch-specific clones are required."
            )
        for requirement_id, node_ids in by_requirement.items():
            for node_id in node_ids:
                node = node_by_id.get(node_id)
                if node is None:
                    continue
                expected_suffix = f"_{requirement_id}"
                if not node.id.endswith(expected_suffix):
                    warnings.append(
                        f"Shared surface node {node.id!r} for requirement {requirement_id!r} "
                        f"must use branch-specific suffix {expected_suffix!r}."
                    )
                if node.branch_of != requirement_id:
                    warnings.append(
                        f"Shared surface node {node.id!r} must have branch_of={requirement_id!r}; "
                        f"got {node.branch_of!r}."
                    )
    return warnings


def _format_relation_hint(frame: ValueSlotCueFrame, source_node: SemanticASTNode) -> str:
    label = _slot_text(source_node.label) or "value"
    return frame.relation_hint.format(label=label.replace("_", " "))


def _next_slot_index(nodes: list[SemanticASTNode], expected_value_slot: str) -> int:
    pattern = re.compile(rf"^{re.escape(expected_value_slot)}_(\d+)$")
    indices = [
        int(match.group(1))
        for node in nodes
        for match in [pattern.match(node.id)]
        if match
    ]
    return max(indices, default=0) + 1


def _unique_node_id(nodes: list[SemanticASTNode], base: str) -> str:
    existing = {node.id for node in nodes}
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _slot_from_surface(surface: str, cue: str) -> str:
    value = surface.strip().lower().replace("-", "_")
    if cue == "fewest" and value.endswith("s"):
        return value[:-1] + "_count"
    return value


def _slot_matches_text(slot: str, normalized_text: str) -> bool:
    normalized_slot = _slot_text(slot)
    if normalized_slot in normalized_text:
        return True
    compact_slot = normalized_slot.replace(" ", "")
    compact_text = normalized_text.replace(" ", "")
    return compact_slot in compact_text


def _slot_text(value: str) -> str:
    text = _norm(value).replace("_", " ").replace("-", " ")
    aliases = {
        "birthdate": "birth date",
        "birth date": "birth date",
        "date of birth": "birth date",
        "releasedate": "release date",
        "release date": "release date",
        "publication date": "publication date",
        "founding date": "founding date",
        "death date": "death date",
        "launch date": "launch date",
        "award count": "award count",
    }
    return aliases.get(text, text)


def _semantic_type_for_slot(slot: str) -> str:
    if slot.endswith("_date"):
        return "Date"
    if slot.endswith("_count") or slot in {"population", "height", "elevation", "length", "age"}:
        return "Measure"
    if slot in {"nationality", "country"}:
        return slot.title()
    return "Value"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
