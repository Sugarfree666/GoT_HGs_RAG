from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from ast_validator import repair_missing_value_slots
from models import (
    ASTResult,
    AtomicQuestionDAG,
    AtomicQuestionEdge,
    AtomicQuestionNode,
    AtomicSubquestion,
    ExecutionPlan,
    ExecutionPlanStep,
    ExtractionResult,
    SemanticASTEdge,
    SemanticASTNode,
    SemanticASTResult,
)
from prompts import (
    ATOMIC_PLAN_STEP_SURFACE_SYSTEM,
    ONE_HOP_SUBQUESTION_SYSTEM,
    build_atomic_plan_step_surface_prompt,
    build_one_hop_prompt,
)
from surface_validator import contains_bare_variable, validate_atomic_dag_surface

if TYPE_CHECKING:
    from llm_client import LLMClient


@dataclass
class _GenerationState:
    counter: int = 1

    def next_serial_var(self) -> str:
        value = f"X{self.counter}"
        self.counter += 1
        return value


class SubquestionGenerator:
    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def generate(
        self,
        original_question: str,
        ast: ASTResult | SemanticASTResult,
        extraction: ExtractionResult | None = None,
    ) -> list[AtomicSubquestion]:
        if isinstance(ast, SemanticASTResult):
            return self.generate_dag(original_question, ast).to_subquestions()
        if extraction is None:
            raise TypeError("Legacy ASTResult generation requires extraction.")
        return self._generate_general(original_question, ast, extraction)

    def generate_dag(
        self,
        original_question: str,
        semantic_ast: SemanticASTResult | None = None,
        ast: SemanticASTResult | None = None,
    ) -> AtomicQuestionDAG:
        semantic_ast = semantic_ast if semantic_ast is not None else ast
        if semantic_ast is None:
            raise TypeError("generate_dag requires a SemanticASTResult.")
        semantic_ast = repair_missing_value_slots(original_question, semantic_ast)
        execution_plan = compile_execution_plan(semantic_ast)
        nodes: list[AtomicQuestionNode] = []
        edges: list[AtomicQuestionEdge] = []
        variable_to_question: dict[str, str] = {}
        variable_descriptions: dict[str, str] = {}

        for plan_step in execution_plan.steps:
            inputs = _dag_step_inputs(plan_step)
            depends_on: list[str] = []
            for input_value in inputs:
                upstream_question = variable_to_question.get(input_value)
                if upstream_question is None or upstream_question in depends_on:
                    continue
                depends_on.append(upstream_question)
                edges.append(
                    AtomicQuestionEdge(
                        source=upstream_question,
                        target=plan_step.step_id,
                        variable=input_value,
                    )
                )

            question_text, source = self._surface_plan_step(
                original_question=original_question,
                plan_step=plan_step,
                variable_descriptions=variable_descriptions,
            )
            output = _dag_step_output(plan_step)
            candidate_bindings = _dag_candidate_bindings(plan_step, original_question, variable_to_question)
            node = AtomicQuestionNode(
                id=plan_step.step_id,
                question=question_text,
                type=_dag_node_type(plan_step, candidate_bindings, original_question),
                inputs=inputs,
                output=output,
                depends_on=depends_on,
                source_node=plan_step.source_node,
                target_node=plan_step.target_node,
                ast_edge=plan_step.ast_edge,
                candidate_bindings=candidate_bindings,
                source=source,
            )
            nodes.append(node)
            if output:
                variable_to_question[output] = node.id
                variable_descriptions[output] = _output_description(plan_step, variable_descriptions)
            if plan_step.answer_variable:
                variable_to_question[plan_step.answer_variable] = node.id
                variable_descriptions[plan_step.answer_variable] = _output_description(plan_step, variable_descriptions)

        dag = AtomicQuestionDAG(
            nodes=nodes,
            edges=edges,
            variable_to_question=variable_to_question,
            warnings=[
                *execution_plan.warnings,
                *semantic_ast.validation_warnings,
                *semantic_ast.fallback_repair_actions,
            ],
        )
        _repair_atomic_dag_surface(dag, original_question, variable_descriptions)
        validation_messages = validate_atomic_dag_surface(dag)
        if validation_messages:
            raise ValueError("Invalid atomic DAG surface after repair: " + "; ".join(validation_messages))
        return dag

    def _generate_from_semantic_ast(
        self,
        original_question: str,
        semantic_ast: SemanticASTResult,
    ) -> list[AtomicSubquestion]:
        return self.generate_dag(original_question, semantic_ast).to_subquestions()

    def _surface_plan_step(
        self,
        original_question: str,
        plan_step: ExecutionPlanStep,
        variable_descriptions: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        variable_descriptions = variable_descriptions or {}
        try:
            payload = self.llm_client.chat_json(
                ATOMIC_PLAN_STEP_SURFACE_SYSTEM,
                build_atomic_plan_step_surface_prompt(
                    original_question=original_question,
                    plan_step=plan_step.to_dict(),
                    resolved_known_subject=_known_description(plan_step, variable_descriptions),
                    input_descriptions=_input_descriptions(plan_step.inputs, variable_descriptions),
                ),
            )
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("empty question")
            if contains_bare_variable(question):
                raise ValueError("surface question exposed an internal variable")
            if _contains_operator_cue(question):
                raise ValueError("ordinary edge question included operator cue")
            return question, "llm"
        except Exception:
            return _fallback_plan_edge_question(plan_step, variable_descriptions), "fallback_template"

    def _generate_general(
        self,
        original_question: str,
        ast: ASTResult,
        extraction: ExtractionResult,
    ) -> list[AtomicSubquestion]:
        graph = self._anchor_only_graph(ast.graph)
        starts = [node.placeholder for node in extraction.entities if node.placeholder in graph]
        if not starts:
            starts = [node for node in graph.nodes if graph.degree(node) <= 1] or list(graph.nodes)
        starts = sorted(starts, key=lambda node: graph.nodes[node].get("order", 10**9))

        state = _GenerationState()
        questions: list[AtomicSubquestion] = []
        visited_edges: set[frozenset[str]] = set()

        for start in starts:
            self._walk_general(
                original_question=original_question,
                ast=ast,
                graph=graph,
                current=start,
                parent=None,
                current_display=ast.display_label(start),
                current_original=ast.display_label(start),
                state=state,
                questions=questions,
                visited_edges=visited_edges,
            )

        return questions

    def _walk_general(
        self,
        original_question: str,
        ast: ASTResult,
        graph: nx.Graph,
        current: str,
        parent: str | None,
        current_display: str,
        current_original: str,
        state: _GenerationState,
        questions: list[AtomicSubquestion],
        visited_edges: set[frozenset[str]],
    ) -> None:
        neighbors = sorted(graph.neighbors(current), key=lambda node: graph.nodes[node].get("order", 10**9))
        for neighbor in neighbors:
            if neighbor == parent:
                continue
            edge_key = frozenset({current, neighbor})
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            edge_hint = _edge_hint(graph, current, neighbor)
            answer_var = state.next_serial_var()
            question_text = self._one_hop_question(
                original_question=original_question,
                source_display=current_display,
                target_display=ast.display_label(neighbor),
                source_original=current_original,
                target_original=ast.display_label(neighbor),
                answer_variable=answer_var,
                edge_hint=edge_hint,
            )
            questions.append(
                AtomicSubquestion(
                    index=len(questions) + 1,
                    question=question_text,
                    answer_variable=answer_var,
                    source_node=current,
                    target_node=neighbor,
                )
            )
            self._walk_general(
                original_question=original_question,
                ast=ast,
                graph=graph,
                current=neighbor,
                parent=current,
                current_display=answer_var,
                current_original=ast.display_label(neighbor),
                state=state,
                questions=questions,
                visited_edges=visited_edges,
            )

    def _one_hop_question(
        self,
        original_question: str,
        source_display: str,
        target_display: str,
        source_original: str,
        target_original: str,
        answer_variable: str,
        edge_hint: str | None,
    ) -> str:
        payload = self.llm_client.chat_json(
            ONE_HOP_SUBQUESTION_SYSTEM,
            build_one_hop_prompt(
                original_question=original_question,
                source_display=source_display,
                target_display=target_display,
                source_original=source_original,
                target_original=target_original,
                answer_variable=answer_variable,
                edge_hint=edge_hint,
            ),
        )
        question = str(payload.get("question", "")).strip()
        if not question:
            raise RuntimeError("LLM returned an empty one-hop subquestion.")
        return _enforce_source_variable_binding(question, source_display, source_original)

    @staticmethod
    def _anchor_only_graph(graph: nx.Graph) -> nx.Graph:
        result = graph.copy()
        operator_nodes = [
            node for node, attrs in result.nodes(data=True) if attrs.get("kind") == "operator"
        ]
        result.remove_nodes_from(operator_nodes)
        return result

def _edge_hint(graph: nx.Graph, source: str, target: str) -> str | None:
    if not graph.has_edge(source, target):
        return None
    attrs = graph.edges[source, target]
    relations = attrs.get("relations") or []
    path_words = attrs.get("path_words") or []
    pieces = []
    if relations:
        pieces.append("relations=" + "/".join(str(item) for item in relations if item))
    if path_words:
        pieces.append("dependency_path=" + " -> ".join(str(item) for item in path_words))
    return "; ".join(pieces) if pieces else None


def _slug(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    if not words:
        return "value"
    return "_".join(words[-2:]) if len(words) > 2 else "_".join(words)


def _known_description(
    plan_step: ExecutionPlanStep,
    variable_descriptions: dict[str, str],
) -> str:
    known = str(plan_step.known or "").strip()
    if known in variable_descriptions:
        return variable_descriptions[known]
    if known:
        return known
    return str(plan_step.known_node_label or "the source").strip()


def _input_descriptions(
    inputs: list[str],
    variable_descriptions: dict[str, str],
) -> dict[str, str]:
    return {
        input_value: variable_descriptions.get(input_value, input_value)
        for input_value in inputs
        if str(input_value).strip()
    }


def _output_description(
    plan_step: ExecutionPlanStep,
    variable_descriptions: dict[str, str],
) -> str:
    known = _known_description(plan_step, variable_descriptions)
    ask = str(plan_step.ask or "value").strip()
    if not ask:
        return f"the answer related to {known}"
    return _attribute_phrase(ask, known)


def _attribute_phrase(attribute: str, subject: str) -> str:
    attribute = re.sub(r"\s+", " ", str(attribute).strip())
    subject = re.sub(r"\s+", " ", str(subject).strip())
    if not subject:
        subject = "the subject"
    if not attribute:
        return f"the value of {subject}"
    normalized_attribute = attribute.lower()
    if normalized_attribute.startswith(("who ", "what ", "which ", "where ", "when ")):
        return f"the answer to {attribute}"
    return f"the {attribute} of {subject}"


def _enforce_source_variable_binding(question: str, source_display: str, source_original: str) -> str:
    if not _is_answer_variable(source_display) or _contains_variable(question, source_display):
        return question

    fixed = _replace_source_text(question, source_original, source_display)
    if fixed != question:
        return fixed
    return f"For {source_display}, {question[:1].lower()}{question[1:]}" if question else question


def _is_answer_variable(value: str) -> bool:
    return bool(re.fullmatch(r"X\d+(?:_[A-Za-z0-9_]+)?", value.strip()))


def _contains_variable(question: str, variable: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])", question))


def _replace_source_text(question: str, source_original: str, variable: str) -> str:
    source_words = re.findall(r"[A-Za-z0-9]+", source_original)
    if not source_words:
        return question
    escaped = r"\s+".join(re.escape(word) for word in source_words)
    pattern = re.compile(rf"\b(?:the|a|an)?\s*{escaped}\b", flags=re.IGNORECASE)
    return pattern.sub(variable, question, count=1)


def _contains_operator_cue(question: str) -> bool:
    cues = {
        "after",
        "before",
        "different",
        "first",
        "highest",
        "larger",
        "largest",
        "older",
        "same",
        "smaller",
        "youngest",
        "younger",
    }
    words = set(re.findall(r"[A-Za-z]+", question.lower()))
    return bool(words & cues)


def _ordered_semantic_edges(semantic_ast: SemanticASTResult) -> list[SemanticASTEdge]:
    if not semantic_ast.edges:
        return []
    graph = nx.DiGraph()
    node_order = {node.id: index for index, node in enumerate(semantic_ast.nodes)}
    edge_order: dict[tuple[str, str], int] = {}
    edge_by_pair: dict[tuple[str, str], SemanticASTEdge] = {}
    for index, edge in enumerate(semantic_ast.edges):
        graph.add_edge(edge.source, edge.target)
        edge_order.setdefault((edge.source, edge.target), index)
        edge_by_pair.setdefault((edge.source, edge.target), edge)
    if not nx.is_directed_acyclic_graph(graph):
        return list(semantic_ast.edges)

    roots = sorted(
        [node for node in graph.nodes if graph.in_degree(node) == 0],
        key=lambda node: node_order.get(node, 10**9),
    )
    ordered: list[SemanticASTEdge] = []
    visited_edges: set[tuple[str, str]] = set()

    def visit(node: str) -> None:
        outgoing = sorted(
            graph.successors(node),
            key=lambda target: (
                edge_order.get((node, target), 10**9),
                node_order.get(target, 10**9),
            ),
        )
        for target in outgoing:
            key = (node, target)
            if key in visited_edges:
                continue
            visited_edges.add(key)
            edge = edge_by_pair.get(key)
            if edge is not None:
                ordered.append(edge)
            visit(target)

    for root in roots:
        visit(root)
    for edge in semantic_ast.edges:
        key = (edge.source, edge.target)
        if key not in visited_edges:
            ordered.append(edge)
            visited_edges.add(key)
    return ordered


def _dag_step_inputs(plan_step: ExecutionPlanStep) -> list[str]:
    if plan_step.known:
        return [plan_step.known]
    return []


def _dag_step_output(plan_step: ExecutionPlanStep) -> str:
    return plan_step.answer_variable or plan_step.output


def _dag_node_type(
    plan_step: ExecutionPlanStep,
    candidate_bindings: list[dict[str, object]],
    original_question: str,
) -> str:
    del plan_step, candidate_bindings, original_question
    return "lookup"


def _dag_candidate_bindings(
    plan_step: ExecutionPlanStep,
    original_question: str,
    variable_to_question: dict[str, str],
) -> list[dict[str, object]]:
    del plan_step, original_question, variable_to_question
    return []


def compile_execution_plan(semantic_ast: SemanticASTResult) -> ExecutionPlan:
    """Compile a directed semantic AST into a deterministic variable-bound DAG.

    The AST stores semantic structure. This plan stores execution order and
    variable bindings, so the LLM only surfaces a single already-decided step.
    """

    node_by_id = semantic_ast.node_by_id()
    node_bindings: dict[str, list[str]] = {}
    warnings: list[str] = []
    steps: list[ExecutionPlanStep] = []
    serial = 1

    for edge in _ordered_ordinary_semantic_edges(semantic_ast):
        source_node = node_by_id.get(edge.source)
        target_node = node_by_id.get(edge.target)
        if source_node is None or target_node is None:
            warnings.append(f"Skipped execution edge with missing endpoint: {edge.source}->{edge.target}.")
            continue

        answer_variable = f"X{serial}"
        serial += 1
        known = _node_binding_display(source_node, node_bindings)
        step = ExecutionPlanStep(
            step_id=f"q{len(steps) + 1}",
            step_type="edge",
            source_node=edge.source,
            target_node=edge.target,
            known=known,
            known_node_label=source_node.label,
            ask=target_node.label,
            relation_hint=edge.relation_hint,
            answer_variable=answer_variable,
            ast_edge=edge.to_dict(),
        )
        steps.append(step)
        node_bindings.setdefault(edge.target, []).append(answer_variable)

    return ExecutionPlan(steps=steps, node_bindings=node_bindings, warnings=warnings)


def _ordered_ordinary_semantic_edges(semantic_ast: SemanticASTResult) -> list[SemanticASTEdge]:
    ordinary_edges = [edge for edge in semantic_ast.edges if edge.edge_type != "operator"]
    if len(ordinary_edges) == len(semantic_ast.edges):
        return _ordered_semantic_edges(semantic_ast)
    ordinary_ast = SemanticASTResult(
        status=semantic_ast.status,
        primary_operator=semantic_ast.primary_operator,
        nodes=semantic_ast.nodes,
        edges=ordinary_edges,
        warnings=semantic_ast.warnings,
        raw_payload=semantic_ast.raw_payload,
    )
    return _ordered_semantic_edges(ordinary_ast)


def _node_binding_display(
    node: SemanticASTNode,
    node_bindings: dict[str, list[str]],
) -> str:
    bindings = node_bindings.get(node.id, [])
    if bindings:
        return bindings[-1]
    return node.label


def _fallback_plan_edge_question(
    plan_step: ExecutionPlanStep,
    variable_descriptions: dict[str, str] | None = None,
) -> str:
    variable_descriptions = variable_descriptions or {}
    known = _known_description(plan_step, variable_descriptions)
    ask = plan_step.ask or "value"
    relation = plan_step.relation_hint.lower()
    if _is_person_answer_label(ask):
        return f"Who is the {ask} of {known}?"
    if relation.startswith(("develop", "create", "invent", "found", "write", "direct")):
        return f"What {ask} is related to {known}?"
    return f"What is the {ask} of {known}?"


def _repair_atomic_dag_surface(
    dag: AtomicQuestionDAG,
    original_question: str,
    variable_descriptions: dict[str, str],
) -> None:
    node_by_id = {node.id: node for node in dag.nodes}
    for node in dag.nodes:
        repair_notes: list[str] = []
        if contains_bare_variable(node.question):
            original_surface = node.question
            node.question = _replace_internal_variables(
                node.question,
                variable_descriptions=variable_descriptions,
                dependency_questions=[node_by_id[dependency].question for dependency in node.depends_on if dependency in node_by_id],
            )
            if contains_bare_variable(node.question):
                node.question = _generic_dependency_question(node)
            repair_notes.append(f"Repaired internal variable exposure in question: {original_surface}")

        for binding in node.candidate_bindings:
            for key in ("candidate", "label"):
                label = str(binding.get(key) or "").strip()
                if label and contains_bare_variable(label):
                    binding[key] = _replace_internal_variables(
                        label,
                        variable_descriptions=variable_descriptions,
                        dependency_questions=[],
                    )
                    repair_notes.append(f"Repaired internal variable exposure in candidate label: {label}")

        if repair_notes:
            existing_warning = str(node.metadata.get("warning", "") or "").strip()
            combined = "; ".join([existing_warning, *repair_notes] if existing_warning else repair_notes)
            node.metadata["warning"] = combined


def _replace_internal_variables(
    text: str,
    *,
    variable_descriptions: dict[str, str],
    dependency_questions: list[str],
) -> str:
    dependency_iter = iter(dependency_questions)

    def replacement(match: re.Match[str]) -> str:
        variable = match.group(0)
        if variable in variable_descriptions:
            return variable_descriptions[variable]
        dependency_question = next(dependency_iter, "")
        if dependency_question:
            return f"the answer to '{dependency_question}'"
        return "the dependent answer"

    return re.sub(r"(?<![A-Za-z0-9_])(?:X\d+(?:_[A-Za-z0-9_]+)?|V\d+|VAR_[A-Za-z0-9_]+)(?![A-Za-z0-9_])", replacement, text)


def _generic_dependency_question(node: AtomicQuestionNode) -> str:
    if node.depends_on:
        return "What fact is needed from the dependency evidence?"
    return "What is the requested fact?"


def _is_person_answer_label(label: str) -> bool:
    normalized = label.strip().lower()
    person_labels = {
        "actor",
        "author",
        "ceo",
        "director",
        "founder",
        "person",
        "player",
        "president",
        "producer",
        "writer",
    }
    return normalized in person_labels or normalized.endswith((" actor", " author", " director", " person"))
