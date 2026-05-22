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
    ATOMIC_SUBQUESTION_GENERATION_SYSTEM,
    ONE_HOP_SUBQUESTION_SYSTEM,
    build_atomic_plan_step_surface_prompt,
    build_atomic_subquestion_generation_prompt,
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
        operator_node = self._first_operator(ast)
        if operator_node:
            return self._generate_operator(original_question, ast, extraction, operator_node)
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
                operator=plan_step.operator,
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
            if plan_step.step_type == "operator":
                return question, "llm"
            if _contains_operator_cue(question):
                raise ValueError("ordinary edge question included operator cue")
            return question, "llm"
        except Exception:
            if plan_step.step_type == "operator":
                return (
                    _operator_surface_question(original_question, plan_step, variable_descriptions),
                    "fallback_template",
                )
            return _fallback_plan_edge_question(plan_step, variable_descriptions), "fallback_template"

    def _semantic_edge_question(
        self,
        original_question: str,
        semantic_ast: SemanticASTResult,
        edge: SemanticASTEdge,
        source_node: SemanticASTNode | None,
        target_node: SemanticASTNode | None,
        source_display: str,
        source_original: str,
        answer_variable: str,
    ) -> tuple[str, str]:
        prompt = build_atomic_subquestion_generation_prompt(
            original_question=original_question,
            semantic_ast=semantic_ast.to_dict(),
            current_edge={
                **edge.to_dict(),
                "source_display": source_display,
                "source_label": source_original,
                "target_label": target_node.label if target_node is not None else edge.target,
                "answer_variable": answer_variable,
            },
            source_node=source_node.to_dict() if source_node else None,
            target_node=target_node.to_dict() if target_node else None,
            primary_operator=semantic_ast.primary_operator.to_dict(),
        )
        try:
            payload = self.llm_client.chat_json(ATOMIC_SUBQUESTION_GENERATION_SYSTEM, prompt)
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("empty question")
            if _contains_operator_cue(question) and edge.edge_type != "operator":
                raise ValueError("ordinary edge question included operator cue")
            if _expands_bound_source(question, source_display, source_original):
                raise ValueError("ordinary edge question expanded an already-bound source variable")
            return _enforce_source_variable_binding(question, source_display, source_original), "llm"
        except Exception:
            return _fallback_semantic_edge_question(source_display, target_node), "fallback_template"

    def _semantic_operator_question(
        self,
        original_question: str,
        semantic_ast: SemanticASTResult,
        operator_inputs: list[str],
    ) -> tuple[str, str]:
        operator = semantic_ast.primary_operator
        current_edge = {
            "type": "operator_step",
            "operator": operator.operator,
            "inputs": operator_inputs,
            "semantic_inputs": operator.inputs,
            "output": operator.output,
            "cue_text": operator.cue_text,
        }
        try:
            payload = self.llm_client.chat_json(
                ATOMIC_SUBQUESTION_GENERATION_SYSTEM,
                build_atomic_subquestion_generation_prompt(
                    original_question=original_question,
                    semantic_ast=semantic_ast.to_dict(),
                    current_edge=current_edge,
                    source_node=None,
                    target_node=None,
                    primary_operator=operator.to_dict(),
                ),
            )
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("empty operator question")
            if not _operator_question_uses_inputs(question, operator_inputs):
                raise ValueError("operator question did not use bound operator inputs")
            return question, "llm"
        except Exception:
            return _operator_question(operator.operator, operator_inputs, operator.cue_text), "fallback_template"

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

    def _generate_operator(
        self,
        original_question: str,
        ast: ASTResult,
        extraction: ExtractionResult,
        operator_node: str,
    ) -> list[AtomicSubquestion]:
        graph = self._anchor_only_graph(ast.graph)
        attach_nodes = [node for node in ast.graph.neighbors(operator_node) if node in graph]
        if not attach_nodes:
            attach_nodes = [self._choose_compare_target(graph, extraction)]
        target = attach_nodes[0]

        entities = [node.placeholder for node in extraction.entities if node.placeholder in graph]
        if not entities:
            entities = [node for node in graph.nodes if graph.degree(node) <= 1 and node != target]
        entities = sorted(entities, key=lambda node: graph.nodes[node].get("order", 10**9))

        operator_name = str(ast.graph.nodes[operator_node].get("text", operator_node))
        use_direct_implicit_attribute = (
            operator_name.startswith("COMPARE")
            and _is_implicit_type_variable(target, extraction)
        )
        questions: list[AtomicSubquestion] = []
        final_vars: list[str] = []
        used_edges: set[tuple[str, str, int]] = set()

        for branch_index, entity in enumerate(entities, start=1):
            if use_direct_implicit_attribute:
                path = [entity, target]
            else:
                try:
                    path = nx.shortest_path(graph, entity, target)
                except nx.NetworkXNoPath:
                    continue
            current_display = ast.display_label(entity)
            current_original = ast.display_label(entity)
            branch_final = None
            for step_index, (source, target_node) in enumerate(zip(path, path[1:]), start=1):
                edge_identity = (source, target_node, branch_index)
                if edge_identity in used_edges:
                    continue
                used_edges.add(edge_identity)
                edge_hint = _edge_hint(graph, source, target_node)
                if step_index == 1:
                    answer_var = f"X{branch_index}"
                elif target_node == target:
                    answer_var = f"X{branch_index}_{_slug(ast.display_label(target_node))}"
                else:
                    answer_var = f"X{branch_index}_{step_index}"
                question_text = self._one_hop_question(
                    original_question=original_question,
                    source_display=current_display,
                    target_display=ast.display_label(target_node),
                    source_original=current_original,
                    target_original=ast.display_label(target_node),
                    answer_variable=answer_var,
                    edge_hint=edge_hint,
                )
                questions.append(
                    AtomicSubquestion(
                        index=len(questions) + 1,
                        question=question_text,
                        answer_variable=answer_var,
                        source_node=source,
                        target_node=target_node,
                    )
                )
                current_display = answer_var
                current_original = ast.display_label(target_node)
                branch_final = answer_var
            if branch_final:
                final_vars.append(branch_final)

        min_vars = 2 if operator_name.startswith("COMPARE") or operator_name in {"INTERSECTION", "UNION", "DIFFERENCE"} else 1
        if len(final_vars) >= min_vars:
            compare_question = _operator_question(operator_name, final_vars)
            questions.append(
                AtomicSubquestion(
                    index=len(questions) + 1,
                    question=compare_question,
                    answer_variable=None,
                    operator=operator_name,
                )
            )
        return questions

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

    @staticmethod
    def _first_operator(ast: ASTResult) -> str | None:
        for node, attrs in ast.graph.nodes(data=True):
            if attrs.get("kind") == "operator":
                return node
        return None

    @staticmethod
    def _choose_compare_target(graph: nx.Graph, extraction: ExtractionResult) -> str:
        type_nodes = [node.placeholder for node in extraction.type_variables if node.placeholder in graph]
        if type_nodes:
            return max(type_nodes, key=lambda node: graph.nodes[node].get("order", 0))
        return max(graph.nodes, key=lambda node: graph.degree(node))


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
    if plan_step.step_type == "operator":
        return "the final answer"
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


def _operator_surface_question(
    original_question: str,
    plan_step: ExecutionPlanStep,
    variable_descriptions: dict[str, str],
) -> str:
    question = original_question.strip()
    if question and not contains_bare_variable(question):
        return question
    inputs = [variable_descriptions.get(input_value, input_value) for input_value in plan_step.inputs]
    return _operator_question(plan_step.operator or "NONE", inputs, plan_step.cue_text)


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


def _is_implicit_type_variable(placeholder: str, extraction: ExtractionResult) -> bool:
    for node in extraction.type_variables:
        if node.placeholder == placeholder:
            return node.occurrence == 0
    return False


def _operator_question(operator: str, variables: list[str], cue_text: str = "") -> str:
    if operator == "COMPARE_DIFF":
        return f"Are {' and '.join(variables)} different?"
    if operator == "COMPARE_SAME":
        return f"Are {' and '.join(variables)} the same?"
    if operator == "INTERSECTION":
        return f"What values are common to {' and '.join(variables)}?"
    if operator == "UNION":
        return f"What values are in either {' or '.join(variables)}?"
    if operator == "DIFFERENCE":
        return f"What values are in {variables[0]} but not in {variables[1]}?" if len(variables) >= 2 else f"What is the difference for {', '.join(variables)}?"
    if operator == "COMPARE_GREATER":
        if cue_text.strip():
            return _cue_based_compare_question(variables, cue_text)
        return f"Which is greater, {' or '.join(variables)}?"
    if operator == "COMPARE_LESS":
        if cue_text.strip():
            return _cue_based_compare_question(variables, cue_text)
        return f"Which is less, {' or '.join(variables)}?"
    if operator == "ARGMAX":
        return f"Which has the maximum value among {', '.join(variables)}?"
    if operator == "ARGMIN":
        return f"Which has the minimum value among {', '.join(variables)}?"
    if operator == "LOGICAL_OR":
        return f"Does either {' or '.join(variables)} satisfy the condition?"
    if operator == "LOGICAL_AND":
        return f"Do {' and '.join(variables)} all satisfy the condition?"
    return f"Apply {operator} to {', '.join(variables)}."


def _cue_based_compare_question(variables: list[str], cue_text: str) -> str:
    cue = re.sub(r"\s+", " ", cue_text.strip())
    variable_phrase = _choice_variable_phrase(variables)
    lowered = cue.lower()
    if lowered.startswith(("is ", "are ", "was ", "were ", "has ", "have ")):
        return f"Which of {variable_phrase} {cue}?"
    if lowered.startswith(("born ", "released ", "published ", "founded ", "created ", "established ")):
        return f"Which of {variable_phrase} was {cue}?"
    return f"Which of {variable_phrase} is {cue}?"


def _choice_variable_phrase(variables: list[str]) -> str:
    if len(variables) <= 2:
        return " and ".join(variables)
    return f"{', '.join(variables[:-1])}, and {variables[-1]}"


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


def _expands_bound_source(question: str, source_display: str, source_original: str) -> bool:
    if not _is_answer_variable(source_display):
        return False
    if not _contains_variable(question, source_display):
        return True
    original = source_original.strip()
    if not original or _is_answer_variable(original):
        return False
    words = re.findall(r"[A-Za-z0-9]+", original)
    if not words:
        return False
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])" + r"\s+".join(re.escape(word) for word in words) + r"(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(question))


def _operator_question_uses_inputs(question: str, operator_inputs: list[str]) -> bool:
    if not operator_inputs:
        return True
    return all(_contains_variable(question, item) for item in operator_inputs if _is_answer_variable(item))


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


def _bound_source_display(
    source_node: SemanticASTNode | None,
    node_bindings: dict[str, list[str]],
) -> str:
    if source_node is None:
        return "the source"
    bindings = node_bindings.get(source_node.id, [])
    if bindings:
        return bindings[-1]
    return source_node.label


def _operator_input_variables(
    semantic_ast: SemanticASTResult,
    node_bindings: dict[str, list[str]],
) -> list[str]:
    variables: list[str] = []
    for semantic_input in semantic_ast.primary_operator.inputs:
        bindings = node_bindings.get(semantic_input, [])
        if bindings:
            variables.extend(bindings)
        else:
            variables.append(semantic_input)
    return variables


def _dag_step_inputs(plan_step: ExecutionPlanStep) -> list[str]:
    if plan_step.step_type == "operator":
        return _dedupe_preserve_order(plan_step.inputs)
    if plan_step.known:
        return [plan_step.known]
    return []


def _dag_step_output(plan_step: ExecutionPlanStep) -> str:
    if plan_step.step_type == "operator":
        return "FINAL"
    return plan_step.answer_variable or plan_step.output


def _dag_node_type(
    plan_step: ExecutionPlanStep,
    candidate_bindings: list[dict[str, object]],
    original_question: str,
) -> str:
    if plan_step.step_type != "operator":
        return "lookup"

    operator = plan_step.operator or "operator"
    if operator in {"ARGMIN", "ARGMAX"}:
        return "selection"
    if operator.startswith("COMPARE"):
        if candidate_bindings and _asks_for_candidate(original_question):
            return "selection"
        return "comparison"
    return "operator"


def _dag_candidate_bindings(
    plan_step: ExecutionPlanStep,
    original_question: str,
    variable_to_question: dict[str, str],
) -> list[dict[str, object]]:
    if plan_step.step_type != "operator":
        return []
    bindings = _operator_candidate_bindings(plan_step)
    if not bindings:
        return []
    for binding in bindings:
        value = str(binding.get("value") or "").strip()
        source_node_id = variable_to_question.get(value, "")
        if source_node_id:
            binding["source_node_id"] = source_node_id
        if "candidate" in binding:
            binding["label"] = binding["candidate"]
    operator = plan_step.operator or ""
    if operator in {"ARGMIN", "ARGMAX"}:
        return bindings
    if operator.startswith("COMPARE") and _asks_for_candidate(original_question):
        return bindings
    return []


def _asks_for_candidate(original_question: str) -> bool:
    return bool(re.match(r"\s*(which|what)\b", original_question, flags=re.IGNORECASE))


def _operator_candidate_bindings(plan_step: ExecutionPlanStep) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for branch in plan_step.operator_branches:
        value = str(branch.get("input_variable") or "").strip()
        if not value:
            continue
        branch_steps = _branch_steps(branch.get("branch_steps"))
        candidate = _candidate_for_branch(branch, branch_steps)
        if not candidate:
            continue
        binding: dict[str, object] = {
            "candidate": candidate,
            "value": value,
        }
        context = _branch_context(branch_steps, value)
        if context:
            binding["context"] = context
        bindings.append(binding)
    return bindings


def _branch_steps(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _candidate_for_branch(
    branch: dict[str, object],
    branch_steps: list[dict[str, object]],
) -> str:
    if branch_steps:
        first_step = branch_steps[0]
        known = str(first_step.get("known") or "").strip()
        if known and not _is_answer_variable(known):
            return known
        source_label = str(first_step.get("source_label") or "").strip()
        if source_label:
            return source_label
    return str(branch.get("input_label") or branch.get("semantic_input") or "").strip()


def _branch_context(
    branch_steps: list[dict[str, object]],
    value_variable: str,
) -> dict[str, str]:
    context: dict[str, str] = {}
    for step in branch_steps:
        answer_variable = str(step.get("answer_variable") or "").strip()
        if not answer_variable or answer_variable == value_variable:
            continue
        key = str(step.get("target_label") or step.get("target_node") or answer_variable).strip()
        if not key:
            key = answer_variable
        if key in context and context[key] != answer_variable:
            alternate = str(step.get("target_node") or "").strip()
            if alternate and alternate not in context:
                key = alternate
            else:
                key = f"{key}_{len(context) + 1}"
        context[key] = answer_variable
    return context


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def compile_execution_plan(semantic_ast: SemanticASTResult) -> ExecutionPlan:
    """Compile a directed semantic AST into a deterministic variable-bound DAG.

    The AST stores semantic structure. This plan stores execution order and
    variable bindings, so the LLM only surfaces a single already-decided step.
    """

    node_by_id = semantic_ast.node_by_id()
    node_bindings: dict[str, list[str]] = {}
    warnings: list[str] = []
    steps: list[ExecutionPlanStep] = []
    branch_paths_by_node: dict[str, list[dict[str, object]]] = {}
    branch_by_variable: dict[str, dict[str, object]] = {}
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
        branch_steps = [
            *branch_paths_by_node.get(edge.source, []),
            _execution_branch_step(step, source_node, target_node),
        ]
        branch_paths_by_node[edge.target] = branch_steps
        branch_by_variable[answer_variable] = _operator_branch_for_binding(
            semantic_input=edge.target,
            input_variable=answer_variable,
            input_node=target_node,
            branch_steps=branch_steps,
        )

    if semantic_ast.primary_operator.operator != "NONE":
        semantic_inputs = _operator_semantic_inputs(semantic_ast)
        operator_inputs = _operator_input_variables_for_inputs(semantic_inputs, node_bindings)
        operator_branches = _operator_branches_for_inputs(
            semantic_inputs=semantic_inputs,
            node_by_id=node_by_id,
            node_bindings=node_bindings,
            branch_by_variable=branch_by_variable,
        )
        steps.append(
            ExecutionPlanStep(
                step_id=f"q{len(steps) + 1}",
                step_type="operator",
                operator=semantic_ast.primary_operator.operator,
                inputs=operator_inputs,
                semantic_inputs=semantic_inputs,
                output=semantic_ast.primary_operator.output,
                cue_text=semantic_ast.primary_operator.cue_text,
                answer_variable=None,
                operator_branches=operator_branches,
            )
        )

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


def _operator_semantic_inputs(semantic_ast: SemanticASTResult) -> list[str]:
    if semantic_ast.primary_operator.inputs:
        return list(semantic_ast.primary_operator.inputs)
    operator_node_ids = {
        node.id
        for node in semantic_ast.nodes
        if node.kind == "operator" and node.label == semantic_ast.primary_operator.operator
    }
    return [
        edge.source
        for edge in semantic_ast.edges
        if edge.edge_type == "operator" and edge.target in operator_node_ids
    ]


def _operator_input_variables_for_inputs(
    semantic_inputs: list[str],
    node_bindings: dict[str, list[str]],
) -> list[str]:
    variables: list[str] = []
    for semantic_input in semantic_inputs:
        bindings = node_bindings.get(semantic_input, [])
        if bindings:
            variables.extend(bindings)
        else:
            variables.append(semantic_input)
    return variables


def _execution_branch_step(
    step: ExecutionPlanStep,
    source_node: SemanticASTNode,
    target_node: SemanticASTNode,
) -> dict[str, object]:
    return {
        "step_id": step.step_id,
        "source_node": step.source_node,
        "target_node": step.target_node,
        "source_label": source_node.label,
        "source_kind": source_node.kind,
        "target_label": target_node.label,
        "target_kind": target_node.kind,
        "known": step.known,
        "known_node_label": step.known_node_label,
        "ask": step.ask,
        "relation_hint": step.relation_hint,
        "answer_variable": step.answer_variable,
    }


def _operator_branch_for_binding(
    semantic_input: str,
    input_variable: str,
    input_node: SemanticASTNode,
    branch_steps: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "semantic_input": semantic_input,
        "input_variable": input_variable,
        "input_label": input_node.label,
        "input_kind": input_node.kind,
        "branch_steps": branch_steps,
        "branch_summary": _operator_branch_summary(input_variable, branch_steps),
    }


def _operator_branches_for_inputs(
    semantic_inputs: list[str],
    node_by_id: dict[str, SemanticASTNode],
    node_bindings: dict[str, list[str]],
    branch_by_variable: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    branches: list[dict[str, object]] = []
    for semantic_input in semantic_inputs:
        input_node = node_by_id.get(semantic_input)
        bindings = node_bindings.get(semantic_input, [])
        if bindings:
            for input_variable in bindings:
                branch = dict(branch_by_variable.get(input_variable, {}))
                branch["semantic_input"] = semantic_input
                branch["input_variable"] = input_variable
                if input_node is not None:
                    branch["input_label"] = input_node.label
                    branch["input_kind"] = input_node.kind
                branch.setdefault("branch_steps", [])
                branch.setdefault(
                    "branch_summary",
                    _operator_branch_summary(input_variable, list(branch["branch_steps"])),
                )
                branches.append(branch)
            continue

        input_label = input_node.label if input_node is not None else semantic_input
        input_kind = input_node.kind if input_node is not None else ""
        branches.append(
            {
                "semantic_input": semantic_input,
                "input_variable": semantic_input,
                "input_label": input_label,
                "input_kind": input_kind,
                "branch_steps": [],
                "branch_summary": f"{semantic_input}: {input_label}",
            }
        )
    return branches


def _operator_branch_summary(input_variable: str, branch_steps: list[dict[str, object]]) -> str:
    if not branch_steps:
        return f"{input_variable}: operator input"
    parts: list[str] = []
    for step in branch_steps:
        answer_variable = str(step.get("answer_variable") or "value")
        known = str(step.get("known") or step.get("source_label") or "source")
        ask = str(step.get("ask") or step.get("target_label") or "value")
        relation_hint = str(step.get("relation_hint") or "").strip()
        if relation_hint:
            parts.append(f"{answer_variable}: {ask} of {known} ({relation_hint})")
        else:
            parts.append(f"{answer_variable}: {ask} of {known}")
    return "; ".join(parts)


def _fallback_semantic_edge_question(
    source_display: str,
    target_node: SemanticASTNode | None,
) -> str:
    target = target_node.label if target_node is not None else "the target"
    if target_node is not None and target_node.kind in {"type_variable", "implicit_type_variable"}:
        return f"What is the {target} of {source_display}?"
    return f"What is the {target} related to {source_display}?"


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
                if node.operator:
                    node.question = _repair_operator_question_from_metadata(original_question, node, variable_descriptions)
                else:
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


def _repair_operator_question_from_metadata(
    original_question: str,
    node: AtomicQuestionNode,
    variable_descriptions: dict[str, str],
) -> str:
    if original_question and not contains_bare_variable(original_question):
        return original_question
    input_descriptions = [variable_descriptions.get(item, item) for item in node.inputs]
    return _operator_question(node.operator or "NONE", input_descriptions)


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
