from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import (
    AnchorSelectionResult,
    ASTSkeleton,
    AtomicQuestionDAG,
    AtomicSubquestion,
    CandidateNode,
    CandidatePath,
    MaskReplacement,
    MaskSpanResult,
    ProblemFrame,
    QuestionRecord,
    RestoredAnchorConnectedSubgraph,
    RestoredGraphNodeCandidate,
    SelectedPath,
    EntityOriginPath,
    EntityStartNode,
    SelectedEntityPath,
    SemanticNormalizationResult,
    SemanticASTResult,
)

if TYPE_CHECKING:
    from anchor_selector import AnchorSelector
    from ast_builder import SemanticASTOptimizer
    from corenlp_parser import CoreNLPParser
    from graph_builder import GraphBuilder
    from mask_span_extractor import MaskSpanExtractor
    from entity_path_pipeline import EntityPathSemanticParser
    from question_normalizer import SemanticQuestionNormalizer
    from subquestion_generator import SubquestionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEPO decomposition with mask-only parsing, restored anchor selection, semantic AST, and one-hop subquestions."
    )
    parser.add_argument("--question", help="Run one manually supplied question instead of questions.json.")
    parser.add_argument("--questions-file", default="questions.json", help="Path to questions.json.")
    parser.add_argument("--api-key", help="OpenAI API key. Used only if OPENAI_API_KEY is not set.")
    parser.add_argument("--base-url", help="OpenAI base URL. Used only if OPENAI_BASE_URL is not set.")
    parser.add_argument(
        "--corenlp-url",
        default="http://localhost:9000",
        help="Endpoint used by Stanza CoreNLPClient for the managed CoreNLP server.",
    )
    parser.add_argument("--corenlp-memory", default="4G", help="Java heap memory for managed CoreNLP.")
    parser.add_argument(
        "--corenlp-home",
        help="Path to a Stanford CoreNLP directory containing stanford-corenlp*.jar files.",
    )
    parser.add_argument(
        "--corenlp-timeout-ms",
        type=int,
        default=60000,
        help="CoreNLP annotation timeout in milliseconds.",
    )
    parser.add_argument("--debug", action="store_true", help="Print detailed intermediate structures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    records = [QuestionRecord(question=args.question)] if args.question else read_questions(args.questions_file)

    try:
        from corenlp_parser import CoreNLPConnectionError, CoreNLPParser
        from graph_builder import GraphBuilder
        from llm_client import LLMClient
        from mask_span_extractor import MaskSpanExtractor
        from entity_path_pipeline import EntityPathSemanticParser
        from question_normalizer import SemanticQuestionNormalizer
        from subquestion_generator import SubquestionGenerator

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
        question_normalizer = SemanticQuestionNormalizer(llm_client)
        mask_span_extractor = MaskSpanExtractor(llm_client)
        graph_builder = GraphBuilder()
        path_semantic_parser = EntityPathSemanticParser(llm_client)
        subquestion_generator = SubquestionGenerator(llm_client)

        with CoreNLPParser(
            args.corenlp_url,
            timeout_ms=args.corenlp_timeout_ms,
            memory=args.corenlp_memory,
            corenlp_home=args.corenlp_home,
        ) as parser:
            for index, record in enumerate(records, start=1):
                result = run_pipeline(
                    record=record,
                    index=index,
                    mask_span_extractor=mask_span_extractor,
                    parser=parser,
                    graph_builder=graph_builder,
                    anchor_selector=None,
                    semantic_ast_optimizer=None,
                    subquestion_generator=subquestion_generator,
                    question_normalizer=question_normalizer,
                    path_semantic_parser=path_semantic_parser,
                    debug=args.debug,
                )
                print_result(index, record, result, debug=args.debug)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (CoreNLPConnectionError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def run_pipeline(
    record: QuestionRecord,
    index: int,
    mask_span_extractor: "MaskSpanExtractor",
    parser: "CoreNLPParser",
    graph_builder: "GraphBuilder",
    anchor_selector: "AnchorSelector",
    semantic_ast_optimizer: "SemanticASTOptimizer",
    subquestion_generator: "SubquestionGenerator",
    question_normalizer: "SemanticQuestionNormalizer | None" = None,
    path_semantic_parser: "EntityPathSemanticParser | None" = None,
    debug: bool = False,
) -> dict[str, Any]:
    del index, debug
    del anchor_selector, semantic_ast_optimizer
    from placeholder import selective_entity_masking
    from entity_path_pipeline import EntityPathSemanticParser
    from entity_path_projector import (
        enumerate_entity_origin_paths,
        extract_entity_start_nodes,
        undirected_graph_edge_payloads,
    )
    from path_projector import (
        build_undirected_dependency_graph,
    )

    semantic_normalization = (
        question_normalizer.normalize(record.question)
        if question_normalizer is not None
        else SemanticNormalizationResult(
            original_question=record.question,
            normalized_question=record.question,
            changed=False,
        )
    )
    processing_question = semantic_normalization.normalized_question
    mask_spans = mask_span_extractor.extract(processing_question)
    replacement = selective_entity_masking(
        original_question=processing_question,
        extracted_nodes=mask_spans,
    )
    dependency_parse = parser.parse(replacement.masked_question)
    graph_node_candidates = graph_builder.build_graph_node_candidates(
        dependency_parse=dependency_parse,
        replacement=replacement,
    )
    restored_graph_node_candidates = graph_builder.restore_graph_node_candidates(
        graph_node_candidates=graph_node_candidates,
        replacement=replacement,
    )

    dependency_graph = build_undirected_dependency_graph(
        dependency_parse=dependency_parse,
        restored_graph_node_candidates=restored_graph_node_candidates,
    )
    if path_semantic_parser is None:
        path_semantic_parser = EntityPathSemanticParser(getattr(subquestion_generator, "llm_client", None))

    entity_start_nodes = extract_entity_start_nodes(
        dependency_graph=dependency_graph,
        restored_graph_node_candidates=restored_graph_node_candidates,
        replacement=replacement,
    )
    if not entity_start_nodes:
        raise ValueError("No entity start nodes found for entity-origin path pipeline.")

    entity_origin_paths = enumerate_entity_origin_paths(
        dependency_graph=dependency_graph,
        entity_starts=entity_start_nodes,
    )
    if not entity_origin_paths:
        raise ValueError("No entity-origin dependency paths were enumerated.")

    selected_entity_paths, selected_paths_payload = path_semantic_parser.select_entity_paths(
        original_question=record.question,
        restored_question=processing_question,
        entity_start_nodes=entity_start_nodes,
        entity_origin_paths=entity_origin_paths,
    )
    graph_edge_payloads = undirected_graph_edge_payloads(dependency_graph)
    semantic_ast, path_pruned_ast_payload = path_semantic_parser.build_path_pruned_ast(
        original_question=record.question,
        restored_question=processing_question,
        selected_entity_paths=selected_entity_paths,
        entity_origin_paths=entity_origin_paths,
        undirected_graph_edges=graph_edge_payloads,
    )

    subquestion_dag: AtomicQuestionDAG | None = None
    generate_dag = getattr(subquestion_generator, "generate_dag", None)
    if callable(generate_dag):
        subquestion_dag = generate_dag(
            original_question=processing_question,
            semantic_ast=semantic_ast,
        )
        subquestions = subquestion_dag.to_subquestions()
    else:
        subquestions = subquestion_generator.generate(
            original_question=processing_question,
            ast=semantic_ast,
        )
    return {
        "semantic_normalization": semantic_normalization,
        "mask_spans": mask_spans,
        "replacement": replacement,
        "dependency_parse": dependency_parse,
        "weighted_graph": dependency_graph,
        "graph_node_candidates": graph_node_candidates,
        "restored_graph_node_candidates": restored_graph_node_candidates,
        "dependency_graph": dependency_graph,
        "entity_start_nodes": entity_start_nodes,
        "entity_origin_paths": entity_origin_paths,
        "selected_entity_paths": selected_entity_paths,
        "selected_paths_payload": selected_paths_payload,
        "path_pruned_ast_payload": path_pruned_ast_payload,
        "semantic_ast": semantic_ast,
        "subquestions": subquestions,
        "subquestion_dag": subquestion_dag,
        "candidate_nodes": [],
        "problem_frame": None,
        "candidate_frame_payload": None,
        "candidate_projected_graph": None,
        "enumerated_candidate_paths": [],
        "filtered_candidate_paths": [],
        "selected_paths": [],
        "selected_path_repair_actions": [],
        "ast_skeleton": None,
        "relation_label_payload": None,
    }


def print_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    from graph_builder import format_undirected_graph_edges

    semantic_normalization: SemanticNormalizationResult = result["semantic_normalization"]
    mask_spans: MaskSpanResult = result["mask_spans"]
    replacement: MaskReplacement = result["replacement"]
    dependency_parse = result["dependency_parse"]
    dependency_graph = result.get("dependency_graph", result["weighted_graph"])
    restored_graph_node_candidates: list[RestoredGraphNodeCandidate] = result["restored_graph_node_candidates"]
    entity_start_nodes: list[EntityStartNode] = result["entity_start_nodes"]
    entity_origin_paths: list[EntityOriginPath] = result["entity_origin_paths"]
    selected_entity_paths: list[SelectedEntityPath] = result["selected_entity_paths"]
    semantic_ast: SemanticASTResult = result["semantic_ast"]
    subquestions: list[AtomicSubquestion] = result["subquestions"]
    subquestion_dag: AtomicQuestionDAG | None = result.get("subquestion_dag")

    separator = "=" * 60
    print(separator)
    title = f"Question {index}"
    if record.qid:
        title += f" ({record.qid})"
    print(title)
    print(separator)
    print()

    print("[Original Question]")
    print(record.question)
    print()

    print("[1. Semantic-Normalized Question]")
    print(semantic_normalization.normalized_question)
    if debug:
        _print_warnings(semantic_normalization.warnings)
    print()

    print("[2. Mask Spans]")
    if replacement.mask_mappings:
        for mapping in replacement.mask_mappings:
            print(f"  - {mapping.original_text} -> {mapping.placeholder}")
    else:
        print("  (none)")
    if debug:
        _print_warnings(mask_spans.warnings)
    print()

    print("[3. Selective Masked Question]")
    print(replacement.masked_question)
    print()

    print("[4. CoreNLP Dependency Parse]")
    print("Edges:")
    if dependency_parse.edges:
        for edge in dependency_parse.edges:
            print(f"  - {edge.display()}")
    else:
        print("  (no dependency edges)")
    print()

    print("[5. Undirected Dependency Graph]")
    graph_lines = format_undirected_graph_edges(dependency_graph)
    if graph_lines:
        for line in graph_lines:
            print(line)
    else:
        print("  (no dependency graph edges)")
    print()

    print("[6. Entity Start Nodes]")
    _print_entity_start_nodes(entity_start_nodes)
    print()

    print("[7. Entity-Origin Dependency Paths]")
    _print_entity_origin_paths(entity_origin_paths, include_evidence=debug)
    print()

    print("[8. LLM Selected Entity Paths]")
    _print_selected_entity_paths(selected_entity_paths, entity_origin_paths)
    print()

    print("[9. Path-Pruned Semantic AST]")
    _print_semantic_ast(semantic_ast)
    if debug:
        _print_warnings(semantic_ast.warnings)
    print()

    print("[10. Atomic Subquestion DAG]")
    if subquestion_dag is not None:
        _print_atomic_question_dag(subquestion_dag)
    elif not subquestions:
        print("  (no atomic subquestions generated)")
    else:
        for item in subquestions:
            print(f"  q{item.index}: {item.question}")
    print()


def _format_restored_subgraph_edges(
    restored_anchor_connected_subgraph: RestoredAnchorConnectedSubgraph,
) -> list[str]:
    lines: list[str] = []
    for edge in restored_anchor_connected_subgraph.edges:
        source = edge.get("source")
        target = edge.get("target")
        source_text = edge.get("source_text", source)
        target_text = edge.get("target_text", target)
        relation = edge.get("relation") or "|".join(edge.get("relations", []))
        relation_text = relation or "related"
        lines.append(f"  - {source_text}[{source}] --{relation_text}--> {target_text}[{target}]")
    return lines


def _print_entity_start_nodes(entity_start_nodes: list[EntityStartNode]) -> None:
    if not entity_start_nodes:
        print("  (none)")
        return
    for entity in entity_start_nodes:
        semantic = f" semantic_type={entity.semantic_type_hint}" if entity.semantic_type_hint else ""
        tokens = f" token_ids={entity.token_ids}" if entity.token_ids else ""
        print(f"  - {entity.entity_id}: {entity.text} graph_node_ids={entity.graph_node_ids}{tokens}{semantic}")


def _print_entity_origin_paths(
    entity_origin_paths: list[EntityOriginPath],
    *,
    include_evidence: bool = False,
) -> None:
    if not entity_origin_paths:
        print("  (none)")
        return
    for path in entity_origin_paths:
        print(f"  - {path.path_id} ({path.entity_id}): {' -- '.join(path.nodes)}")
        if include_evidence:
            for evidence in path.evidence:
                relations = "/".join(str(item) for item in evidence.get("relations", []) if item)
                relation_text = f" relations={relations}" if relations else ""
                print(
                    f"    {evidence.get('source_text', evidence.get('source'))}"
                    f" -> {evidence.get('target_text', evidence.get('target'))}{relation_text}"
                )


def _print_selected_entity_paths(
    selected_entity_paths: list[SelectedEntityPath],
    entity_origin_paths: list[EntityOriginPath],
) -> None:
    if not selected_entity_paths:
        print("  (none)")
        return
    path_by_id = {path.path_id: path for path in entity_origin_paths}
    for selected in selected_entity_paths:
        path = path_by_id.get(selected.path_id)
        path_text = f" {' -- '.join(path.nodes)}" if path is not None else ""
        reason = f" reason={selected.reason}" if selected.reason else ""
        print(f"  - {selected.entity_id}: {selected.path_id}{path_text}{reason}")


def _print_candidate_nodes(candidate_nodes: list[CandidateNode]) -> None:
    if not candidate_nodes:
        print("  (none)")
        return
    for node in candidate_nodes:
        grounding = f" graph_node_ids={node.graph_node_ids}" if node.graph_node_ids else ""
        print(f"  - {node.id}: {node.text} kind={node.kind} confidence={node.confidence}{grounding}")


def _print_problem_frame(problem_frame: ProblemFrame) -> None:
    if problem_frame.answer_mode:
        print(f"Answer mode: {problem_frame.answer_mode}")
    if problem_frame.answer_focus:
        print(f"Answer focus: {problem_frame.answer_focus}")
    if problem_frame.notes:
        print(f"Notes: {problem_frame.notes}")
    print("Requirements:")
    if not problem_frame.requirements:
        print("  (none)")
        return
    for requirement in problem_frame.requirements:
        description = f" - {requirement.description}" if requirement.description else ""
        context = f" context={requirement.context}" if requirement.context else ""
        print(f"  - {requirement.id}: {requirement.root} -> {requirement.target}{context}{description}")


def _print_candidate_paths(candidate_paths: list[CandidatePath]) -> None:
    if not candidate_paths:
        print("  (none)")
        return
    for path in candidate_paths:
        candidate_for = ", ".join(path.candidate_for) if path.candidate_for else "unassigned"
        print(f"  - {path.path_id}: {' -- '.join(path.nodes)} candidate_for={candidate_for}")
        for evidence in path.evidence:
            evidence_text = " -- ".join(str(item) for item in evidence.get("evidence_text_path", []))
            source = evidence.get("source_text", evidence.get("source", ""))
            target = evidence.get("target_text", evidence.get("target", ""))
            print(f"    evidence {source} -> {target}: {evidence_text}")


def _print_selected_paths(selected_paths: list[SelectedPath]) -> None:
    if not selected_paths:
        print("  (none)")
        return
    for selected in selected_paths:
        print(f"  - {selected.requirement_id}: {selected.path_id}")


def _print_repair_actions(actions: list[str]) -> None:
    if not actions:
        return
    print("Repair actions:")
    for action in actions:
        print(f"  - {action}")


def _print_ast_skeleton(ast_skeleton: ASTSkeleton) -> None:
    print("Branch terminals:")
    if ast_skeleton.branch_terminals:
        for requirement_id, terminal in ast_skeleton.branch_terminals.items():
            print(f"  - {requirement_id}: {terminal}")
    else:
        print("  (none)")
    print("Nodes:")
    if ast_skeleton.nodes:
        for node in ast_skeleton.nodes:
            branch = f" branch_of={node.branch_of}" if node.branch_of else ""
            print(f"  - {node.id}: {node.label} kind={node.kind}{branch}")
    else:
        print("  (none)")
    print("Edges:")
    if ast_skeleton.edges:
        for edge in ast_skeleton.edges:
            support = f" support={' -- '.join(edge.support_path)}" if edge.support_path else ""
            print(f"  - {edge.source} -> {edge.target}{support}")
    else:
        print("  (none)")


def _print_semantic_ast(semantic_ast: SemanticASTResult) -> None:
    print("Nodes:")
    if semantic_ast.nodes:
        for node in semantic_ast.nodes:
            print(f"  - {node.id}: {node.label}")
    else:
        print("  (none)")
    print("Edges:")
    if semantic_ast.edges:
        for edge in semantic_ast.edges:
            hint = f" ({edge.relation_hint})" if edge.relation_hint else ""
            print(f"  - {edge.source} -> {edge.target}{hint}")
    else:
        print("  (none)")
    if semantic_ast.detected_cue_frame:
        frame = semantic_ast.detected_cue_frame
        cue = frame.get("cue_text", "")
        slot = frame.get("expected_value_slot", "")
        print(f"Detected cue frame: cue={cue} expected_value_slot={slot}")
    if semantic_ast.retry_count:
        print(f"Retry count: {semantic_ast.retry_count}")
    if semantic_ast.validation_warnings:
        print("Validation warnings:")
        for warning in semantic_ast.validation_warnings:
            print(f"  - {warning}")
    if semantic_ast.fallback_repair_actions:
        print("Fallback repair actions:")
        for action in semantic_ast.fallback_repair_actions:
            print(f"  - {action}")


def _print_atomic_question_dag(dag: AtomicQuestionDAG) -> None:
    if not dag.nodes:
        print("  (no atomic subquestions generated)")
        return

    print("Nodes:")
    for node in dag.nodes:
        inputs = ", ".join(node.inputs) if node.inputs else "none"
        depends_on = ", ".join(node.depends_on) if node.depends_on else "none"
        print(f"  - {node.id} [{node.type}]: inputs=({inputs}) -> {node.output}; depends_on={depends_on}")
        print(f"    {node.question}")
        if node.candidate_bindings:
            print(f"    candidate_bindings={node.candidate_bindings}")

    print("Edges:")
    if dag.edges:
        for edge in dag.edges:
            print(f"  - {edge.source} -> {edge.target} via {edge.variable}")
    else:
        print("  (none)")
    if dag.warnings:
        print("Warnings:")
        for warning in dag.warnings:
            print(f"  - {warning}")


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print("Warnings:")
    for warning in warnings:
        print(f"  - {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
