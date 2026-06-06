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
    MaskSpan,
    MaskSpanResult,
    ExplicitEntity,
    ExplicitEntityResult,
    ProblemFrame,
    QuestionRecord,
    RestoredAnchorConnectedSubgraph,
    RestoredGraphNodeCandidate,
    SelectedPath,
    EntityOriginPath,
    EntityStartNode,
    CandidateSemanticAST,
    PathSetCandidate,
    ScoredEntityPath,
    SelectedEntityPath,
    SemanticNormalizationResult,
    SemanticASTResult,
)

if TYPE_CHECKING:
    from anchor_selector import AnchorSelector
    from ast_builder import SemanticASTOptimizer
    from corenlp_parser import CoreNLPParser
    from graph_builder import GraphBuilder
    from mask_span_extractor import ExplicitEntityExtractor, MaskSpanExtractor
    from entity_path_pipeline import (
        EntityPathSemanticParser,
        build_path_set_candidates,
        select_top_paths_by_entity,
    )
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
        mask_span_extractor = ExplicitEntityExtractor(llm_client)
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
    from entity_path_pipeline import (
        EntityPathSemanticParser,
        build_path_set_candidates,
        select_top_paths_by_entity,
    )
    from entity_path_projector import (
        build_entity_start_nodes_from_explicit_entities,
        enumerate_entity_origin_paths,
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
    explicit_entities = _coerce_explicit_entity_result(
        mask_span_extractor.extract(processing_question),
    )
    mask_spans = _mask_span_result_from_explicit_entities(explicit_entities)
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

    entity_start_nodes = build_entity_start_nodes_from_explicit_entities(
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

    graph_edge_payloads = undirected_graph_edge_payloads(dependency_graph)
    scored_entity_paths, path_scoring_payload = path_semantic_parser.score_entity_paths(
        original_question=record.question,
        restored_question=processing_question,
        entity_start_nodes=entity_start_nodes,
        entity_origin_paths=entity_origin_paths,
    )

    top_paths_by_entity = select_top_paths_by_entity(
        scored_paths=scored_entity_paths,
        entity_start_nodes=entity_start_nodes,
        entity_origin_paths=entity_origin_paths,
        top_k=2,
    )
    path_set_candidates = build_path_set_candidates(
        top_paths_by_entity=top_paths_by_entity,
        max_path_sets=16,
    )
    if not path_set_candidates:
        raise ValueError("No candidate path sets were constructed for entity-origin path pipeline.")

    candidate_asts = path_semantic_parser.build_candidate_semantic_asts(
        original_question=record.question,
        restored_question=processing_question,
        path_set_candidates=path_set_candidates,
        entity_origin_paths=entity_origin_paths,
        scored_paths=scored_entity_paths,
        undirected_graph_edges=graph_edge_payloads,
    )
    semantic_ast, best_ast_selection_payload = path_semantic_parser.select_best_candidate_ast(
        original_question=record.question,
        restored_question=processing_question,
        entity_start_nodes=entity_start_nodes,
        path_set_candidates=path_set_candidates,
        scored_paths=scored_entity_paths,
        candidate_asts=candidate_asts,
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
        "explicit_entities": explicit_entities,
        "explicit_entity_payload": explicit_entities.raw_payload,
        "mask_spans": mask_spans,
        "masked_question": replacement.masked_question,
        "entity_mask_mappings": replacement.mask_mappings,
        "replacement": replacement,
        "dependency_parse": dependency_parse,
        "weighted_graph": dependency_graph,
        "graph_node_candidates": graph_node_candidates,
        "restored_graph_node_candidates": restored_graph_node_candidates,
        "dependency_graph": dependency_graph,
        "entity_start_nodes": entity_start_nodes,
        "entity_origin_paths": entity_origin_paths,
        "scored_entity_paths": scored_entity_paths,
        "path_scoring_payload": path_scoring_payload,
        "top_paths_by_entity": top_paths_by_entity,
        "path_set_candidates": path_set_candidates,
        "candidate_asts": candidate_asts,
        "best_ast_selection_payload": best_ast_selection_payload,
        "selected_entity_paths": [],
        "selected_paths_payload": None,
        "selected_path_semantic_transduction_payload": None,
        "path_pruned_ast_payload": best_ast_selection_payload,
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


def _coerce_explicit_entity_result(raw_result: Any) -> ExplicitEntityResult:
    if isinstance(raw_result, ExplicitEntityResult):
        return raw_result
    if isinstance(raw_result, MaskSpanResult):
        return ExplicitEntityResult(
            entities=[
                ExplicitEntity(
                    text=span.text,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    semantic_type_hint=span.semantic_type_hint or "Entity",
                    confidence=1.0,
                    reason=span.reason,
                )
                for span in raw_result.mask_spans
                if span.kind_hint == "entity"
            ],
            warnings=list(raw_result.warnings),
            raw_payload=raw_result.raw_payload,
        )
    raise TypeError("Step 2 extractor must return ExplicitEntityResult or MaskSpanResult.")


def _mask_span_result_from_explicit_entities(explicit_entities: ExplicitEntityResult) -> MaskSpanResult:
    return MaskSpanResult(
        mask_spans=[
            MaskSpan(
                text=entity.text,
                start_char=entity.start_char,
                end_char=entity.end_char,
                kind_hint="entity",
                semantic_type_hint=entity.semantic_type_hint or "Entity",
                reason=entity.reason,
            )
            for entity in explicit_entities.entities
        ],
        warnings=list(explicit_entities.warnings),
        raw_payload=explicit_entities.raw_payload,
    )


def print_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    from graph_builder import format_undirected_graph_edges

    semantic_normalization: SemanticNormalizationResult = result["semantic_normalization"]
    explicit_entities: ExplicitEntityResult = result.get("explicit_entities") or ExplicitEntityResult()
    mask_spans: MaskSpanResult = result["mask_spans"]
    replacement: MaskReplacement = result["replacement"]
    dependency_parse = result["dependency_parse"]
    dependency_graph = result.get("dependency_graph", result["weighted_graph"])
    restored_graph_node_candidates: list[RestoredGraphNodeCandidate] = result["restored_graph_node_candidates"]
    entity_start_nodes: list[EntityStartNode] = result["entity_start_nodes"]
    entity_origin_paths: list[EntityOriginPath] = result["entity_origin_paths"]
    scored_entity_paths: list[ScoredEntityPath] = result.get("scored_entity_paths", [])
    top_paths_by_entity: dict[str, list[ScoredEntityPath]] = result.get("top_paths_by_entity", {})
    path_set_candidates: list[PathSetCandidate] = result.get("path_set_candidates", [])
    candidate_asts: list[CandidateSemanticAST] = result.get("candidate_asts", [])
    best_ast_selection_payload: dict[str, Any] = result.get("best_ast_selection_payload") or {}
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

    print("[2. Explicit Entities]")
    if explicit_entities.entities:
        for entity in explicit_entities.entities:
            semantic = f" [{entity.semantic_type_hint}]" if entity.semantic_type_hint else ""
            print(
                f"  - {entity.text}{semantic} span=({entity.start_char}, {entity.end_char}) "
                f"confidence={entity.confidence:.2f}"
            )
    else:
        print("  (none)")
    if debug:
        _print_warnings(explicit_entities.warnings or mask_spans.warnings)
    print()

    print("[3. Entity Masking]")
    if replacement.mask_mappings:
        for mapping in replacement.mask_mappings:
            print(f"  - {mapping.placeholder} -> {mapping.original_text}")
    else:
        print("  (none)")
    print("Masked question:")
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

    print("[6. Entity Start Nodes from Explicit Entities]")
    _print_entity_start_nodes(entity_start_nodes)
    print()

    print("[7. Entity-Origin Dependency Paths]")
    _print_entity_origin_paths(entity_origin_paths, include_evidence=debug)
    print()

    print("[8. LLM Path Scores]")
    _print_scored_entity_paths(scored_entity_paths, entity_origin_paths)
    print()

    print("[8.1 Top-2 Paths per Entity]")
    _print_top_paths_by_entity(top_paths_by_entity, entity_origin_paths)
    print()

    print("[8.2 Candidate Path Sets]")
    _print_path_set_candidates(path_set_candidates)
    print()

    print("[9. Candidate Path-Set Semantic ASTs]")
    _print_candidate_semantic_asts(candidate_asts)
    print()

    print("[10. LLM Best AST Selection]")
    _print_best_ast_selection(best_ast_selection_payload)
    print("Selected Semantic AST:")
    _print_semantic_ast(semantic_ast)
    if debug:
        _print_warnings(semantic_ast.warnings)
    print()

    print("[11. Atomic Subquestion DAG]")
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


def _print_scored_entity_paths(
    scored_paths: list[ScoredEntityPath],
    entity_origin_paths: list[EntityOriginPath],
) -> None:
    if not scored_paths:
        print("  (none)")
        return
    path_by_id = {path.path_id: path for path in entity_origin_paths}
    for score in sorted(scored_paths, key=lambda item: (_entity_sort_key_for_print(item.entity_id), -item.score, item.path_id)):
        path = path_by_id.get(score.path_id)
        path_text = f" {' -- '.join(path.nodes)}" if path is not None else ""
        terminal = f" terminal_hint={score.terminal_hint}" if score.terminal_hint else ""
        chain = f" semantic_chain_hint={score.semantic_chain_hint}" if score.semantic_chain_hint else ""
        reason = f" reason={score.reason}" if score.reason else ""
        print(
            f"  - {score.entity_id}: {score.path_id} score={score.score:.1f} "
            f"valid={score.valid}{terminal}{chain}{path_text}{reason}"
        )


def _print_top_paths_by_entity(
    top_paths_by_entity: dict[str, list[ScoredEntityPath]],
    entity_origin_paths: list[EntityOriginPath],
) -> None:
    if not top_paths_by_entity:
        print("  (none)")
        return
    path_by_id = {path.path_id: path for path in entity_origin_paths}
    for entity_id in sorted(top_paths_by_entity, key=_entity_sort_key_for_print):
        parts = []
        for score in top_paths_by_entity[entity_id]:
            path = path_by_id.get(score.path_id)
            path_text = f" ({' -- '.join(path.nodes)})" if path is not None else ""
            parts.append(f"{score.path_id}:{score.score:.1f}{path_text}")
        print(f"  - {entity_id}: {', '.join(parts)}")


def _print_path_set_candidates(path_set_candidates: list[PathSetCandidate]) -> None:
    if not path_set_candidates:
        print("  (none)")
        return
    for candidate in path_set_candidates:
        mapping = ", ".join(
            f"{entity_id}={path_id}"
            for entity_id, path_id in sorted(candidate.path_ids_by_entity.items(), key=lambda item: _entity_sort_key_for_print(item[0]))
        )
        print(f"  - {candidate.path_set_id}: {mapping}; mean_path_score={candidate.mean_path_score:.1f}")


def _print_candidate_semantic_asts(candidate_asts: list[CandidateSemanticAST]) -> None:
    if not candidate_asts:
        print("  (none)")
        return
    for candidate in candidate_asts:
        print(f"  - {candidate.candidate_id} ({candidate.path_set_id}) paths={candidate.path_ids_by_entity}")
        if candidate.parse_error:
            print(f"    parse_error={candidate.parse_error}")
        if candidate.generation_error:
            print(f"    generation_error={candidate.generation_error}")
        if candidate.semantic_ast is None:
            continue
        print("    Nodes:")
        if candidate.semantic_ast.nodes:
            for node in candidate.semantic_ast.nodes:
                print(f"      - {node.id}: {node.label}")
        else:
            print("      (none)")
        print("    Edges:")
        if candidate.semantic_ast.edges:
            for edge in candidate.semantic_ast.edges:
                hint = f" ({edge.relation_hint})" if edge.relation_hint else ""
                print(f"      - {edge.source} -> {edge.target}{hint}")
        else:
            print("      (none)")


def _print_best_ast_selection(payload: dict[str, Any]) -> None:
    if not payload:
        print("  (none)")
        return
    reviews = payload.get("ast_reviews", [])
    if isinstance(reviews, list) and reviews:
        for review in reviews:
            if not isinstance(review, dict):
                continue
            fatal = review.get("fatal_errors") or []
            fatal_text = f" fatal_errors={fatal}" if fatal else ""
            reason = f" reason={review.get('reason')}" if review.get("reason") else ""
            print(
                f"  - {review.get('candidate_id')} ({review.get('path_set_id')}): "
                f"score={review.get('score')} valid={review.get('valid_for_decomposition')}{fatal_text}{reason}"
            )
    else:
        print("  Reviews: (none)")
    best = payload.get("best_candidate_id")
    selected = payload.get("selected_candidate_id")
    fallback = payload.get("selection_fallback")
    print(f"  best_candidate_id={best}")
    if selected:
        print(f"  selected_candidate_id={selected}")
    if fallback:
        print(f"  selection_fallback={fallback}")


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


def _entity_sort_key_for_print(entity_id: str) -> tuple[int, str]:
    text = str(entity_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10**9, text)


if __name__ == "__main__":
    raise SystemExit(main())
