from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import (
    AtomicQuestionDAG,
    AtomicSubquestion,
    MaskReplacement,
    MaskSpan,
    MaskSpanResult,
    ExplicitEntity,
    ExplicitEntityResult,
    QuestionRecord,
    RestoredGraphNodeCandidate,
    EntityOriginPath,
    EntityStartNode,
    PathSetCandidate,
    ScoredEntityPath,
    SemanticNormalizationResult,
)

if TYPE_CHECKING:
    from anchor_selector import AnchorSelector
    from ast_builder import SemanticASTOptimizer
    from corenlp_parser import CoreNLPParser
    from graph_builder import GraphBuilder
    from mask_span_extractor import ExplicitEntityExtractor, MaskSpanExtractor
    from entity_path_pipeline import (
        EntityPathSemanticParser,
        build_selected_dependency_path_evidence,
        build_path_set_candidates,
        select_top_paths_by_entity,
    )
    from question_normalizer import SemanticQuestionNormalizer
    from subquestion_generator import SubquestionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEPO decomposition with explicit entity masking, grounded paths, and grounded atomic subquestions."
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
        from mask_span_extractor import ExplicitEntityExtractor
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
        build_selected_dependency_path_evidence,
        build_path_set_candidates,
        select_top_paths_by_entity,
    )
    from entity_path_projector import (
        build_entity_start_nodes_from_explicit_entities,
        enumerate_entity_origin_paths,
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

    selected_dependency_path_evidence = build_selected_dependency_path_evidence(
        path_set_candidates=path_set_candidates,
        entity_origin_paths=entity_origin_paths,
        max_path_sets=4,
    )
    subquestion_dag, grounded_atomic_dag_payload = path_semantic_parser.build_grounded_atomic_dag(
        original_question=record.question,
        selected_dependency_path_evidence=selected_dependency_path_evidence,
    )
    subquestions = subquestion_dag.to_subquestions()
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
        "selected_dependency_path_evidence": selected_dependency_path_evidence,
        "grounded_atomic_dag_payload": grounded_atomic_dag_payload,
        "subquestions": subquestions,
        "subquestion_dag": subquestion_dag,
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
    selected_dependency_path_evidence: list[dict[str, Any]] = result.get("selected_dependency_path_evidence", [])
    grounded_atomic_dag_payload: dict[str, Any] = result.get("grounded_atomic_dag_payload") or {}
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

    print("[9. Grounded Atomic DAG Generation]")
    print("Inputs:")
    print(f"  Original question: {record.question}")
    _print_selected_dependency_path_evidence(selected_dependency_path_evidence)
    print("Output:")
    _print_grounded_atomic_dag_payload(grounded_atomic_dag_payload)
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


def _print_selected_dependency_path_evidence(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        print("  Selected dependency path evidence: (none)")
        return
    print("  Selected dependency path evidence:")
    for path_set in evidence:
        if not isinstance(path_set, dict):
            continue
        path_set_id = path_set.get("path_set_id")
        print(f"    - {path_set_id}:")
        paths = path_set.get("paths") or []
        if not isinstance(paths, list) or not paths:
            print("      (no paths)")
            continue
        for path in paths:
            if not isinstance(path, dict):
                continue
            print(f"      - {path.get('path_id')}: {path.get('path_text')}")


def _print_grounded_atomic_dag_payload(payload: dict[str, Any]) -> None:
    if not payload:
        print("  (none)")
        return
    reason = str(payload.get("reason") or "").strip()
    selected_path_sets = payload.get("selected_path_set_ids")
    if selected_path_sets:
        print(f"  selected_path_set_ids={selected_path_sets}")
    if reason:
        print(f"  reason={reason}")
    nodes = payload.get("nodes", [])
    if isinstance(nodes, list) and nodes:
        print("  Nodes:")
        for node in nodes:
            if not isinstance(node, dict):
                continue
            dependencies = node.get("dependencies") or []
            support = node.get("support") or []
            support_ids = []
            if isinstance(support, list):
                support_ids = [str(item.get("path_id")) for item in support if isinstance(item, dict) and item.get("path_id")]
            print(f"    - {node.get('node_id')}: depends_on={dependencies or 'none'} support={support_ids or 'none'}")
            print(f"      {node.get('question')}")
    warnings = payload.get("normalization_warnings") or []
    if warnings:
        print("  Warnings:")
        for warning in warnings:
            print(f"    - {warning}")


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
        support_ids = node.metadata.get("support_path_ids") if isinstance(node.metadata, dict) else None
        if support_ids:
            print(f"    support_path_ids={support_ids}")
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
