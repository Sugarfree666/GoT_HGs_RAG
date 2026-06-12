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
    SemanticReasoningPathResult,
    SemanticNormalizationResult,
)

if TYPE_CHECKING:
    from corenlp_parser import CoreNLPParser
    from graph_builder import GraphBuilder
    from mask_span_extractor import ExplicitEntityExtractor, MaskSpanExtractor
    from entity_path_pipeline import (
        EntityPathSemanticParser,
        build_selected_dependency_path_evidence,
        build_single_path_set_candidate,
        select_best_path_by_entity,
    )
    from question_normalizer import SemanticQuestionNormalizer


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

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
        question_normalizer = SemanticQuestionNormalizer(llm_client)
        mask_span_extractor = ExplicitEntityExtractor(llm_client)
        graph_builder = GraphBuilder()
        path_semantic_parser = EntityPathSemanticParser(llm_client)

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
    question_normalizer: "SemanticQuestionNormalizer | None" = None,
    path_semantic_parser: "EntityPathSemanticParser | None" = None,
    debug: bool = False,
) -> dict[str, Any]:
    del index, debug
    from placeholder import selective_entity_masking
    from entity_path_pipeline import (
        build_selected_dependency_path_evidence,
        build_single_path_set_candidate,
        select_best_path_by_entity,
    )
    from entity_path_projector import (
        build_entity_start_nodes_from_explicit_entities,
        enumerate_entity_origin_paths,
        prune_terminal_glue_paths,
    )
    from path_projector import (
        build_undirected_dependency_graph,
    )
    from dependency_graph_collapser import collapse_dependency_graph

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

    raw_dependency_graph = build_undirected_dependency_graph(
        dependency_parse=dependency_parse,
        restored_graph_node_candidates=restored_graph_node_candidates,
    )
    dependency_graph = collapse_dependency_graph(raw_dependency_graph)
    if path_semantic_parser is None:
        raise TypeError("run_pipeline requires path_semantic_parser.")

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

    pruned_entity_origin_paths, path_pruning_stats = prune_terminal_glue_paths(
        entity_origin_paths=entity_origin_paths,
        dependency_graph=dependency_graph,
        entity_start_nodes=entity_start_nodes,
    )
    if not pruned_entity_origin_paths:
        raise ValueError("No entity-origin dependency paths remained after terminal glue pruning.")

    scored_entity_paths, path_scoring_payload = path_semantic_parser.score_entity_paths(
        original_question=record.question,
        restored_question=processing_question,
        entity_start_nodes=entity_start_nodes,
        entity_origin_paths=pruned_entity_origin_paths,
    )

    best_paths_by_entity = select_best_path_by_entity(
        scored_paths=scored_entity_paths,
        entity_start_nodes=entity_start_nodes,
        entity_origin_paths=pruned_entity_origin_paths,
    )
    path_set_candidates = build_single_path_set_candidate(
        best_paths_by_entity=best_paths_by_entity,
    )
    if not path_set_candidates:
        raise ValueError("No candidate path sets were constructed for entity-origin path pipeline.")

    selected_dependency_path_evidence = build_selected_dependency_path_evidence(
        path_set_candidates=path_set_candidates,
        entity_origin_paths=pruned_entity_origin_paths,
        max_path_sets=1,
    )
    semantic_reasoning_paths, semantic_reasoning_path_payload = path_semantic_parser.build_semantic_reasoning_paths(
        original_question=record.question,
        selected_dependency_path_evidence=selected_dependency_path_evidence,
    )
    subquestion_dag, grounded_atomic_dag_payload = path_semantic_parser.build_grounded_atomic_dag(
        original_question=record.question,
        semantic_reasoning_paths=semantic_reasoning_paths,
    )
    atomic_evidences = (
        semantic_reasoning_path_payload.get("atomic_evidences")
        or semantic_reasoning_path_payload.get("evidence_atoms", [])
        if isinstance(semantic_reasoning_path_payload, dict)
        else []
    )
    step9_llm_input_contains_raw_dependency_paths = (
        bool(semantic_reasoning_path_payload.get("step9_llm_input_contains_raw_dependency_paths"))
        if isinstance(semantic_reasoning_path_payload, dict)
        else False
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
        "raw_dependency_graph": raw_dependency_graph,
        "dependency_collapse_stats": {
            "enabled": bool(dependency_graph.graph.get("dependency_collapsing_enabled")),
            "relations": list(dependency_graph.graph.get("collapse_relations") or []),
            "raw_node_count": dependency_graph.graph.get("raw_node_count"),
            "raw_edge_count": dependency_graph.graph.get("raw_edge_count"),
            "collapsed_node_count": dependency_graph.graph.get("collapsed_node_count"),
            "collapsed_edge_count": dependency_graph.graph.get("collapsed_edge_count"),
            "decisions": list(dependency_graph.graph.get("collapse_decisions") or []),
        },
        "weighted_graph": dependency_graph,
        "graph_node_candidates": graph_node_candidates,
        "restored_graph_node_candidates": restored_graph_node_candidates,
        "dependency_graph": dependency_graph,
        "entity_start_nodes": entity_start_nodes,
        "entity_origin_paths": entity_origin_paths,
        "pruned_entity_origin_paths": pruned_entity_origin_paths,
        "path_pruning_stats": path_pruning_stats,
        "scored_entity_paths": scored_entity_paths,
        "path_scoring_payload": path_scoring_payload,
        "best_paths_by_entity": best_paths_by_entity,
        "path_set_candidates": path_set_candidates,
        "selected_dependency_path_evidence": selected_dependency_path_evidence,
        "atomic_evidences": atomic_evidences,
        "evidence_atoms": atomic_evidences,
        "step9_llm_input_contains_raw_dependency_paths": step9_llm_input_contains_raw_dependency_paths,
        "semantic_reasoning_paths": semantic_reasoning_paths,
        "semantic_reasoning_path_payload": semantic_reasoning_path_payload,
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
    semantic_normalization: SemanticNormalizationResult = result["semantic_normalization"]
    explicit_entities: ExplicitEntityResult = result.get("explicit_entities") or ExplicitEntityResult()
    mask_spans: MaskSpanResult = result["mask_spans"]
    replacement: MaskReplacement = result["replacement"]
    dependency_parse = result["dependency_parse"]
    entity_start_nodes: list[EntityStartNode] = result["entity_start_nodes"]
    entity_origin_paths: list[EntityOriginPath] = result["entity_origin_paths"]
    pruned_entity_origin_paths: list[EntityOriginPath] = result.get("pruned_entity_origin_paths") or entity_origin_paths
    selected_dependency_path_evidence: list[dict[str, Any]] = result.get("selected_dependency_path_evidence", [])
    atomic_evidences: list[dict[str, Any]] = result.get("atomic_evidences") or result.get("evidence_atoms", [])
    semantic_reasoning_paths: SemanticReasoningPathResult | None = result.get("semantic_reasoning_paths")
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
    if semantic_normalization.normalized_question == record.question:
        print("  unchanged")
    else:
        print(f"  {semantic_normalization.normalized_question}")
    if debug:
        _print_warnings(semantic_normalization.warnings)
    print()

    print("[2. Explicit Entities]")
    if explicit_entities.entities:
        for entity in explicit_entities.entities:
            print(f"  - {entity.text}")
    else:
        print("  (none)")
    print()

    print("[3. Entity Masking]")
    if replacement.mask_mappings:
        for mapping in replacement.mask_mappings:
            print(f"  - {mapping.placeholder} -> {mapping.original_text}")
    else:
        print("  (none)")
    print(f"  Masked question: {replacement.masked_question}")
    print()

    print("[4. CoreNLP Dependency Parse]")
    print(f"  tokens={len(dependency_parse.tokens)} edges={len(dependency_parse.edges)}")
    _print_dependency_parse_edges(dependency_parse.edges)
    print()

    print("[5. Undirected Dependency Graph]")
    graph = result.get("dependency_graph", result["weighted_graph"])
    print(f"  nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    _print_dependency_graph_edges(graph)
    print()

    print("[6. Entity Start Nodes from Explicit Entities]")
    _print_entity_start_nodes(entity_start_nodes)
    print()

    print("[7. Entity-Origin Dependency Paths]")
    _print_entity_origin_paths(pruned_entity_origin_paths)
    print()

    print("[8. Selected Dependency Paths and Atomic Evidences]")
    _print_selected_dependency_path_evidence(selected_dependency_path_evidence)
    _print_atomic_evidences(atomic_evidences)
    print()

    print("[9. Semantic Reasoning Path Induction]")
    _print_semantic_reasoning_paths(semantic_reasoning_paths, atomic_evidences=atomic_evidences)
    print()

    print("[10. Semantic-Path-Guided Atomic DAG Generation]")
    print("Output:")
    _print_grounded_atomic_dag_payload(grounded_atomic_dag_payload)
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


def _print_entity_start_nodes(entity_start_nodes: list[EntityStartNode]) -> None:
    if not entity_start_nodes:
        print("  (none)")
        return
    for entity in entity_start_nodes:
        semantic = f" semantic_type={entity.semantic_type_hint}" if entity.semantic_type_hint else ""
        print(f"  - {entity.entity_id}: {entity.text} graph_node_ids={entity.graph_node_ids}{semantic}")


def _print_sample_dependency_edges(edges: list[Any], limit: int = 5) -> None:
    if not edges:
        return
    for edge in edges[:limit]:
        print(f"  - {edge.display()}")
    remaining = len(edges) - limit
    if remaining > 0:
        print(f"  ... {remaining} more edges")


def _print_dependency_parse_edges(edges: list[Any]) -> None:
    if not edges:
        print("  (none)")
        return
    for edge in edges:
        print(f"  - {edge.display()}")


def _print_dependency_graph_edges(graph: Any) -> None:
    edges = list(graph.edges(data=True))
    if not edges:
        print("    (none)")
        return
    for source, target, attrs in sorted(edges, key=lambda item: _dependency_graph_edge_sort_key(graph, item)):
        relations = [str(item) for item in attrs.get("relations", []) if str(item)]
        relation = "/".join(relations) or str(attrs.get("relation") or attrs.get("dependency_label") or "related")
        collapsed_via = attrs.get("collapsed_via") or []
        collapsed_text = f" collapsed_via={len(collapsed_via)}" if collapsed_via else ""
        print(
            f"    - {_dependency_graph_node_text(graph, source)}[{source}] "
            f"--{relation}-- {_dependency_graph_node_text(graph, target)}[{target}]"
            f"{collapsed_text}"
        )


def _dependency_graph_edge_sort_key(graph: Any, edge: tuple[Any, Any, dict[str, Any]]) -> tuple[int, int, str, str]:
    source, target, _attrs = edge
    source_order = _dependency_graph_node_order(graph, source)
    target_order = _dependency_graph_node_order(graph, target)
    return (min(source_order, target_order), max(source_order, target_order), str(source), str(target))


def _dependency_graph_node_order(graph: Any, node_id: Any) -> int:
    attrs = graph.nodes[node_id]
    value = attrs.get("order")
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(node_id)
        except (TypeError, ValueError):
            return 10**9


def _dependency_graph_node_text(graph: Any, node_id: Any) -> str:
    attrs = graph.nodes[node_id]
    return str(attrs.get("text") or attrs.get("word") or attrs.get("label") or node_id)


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


def _print_entity_path_summary(
    entity_origin_paths: list[EntityOriginPath],
    entity_start_nodes: list[EntityStartNode],
) -> None:
    if not entity_origin_paths:
        print("  (none)")
        return
    text_by_entity = {entity.entity_id: entity.text for entity in entity_start_nodes}
    paths_by_entity: dict[str, list[EntityOriginPath]] = {}
    for path in entity_origin_paths:
        paths_by_entity.setdefault(path.entity_id, []).append(path)
    print(f"  total_paths={len(entity_origin_paths)}")
    for entity_id in sorted(paths_by_entity, key=_entity_sort_key_for_print):
        paths = paths_by_entity[entity_id]
        shortest = min((path.length for path in paths), default=0)
        longest = max((path.length for path in paths), default=0)
        entity_text = text_by_entity.get(entity_id, "")
        label = f"{entity_id} / {entity_text}" if entity_text else entity_id
        print(f"  - {label}: paths={len(paths)} length_range={shortest}-{longest}")


def _print_path_pruning_stats(
    pruning_stats: dict[str, Any],
    entity_start_nodes: list[EntityStartNode],
) -> None:
    if not pruning_stats:
        print("  (none)")
        return
    total_raw = int(pruning_stats.get("total_raw_paths") or 0)
    total_kept = int(pruning_stats.get("total_kept_paths") or 0)
    total_pruned = int(pruning_stats.get("total_pruned_paths") or 0)
    total_ratio = float(pruning_stats.get("total_pruned_ratio") or 0.0)
    print(f"  Total raw paths: {total_raw}")
    print(f"  Total kept paths: {total_kept}")
    print(f"  Total pruned paths: {total_pruned}")
    print(f"  Total pruned ratio: {total_ratio:.2%}")
    by_entity = pruning_stats.get("by_entity") or {}
    if not isinstance(by_entity, dict) or not by_entity:
        return
    text_by_entity = {entity.entity_id: entity.text for entity in entity_start_nodes}
    print("  By entity:")
    for entity_id in sorted(by_entity, key=_entity_sort_key_for_print):
        stats = by_entity.get(entity_id) or {}
        entity_text = text_by_entity.get(entity_id, "")
        heading = f"{entity_id} / {entity_text}" if entity_text else entity_id
        print(
            f"    - {heading}: raw={stats.get('raw', 0)} kept={stats.get('kept', 0)} "
            f"pruned={stats.get('pruned', 0)} fallback_used={bool(stats.get('fallback_used'))}"
        )


def _print_selected_dependency_path_evidence(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        print("  Selected dependency paths: (none)")
        return
    print("  Selected dependency paths:")
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


def _print_atomic_evidences(atomic_evidences: list[dict[str, Any]]) -> None:
    if not atomic_evidences:
        print("  Atomic evidences: (none)")
        return
    print("  Atomic evidences:")
    for atom in atomic_evidences:
        if not isinstance(atom, dict):
            continue
        atom_id = atom.get("id") or "?"
        kind = atom.get("kind") or "unknown"
        text = atom.get("text") or ""
        print(f"    - {atom_id} [{kind}]: {text}")


def _print_semantic_reasoning_paths(
    result: SemanticReasoningPathResult | None,
    *,
    atomic_evidences: list[dict[str, Any]] | None = None,
) -> None:
    if result is None or not result.paths:
        print("  paths: (none)")
        return
    atom_text_by_id = _atomic_evidence_text_index(atomic_evidences or [])
    for path in result.paths:
        label_by_id = {node.node_id: node.label for node in path.nodes}
        if path.edges:
            chain_parts: list[str] = []
            first_edge = path.edges[0]
            chain_parts.append(label_by_id.get(first_edge.source, first_edge.source))
            for edge in path.edges:
                chain_parts.append(f"--{edge.relation}-->")
                chain_parts.append(label_by_id.get(edge.target, edge.target))
            print(f"  - {path.branch_id}: {' '.join(chain_parts)}")
        else:
            node_labels = [node.label for node in path.nodes]
            print(f"  - {path.branch_id}: {' -> '.join(node_labels) if node_labels else '(empty)'}")
        for edge in path.edges:
            print(
                f"    {edge.edge_id}: "
                f"{label_by_id.get(edge.source, edge.source)} --{edge.relation}--> "
                f"{label_by_id.get(edge.target, edge.target)}"
            )
            support_atom_ids = _semantic_edge_supported_atom_ids(edge)
            if support_atom_ids:
                print("      supported_by:")
                for atom_id in support_atom_ids:
                    atom_text = atom_text_by_id.get(atom_id, "")
                    suffix = f": {atom_text}" if atom_text else ""
                    print(f"        - {atom_id}{suffix}")
            else:
                print("      supported_by: (none)")
        if path.warnings:
            print("    Warnings:")
            for warning in path.warnings:
                print(f"      - {warning}")
    if result.warnings:
        print("  Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")


def _atomic_evidence_text_index(atomic_evidences: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for atom in atomic_evidences:
        if not isinstance(atom, dict):
            continue
        atom_id = str(atom.get("id") or "").strip()
        if not atom_id:
            continue
        result[atom_id] = str(atom.get("text") or "").strip()
    return result


def _semantic_edge_supported_atom_ids(edge: Any) -> list[str]:
    atom_ids: list[str] = []
    seen: set[str] = set()
    for support in getattr(edge, "support", []) or []:
        if not isinstance(support, dict):
            continue
        raw_ids = support.get("atom_ids") or support.get("supported_by") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list):
            continue
        for raw_id in raw_ids:
            atom_id = str(raw_id or "").strip()
            if atom_id and atom_id not in seen:
                seen.add(atom_id)
                atom_ids.append(atom_id)
    return atom_ids


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
            semantic_edge_id = node.get("source_semantic_edge_id")
            semantic_path_id = node.get("source_semantic_path_id")
            relation = node.get("one_hop_relation")
            semantic = ""
            if semantic_path_id or semantic_edge_id:
                semantic = f" source={semantic_path_id or '?'}:{semantic_edge_id or '?'}"
            relation_text = f" relation={relation}" if relation else ""
            print(
                f"    - {node.get('node_id')}: depends_on={dependencies or 'none'} "
                f"{semantic}{relation_text}"
            )
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
        depends_on = ", ".join(node.depends_on) if node.depends_on else "none"
        print(f"  - {node.id}: depends_on={depends_on}")
        print(f"    {node.question}")

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
