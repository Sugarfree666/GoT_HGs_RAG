from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import (
    AtomicQuestionDAG,
    AtomicSubquestion,
    CoreNLPViewAnnotation,
    DeclarativeView,
    HanLPSDPPreprocessResult,
    HanLPSDPResult,
    MaskReplacement,
    MaskSpan,
    MaskSpanResult,
    ExplicitEntity,
    ExplicitEntityResult,
    QuestionRecord,
    RestoredGraphNodeCandidate,
    EntityStartNode,
    SemanticReasoningPathResult,
    SemanticNormalizationResult,
)

if TYPE_CHECKING:
    from corenlp_parser import CoreNLPParser
    from graph_builder import GraphBuilder
    from mask_span_extractor import ExplicitEntityExtractor, MaskSpanExtractor
    from entity_path_pipeline import EntityPathSemanticParser
    from question_normalizer import SemanticQuestionNormalizer
    from hanlp_sdp_parser import HanLPSDPParser
    from hanlp_sdp_preprocessor import HanLPSDPPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEPO decomposition with explicit entity masking, grounded paths, and grounded atomic subquestions."
    )
    parser.add_argument(
        "--pipeline",
        choices=("hanlp_sdp", "corenlp_openie"),
        default="hanlp_sdp",
        help="Pipeline mode. Default: hanlp_sdp.",
    )
    parser.add_argument("--question", help="Run one manually supplied question instead of questions.json.")
    parser.add_argument("--questions-file", default="questions.json", help="Path to questions.json.")
    parser.add_argument("--api-key", help="OpenAI API key. Used only if OPENAI_API_KEY is not set.")
    parser.add_argument("--base-url", help="OpenAI base URL. Used only if OPENAI_BASE_URL is not set.")
    parser.add_argument(
        "--hanlp-model",
        help="HanLP pretrained constant name from hanlp.pretrained.mtl/sdp, or a local model path.",
    )
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
    records = [QuestionRecord(question=args.question)] if args.question else read_questions(args.questions_file)
    if args.pipeline == "hanlp_sdp":
        return _run_hanlp_sdp_cli(args, records)
    return _run_corenlp_openie_cli(args, records)


def _run_hanlp_sdp_cli(args: argparse.Namespace, records: list[QuestionRecord]) -> int:
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print(
            "This HanLP SDP branch requires one LLM call for entity masking and SDP-oriented rewrite.",
            file=sys.stderr,
        )
        print("Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        from hanlp_sdp_parser import HanLPSDPParser
        from hanlp_sdp_preprocessor import HanLPSDPPreprocessor
        from llm_client import LLMClient

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
        preprocessor = HanLPSDPPreprocessor(llm_client)
        parser = HanLPSDPParser(args.hanlp_model)

        print("If this is the first run, HanLP may download the model automatically.")
        print("You can set HANLP_HOME to control the cache directory.")
        print()

        for index, record in enumerate(records, start=1):
            result = run_hanlp_sdp_pipeline(
                record=record,
                index=index,
                preprocessor=preprocessor,
                parser=parser,
                debug=args.debug,
            )
            print_hanlp_sdp_result(index, record, result, debug=args.debug)
    except ModuleNotFoundError as exc:
        if "hanlp" in str(exc).lower() or getattr(exc, "name", "") == "hanlp":
            print("Missing dependency: hanlp", file=sys.stderr)
            print("Run: pip install hanlp", file=sys.stderr)
            return 2
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def _run_corenlp_openie_cli(args: argparse.Namespace, records: list[QuestionRecord]) -> int:
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

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
                result = run_corenlp_openie_pipeline(
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


def run_hanlp_sdp_pipeline(
    record: QuestionRecord,
    index: int,
    preprocessor: "HanLPSDPPreprocessor",
    parser: "HanLPSDPParser",
    debug: bool = False,
) -> dict[str, Any]:
    del index, debug
    from content_chain_compiler import compile_content_chains

    preprocess_result = preprocessor.preprocess(record.question)
    hanlp_sdp_result = parser.parse(
        preprocess_result.sdp_input_sentence,
        placeholders=[mapping.placeholder for mapping in preprocess_result.mask_mappings],
    )
    content_chains = compile_content_chains(
        hanlp_sdp_result,
        explicit_entities=[mapping.placeholder for mapping in preprocess_result.mask_mappings],
    )
    return {
        "preprocess_result": preprocess_result,
        "explicit_entities": preprocess_result.explicit_entities,
        "explicit_entity_payload": preprocess_result.explicit_entities.raw_payload,
        "masked_question": preprocess_result.masked_question,
        "sdp_input_sentence": preprocess_result.sdp_input_sentence,
        "entity_mask_mappings": preprocess_result.mask_mappings,
        "hanlp_sdp_result": hanlp_sdp_result,
        "content_chain_result": content_chains,
    }


def run_corenlp_openie_pipeline(
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
    from atomic_evidence_extractor import AtomicEvidenceExtractor
    from entity_path_projector import (
        build_entity_start_nodes_from_explicit_entities,
    )
    from path_projector import (
        build_undirected_dependency_graph,
    )
    from dependency_graph_collapser import collapse_dependency_graph
    from question_normalizer import RelationCarrierDeclarativeGenerator

    semantic_normalization = SemanticNormalizationResult(
        original_question=record.question,
        normalized_question=record.question,
        changed=False,
        warnings=["Deprecated stage bypassed: relation-carrier declarative views are generated after entity masking."],
    )
    processing_question = record.question
    explicit_entities = _coerce_explicit_entity_result(
        mask_span_extractor.extract(processing_question),
    )
    mask_spans = _mask_span_result_from_explicit_entities(explicit_entities)
    replacement = selective_entity_masking(
        original_question=processing_question,
        extracted_nodes=mask_spans,
    )
    view_generator = question_normalizer or RelationCarrierDeclarativeGenerator(None)
    if hasattr(view_generator, "generate_relation_carrier_views"):
        relation_carrier_result = view_generator.generate_relation_carrier_views(
            original_question=record.question,
            masked_question=replacement.masked_question,
            placeholders=[mapping.placeholder for mapping in replacement.mask_mappings],
        )
    else:
        relation_carrier_result = RelationCarrierDeclarativeGenerator(None).generate_relation_carrier_views(
            original_question=record.question,
            masked_question=replacement.masked_question,
            placeholders=[mapping.placeholder for mapping in replacement.mask_mappings],
        )
    declarative_views = relation_carrier_result.declarative_views or [
        DeclarativeView(id="view_1", sentence=replacement.masked_question, purpose="relation_carrier")
    ]

    corenlp_annotations = _annotate_declarative_views(parser, declarative_views)
    dependency_parse = (
        corenlp_annotations[0].to_dependency_parse()
        if corenlp_annotations
        else parser.parse(replacement.masked_question)
    )
    if path_semantic_parser is None:
        raise TypeError("run_pipeline requires path_semantic_parser.")

    graph_node_candidates = []
    restored_graph_node_candidates = []
    raw_dependency_graph = None
    dependency_graph = None
    entity_start_nodes = []
    try:
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
        entity_start_nodes = build_entity_start_nodes_from_explicit_entities(
            dependency_graph=dependency_graph,
            restored_graph_node_candidates=restored_graph_node_candidates,
            replacement=replacement,
        )
    except Exception:
        raw_dependency_graph = None
        dependency_graph = None
        entity_start_nodes = []

    atomic_evidence_extractor = AtomicEvidenceExtractor()
    atomic_evidence_objects = atomic_evidence_extractor.extract(
        masked_question=replacement.masked_question,
        explicit_entities=explicit_entities,
        mask_mappings=replacement.mask_mappings,
        declarative_views=declarative_views,
        corenlp_annotations=corenlp_annotations,
        operator_intent=relation_carrier_result.operator_intent,
    )
    atomic_evidences = [evidence.to_dict() for evidence in atomic_evidence_objects]

    semantic_reasoning_paths, semantic_reasoning_path_payload = path_semantic_parser.build_semantic_reasoning_paths(
        original_question=record.question,
        masked_question=replacement.masked_question,
        explicit_entities=[entity.to_dict() for entity in explicit_entities.entities],
        declarative_views=[view.to_dict() for view in declarative_views],
        operator_intent=relation_carrier_result.operator_intent,
        atomic_evidence_pool=atomic_evidence_objects,
    )
    subquestion_dag, grounded_atomic_dag_payload = path_semantic_parser.build_grounded_atomic_dag(
        original_question=record.question,
        semantic_reasoning_paths=semantic_reasoning_paths,
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
        "relation_carrier_views": relation_carrier_result,
        "declarative_views": declarative_views,
        "operator_intent": relation_carrier_result.operator_intent,
        "corenlp_view_annotations": corenlp_annotations,
        "dependency_parse": dependency_parse,
        "raw_dependency_graph": raw_dependency_graph,
        "dependency_collapse_stats": {
            "enabled": bool(dependency_graph is not None and dependency_graph.graph.get("dependency_collapsing_enabled")),
            "relations": list(dependency_graph.graph.get("collapse_relations") or []) if dependency_graph is not None else [],
            "raw_node_count": dependency_graph.graph.get("raw_node_count") if dependency_graph is not None else 0,
            "raw_edge_count": dependency_graph.graph.get("raw_edge_count") if dependency_graph is not None else 0,
            "collapsed_node_count": dependency_graph.graph.get("collapsed_node_count") if dependency_graph is not None else 0,
            "collapsed_edge_count": dependency_graph.graph.get("collapsed_edge_count") if dependency_graph is not None else 0,
            "decisions": list(dependency_graph.graph.get("collapse_decisions") or []) if dependency_graph is not None else [],
        },
        "weighted_graph": dependency_graph,
        "graph_node_candidates": graph_node_candidates,
        "restored_graph_node_candidates": restored_graph_node_candidates,
        "dependency_graph": dependency_graph,
        "entity_start_nodes": entity_start_nodes,
        "atomic_evidences": atomic_evidences,
        "evidence_atoms": atomic_evidences,
        "step9_llm_input_contains_raw_dependency_paths": step9_llm_input_contains_raw_dependency_paths,
        "semantic_reasoning_paths": semantic_reasoning_paths,
        "semantic_reasoning_path_payload": semantic_reasoning_path_payload,
        "grounded_atomic_dag_payload": grounded_atomic_dag_payload,
        "subquestions": subquestions,
        "subquestion_dag": subquestion_dag,
    }


def run_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return run_corenlp_openie_pipeline(*args, **kwargs)


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


def _annotate_declarative_views(parser: Any, declarative_views: list[DeclarativeView]) -> list[CoreNLPViewAnnotation]:
    view_payloads = [view.to_dict() for view in declarative_views]
    if hasattr(parser, "annotate_views"):
        return parser.annotate_views(view_payloads, enable_openie=True)

    annotations: list[CoreNLPViewAnnotation] = []
    for view in declarative_views:
        dependency_parse = parser.parse(view.sentence)
        annotations.append(
            CoreNLPViewAnnotation(
                view_id=view.id,
                text=view.sentence,
                tokens=list(dependency_parse.tokens),
                edges=list(dependency_parse.edges),
                raw=dependency_parse.raw,
                warnings=["Parser does not expose OpenIE view annotation; using dependency parse only."],
            )
        )
    return annotations


def print_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    semantic_normalization: SemanticNormalizationResult = result["semantic_normalization"]
    explicit_entities: ExplicitEntityResult = result.get("explicit_entities") or ExplicitEntityResult()
    mask_spans: MaskSpanResult = result["mask_spans"]
    replacement: MaskReplacement = result["replacement"]
    declarative_views: list[DeclarativeView] = result.get("declarative_views") or []
    corenlp_view_annotations: list[CoreNLPViewAnnotation] = result.get("corenlp_view_annotations") or []
    dependency_parse = result["dependency_parse"]
    entity_start_nodes: list[EntityStartNode] = result["entity_start_nodes"]
    atomic_evidences: list[dict[str, Any]] = result.get("atomic_evidences") or result.get("evidence_atoms", [])
    semantic_reasoning_paths: SemanticReasoningPathResult | None = result.get("semantic_reasoning_paths")
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

    print("[1. Explicit Entities]")
    if explicit_entities.entities:
        for entity in explicit_entities.entities:
            print(f"  - {entity.text}")
    else:
        print("  (none)")
    if debug:
        _print_warnings(semantic_normalization.warnings)
    print()

    print("[2. Entity Masking]")
    if replacement.mask_mappings:
        for mapping in replacement.mask_mappings:
            print(f"  - {mapping.placeholder} -> {mapping.original_text}")
    else:
        print("  (none)")
    print(f"  Masked question: {replacement.masked_question}")
    print()

    print("[3. Relation-Carrier Declarative Views]")
    if declarative_views:
        for view in declarative_views:
            print(f"  - {view.id}: {view.sentence}")
    else:
        print("  (none)")
    print()

    print("[4. CoreNLP + OpenIE View Annotations]")
    if corenlp_view_annotations:
        for annotation in corenlp_view_annotations:
            print(
                f"  - {annotation.view_id}: tokens={len(annotation.tokens)} "
                f"deps={len(annotation.edges)} openie={len(annotation.openie_triples)}"
            )
            if debug:
                _print_warnings(annotation.warnings)
    else:
        print("  (none)")
    print()

    print("[5. CoreNLP Structural Evidence Graph]")
    graph = result.get("dependency_graph", result["weighted_graph"])
    if graph is None:
        print("  (none)")
    else:
        print(f"  nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
        _print_dependency_graph_edges(graph)
    print()

    print("[6. Entity Start Nodes from Explicit Entities]")
    _print_entity_start_nodes(entity_start_nodes)
    print()

    print("[7. Atomic Evidence Pool]")
    structural_count = _atomic_evidence_source_count(atomic_evidences, "corenlp")
    openie_count = _atomic_evidence_source_count(atomic_evidences, "openie")
    print(f"  CoreNLP structural evidence count: {structural_count}")
    print(f"  OpenIE relational evidence count: {openie_count}")
    _print_atomic_evidences(atomic_evidences)
    print()

    print("[8. Semantic Reasoning Path Induction]")
    _print_semantic_reasoning_paths(semantic_reasoning_paths, atomic_evidences=atomic_evidences)
    print()

    print("[9. Semantic-Path-Guided Atomic DAG]")
    if subquestion_dag is not None:
        _print_atomic_question_dag(subquestion_dag)
    elif not subquestions:
        print("  (no atomic subquestions generated)")
    else:
        for item in subquestions:
            print(f"  q{item.index}: {item.question}")
    print()


def print_hanlp_sdp_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    preprocess_result: HanLPSDPPreprocessResult = result["preprocess_result"]
    explicit_entities: ExplicitEntityResult = preprocess_result.explicit_entities
    hanlp_result: HanLPSDPResult = result["hanlp_sdp_result"]
    content_chain_result = result["content_chain_result"]

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

    print("[1. Explicit Entities]")
    if explicit_entities.entities:
        for entity in explicit_entities.entities:
            semantic_type = entity.semantic_type_hint or "Entity"
            print(f" - {entity.text} [{semantic_type}]")
    else:
        print(" (none)")
    print()

    print("[2. SDP-Oriented Rewrite]")
    if preprocess_result.mask_mappings:
        for mapping in preprocess_result.mask_mappings:
            print(f" - {mapping.placeholder} -> {mapping.original_text}")
    else:
        print(" (none)")
    print(f"Masked question: {preprocess_result.masked_question}")
    print(f"SDP input sentence: {preprocess_result.sdp_input_sentence}")
    print()

    print("[3. HanLP SDP Parsing]")
    print(f" Model: {hanlp_result.model or '(unknown)'}")
    print()

    print("[Mask Token Check]")
    if hanlp_result.mask_token_checks:
        for placeholder, status in hanlp_result.mask_token_checks.items():
            print(f"{placeholder}: {status}")
    else:
        print("(none)")
    print()

    print("[Raw SDP/DM Edges]")
    _print_hanlp_edges_for_formalism(hanlp_result, "sdp/dm")
    if debug:
        _print_non_dm_hanlp_edges(hanlp_result)
    print()

    print("[4. Content Reasoning Chains]")
    chain_entities = [mapping.placeholder for mapping in preprocess_result.mask_mappings]
    if not chain_entities:
        chain_entities = list(content_chain_result.chains)
    for entity in chain_entities:
        chain = content_chain_result.chains.get(entity) or [entity]
        print(" -- ".join(chain))
    combined_warnings = [*preprocess_result.warnings, *hanlp_result.warnings]
    if debug and combined_warnings:
        print()
        print("[HanLP SDP Warnings]")
        for warning in combined_warnings:
            print(f" - {warning}")
    print()


def _print_hanlp_edges_for_formalism(hanlp_result: HanLPSDPResult, formalism: str) -> None:
    if formalism not in hanlp_result.sdp_graphs:
        print("(none)")
        return
    print(f"[SDP: {formalism}]")
    edges = [edge for edge in hanlp_result.edges if edge.formalism == formalism]
    if not edges:
        print("(no readable edges)")
        return
    for edge in edges:
        print(edge.display())


def _print_non_dm_hanlp_edges(hanlp_result: HanLPSDPResult) -> None:
    for formalism in hanlp_result.sdp_graphs:
        if formalism == "sdp/dm":
            continue
        print(f"[SDP: {formalism}]")
        edges = [edge for edge in hanlp_result.edges if edge.formalism == formalism]
        if not edges:
            print("(no readable edges)")
            continue
        for edge in edges:
            print(edge.display())


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


def _print_atomic_evidences(atomic_evidences: list[dict[str, Any]]) -> None:
    if not atomic_evidences:
        print("  Atomic evidences: (none)")
        return
    print("  Atomic evidences:")
    for atom in atomic_evidences:
        if not isinstance(atom, dict):
            continue
        atom_id = atom.get("id") or "?"
        kind = atom.get("type") or atom.get("kind") or "unknown"
        text = atom.get("text") or ""
        print(f"    - {atom_id} [{kind}]: {text}")


def _atomic_evidence_source_count(atomic_evidences: list[dict[str, Any]], source: str) -> int:
    return sum(1 for atom in atomic_evidences if isinstance(atom, dict) and atom.get("source") == source)


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
