from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DEPO decomposition over questions/*.json and save per-question "
            "decomposition artifacts for later LLM/manual analysis."
        )
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/, e.g. 2wikimultihopqa.")
    parser.add_argument("--questions-file", help="Specific questions.json path. Overrides --dataset.")
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Process every questions/*/questions.json file.",
    )
    parser.add_argument("--questions-root", default="questions", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--output-root",
        default="runs/depo_decomposition",
        help="Root output directory for decomposition artifacts.",
    )
    parser.add_argument(
        "--run-id",
        help="Output run id under output-root/dataset/. Defaults to current timestamp.",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of questions per dataset.")
    parser.add_argument("--start", type=int, default=1, help="1-based start index within each questions file.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a question if its decomposition.json already exists.",
    )
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model used by DEPO stages.")
    parser.add_argument(
        "--corenlp-url",
        default="http://localhost:9000",
        help="Endpoint used by the managed CoreNLP server.",
    )
    parser.add_argument("--corenlp-memory", default="4G", help="Java heap memory for managed CoreNLP.")
    parser.add_argument("--corenlp-home", help="Stanford CoreNLP directory containing jar files.")
    parser.add_argument(
        "--corenlp-timeout-ms",
        type=int,
        default=60000,
        help="CoreNLP annotation timeout in milliseconds.",
    )
    parser.add_argument("--debug", action="store_true", help="Keep full verbose fields in outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        question_files = _resolve_question_files(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not question_files:
        print("No questions.json files found.", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)

    try:
        from corenlp_parser import CoreNLPParser
        from entity_path_pipeline import EntityPathSemanticParser
        from graph_builder import GraphBuilder
        from llm_client import LLMClient
        from main import run_pipeline
        from mask_span_extractor import ExplicitEntityExtractor
        from models import QuestionRecord
        from question_normalizer import SemanticQuestionNormalizer
        from subquestion_generator import SubquestionGenerator

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.model)
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
            for questions_file in question_files:
                dataset = questions_file.parent.name
                dataset_output_dir = output_root / dataset / run_id
                dataset_output_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = dataset_output_dir / "manifest.jsonl"
                question_items = _read_question_items(questions_file)
                selected_items = _slice_items(question_items, start=args.start, limit=args.limit)
                print(
                    f"Running DEPO decomposition: dataset={dataset}, "
                    f"questions={len(selected_items)}, output={dataset_output_dir}"
                )

                with manifest_path.open("a", encoding="utf-8") as manifest:
                    for item in selected_items:
                        record = QuestionRecord(question=item["question"], qid=item.get("qid"))
                        question_dir = dataset_output_dir / _question_dir_name(item["index"], record.qid, record.question)
                        decomposition_path = question_dir / "decomposition.json"
                        if args.resume and decomposition_path.exists():
                            print(f"[skip] {dataset} #{item['index']} {record.question}")
                            manifest.write(
                                json.dumps(
                                    {
                                        "dataset": dataset,
                                        "index": item["index"],
                                        "qid": record.qid,
                                        "question": record.question,
                                        "status": "skipped",
                                        "output_dir": str(question_dir),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            manifest.flush()
                            continue

                        question_dir.mkdir(parents=True, exist_ok=True)
                        print(f"[run] {dataset} #{item['index']} {record.question}")
                        try:
                            result = run_pipeline(
                                record=record,
                                index=item["index"],
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
                            payload = build_decomposition_payload(
                                dataset=dataset,
                                questions_file=questions_file,
                                item=item,
                                result=result,
                                debug=args.debug,
                            )
                            _write_json(decomposition_path, payload)
                            (question_dir / "decomposition.md").write_text(
                                build_markdown_report(payload),
                                encoding="utf-8",
                            )
                            manifest_item = _manifest_item(payload, question_dir)
                            print(f"[ok]  {dataset} #{item['index']} -> {question_dir}")
                        except Exception as exc:  # Keep batch jobs moving across bad items.
                            payload = build_error_payload(dataset, questions_file, item, exc)
                            _write_json(question_dir / "error.json", payload)
                            (question_dir / "error.md").write_text(build_error_markdown(payload), encoding="utf-8")
                            manifest_item = {
                                "dataset": dataset,
                                "index": item["index"],
                                "qid": item.get("qid"),
                                "question": item["question"],
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "output_dir": str(question_dir),
                            }
                            print(f"[err] {dataset} #{item['index']} {type(exc).__name__}: {exc}")

                        manifest.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                        manifest.flush()
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def build_decomposition_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    result: dict[str, Any],
    debug: bool,
) -> dict[str, Any]:
    dependency_parse = result["dependency_parse"]
    dependency_graph = result["dependency_graph"]
    raw_dependency_graph = result.get("raw_dependency_graph") or dependency_graph
    subquestion_dag = result.get("subquestion_dag")
    subquestions = result.get("subquestions", [])

    best_paths_by_entity = {
        entity_id: _dataclass_to_jsonable(path)
        for entity_id, path in (result.get("best_paths_by_entity") or {}).items()
    }

    payload: dict[str, Any] = {
        "status": "ok",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "stages": {
            "1_semantic_normalized_question": _dataclass_to_jsonable(result["semantic_normalization"]),
            "2_explicit_entities": _dataclass_to_jsonable(result.get("explicit_entities")),
            "2_mask_spans": _dataclass_to_jsonable(result["mask_spans"]),
            "3_entity_masking": {
                "masked_question": result["replacement"].masked_question,
                "mask_mappings": [_dataclass_to_jsonable(mapping) for mapping in result.get("entity_mask_mappings", [])],
            },
            "3_selective_masked_question": result["replacement"].masked_question,
            "4_corenlp_dependency_parse": {
                "tokens": [_dataclass_to_jsonable(token) for token in dependency_parse.tokens],
                "edges": [_dataclass_to_jsonable(edge) for edge in dependency_parse.edges],
                "edge_display": [edge.display() for edge in dependency_parse.edges],
            },
            "5_undirected_dependency_graph": _graph_payload(raw_dependency_graph),
            "5_1_collapsed_dependency_graph": _graph_payload(dependency_graph),
            "5_2_dependency_graph_collapse_stats": _dataclass_to_jsonable(result.get("dependency_collapse_stats") or {}),
            "6_entity_start_nodes": [_dataclass_to_jsonable(entity) for entity in result["entity_start_nodes"]],
            "7_entity_origin_paths": [_dataclass_to_jsonable(path) for path in result["entity_origin_paths"]],
            "7_5_terminal_glue_path_pruning": _dataclass_to_jsonable(result.get("path_pruning_stats") or {}),
            "8_path_scores": [_dataclass_to_jsonable(path) for path in result.get("scored_entity_paths", [])],
            "8_1_best_paths_by_entity": best_paths_by_entity,
            "8_2_path_set_candidates": [_dataclass_to_jsonable(candidate) for candidate in result.get("path_set_candidates", [])],
            "9_selected_dependency_path_evidence": _dataclass_to_jsonable(result.get("selected_dependency_path_evidence") or []),
            "9a_evidence_atoms": _dataclass_to_jsonable(result.get("evidence_atoms") or []),
            "9b_semantic_reasoning_paths": _dataclass_to_jsonable(result.get("semantic_reasoning_paths")),
            "10_grounded_atomic_dag_generation": _dataclass_to_jsonable(result.get("grounded_atomic_dag_payload") or {}),
            "10_atomic_subquestion_dag": _dataclass_to_jsonable(subquestion_dag) if subquestion_dag else None,
            "10_subquestions": [_dataclass_to_jsonable(item) for item in subquestions],
        },
    }
    if debug:
        payload["debug_payloads"] = {
            "path_scoring_payload": result.get("path_scoring_payload"),
            "grounded_atomic_dag_payload": result.get("grounded_atomic_dag_payload"),
        }
    return payload


def build_error_payload(
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "status": "error",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    stages = payload["stages"]
    lines: list[str] = []
    lines.append(f"# DEPO Decomposition #{payload['index']}")
    lines.append("")
    lines.append(f"- Dataset: `{payload['dataset']}`")
    if payload.get("qid"):
        lines.append(f"- QID: `{payload['qid']}`")
    lines.append(f"- Question: {payload['question']}")
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.append("")

    normalized = stages["1_semantic_normalized_question"].get("normalized_question", payload["question"])
    lines.append("## 1. Semantic-Normalized Question")
    lines.append(normalized)
    lines.append("")

    lines.append("## 2. Explicit Entities")
    explicit_entities = (stages.get("2_explicit_entities") or {}).get("entities", [])
    if explicit_entities:
        for entity in explicit_entities:
            lines.append(
                f"- {entity.get('text')} ({entity.get('semantic_type_hint')}) "
                f"span=({entity.get('start_char')}, {entity.get('end_char')})"
            )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## 3. Entity Masking")
    entity_masking = stages.get("3_entity_masking") or {}
    for mapping in entity_masking.get("mask_mappings", []):
        lines.append(f"- {mapping.get('placeholder')} -> {mapping.get('original_text')}")
    if not entity_masking.get("mask_mappings"):
        lines.append("(none)")
    lines.append("")
    lines.append(str(entity_masking.get("masked_question", stages.get("3_selective_masked_question", ""))))
    lines.append("")

    lines.append("## 4. CoreNLP Dependency Parse")
    for edge in stages["4_corenlp_dependency_parse"]["edge_display"]:
        lines.append(f"- {edge}")
    if not stages["4_corenlp_dependency_parse"]["edge_display"]:
        lines.append("(none)")
    lines.append("")

    lines.append("## 5. Undirected Dependency Graph")
    for edge in stages["5_undirected_dependency_graph"]["edges"]:
        relation = "/".join(edge.get("relations", [])) or "related"
        lines.append(f"- {edge.get('source_text')}[{edge.get('source')}] --{relation}-- {edge.get('target_text')}[{edge.get('target')}]")
    if not stages["5_undirected_dependency_graph"]["edges"]:
        lines.append("(none)")
    lines.append("")

    collapsed_stage = stages.get("5_1_collapsed_dependency_graph") or {"edges": []}
    lines.append("## 5.1 Collapsed Dependency Graph")
    for edge in collapsed_stage.get("edges", []):
        relation = "/".join(edge.get("relations", [])) or "related"
        lines.append(f"- {edge.get('source_text')}[{edge.get('source')}] --{relation}-- {edge.get('target_text')}[{edge.get('target')}]")
    if not collapsed_stage.get("edges"):
        lines.append("(none)")
    lines.append("")

    collapse_stats = stages.get("5_2_dependency_graph_collapse_stats") or {}
    lines.append("## 5.2 Dependency Graph Collapse Stats")
    lines.append(f"- Enabled: {bool(collapse_stats.get('enabled'))}")
    lines.append(f"- Relations: {collapse_stats.get('relations') or []}")
    lines.append(
        f"- Counts: raw={collapse_stats.get('raw_node_count')} nodes/{collapse_stats.get('raw_edge_count')} edges; "
        f"collapsed={collapse_stats.get('collapsed_node_count')} nodes/{collapse_stats.get('collapsed_edge_count')} edges"
    )
    decisions = collapse_stats.get("decisions") or []
    for decision in decisions:
        if isinstance(decision, dict):
            lines.append(
                f"- collapse {decision.get('relation')}: "
                f"{decision.get('child_text')} -> {decision.get('head_text_before')} => {decision.get('head_text_after')}"
            )
        else:
            lines.append(f"- {decision}")
    if not decisions:
        lines.append("- Decisions: none")
    lines.append("")

    lines.append("## 6. Entity Start Nodes from Explicit Entities")
    for entity in stages["6_entity_start_nodes"]:
        lines.append(f"- {entity.get('entity_id')}: {entity.get('text')} graph_node_ids={entity.get('graph_node_ids')}")
    if not stages["6_entity_start_nodes"]:
        lines.append("(none)")
    lines.append("")

    lines.append("## 7. Entity-Origin Dependency Paths")
    for path in stages["7_entity_origin_paths"]:
        lines.append(f"- {path.get('path_id')} ({path.get('entity_id')}): {' -- '.join(path.get('nodes', []))}")
    if not stages["7_entity_origin_paths"]:
        lines.append("(none)")
    lines.append("")

    lines.append("## 7.5 Terminal Glue Path Pruning")
    pruning_stats = stages.get("7_5_terminal_glue_path_pruning") or {}
    if pruning_stats:
        lines.append(f"Total raw paths: {pruning_stats.get('total_raw_paths', 0)}")
        lines.append(f"Total kept paths: {pruning_stats.get('total_kept_paths', 0)}")
        lines.append(f"Total pruned paths: {pruning_stats.get('total_pruned_paths', 0)}")
        ratio = float(pruning_stats.get("total_pruned_ratio") or 0.0)
        lines.append(f"Total pruned ratio: {ratio:.2%}")
        by_entity = pruning_stats.get("by_entity") or {}
        if by_entity:
            lines.append("")
            lines.append("### By Entity")
            entity_text_by_id = {
                entity.get("entity_id"): entity.get("text")
                for entity in stages.get("6_entity_start_nodes", [])
                if isinstance(entity, dict)
            }
            for entity_id, stats in by_entity.items():
                label = entity_text_by_id.get(entity_id)
                heading = f"{entity_id} / {label}" if label else str(entity_id)
                lines.append(f"- {heading}")
                lines.append(f"  - raw: {stats.get('raw', 0)}")
                lines.append(f"  - kept: {stats.get('kept', 0)}")
                lines.append(f"  - pruned: {stats.get('pruned', 0)}")
                lines.append(f"  - fallback_used: {stats.get('fallback_used', False)}")
                examples = stats.get("pruned_examples") or []
                if examples:
                    lines.append("  - examples:")
                    for example in examples[:5]:
                        lines.append(
                            f"    - {example.get('path_id')}: {example.get('path_text')} "
                            f"[terminal={example.get('terminal')}, reason={example.get('reason')}]"
                        )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## 8. LLM Path Scores")
    for score in stages["8_path_scores"]:
        reason = score.get("reason") or ""
        terminal = f" terminal={score.get('terminal_hint')}" if score.get("terminal_hint") else ""
        lines.append(
            f"- {score.get('entity_id')}: {score.get('path_id')} "
            f"score={score.get('score')} valid={score.get('valid')}{terminal}"
        )
        if reason:
            lines.append(f"  Reason: {reason}")
    if not stages["8_path_scores"]:
        lines.append("(none)")
    lines.append("")

    lines.append("## 8.1 Highest-Scored Path per Entity")
    for entity_id, path in stages["8_1_best_paths_by_entity"].items():
        lines.append(f"- {entity_id}: {path.get('path_id', '')} score={path.get('score')}")
    if not stages["8_1_best_paths_by_entity"]:
        lines.append("(none)")
    lines.append("")

    lines.append("## 8.2 Selected Path Set")
    for candidate in stages["8_2_path_set_candidates"]:
        lines.append(
            f"- {candidate.get('path_set_id')}: {candidate.get('path_ids_by_entity')} "
            f"mean_path_score={candidate.get('mean_path_score')}"
        )
    if not stages["8_2_path_set_candidates"]:
        lines.append("(none)")
    lines.append("")

    lines.append("## 9. Semantic Reasoning Path Induction")
    lines.append("Inputs:")
    lines.append(f"- Original question: {payload['question']}")
    selected_evidence = stages.get("9_selected_dependency_path_evidence") or []
    for path_set in selected_evidence:
        lines.append(f"- {path_set.get('path_set_id')}")
        for path in path_set.get("paths", []):
            lines.append(f"  - {path.get('path_id')}: {path.get('path_text')}")
    if not selected_evidence:
        lines.append("- Selected dependency path evidence: (none)")
    lines.append("")

    evidence_atoms = stages.get("9a_evidence_atoms") or []
    lines.append("### 9A Evidence Atoms")
    for atom in evidence_atoms:
        lines.append(
            f"- {atom.get('id')}: {atom.get('text')} "
            f"(path={atom.get('path_id')}, source={atom.get('source')}, relation_hint={atom.get('relation_hint')}, target={atom.get('target')})"
        )
    if not evidence_atoms:
        lines.append("(none)")
    lines.append("")

    lines.append("### 9B Semantic Reasoning Paths")
    semantic_payload = stages.get("9b_semantic_reasoning_paths") or {}
    for path in semantic_payload.get("paths", []) if isinstance(semantic_payload, dict) else []:
        lines.append(f"- {path.get('branch_id')} entity={path.get('entity_id')} source_path={path.get('source_path_id')}")
        node_labels = {
            node.get("node_id"): node.get("label")
            for node in path.get("nodes", [])
            if isinstance(node, dict)
        }
        for edge in path.get("edges", []):
            supported_by = []
            for support in edge.get("support", []) or []:
                if isinstance(support, dict):
                    supported_by.extend(support.get("atom_ids") or support.get("supported_by") or [])
            lines.append(
                f"  - {node_labels.get(edge.get('source'), edge.get('source'))} "
                f"--{edge.get('relation')}--> {node_labels.get(edge.get('target'), edge.get('target'))} "
                f"supported_by={supported_by}"
            )
    if not isinstance(semantic_payload, dict) or not semantic_payload.get("paths"):
        lines.append("(none)")
    lines.append("")

    lines.append("## 10. Grounded Atomic DAG Generation")
    lines.append("Output:")
    grounded_payload = stages.get("10_grounded_atomic_dag_generation") or {}
    if grounded_payload.get("selected_path_set_ids"):
        lines.append(f"- selected_path_set_ids: {grounded_payload.get('selected_path_set_ids')}")
    if grounded_payload.get("reason"):
        lines.append(f"- reason: {grounded_payload.get('reason')}")
    for node in grounded_payload.get("nodes", []):
        support = node.get("support") or []
        support_ids = [
            item.get("path_id")
            for item in support
            if isinstance(item, dict) and item.get("path_id")
        ]
        lines.append(
            f"- {node.get('node_id')}: {node.get('question')} "
            f"depends_on={node.get('dependencies') or []} support={support_ids}"
        )
    if not grounded_payload.get("nodes"):
        lines.append("(none)")
    warnings = grounded_payload.get("normalization_warnings") or []
    for warning in warnings:
        lines.append(f"- warning: {warning}")
    lines.append("")

    lines.append("## 10. Atomic Subquestion DAG")
    dag = stages["10_atomic_subquestion_dag"] or {}
    for node in dag.get("nodes", []):
        metadata = node.get("metadata", {})
        operator = f" operator={metadata.get('operator')}" if metadata.get("operator") else ""
        lines.append(f"- {node.get('node_id')}{operator}: {node.get('question')}")
        dependencies = node.get("dependencies") or []
        if dependencies:
            lines.append(f"  Depends on: {', '.join(dependencies)}")
    if not dag.get("nodes"):
        lines.append("(none)")
    lines.append("")
    return "\n".join(lines)


def build_error_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# DEPO Decomposition Error #{payload['index']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Question: {payload['question']}",
    ]
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.extend(
        [
            f"- Error type: `{payload['error_type']}`",
            "",
            "```text",
            str(payload["error"]),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_question_files(args: argparse.Namespace) -> list[Path]:
    if args.questions_file:
        return [Path(args.questions_file)]
    questions_root = Path(args.questions_root)
    if args.dataset:
        return [questions_root / args.dataset / "questions.json"]
    if args.all_datasets:
        return sorted(questions_root.glob("*/questions.json"))
    raise ValueError("Specify --dataset, --questions-file, or --all-datasets.")


def _read_question_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            question = item.strip()
            raw = item
            qid = None
            answer = None
        elif isinstance(item, dict):
            question = str(item.get("question", "")).strip()
            raw = item
            qid_value = item.get("id", item.get("qid"))
            qid = str(qid_value) if qid_value is not None else None
            answer = item.get("answer")
        else:
            raise ValueError(f"Unsupported question item at index {index}: {item!r}")
        if not question:
            raise ValueError(f"Question at index {index} is empty.")
        items.append(
            {
                "index": index,
                "qid": qid,
                "question": question,
                "answer": answer,
                "raw": raw,
            }
        )
    return items


def _slice_items(items: list[dict[str, Any]], *, start: int, limit: int | None) -> list[dict[str, Any]]:
    if start < 1:
        raise ValueError("--start must be >= 1.")
    selected = items[start - 1 :]
    return selected[:limit] if limit is not None else selected


def _question_dir_name(index: int, qid: str | None, question: str) -> str:
    prefix = f"{index:05d}"
    if qid:
        prefix += f"_{_slug(qid, max_len=48)}"
    return f"{prefix}_{_slug(question, max_len=80)}"


def _slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return (slug[:max_len].strip("-") or "question")


def _graph_payload(graph: Any) -> dict[str, Any]:
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append({"id": str(node_id), **_dataclass_to_jsonable(dict(attrs))})
    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append(
            {
                "source": str(source),
                "target": str(target),
                "source_text": str(graph.nodes[source].get("text") or graph.nodes[source].get("word") or source),
                "target_text": str(graph.nodes[target].get("text") or graph.nodes[target].get("word") or target),
                **_dataclass_to_jsonable(dict(attrs)),
            }
        )
    return {"nodes": nodes, "edges": edges}


def _dataclass_to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return _dataclass_to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _dataclass_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dataclass_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_item(payload: dict[str, Any], question_dir: Path) -> dict[str, Any]:
    stages = payload["stages"]
    dag = stages.get("10_atomic_subquestion_dag") or {}
    return {
        "dataset": payload["dataset"],
        "index": payload["index"],
        "qid": payload.get("qid"),
        "question": payload["question"],
        "gold_answer": payload.get("gold_answer"),
        "status": "ok",
        "path_score_count": len(stages.get("8_path_scores", [])),
        "atomic_question_count": len(dag.get("nodes", [])),
        "output_dir": str(question_dir),
    }


if __name__ == "__main__":
    raise SystemExit(main())
