from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
EVAL_ROOT = PROJECT_ROOT / "eval"
for path in (DEPO_ROOT, PROJECT_ROOT, SCRIPTS_ROOT, EVAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import get_score as eval_score  # noqa: E402
import run_depo_decomposition_batch as depo_batch  # noqa: E402
from hyper_branch.config import load_config  # noqa: E402
from hyper_branch.logging_utils import TraceStore, configure_logging  # noqa: E402
from hyper_branch.pipeline import HyperBranchPipeline  # noqa: E402


METHOD = "depo_hanlp_sdp_atomic_dag_plus_hyperbranch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DEPO Step1-5 atomic DAG generation and execute that DAG with HyperBranch."
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/, e.g. musique.")
    parser.add_argument("--questions-file", help="Specific questions file path. Overrides --dataset.")
    parser.add_argument("--all-datasets", action="store_true", help="Process every questions/*/questions.json file.")
    parser.add_argument("--questions-root", default="questions", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--config",
        help="HyperBranch YAML config. Defaults to configs/<dataset>.yaml for each dataset.",
    )
    parser.add_argument(
        "--output-root",
        default="runs/depo_hyperbranch",
        help="Root output directory for combined DEPO + HyperBranch artifacts.",
    )
    parser.add_argument("--run-id", help="Run id under output-root/dataset/. Defaults to current timestamp.")
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive start index.")
    parser.add_argument("--end", type=int, help="1-based inclusive end index.")
    parser.add_argument("--limit", type=int, help="Maximum number of questions after applying --start/--end.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed questions and reuse existing DEPO decomposition.json when HyperBranch is missing.",
    )
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model used by DEPO Step2/Step5.")
    parser.add_argument(
        "--hyperbranch-llm-model",
        help="Override the HyperBranch chat model. Defaults to --llm-model/config value.",
    )
    parser.add_argument("--embedding-model", help="Override the HyperBranch embedding model.")
    parser.add_argument(
        "--hanlp-model",
        help="HanLP pretrained constant name from hanlp.pretrained.mtl/sdp, or a local model path.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Tri-SDP Step4 debug JSON files.")
    parser.add_argument(
        "--debug-dir",
        default="debug/hanlp_sdp",
        help="Directory for Tri-SDP debug JSON files when --debug is enabled.",
    )
    parser.add_argument(
        "--hyperbranch-mock-llm",
        action="store_true",
        help="Use HyperBranch mock LLM/local hash embeddings after DEPO generation.",
    )
    parser.add_argument(
        "--live-eval",
        action="store_true",
        help="After each processed question, compute cumulative EM/F1 with eval/get_score.py helpers.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Print live EM/F1 every N processed questions when --live-eval is enabled.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show verbose HyperBranch console logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2
    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    try:
        question_files = depo_batch._resolve_question_files(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not question_files:
        print("No questions files found.", file=sys.stderr)
        return 2

    try:
        from atomic_question_dag import restore_global_best_paths
        from entity_masking_preprocessor import EntityMaskingPreprocessor
        from hanlp_sdp_parser import HanLPSDPParser
        from llm_client import LLMClient
        from main import run_hanlp_sdp_pipeline
        from models import QuestionRecord
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _repo_path(args.output_root)
    debug_dir = str(_repo_path(args.debug_dir))

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
    preprocessor = EntityMaskingPreprocessor(llm_client)
    parser = HanLPSDPParser(args.hanlp_model)

    for questions_file in question_files:
        dataset = depo_batch._dataset_name(questions_file)
        config_path = _config_path_for_dataset(args, dataset)
        dataset_output_dir = output_root / dataset / run_id
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_output_dir / "manifest.jsonl"
        summary_path = dataset_output_dir / "summary.md"

        items = depo_batch._slice_items(
            depo_batch._read_question_items(questions_file),
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
        print(
            f"Running DEPO + HyperBranch: dataset={dataset}, questions={len(items)}, "
            f"config={config_path}, output={dataset_output_dir}"
        )

        summary_lines = _summary_header(
            dataset=dataset,
            questions_file=questions_file,
            config_path=config_path,
            run_id=run_id,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
        live_eval_records: list[dict[str, Any]] = []
        hyperbranch_runner: _ReusableHyperBranchRunner | None = None

        with manifest_path.open("a", encoding="utf-8") as manifest:
            for offset, item in enumerate(items, start=1):
                record = QuestionRecord(question=item["question"], qid=item.get("qid"))
                question_dir = dataset_output_dir / depo_batch._question_dir_name(
                    item["index"], record.qid, record.question
                )
                combined_path = question_dir / "pipeline.json"
                decomposition_path = question_dir / "decomposition.json"
                hyperbranch_result_path = question_dir / "hyperbranch_result.json"
                if args.resume and combined_path.exists() and hyperbranch_result_path.exists():
                    print(f"[skip] {dataset} #{item['index']} {record.question}")
                    if args.live_eval:
                        try:
                            existing_payload = json.loads(combined_path.read_text(encoding="utf-8"))
                            _append_live_eval_record(
                                live_eval_records,
                                item=item,
                                final_answer=existing_payload.get("final_answer"),
                                question_dir=question_dir,
                                status="success",
                            )
                            _maybe_report_live_eval(
                                records=live_eval_records,
                                eval_every=args.eval_every,
                            )
                        except Exception as exc:
                            print(f"[eval-warn] skipped item #{item['index']} could not be scored: {exc}")
                    manifest_item = _skipped_manifest_item(dataset, item, question_dir)
                    manifest.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                    manifest.flush()
                    continue

                question_dir.mkdir(parents=True, exist_ok=True)
                print(f"[run {offset}/{len(items)}] {dataset} #{item['index']} {record.question}")
                try:
                    if args.resume and decomposition_path.exists():
                        decomposition_payload = json.loads(decomposition_path.read_text(encoding="utf-8"))
                    else:
                        result = run_hanlp_sdp_pipeline(
                            record=record,
                            index=item["index"],
                            preprocessor=preprocessor,
                            parser=parser,
                            debug=args.debug,
                            debug_dir=debug_dir,
                            llm_client=llm_client,
                        )
                        restored_global_best_paths = restore_global_best_paths(
                            result["token_reasoning_structure"].paths,
                            result["preprocess_result"].mask_mappings,
                        )
                        decomposition_payload = depo_batch.build_decomposition_payload(
                            dataset=dataset,
                            questions_file=questions_file,
                            item=item,
                            result=result,
                            restored_global_best_paths=restored_global_best_paths,
                        )
                        depo_batch._write_json(decomposition_path, decomposition_payload)
                        (question_dir / "decomposition.md").write_text(
                            depo_batch.build_markdown_report(decomposition_payload),
                            encoding="utf-8",
                        )

                    dag_payload = _hyperbranch_dag_payload(decomposition_payload)
                    dag_path = question_dir / "hyperbranch_dag.json"
                    _write_json(dag_path, dag_payload)

                    if hyperbranch_runner is None:
                        hyperbranch_runner = _ReusableHyperBranchRunner(
                            config_path=config_path,
                            cache_dir=dataset_output_dir / "_hyperbranch_pipeline",
                            args=args,
                        )
                    hyperbranch_result = hyperbranch_runner.run(
                        question=record.question,
                        dag_payload=dag_payload,
                        original_question_entities=_depo_explicit_entity_texts(decomposition_payload),
                        question_dir=question_dir,
                    )
                    _write_json(hyperbranch_result_path, hyperbranch_result)

                    combined_payload = _combined_payload(
                        dataset=dataset,
                        questions_file=questions_file,
                        item=item,
                        config_path=config_path,
                        question_dir=question_dir,
                        decomposition_payload=decomposition_payload,
                        dag_path=dag_path,
                        hyperbranch_result=hyperbranch_result,
                    )
                    if args.live_eval:
                        _append_live_eval_record(
                            live_eval_records,
                            item=item,
                            final_answer=combined_payload.get("final_answer"),
                            question_dir=question_dir,
                            status="success",
                        )
                    _write_json(combined_path, combined_payload)
                    (question_dir / "pipeline.md").write_text(
                        _combined_markdown(combined_payload),
                        encoding="utf-8",
                    )
                    manifest_item = _manifest_item(combined_payload)
                    summary_lines.extend(_summary_question_lines(combined_payload))
                    print(
                        f"[ok]  {dataset} #{item['index']} "
                        f"dag_nodes={manifest_item['dag_node_count']} answer={manifest_item['final_answer']!r}"
                    )
                    if args.live_eval:
                        _maybe_report_live_eval(
                            records=live_eval_records,
                            eval_every=args.eval_every,
                        )
                except Exception as exc:
                    error_payload = _error_payload(dataset, questions_file, item, config_path, question_dir, exc)
                    if args.live_eval:
                        _append_live_eval_record(
                            live_eval_records,
                            item=item,
                            final_answer={},
                            question_dir=question_dir,
                            status="failed",
                        )
                    _write_json(question_dir / "sample.json", error_payload)
                    (question_dir / "sample.md").write_text(_error_markdown(error_payload), encoding="utf-8")
                    manifest_item = _error_manifest_item(error_payload, question_dir)
                    summary_lines.extend(_summary_error_lines(error_payload))
                    print(f"[err] {dataset} #{item['index']} {type(exc).__name__}: {exc}")
                    if args.live_eval:
                        _maybe_report_live_eval(
                            records=live_eval_records,
                            eval_every=args.eval_every,
                        )

                manifest.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                manifest.flush()
                summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

        summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
        if args.live_eval:
            _maybe_report_live_eval(
                records=live_eval_records,
                eval_every=args.eval_every,
                force=True,
            )
        print(f"Summary written to {summary_path}")

    return 0


class _ReusableHyperBranchRunner:
    def __init__(self, *, config_path: Path, cache_dir: Path, args: argparse.Namespace) -> None:
        self.config = _load_hyperbranch_config(config_path, args)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger = configure_logging(self.cache_dir, self.config.runtime.log_level, verbose_console=args.verbose)
        trace_store = TraceStore(self.cache_dir)
        try:
            self.pipeline = HyperBranchPipeline(
                config=self.config,
                run_dir=self.cache_dir,
                logger=logger,
                trace_store=trace_store,
            )
        finally:
            _close_logger(logger)
        self.verbose = args.verbose

    def run(
        self,
        *,
        question: str,
        dag_payload: dict[str, Any],
        original_question_entities: list[str] | None = None,
        question_dir: Path,
    ) -> dict[str, Any]:
        run_dir = question_dir / "hyperbranch_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = configure_logging(run_dir, self.config.runtime.log_level, verbose_console=self.verbose)
        trace_store = TraceStore(run_dir)
        self._bind_question_context(run_dir=run_dir, logger=logger, trace_store=trace_store)
        try:
            trace_store.save_artifact("dataset_summary.json", self.pipeline.dataset.summary)
            return self.pipeline.run(
                question,
                dag_payload=dag_payload,
                original_question_entities=original_question_entities,
            )
        finally:
            _close_logger(logger)

    def _bind_question_context(self, *, run_dir: Path, logger: Any, trace_store: TraceStore) -> None:
        self.pipeline.run_dir = run_dir
        self.pipeline.logger = logger
        self.pipeline.trace_store = trace_store
        self.pipeline.executor.logger = logger
        retriever = getattr(self.pipeline.executor, "retriever", None)
        if retriever is not None and hasattr(retriever, "logger"):
            retriever.logger = logger
        walker = getattr(self.pipeline.executor, "walker", None)
        if walker is not None and hasattr(walker, "logger"):
            walker.logger = logger
            anchor_resolver = getattr(walker, "_anchor_resolver", None)
            if anchor_resolver is not None and hasattr(anchor_resolver, "logger"):
                anchor_resolver.logger = logger
        _set_trace_store(self.pipeline.embedder, trace_store)
        llm_client = getattr(self.pipeline.llm_service, "client", None)
        _set_trace_store(llm_client, trace_store)


def _load_hyperbranch_config(config_path: Path, args: argparse.Namespace) -> Any:
    config = load_config(config_path, PROJECT_ROOT)
    if args.hyperbranch_mock_llm:
        config.llm.use_mock = True
    if args.hyperbranch_llm_model:
        config.llm.model = args.hyperbranch_llm_model
    elif args.llm_model:
        config.llm.model = args.llm_model
    if args.embedding_model:
        config.llm.embedding_model = args.embedding_model
    return config


def _set_trace_store(client: Any, trace_store: TraceStore) -> None:
    if client is not None and hasattr(client, "trace_store"):
        client.trace_store = trace_store


def _hyperbranch_dag_payload(decomposition_payload: dict[str, Any]) -> dict[str, Any]:
    dag = (((decomposition_payload.get("stages") or {}).get("6_atomic_question_dag")) or {})
    if not isinstance(dag, dict):
        raise ValueError("DEPO decomposition does not contain stages.6_atomic_question_dag.")
    if not dag.get("valid"):
        errors = dag.get("validation_errors") or []
        raise ValueError(f"DEPO atomic DAG is invalid: {errors}")
    nodes = dag.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("DEPO atomic DAG does not contain any nodes.")
    topic_entities = _depo_explicit_entity_texts(decomposition_payload)
    return {
        "question": decomposition_payload.get("question", ""),
        "topic_entities": topic_entities,
        "original_question_entities": topic_entities,
        "nodes": nodes,
        "edges": dag.get("edges") or [],
        "leaf_node_ids": dag.get("leaf_node_ids") or [],
        "source": "depo_stages.6_atomic_question_dag",
    }


def _depo_explicit_entity_texts(decomposition_payload: dict[str, Any]) -> list[str]:
    explicit = (((decomposition_payload.get("stages") or {}).get("1_explicit_entities")) or {})
    entities = explicit.get("entities") if isinstance(explicit, dict) else []
    entity_items = entities if isinstance(entities, list) else []
    texts: list[str] = []
    seen: set[str] = set()
    for item in entity_items:
        raw_text = item.get("text") if isinstance(item, dict) else item
        text = str(raw_text or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            texts.append(text)
    return texts


def _combined_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    config_path: Path,
    question_dir: Path,
    decomposition_payload: dict[str, Any],
    dag_path: Path,
    hyperbranch_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "ok",
        "method": METHOD,
        "dataset": dataset,
        "questions_file": str(questions_file),
        "config": str(config_path),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "output_dir": str(question_dir),
        "depo_decomposition_path": str(question_dir / "decomposition.json"),
        "hyperbranch_dag_path": str(dag_path),
        "hyperbranch_run_dir": hyperbranch_result.get("run_dir"),
        "stages": {
            "depo": decomposition_payload.get("stages"),
            "hyperbranch": {
                "dag_input": hyperbranch_result.get("artifacts", {}).get("dag_input"),
                "atomic_answers": hyperbranch_result.get("artifacts", {}).get("atomic_answers"),
                "final_answer": hyperbranch_result.get("final_answer"),
            },
        },
        "final_answer": hyperbranch_result.get("final_answer"),
    }


def _combined_markdown(payload: dict[str, Any]) -> str:
    final_answer = payload.get("final_answer") or {}
    dag = (((payload.get("stages") or {}).get("depo") or {}).get("6_atomic_question_dag")) or {}
    atomic_answers = (((payload.get("stages") or {}).get("hyperbranch") or {}).get("atomic_answers")) or []
    lines = [
        f"# DEPO + HyperBranch #{payload['index']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Question: {payload['question']}",
    ]
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.extend(
        [
            f"- HyperBranch run: `{payload.get('hyperbranch_run_dir')}`",
            "",
            "## Atomic DAG",
        ]
    )
    for node in dag.get("nodes", []) or []:
        depends_on = node.get("depends_on") or []
        lines.append(f"- {node.get('id')}: {node.get('question')}")
        lines.append(f"  - depends_on: {', '.join(depends_on) if depends_on else '(none)'}")
    lines.extend(["", "## Atomic Answers"])
    for answer in atomic_answers:
        lines.append(f"- {answer.get('node_id')}: {answer.get('answer')}")
        lines.append(f"  - question: {answer.get('question')}")
        lines.append(f"  - confidence: {answer.get('confidence')}")
    lines.extend(
        [
            "",
            "## Final Answer",
            str(final_answer.get("answer", "")),
            "",
            f"- confidence: {final_answer.get('confidence')}",
        ]
    )
    return "\n".join(lines)


def _manifest_item(payload: dict[str, Any]) -> dict[str, Any]:
    dag = (((payload.get("stages") or {}).get("depo") or {}).get("6_atomic_question_dag")) or {}
    final_answer = payload.get("final_answer") or {}
    return {
        "method": payload["method"],
        "dataset": payload["dataset"],
        "index": payload["index"],
        "qid": payload.get("qid"),
        "question": payload["question"],
        "gold_answer": payload.get("gold_answer"),
        "status": payload["status"],
        "dag_valid": dag.get("valid"),
        "dag_node_count": len(dag.get("nodes", []) or []),
        "final_answer": final_answer.get("answer"),
        "confidence": final_answer.get("confidence"),
        "output_dir": payload["output_dir"],
        "hyperbranch_run_dir": payload.get("hyperbranch_run_dir"),
    }


def _skipped_manifest_item(dataset: str, item: dict[str, Any], question_dir: Path) -> dict[str, Any]:
    return {
        "method": METHOD,
        "dataset": dataset,
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "gold_answer": item.get("answer"),
        "status": "skipped",
        "output_dir": str(question_dir),
    }


def _error_payload(
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    config_path: Path,
    question_dir: Path,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "status": "sample",
        "method": METHOD,
        "dataset": dataset,
        "questions_file": str(questions_file),
        "config": str(config_path),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "output_dir": str(question_dir),
        "error_type": type(exc).__name__,
        "sample": str(exc),
        "traceback": traceback.format_exc(),
    }


def _error_manifest_item(payload: dict[str, Any], question_dir: Path) -> dict[str, Any]:
    return {
        "method": METHOD,
        "dataset": payload["dataset"],
        "index": payload["index"],
        "qid": payload.get("qid"),
        "question": payload["question"],
        "gold_answer": payload.get("gold_answer"),
        "status": "sample",
        "error_type": payload["error_type"],
        "sample": payload["sample"],
        "output_dir": str(question_dir),
    }


def _error_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# DEPO + HyperBranch Error #{payload['index']}",
            "",
            f"- Dataset: `{payload['dataset']}`",
            f"- Question: {payload['question']}",
            f"- Error type: `{payload['error_type']}`",
            "",
            "```text",
            str(payload["sample"]),
            "```",
            "",
        ]
    )


def _summary_header(
    *,
    dataset: str,
    questions_file: Path,
    config_path: Path,
    run_id: str,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[str]:
    lines = [
        f"# DEPO + HyperBranch Run: {dataset}",
        "",
        f"- Run id: `{run_id}`",
        f"- Questions file: `{questions_file}`",
        f"- Config: `{config_path}`",
        f"- Range: `{start}-{end if end is not None else 'end'}`",
    ]
    if limit is not None:
        lines.append(f"- Limit: `{limit}`")
    lines.append("")
    return lines


def _summary_question_lines(payload: dict[str, Any]) -> list[str]:
    final_answer = payload.get("final_answer") or {}
    dag = (((payload.get("stages") or {}).get("depo") or {}).get("6_atomic_question_dag")) or {}
    return [
        f"## {payload['index']}. {payload['question']}",
        "",
        f"- Output: `{payload['output_dir']}`",
        f"- DAG valid: `{dag.get('valid')}`",
        f"- DAG nodes: `{len(dag.get('nodes', []) or [])}`",
        f"- Final answer: {final_answer.get('answer')}",
        "",
    ]


def _summary_error_lines(payload: dict[str, Any]) -> list[str]:
    return [
        f"## {payload['index']}. {payload['question']}",
        "",
        f"- Output: `{payload['output_dir']}`",
        f"- Status: sample",
        f"- Error: `{payload['error_type']}: {payload['sample']}`",
        "",
    ]


def _append_live_eval_record(
    records: list[dict[str, Any]],
    *,
    item: dict[str, Any],
    final_answer: Any,
    question_dir: Path,
    status: str,
) -> dict[str, Any]:
    question_entry = item.get("raw") if isinstance(item.get("raw"), dict) else item
    if not isinstance(question_entry, dict):
        question_entry = {"question": item["question"], "answer": item.get("answer")}
    answer = ""
    if isinstance(final_answer, dict):
        answer = str(final_answer.get("answer", "") or "").strip()
    record = {
        "question": item["question"],
        "golden_answers": eval_score._extract_gold_answers(question_entry),
        "context": list(question_entry.get("context", [])) if isinstance(question_entry.get("context"), list) else [],
        "nhops": question_entry.get("nhops"),
        "run_dir": str(question_dir / "hyperbranch_run"),
        "run_status": status if answer or status != "success" else "partial",
        "answer": answer,
        "generation": f"<answer>{answer}</answer>" if answer else "",
        "generation_explanation": "",
        "retrieved": [],
        "retrieved_knowledge": "",
    }
    scored = eval_score.evaluate_one(record)
    records.append(scored)
    return scored


def _maybe_report_live_eval(
    *,
    records: list[dict[str, Any]],
    eval_every: int,
    force: bool = False,
) -> None:
    if not records:
        return
    cadence = max(1, int(eval_every or 1))
    if not force and len(records) % cadence != 0:
        return
    summary = eval_score.summarize(records)
    overall = summary.get("overall", {})
    counts = summary.get("counts", {})
    em = overall.get("em")
    f1 = overall.get("f1")
    em_text = "N/A" if em is None else f"{float(em):.4f}"
    f1_text = "N/A" if f1 is None else f"{float(f1):.4f}"
    print(
        "[eval] "
        f"processed={counts.get('total', len(records))} "
        f"success={counts.get('success', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"missing={counts.get('missing', 0)} "
        f"EM={em_text} F1={f1_text}"
    )


def _config_path_for_dataset(args: argparse.Namespace, dataset: str) -> Path:
    if args.config:
        path = _repo_path(args.config)
    else:
        path = PROJECT_ROOT / "configs" / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"HyperBranch config not found: {path}")
    return path


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(depo_batch._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _close_logger(logger: Any) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
