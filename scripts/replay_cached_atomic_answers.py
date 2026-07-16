from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "eval"
for path in (PROJECT_ROOT, EVAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval import cal_em, cal_f1  # type: ignore  # noqa: E402
from hyper_branch.atomic.dependency_rewrite import resolve_dependency_question  # noqa: E402
from hyper_branch.atomic.executor import AtomicDagExecutor  # noqa: E402
from hyper_branch.atomic.models import (  # noqa: E402
    AtomicAnswerResult,
    AtomicQuestionAnalysis,
    AtomicQuestionNode,
    FusedHyperedgeCandidate,
)
from hyper_branch.atomic.retriever import LocalHyperedgeRetrievalResult  # noqa: E402
from hyper_branch.config import LLMConfig, load_config  # noqa: E402
from hyper_branch.llm.client import OpenAICompatibleClient  # noqa: E402
from hyper_branch.llm.prompts import PromptManager  # noqa: E402
from hyper_branch.llm.service import MockAtomicLLMService, OpenAIAtomicLLMService  # noqa: E402
from hyper_branch.logging_utils import TraceStore, configure_logging  # noqa: E402
from hyper_branch.pipeline import HyperBranchPipeline  # noqa: E402
from hyper_branch.utils import ensure_list, pretty_json, short_text  # noqa: E402


METHOD = "depo_hypermemory_cached_atomic_answer_replay"
FACT_QUERY_METHOD = "depo_hypermemory_fact_query_retrieval_replay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay atomic answering from a previous DEPO + HyperBranch run. "
            "It can reuse cached answerer evidence exactly or rerank cached candidate hyperedges with fact_query."
        )
    )
    parser.add_argument("source_run", help="Existing run directory containing manifest.jsonl.")
    parser.add_argument(
        "--output-root",
        default="runs/depo_hypermemory_answer_replay",
        help="Root for replay outputs. Ignored when --output-dir is supplied.",
    )
    parser.add_argument("--output-dir", help="Exact output directory for this replay run.")
    parser.add_argument("--run-id", help="Replay run id. Defaults to current timestamp.")
    parser.add_argument("--ids", default="", help="Comma-separated question ids/ranges, e.g. 47,163,175-180.")
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive source question index.")
    parser.add_argument("--end", type=int, help="1-based inclusive source question index.")
    parser.add_argument("--limit", type=int, help="Maximum number of selected questions to replay.")
    parser.add_argument(
        "--f1-zero-only",
        action="store_true",
        help="Replay only examples whose old final answer has F1=0 against gold.",
    )
    parser.add_argument(
        "--question-mode",
        choices=("cached", "recompute"),
        default="cached",
        help=(
            "cached: reuse old resolved atomic questions and old dependency payloads; "
            "recompute: rewrite dependency variables with newly replayed answers while reusing old evidence."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help=(
            "cached mode: limit cached evidence per atomic node, 0 means all; "
            "fact-query mode: override reranked local_hyperedge_top_k, 0 means config value."
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("cached", "fact-query"),
        default="cached",
        help=(
            "cached: reuse old answerer evidence exactly; "
            "fact-query: reuse cached DAG/analysis/candidate pools but rewrite each resolved atomic question "
            "to fact_query and rerank cached candidate hyperedges."
        ),
    )
    parser.add_argument(
        "--config",
        help=(
            "HyperBranch YAML config used by --retrieval-mode fact-query. "
            "Defaults to configs/<dataset>.yaml inferred from the source run."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        help="Override embedding model used by --retrieval-mode fact-query.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip questions whose replay pipeline.json exists.")
    parser.add_argument(
        "--store-full-evidence",
        action="store_true",
        help="Store full cached evidence records in hyperbranch_result.json. Defaults to compact prompt evidence.",
    )
    parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock answers; useful for smoke tests.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Chat model.")
    parser.add_argument("--prompt-dir", default="prompts", help="Prompt directory containing atomic_answer.md.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Chat temperature for replay.")
    parser.add_argument("--timeout", type=int, default=120, help="LLM request timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="LLM request retries.")
    parser.add_argument("--eval-every", type=int, default=10, help="Print live EM/F1 every N successful replays.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run = _resolve_path(args.source_run)
    if not source_run.exists():
        print(f"Source run does not exist: {source_run}", file=sys.stderr)
        return 2
    manifest_path = source_run / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"Source run is missing manifest.jsonl: {manifest_path}", file=sys.stderr)
        return 2

    rows = _select_rows(_read_jsonl(manifest_path), args)
    if not rows:
        print("No source rows selected.", file=sys.stderr)
        return 2

    output_dir = _output_dir(args, source_run, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"
    manifest_out = output_dir / "manifest.jsonl"

    fact_query_runtime: _FactQueryReplayRuntime | None = None
    if args.retrieval_mode == "fact-query":
        fact_query_runtime = _FactQueryReplayRuntime(
            config_path=_config_path_for_replay(args, source_run, rows),
            cache_dir=output_dir / "_hyperbranch_pipeline",
            args=args,
        )
        service = fact_query_runtime.service
    else:
        service = _make_llm_service(args)
    summary_lines = _summary_header(source_run, output_dir, args, rows)
    eval_records: list[dict[str, Any]] = []

    print(
        f"Replaying cached atomic answers: source={source_run} selected={len(rows)} "
        f"output={output_dir} question_mode={args.question_mode} retrieval_mode={args.retrieval_mode}"
    )
    with manifest_out.open("a", encoding="utf-8") as manifest_handle:
        for offset, row in enumerate(rows, start=1):
            source_question_dir = Path(str(row.get("output_dir") or ""))
            if not source_question_dir.is_absolute():
                source_question_dir = source_run / source_question_dir
            question_dir = output_dir / source_question_dir.name
            if args.resume and (question_dir / "pipeline.json").exists():
                existing = json.loads((question_dir / "pipeline.json").read_text(encoding="utf-8"))
                manifest_item = _manifest_item(existing)
                manifest_item["status"] = "skipped"
                manifest_handle.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                manifest_handle.flush()
                print(f"[skip] #{row.get('index')} {row.get('question')}")
                continue

            print(f"[run {offset}/{len(rows)}] #{row.get('index')} {row.get('question')}")
            question_dir.mkdir(parents=True, exist_ok=True)
            try:
                payload = replay_one_question(
                    row=row,
                    source_question_dir=source_question_dir,
                    output_question_dir=question_dir,
                    service=service,
                    fact_query_runtime=fact_query_runtime,
                    args=args,
                )
                _write_json(question_dir / "pipeline.json", payload)
                (question_dir / "pipeline.md").write_text(_pipeline_markdown(payload), encoding="utf-8")
                manifest_item = _manifest_item(payload)
                manifest_handle.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                manifest_handle.flush()
                summary_lines.extend(_summary_question_lines(payload))
                _append_eval(eval_records, manifest_item)
                _maybe_print_eval(eval_records, args.eval_every)
                print(
                    f"[ok]  #{payload['index']} old={manifest_item.get('old_final_answer')!r} "
                    f"new={manifest_item.get('final_answer')!r} f1={manifest_item.get('f1')}"
                )
            except Exception as exc:
                error_payload = _error_payload(row, source_question_dir, question_dir, exc)
                _write_json(question_dir / "sample.json", error_payload)
                (question_dir / "sample.md").write_text(_error_markdown(error_payload), encoding="utf-8")
                manifest_item = _error_manifest_item(error_payload)
                manifest_handle.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                manifest_handle.flush()
                summary_lines.extend(_summary_error_lines(error_payload))
                print(f"[err] #{row.get('index')} {type(exc).__name__}: {exc}")
            summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

    if fact_query_runtime is not None:
        fact_query_runtime.close()
    _maybe_print_eval(eval_records, args.eval_every, force=True)
    summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Summary written to {summary_path}")
    return 0


def replay_one_question(
    *,
    row: dict[str, Any],
    source_question_dir: Path,
    output_question_dir: Path,
    service: Any,
    fact_query_runtime: "_FactQueryReplayRuntime | None" = None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_pipeline = _read_optional_json(source_question_dir / "pipeline.json")
    original_question = str(row.get("question") or source_pipeline.get("question") or "")
    if not original_question:
        raise ValueError("Missing question text in manifest/pipeline.json.")

    _copy_if_exists(source_question_dir / "decomposition.json", output_question_dir / "decomposition.json")
    _copy_if_exists(source_question_dir / "decomposition.md", output_question_dir / "decomposition.md")
    _copy_if_exists(source_question_dir / "hyperbranch_dag.json", output_question_dir / "hyperbranch_dag.json")

    if args.retrieval_mode == "fact-query":
        if fact_query_runtime is None:
            raise ValueError("--retrieval-mode fact-query requires an initialized fact-query runtime.")
        return replay_one_question_with_fact_query_retrieval(
            row=row,
            source_pipeline=source_pipeline,
            source_question_dir=source_question_dir,
            output_question_dir=output_question_dir,
            original_question=original_question,
            runtime=fact_query_runtime,
            args=args,
        )

    dag_nodes = _load_dag_nodes(source_question_dir, source_pipeline)
    dag_nodes, dag_repair = AtomicDagExecutor.repair_dag_for_execution(dag_nodes)
    order = AtomicDagExecutor.topological_sort(dag_nodes)
    AtomicDagExecutor.validate_terminal_leaf(order)

    cached_analysis = _load_cached_analysis(source_question_dir)
    cached_retrieval = _load_cached_retrieval(source_question_dir)
    cached_answers = _load_cached_answers(source_question_dir)
    results_by_id: dict[str, AtomicAnswerResult] = {}
    replay_analyses: list[dict[str, Any]] = []
    replay_retrieval: list[dict[str, Any]] = []
    answer_inputs: list[dict[str, Any]] = []

    run_dir = output_question_dir / "hyperbranch_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_store = TraceStore(run_dir)
    _set_trace_store(service, trace_store)

    for node in order:
        retrieval_record = cached_retrieval.get(node.node_id, {})
        analysis_record = cached_analysis.get(node.node_id, {})
        old_answer_record = cached_answers.get(node.node_id, {})
        dependency_context = _dependency_context(node.dependencies, results_by_id)

        if args.question_mode == "recompute":
            rewrite = resolve_dependency_question(node.question, dependency_context)
            resolved_question = rewrite.retrieval_question
            rewrite_payload = rewrite.to_dict()
            dependency_answers_for_prompt = AtomicDagExecutor._answer_dependency_context(dependency_context)
        else:
            resolved_question = str(
                retrieval_record.get("resolved_question")
                or analysis_record.get("resolved_question")
                or old_answer_record.get("question")
                or node.question
            )
            rewrite_payload = dict(
                retrieval_record.get("dependency_question_rewrite")
                or analysis_record.get("dependency_question_rewrite")
                or {}
            )
            dependency_answers_for_prompt = AtomicDagExecutor._answer_dependency_context(
                ensure_list(retrieval_record.get("dependency_answers"))
            )

        analysis = _analysis_from_cached(analysis_record, old_answer_record)
        evidence = _evidence_from_cached(retrieval_record, old_answer_record, args.top_k)
        answer_contract = AtomicDagExecutor._answer_contract(resolved_question)
        evidence_payload = AtomicDagExecutor._answer_evidence_payload(evidence)
        answer_input = {
            "node_id": node.node_id,
            "original_question": original_question,
            "atomic_question": resolved_question,
            "answer_contract": answer_contract,
            "dependency_answers": dependency_answers_for_prompt,
            "evidence": list(evidence_payload.get("evidence", [])),
            "contexts": list(evidence_payload.get("contexts", [])),
            "question_mode": args.question_mode,
            "cached_resolved_question": str(
                retrieval_record.get("resolved_question") or old_answer_record.get("question") or ""
            ),
        }
        answer_inputs.append(answer_input)
        raw_payload = service.answer_atomic_question(
            atomic_question=resolved_question,
            answer_contract=answer_contract,
            dependency_answers=dependency_answers_for_prompt,
            evidence=evidence_payload,
            original_question=original_question,
        )
        answer_payload = _coerce_answer_payload(raw_payload, evidence)
        result = AtomicAnswerResult(
            node_id=node.node_id,
            question=resolved_question,
            analysis=analysis,
            evidence=evidence,
            answer=str(answer_payload.get("answer", "") or ""),
            reasoning_summary=str(answer_payload.get("reasoning_summary", "") or ""),
            used_dependencies=list(node.dependencies),
            used_hyperedge_ids=list(answer_payload.get("used_hyperedge_ids", [])),
            insufficient=bool(answer_payload.get("insufficient", False)),
        )
        results_by_id[node.node_id] = result
        replay_analyses.append(
            {
                "node_id": node.node_id,
                "question": resolved_question,
                "original_question": node.question,
                "resolved_question": resolved_question,
                "retrieval_question": resolved_question,
                "dependency_answers": dependency_answers_for_prompt,
                "dependency_question_rewrite": rewrite_payload,
                "analysis": analysis.to_dict(),
                "replay_source": {
                    "cached_question": old_answer_record.get("question") or retrieval_record.get("resolved_question"),
                    "cached_answer": old_answer_record.get("answer"),
                },
            }
        )
        replay_retrieval.append(
            _replay_retrieval_artifact(
                retrieval_record=retrieval_record,
                node=node,
                resolved_question=resolved_question,
                dependency_answers=dependency_answers_for_prompt,
                rewrite_payload=rewrite_payload,
                evidence=evidence,
                answer_payload=answer_payload,
                store_full_evidence=args.store_full_evidence,
            )
        )

    atomic_results = [results_by_id[node.node_id] for node in order]
    final_answer = AtomicDagExecutor._final_answer_from_terminal_node(atomic_results[-1], atomic_results)
    artifacts = {
        "dag_input": [node.to_dict() for node in dag_nodes],
        "dag_repair": dag_repair,
        "original_question_analysis": _read_optional_json(
            source_question_dir / "hyperbranch_run" / "artifacts" / "original_question_analysis.json"
        ),
        "atomic_question_analyses": replay_analyses,
        "atomic_retrieval": replay_retrieval,
        "atomic_answer_inputs": answer_inputs,
        "atomic_answers": [
            _atomic_result_payload(result, store_full_evidence=args.store_full_evidence)
            for result in atomic_results
        ],
        "final_answer": final_answer,
        "replay": {
            "method": METHOD,
            "source_question_dir": str(source_question_dir),
            "question_mode": args.question_mode,
            "top_k": args.top_k,
            "store_full_evidence": bool(args.store_full_evidence),
        },
    }
    hyperbranch_result = {
        "original_question": original_question,
        "atomic_results": artifacts["atomic_answers"],
        "final_answer": final_answer,
        "artifacts": artifacts,
        "run_dir": str(run_dir),
    }
    _write_json(output_question_dir / "hyperbranch_result.json", hyperbranch_result)
    for name, value in artifacts.items():
        trace_store.save_artifact(f"artifacts/{name}.json", value)

    payload = _combined_payload(
        row=row,
        source_pipeline=source_pipeline,
        source_question_dir=source_question_dir,
        output_question_dir=output_question_dir,
        hyperbranch_result=hyperbranch_result,
        args=args,
    )
    return payload


def replay_one_question_with_fact_query_retrieval(
    *,
    row: dict[str, Any],
    source_pipeline: dict[str, Any],
    source_question_dir: Path,
    output_question_dir: Path,
    original_question: str,
    runtime: "_FactQueryReplayRuntime",
    args: argparse.Namespace,
) -> dict[str, Any]:
    dag_nodes = _load_dag_nodes(source_question_dir, source_pipeline)
    dag_nodes, dag_repair = AtomicDagExecutor.repair_dag_for_execution(dag_nodes)
    order = AtomicDagExecutor.topological_sort(dag_nodes)
    AtomicDagExecutor.validate_terminal_leaf(order)

    cached_analysis = _load_cached_analysis(source_question_dir)
    cached_retrieval = _load_cached_retrieval(source_question_dir)
    cached_answers = _load_cached_answers(source_question_dir)
    results_by_id: dict[str, AtomicAnswerResult] = {}
    replay_analyses: list[dict[str, Any]] = []
    replay_retrieval: list[dict[str, Any]] = []
    answer_inputs: list[dict[str, Any]] = []

    run_dir = output_question_dir / "hyperbranch_run"
    trace_store = runtime.bind_question_run(run_dir)

    for node in order:
        retrieval_record = cached_retrieval.get(node.node_id, {})
        analysis_record = cached_analysis.get(node.node_id, {})
        old_answer_record = cached_answers.get(node.node_id, {})
        dependency_context = _dependency_context(node.dependencies, results_by_id)

        if args.question_mode == "recompute":
            rewrite = resolve_dependency_question(node.question, dependency_context)
            resolved_question = rewrite.retrieval_question
            rewrite_payload = rewrite.to_dict()
            dependency_answers_for_prompt = AtomicDagExecutor._answer_dependency_context(dependency_context)
        else:
            resolved_question = str(
                retrieval_record.get("resolved_question")
                or analysis_record.get("resolved_question")
                or old_answer_record.get("question")
                or node.question
            )
            rewrite_payload = dict(
                retrieval_record.get("dependency_question_rewrite")
                or analysis_record.get("dependency_question_rewrite")
                or {}
            )
            dependency_answers_for_prompt = AtomicDagExecutor._answer_dependency_context(
                ensure_list(retrieval_record.get("dependency_answers"))
            )

        analysis = _analysis_from_cached(analysis_record, old_answer_record)
        fact_query = _fact_query_for_replay(
            service=runtime.service,
            node_id=node.node_id,
            resolved_question=resolved_question,
            analysis=analysis,
            retrieval_record=retrieval_record,
            analysis_record=analysis_record,
        )
        candidate_pool = _candidate_pool_from_cached(retrieval_record)
        retrieval_result = runtime.retriever.rank_candidate_pool(candidate_pool, question=fact_query)
        evidence = retrieval_result.evidence
        answer_contract = AtomicDagExecutor._answer_contract(resolved_question)
        evidence_payload = AtomicDagExecutor._answer_evidence_payload(evidence)
        answer_input = {
            "node_id": node.node_id,
            "original_question": original_question,
            "atomic_question": resolved_question,
            "fact_query": fact_query,
            "hyperedge_retrieval_query": fact_query,
            "answer_contract": answer_contract,
            "dependency_answers": dependency_answers_for_prompt,
            "evidence": list(evidence_payload.get("evidence", [])),
            "contexts": list(evidence_payload.get("contexts", [])),
            "question_mode": args.question_mode,
            "retrieval_mode": args.retrieval_mode,
            "cached_resolved_question": str(
                retrieval_record.get("resolved_question") or old_answer_record.get("question") or ""
            ),
        }
        answer_inputs.append(answer_input)
        raw_payload = runtime.service.answer_atomic_question(
            atomic_question=resolved_question,
            answer_contract=answer_contract,
            dependency_answers=dependency_answers_for_prompt,
            evidence=evidence_payload,
            original_question=original_question,
        )
        answer_payload = _coerce_answer_payload(raw_payload, evidence)
        result = AtomicAnswerResult(
            node_id=node.node_id,
            question=resolved_question,
            analysis=analysis,
            evidence=evidence,
            answer=str(answer_payload.get("answer", "") or ""),
            reasoning_summary=str(answer_payload.get("reasoning_summary", "") or ""),
            used_dependencies=list(node.dependencies),
            used_hyperedge_ids=list(answer_payload.get("used_hyperedge_ids", [])),
            insufficient=bool(answer_payload.get("insufficient", False)),
        )
        results_by_id[node.node_id] = result
        replay_analyses.append(
            {
                "node_id": node.node_id,
                "question": resolved_question,
                "original_question": node.question,
                "resolved_question": resolved_question,
                "retrieval_question": resolved_question,
                "fact_query": fact_query,
                "hyperedge_retrieval_query": fact_query,
                "dependency_answers": dependency_answers_for_prompt,
                "dependency_question_rewrite": rewrite_payload,
                "analysis": analysis.to_dict(),
                "replay_source": {
                    "cached_question": old_answer_record.get("question") or retrieval_record.get("resolved_question"),
                    "cached_answer": old_answer_record.get("answer"),
                    "cached_fact_query": retrieval_record.get("fact_query") or analysis_record.get("fact_query"),
                },
            }
        )
        replay_retrieval.append(
            _fact_query_retrieval_artifact(
                retrieval_record=retrieval_record,
                node=node,
                resolved_question=resolved_question,
                fact_query=fact_query,
                dependency_answers=dependency_answers_for_prompt,
                rewrite_payload=rewrite_payload,
                retrieval_result=retrieval_result,
                answer_payload=answer_payload,
                store_full_evidence=args.store_full_evidence,
            )
        )

    atomic_results = [results_by_id[node.node_id] for node in order]
    final_answer = AtomicDagExecutor._final_answer_from_terminal_node(atomic_results[-1], atomic_results)
    artifacts = {
        "dag_input": [node.to_dict() for node in dag_nodes],
        "dag_repair": dag_repair,
        "original_question_analysis": _read_optional_json(
            source_question_dir / "hyperbranch_run" / "artifacts" / "original_question_analysis.json"
        ),
        "atomic_question_analyses": replay_analyses,
        "atomic_retrieval": replay_retrieval,
        "atomic_answer_inputs": answer_inputs,
        "atomic_answers": [
            _atomic_result_payload(result, store_full_evidence=args.store_full_evidence)
            for result in atomic_results
        ],
        "final_answer": final_answer,
        "replay": {
            "method": FACT_QUERY_METHOD,
            "source_question_dir": str(source_question_dir),
            "question_mode": args.question_mode,
            "retrieval_mode": args.retrieval_mode,
            "top_k": args.top_k,
            "effective_top_k": runtime.config.retrieval.local_hyperedge_top_k,
            "store_full_evidence": bool(args.store_full_evidence),
            "config": str(runtime.config_path),
            "reused_cached_candidate_pools": True,
        },
    }
    hyperbranch_result = {
        "original_question": original_question,
        "atomic_results": artifacts["atomic_answers"],
        "final_answer": final_answer,
        "artifacts": artifacts,
        "run_dir": str(run_dir),
    }
    _write_json(output_question_dir / "hyperbranch_result.json", hyperbranch_result)
    for name, value in artifacts.items():
        trace_store.save_artifact(f"artifacts/{name}.json", value)

    return _combined_payload(
        row=row,
        source_pipeline=source_pipeline,
        source_question_dir=source_question_dir,
        output_question_dir=output_question_dir,
        hyperbranch_result=hyperbranch_result,
        args=args,
    )


class _FactQueryReplayRuntime:
    def __init__(self, *, config_path: Path, cache_dir: Path, args: argparse.Namespace) -> None:
        self.config_path = config_path
        self.config = _load_fact_query_config(config_path, args)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger = configure_logging(self.cache_dir, self.config.runtime.log_level)
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
        self.service = self.pipeline.llm_service
        self.retriever = self.pipeline.retriever
        self._logger: Any | None = None

    def bind_question_run(self, run_dir: Path) -> TraceStore:
        run_dir.mkdir(parents=True, exist_ok=True)
        if self._logger is not None:
            _close_logger(self._logger)
        logger = configure_logging(run_dir, self.config.runtime.log_level)
        trace_store = TraceStore(run_dir)
        self._logger = logger
        self.pipeline.run_dir = run_dir
        self.pipeline.logger = logger
        self.pipeline.trace_store = trace_store
        self.pipeline.executor.logger = logger
        self.retriever.logger = logger
        _set_trace_store(self.pipeline.embedder, trace_store)
        _set_trace_store(self.pipeline.llm_service, trace_store)
        return trace_store

    def close(self) -> None:
        if self._logger is not None:
            _close_logger(self._logger)
            self._logger = None


def _load_fact_query_config(config_path: Path, args: argparse.Namespace) -> Any:
    config = load_config(config_path, PROJECT_ROOT)
    if args.mock_llm:
        config.llm.use_mock = True
    if args.api_key:
        os.environ[config.llm.api_key_env] = args.api_key
    if args.base_url:
        os.environ[config.llm.base_url_env] = args.base_url
    if not os.getenv(config.llm.base_url_env) and os.getenv("OPAI_BASE_URL"):
        os.environ[config.llm.base_url_env] = str(os.getenv("OPAI_BASE_URL"))
    if args.model:
        config.llm.model = args.model
    if args.embedding_model:
        config.llm.embedding_model = args.embedding_model
    config.llm.timeout_seconds = args.timeout
    config.llm.max_retries = args.max_retries
    config.llm.temperature = args.temperature
    if args.top_k > 0:
        config.retrieval.local_hyperedge_top_k = args.top_k
    return config


def _make_llm_service(args: argparse.Namespace) -> Any:
    if args.mock_llm:
        return MockAtomicLLMService()
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if not os.getenv("OPENAI_BASE_URL") and os.getenv("OPAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = str(os.getenv("OPAI_BASE_URL"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    llm_config = LLMConfig(
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        model=args.model,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=1.0,
        temperature=args.temperature,
    )
    client = OpenAICompatibleClient(llm_config)
    return OpenAIAtomicLLMService(client=client, prompts=PromptManager(_resolve_path(args.prompt_dir)))


def _set_trace_store(target: Any, trace_store: TraceStore) -> None:
    if target is not None and hasattr(target, "trace_store"):
        target.trace_store = trace_store
    client = getattr(target, "client", None)
    if client is not None and hasattr(client, "trace_store"):
        client.trace_store = trace_store


def _load_dag_nodes(source_question_dir: Path, source_pipeline: dict[str, Any]) -> list[AtomicQuestionNode]:
    dag_input_path = source_question_dir / "hyperbranch_run" / "artifacts" / "dag_input.json"
    dag_payload_path = source_question_dir / "hyperbranch_dag.json"
    if dag_input_path.exists():
        payload = json.loads(dag_input_path.read_text(encoding="utf-8"))
    elif dag_payload_path.exists():
        payload = json.loads(dag_payload_path.read_text(encoding="utf-8"))
    else:
        payload = (((source_pipeline.get("stages") or {}).get("hyperbranch") or {}).get("dag_input")) or None
    return AtomicDagExecutor.normalize_dag_payload(payload, original_question=str(source_pipeline.get("question") or ""))


def _load_cached_analysis(source_question_dir: Path) -> dict[str, dict[str, Any]]:
    path = source_question_dir / "hyperbranch_run" / "artifacts" / "atomic_question_analyses.json"
    rows = _read_optional_json(path, default=[])
    return {str(item.get("node_id") or ""): item for item in rows if isinstance(item, dict)}


def _load_cached_retrieval(source_question_dir: Path) -> dict[str, dict[str, Any]]:
    path = source_question_dir / "hyperbranch_run" / "artifacts" / "atomic_retrieval.json"
    rows = _read_optional_json(path, default=[])
    return {str(item.get("node_id") or ""): item for item in rows if isinstance(item, dict)}


def _load_cached_answers(source_question_dir: Path) -> dict[str, dict[str, Any]]:
    path = source_question_dir / "hyperbranch_run" / "artifacts" / "atomic_answers.json"
    rows = _read_optional_json(path, default=[])
    if not rows:
        result = _read_optional_json(source_question_dir / "hyperbranch_result.json")
        rows = ensure_list(result.get("atomic_results"))
    return {str(item.get("node_id") or ""): item for item in rows if isinstance(item, dict)}


def _analysis_from_cached(
    analysis_record: dict[str, Any],
    old_answer_record: dict[str, Any],
) -> AtomicQuestionAnalysis:
    raw = analysis_record.get("analysis") if isinstance(analysis_record.get("analysis"), dict) else {}
    if not raw and isinstance(old_answer_record.get("analysis"), dict):
        raw = old_answer_record["analysis"]
    return AtomicQuestionAnalysis(
        entities=[str(item) for item in ensure_list(raw.get("entities")) if str(item).strip()],
        answer_type=str(raw.get("answer_type", "") or ""),
    )


def _fact_query_for_replay(
    *,
    service: Any,
    node_id: str,
    resolved_question: str,
    analysis: AtomicQuestionAnalysis,
    retrieval_record: dict[str, Any],
    analysis_record: dict[str, Any],
) -> str:
    cached = _clean_fact_query_text(
        retrieval_record.get("fact_query")
        or retrieval_record.get("hyperedge_retrieval_query")
        or analysis_record.get("fact_query")
        or analysis_record.get("hyperedge_retrieval_query")
    )
    if cached:
        return cached
    if service is None or not hasattr(service, "rewrite_atomic_fact_query"):
        return resolved_question
    try:
        payload = service.rewrite_atomic_fact_query(
            atomic_question=resolved_question,
            answer_type=analysis.answer_type,
        )
    except Exception as exc:
        logging.getLogger("hyper_branch.replay").warning(
            "Atomic fact query rewrite failed for %s; using resolved question: %s",
            node_id,
            exc,
        )
        return resolved_question
    fact_query = _clean_fact_query_payload(payload)
    return fact_query or resolved_question


def _candidate_pool_from_cached(retrieval_record: dict[str, Any]) -> LocalHyperedgeRetrievalResult:
    candidate_ids = _candidate_hyperedge_ids_from_cached(retrieval_record)
    candidate_sources = _candidate_sources_from_cached(retrieval_record, candidate_ids)
    return LocalHyperedgeRetrievalResult(
        method=str(retrieval_record.get("method") or "cached_candidate_pool_fact_query_rerank"),
        primary_anchor_mention=str(retrieval_record.get("primary_anchor_mention", "") or ""),
        linked_entity_id=str(retrieval_record.get("linked_entity_id", "") or ""),
        anchor_match=dict(retrieval_record.get("anchor_match") or {}),
        anchor_mentions=[str(item) for item in ensure_list(retrieval_record.get("anchor_mentions"))],
        linked_entities=[dict(item) for item in ensure_list(retrieval_record.get("linked_entities")) if isinstance(item, dict)],
        anchor_matches=[dict(item) for item in ensure_list(retrieval_record.get("anchor_matches")) if isinstance(item, dict)],
        unlinked_anchor_mentions=[
            str(item) for item in ensure_list(retrieval_record.get("unlinked_anchor_mentions"))
        ],
        adjacent_hyperedge_ids=[str(item) for item in ensure_list(retrieval_record.get("adjacent_hyperedge_ids"))],
        expansion_entity_ids=[str(item) for item in ensure_list(retrieval_record.get("expansion_entity_ids"))],
        second_hop_hyperedge_ids=[
            str(item) for item in ensure_list(retrieval_record.get("second_hop_hyperedge_ids"))
        ],
        candidate_hyperedge_ids=candidate_ids,
        shared_candidate_hyperedge_ids=[
            str(item) for item in ensure_list(retrieval_record.get("shared_candidate_hyperedge_ids"))
        ],
        local_candidate_hyperedge_ids=[
            str(item) for item in ensure_list(retrieval_record.get("local_candidate_hyperedge_ids"))
        ],
        candidate_sources=candidate_sources,
        insufficient_reason=str(retrieval_record.get("insufficient_reason", "") or ""),
        local_insufficient_reason=str(retrieval_record.get("local_insufficient_reason", "") or ""),
        shared_insufficient_reason=str(retrieval_record.get("shared_insufficient_reason", "") or ""),
        fallback_reason=str(retrieval_record.get("fallback_reason", "") or ""),
    )


def _candidate_hyperedge_ids_from_cached(retrieval_record: dict[str, Any]) -> list[str]:
    raw_ids = ensure_list(retrieval_record.get("candidate_hyperedge_ids"))
    if not raw_ids:
        raw_ids = [item.get("hyperedge_id") for item in ensure_list(retrieval_record.get("top_hyperedges")) if isinstance(item, dict)]
    if not raw_ids:
        raw_ids = [
            item.get("hyperedge_id")
            for item in ensure_list(retrieval_record.get("answerer_evidence") or retrieval_record.get("top_evidence"))
            if isinstance(item, dict)
        ]
    return _dedupe_strings(str(item) for item in raw_ids if str(item or "").strip())


def _candidate_sources_from_cached(
    retrieval_record: dict[str, Any],
    candidate_ids: list[str],
) -> list[dict[str, Any]]:
    sources = [
        dict(item)
        for item in ensure_list(retrieval_record.get("candidate_sources"))
        if isinstance(item, dict) and str(item.get("hyperedge_id", "") or "").strip()
    ]
    existing = {str(item.get("hyperedge_id")) for item in sources}
    top_metadata: dict[str, dict[str, Any]] = {}
    for item in ensure_list(retrieval_record.get("top_hyperedges")):
        if isinstance(item, dict) and str(item.get("hyperedge_id", "") or "").strip():
            top_metadata[str(item["hyperedge_id"])] = item
    for hyperedge_id in candidate_ids:
        if hyperedge_id in existing:
            continue
        source = dict(top_metadata.get(hyperedge_id) or {})
        source.update(
            {
                "hyperedge_id": hyperedge_id,
                "hop": int(source.get("hop", 1) or 1),
                "pool_sources": ensure_list(source.get("pool_sources")) or ["cached_candidate_pool"],
            }
        )
        sources.append(source)
    return sources


def _evidence_from_cached(
    retrieval_record: dict[str, Any],
    old_answer_record: dict[str, Any],
    top_k: int,
) -> list[FusedHyperedgeCandidate]:
    raw_evidence = ensure_list(
        retrieval_record.get("answerer_evidence")
        or retrieval_record.get("top_evidence")
        or old_answer_record.get("evidence")
    )
    if top_k > 0:
        raw_evidence = raw_evidence[:top_k]
    return [_fused_candidate_from_payload(item) for item in raw_evidence if isinstance(item, dict)]


def _fused_candidate_from_payload(item: dict[str, Any]) -> FusedHyperedgeCandidate:
    return FusedHyperedgeCandidate(
        hyperedge_id=str(item.get("hyperedge_id", "") or ""),
        hyperedge_text=str(item.get("hyperedge_text", "") or ""),
        branch_support={str(value) for value in ensure_list(item.get("branch_support"))},
        anchor_score=float(item.get("anchor_score", 0.0) or 0.0),
        relation_score=float(item.get("relation_score", 0.0) or 0.0),
        semantic_score=float(item.get("semantic_score", 0.0) or 0.0),
        fusion_score=float(item.get("fusion_score", 0.0) or 0.0),
        entity_ids=[str(value) for value in ensure_list(item.get("entity_ids"))],
        entity_records=[value for value in ensure_list(item.get("entity_records")) if isinstance(value, dict)],
        chunk_ids=[str(value) for value in ensure_list(item.get("chunk_ids"))],
        chunk_texts=[str(value) for value in ensure_list(item.get("chunk_texts"))],
        evidence_texts=[str(value) for value in ensure_list(item.get("evidence_texts"))],
        rank=int(item["rank"]) if str(item.get("rank", "")).strip().isdigit() else None,
        score_breakdown=dict(item.get("score_breakdown") or {}),
    )


def _coerce_answer_payload(payload: Any, evidence: list[FusedHyperedgeCandidate]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    answer = str(payload.get("answer", "") or "").strip() or "INSUFFICIENT_EVIDENCE"
    insufficient = answer.upper() == "INSUFFICIENT_EVIDENCE"
    supporting_evidence_ids = [
        str(item).strip()
        for item in ensure_list(payload.get("supporting_evidence_ids"))
        if str(item).strip()
    ]
    used_hyperedge_ids = [
        str(item).strip()
        for item in ensure_list(payload.get("used_hyperedge_ids"))
        if str(item).strip()
    ]
    if not used_hyperedge_ids:
        used_hyperedge_ids = _hyperedge_ids_from_evidence_ids(supporting_evidence_ids, evidence)
    if not used_hyperedge_ids and not insufficient and evidence:
        used_hyperedge_ids = [evidence[0].hyperedge_id]
    return {
        "answer": "INSUFFICIENT_EVIDENCE" if insufficient else answer,
        "reasoning_summary": str(payload.get("reasoning_summary", "") or ""),
        "used_hyperedge_ids": _dedupe_strings(used_hyperedge_ids),
        "supporting_evidence_ids": _dedupe_strings(supporting_evidence_ids),
        "raw_answer_payload": payload,
        "insufficient": insufficient,
    }


def _hyperedge_ids_from_evidence_ids(
    evidence_ids: Iterable[str],
    evidence: list[FusedHyperedgeCandidate],
) -> list[str]:
    mapped: list[str] = []
    for evidence_id in evidence_ids:
        text = str(evidence_id).strip().upper()
        if not text.startswith("E"):
            continue
        index_text = text[1:]
        if not index_text.isdigit():
            continue
        index = int(index_text) - 1
        if 0 <= index < len(evidence):
            mapped.append(evidence[index].hyperedge_id)
    return mapped


def _dependency_context(
    dependencies: list[str],
    results_by_id: dict[str, AtomicAnswerResult],
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for node_id in dependencies:
        result = results_by_id.get(node_id)
        if result is None:
            continue
        context.append(
            {
                "node_id": result.node_id,
                "question": result.question,
                "resolved_question": result.question,
                "answer": result.answer,
                "answer_type": result.analysis.answer_type,
                "reasoning_summary": result.reasoning_summary,
                "used_hyperedge_ids": list(result.used_hyperedge_ids),
                "insufficient": result.insufficient,
                "evidence_summary": [
                    {
                        "hyperedge_id": evidence.hyperedge_id,
                        "semantic_score": evidence.semantic_score,
                        "rank": evidence.rank,
                        "evidence_texts": evidence.evidence_texts[:2],
                    }
                    for evidence in result.evidence[:3]
                ],
            }
        )
    return context


def _replay_retrieval_artifact(
    *,
    retrieval_record: dict[str, Any],
    node: AtomicQuestionNode,
    resolved_question: str,
    dependency_answers: list[dict[str, Any]],
    rewrite_payload: dict[str, Any],
    evidence: list[FusedHyperedgeCandidate],
    answer_payload: dict[str, Any],
    store_full_evidence: bool,
) -> dict[str, Any]:
    if store_full_evidence:
        artifact = dict(retrieval_record)
    else:
        artifact = {
            "method": retrieval_record.get("method"),
            "primary_anchor_mention": retrieval_record.get("primary_anchor_mention"),
            "linked_entity_id": retrieval_record.get("linked_entity_id"),
            "anchor_match": retrieval_record.get("anchor_match"),
            "anchor_mentions": retrieval_record.get("anchor_mentions", []),
            "linked_entities": retrieval_record.get("linked_entities", []),
            "candidate_hyperedge_count": len(ensure_list(retrieval_record.get("candidate_hyperedge_ids"))),
            "candidate_hyperedge_ids_sample": ensure_list(retrieval_record.get("candidate_hyperedge_ids"))[:50],
            "shared_candidate_hyperedge_count": len(ensure_list(retrieval_record.get("shared_candidate_hyperedge_ids"))),
            "shared_candidate_hyperedge_ids_sample": ensure_list(retrieval_record.get("shared_candidate_hyperedge_ids"))[:50],
            "local_candidate_hyperedge_count": len(ensure_list(retrieval_record.get("local_candidate_hyperedge_ids"))),
            "local_candidate_hyperedge_ids_sample": ensure_list(retrieval_record.get("local_candidate_hyperedge_ids"))[:50],
            "insufficient_reason": retrieval_record.get("insufficient_reason", ""),
            "local_insufficient_reason": retrieval_record.get("local_insufficient_reason", ""),
            "shared_insufficient_reason": retrieval_record.get("shared_insufficient_reason", ""),
            "fallback_reason": retrieval_record.get("fallback_reason", ""),
            "cached_answerer_evidence_count": len(ensure_list(retrieval_record.get("answerer_evidence"))),
            "cached_top_hyperedge_count": len(ensure_list(retrieval_record.get("top_hyperedges"))),
        }
    artifact.update(
        {
            "node_id": node.node_id,
            "original_question": node.question,
            "resolved_question": resolved_question,
            "retrieval_question": resolved_question,
            "dependency_answers": dependency_answers,
            "dependency_question_rewrite": rewrite_payload,
            "dependency_replacements": rewrite_payload.get("dependency_replacements", []),
            "dependency_answers_used": rewrite_payload.get("dependency_answers_used", []),
            "unresolved_dependency": rewrite_payload.get("unresolved_dependencies", []),
            "answerer_evidence": _evidence_payload_for_artifact(evidence, store_full_evidence),
            "top_evidence": _evidence_payload_for_artifact(evidence, store_full_evidence),
            "atomic_answer": answer_payload,
            "replay_reused_cached_candidates": True,
        }
    )
    return artifact


def _fact_query_retrieval_artifact(
    *,
    retrieval_record: dict[str, Any],
    node: AtomicQuestionNode,
    resolved_question: str,
    fact_query: str,
    dependency_answers: list[dict[str, Any]],
    rewrite_payload: dict[str, Any],
    retrieval_result: LocalHyperedgeRetrievalResult,
    answer_payload: dict[str, Any],
    store_full_evidence: bool,
) -> dict[str, Any]:
    retrieval_payload = retrieval_result.to_artifact()
    if store_full_evidence:
        artifact = dict(retrieval_payload)
    else:
        artifact = {
            "method": retrieval_payload.get("method"),
            "primary_anchor_mention": retrieval_payload.get("primary_anchor_mention"),
            "linked_entity_id": retrieval_payload.get("linked_entity_id"),
            "anchor_match": retrieval_payload.get("anchor_match"),
            "anchor_mentions": retrieval_payload.get("anchor_mentions", []),
            "linked_entities": retrieval_payload.get("linked_entities", []),
            "candidate_hyperedge_count": len(ensure_list(retrieval_payload.get("candidate_hyperedge_ids"))),
            "candidate_hyperedge_ids_sample": ensure_list(retrieval_payload.get("candidate_hyperedge_ids"))[:50],
            "shared_candidate_hyperedge_count": len(ensure_list(retrieval_payload.get("shared_candidate_hyperedge_ids"))),
            "shared_candidate_hyperedge_ids_sample": ensure_list(retrieval_payload.get("shared_candidate_hyperedge_ids"))[:50],
            "local_candidate_hyperedge_count": len(ensure_list(retrieval_payload.get("local_candidate_hyperedge_ids"))),
            "local_candidate_hyperedge_ids_sample": ensure_list(retrieval_payload.get("local_candidate_hyperedge_ids"))[:50],
            "insufficient_reason": retrieval_payload.get("insufficient_reason", ""),
            "local_insufficient_reason": retrieval_payload.get("local_insufficient_reason", ""),
            "shared_insufficient_reason": retrieval_payload.get("shared_insufficient_reason", ""),
            "fallback_reason": retrieval_payload.get("fallback_reason", ""),
            "cached_candidate_hyperedge_count": len(ensure_list(retrieval_record.get("candidate_hyperedge_ids"))),
            "cached_top_hyperedge_count": len(ensure_list(retrieval_record.get("top_hyperedges"))),
        }
    artifact.update(
        {
            "node_id": node.node_id,
            "original_question": node.question,
            "resolved_question": resolved_question,
            "retrieval_question": resolved_question,
            "fact_query": fact_query,
            "hyperedge_retrieval_query": fact_query,
            "dependency_answers": dependency_answers,
            "dependency_question_rewrite": rewrite_payload,
            "dependency_replacements": rewrite_payload.get("dependency_replacements", []),
            "dependency_answers_used": rewrite_payload.get("dependency_answers_used", []),
            "unresolved_dependency": rewrite_payload.get("unresolved_dependencies", []),
            "top_hyperedges": retrieval_payload.get("top_hyperedges", []),
            "answerer_evidence": _evidence_payload_for_artifact(retrieval_result.evidence, store_full_evidence),
            "top_evidence": _evidence_payload_for_artifact(retrieval_result.evidence, store_full_evidence),
            "atomic_answer": answer_payload,
            "replay_reused_cached_candidates": True,
            "replay_retrieval_mode": "fact-query",
        }
    )
    return artifact


def _atomic_result_payload(result: AtomicAnswerResult, *, store_full_evidence: bool) -> dict[str, Any]:
    payload = result.to_dict()
    payload["evidence"] = _evidence_payload_for_artifact(result.evidence, store_full_evidence)
    return payload


def _evidence_payload_for_artifact(
    evidence: list[FusedHyperedgeCandidate],
    store_full_evidence: bool,
) -> list[dict[str, Any]]:
    if store_full_evidence:
        return [item.to_dict() for item in evidence]
    payload: list[dict[str, Any]] = []
    for index, item in enumerate(evidence, start=1):
        payload.append(
            {
                "evidence_id": f"E{index}",
                "hyperedge_id": item.hyperedge_id,
                "hyperedge_text": item.hyperedge_text,
                "chunk_texts": list(item.chunk_texts),
                "rank": item.rank,
                "semantic_score": item.semantic_score,
                "fusion_score": item.fusion_score,
                "score_breakdown": {
                    key: item.score_breakdown.get(key)
                    for key in ("selection_source", "semantic_rank", "primary_anchor_mention")
                    if key in item.score_breakdown
                },
            }
        )
    return payload


def _combined_payload(
    *,
    row: dict[str, Any],
    source_pipeline: dict[str, Any],
    source_question_dir: Path,
    output_question_dir: Path,
    hyperbranch_result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    old_final = str(row.get("final_answer") or "")
    final_answer = hyperbranch_result.get("final_answer") or {}
    new_final = str(final_answer.get("answer", "") or "")
    gold = row.get("gold_answer")
    old_em, old_f1 = _score(gold, old_final)
    new_em, new_f1 = _score(gold, new_final)
    payload = dict(source_pipeline) if source_pipeline else {}
    payload.update(
        {
            "status": "ok",
            "method": _replay_method(args),
            "replay_source_method": row.get("method"),
            "source_output_dir": str(source_question_dir),
            "source_hyperbranch_run_dir": str(row.get("hyperbranch_run_dir") or ""),
            "output_dir": str(output_question_dir),
            "hyperbranch_run_dir": hyperbranch_result.get("run_dir"),
            "index": row.get("index"),
            "qid": row.get("qid"),
            "question": row.get("question") or payload.get("question"),
            "gold_answer": gold,
            "old_final_answer": old_final,
            "final_answer": final_answer,
            "replay": {
                "question_mode": args.question_mode,
                "retrieval_mode": args.retrieval_mode,
                "top_k": args.top_k,
                "model": args.model if not args.mock_llm else "mock",
                "prompt_dir": str(_resolve_path(args.prompt_dir)),
                "config": _replay_config_text(args, row),
                "old_em": old_em,
                "old_f1": old_f1,
                "em": new_em,
                "f1": new_f1,
            },
        }
    )
    stages = dict(payload.get("stages") or {})
    stages["hyperbranch"] = {
        "dag_input": hyperbranch_result.get("artifacts", {}).get("dag_input"),
        "atomic_retrieval": hyperbranch_result.get("artifacts", {}).get("atomic_retrieval"),
        "atomic_answers": hyperbranch_result.get("artifacts", {}).get("atomic_answers"),
        "atomic_answer_inputs": hyperbranch_result.get("artifacts", {}).get("atomic_answer_inputs"),
        "final_answer": final_answer,
        "replay": hyperbranch_result.get("artifacts", {}).get("replay"),
    }
    payload["stages"] = stages
    return payload


def _manifest_item(payload: dict[str, Any]) -> dict[str, Any]:
    final_answer = payload.get("final_answer") or {}
    replay = payload.get("replay") or {}
    atomic_answers = (((payload.get("stages") or {}).get("hyperbranch") or {}).get("atomic_answers")) or []
    return {
        "method": payload.get("method", METHOD),
        "source_method": payload.get("replay_source_method"),
        "dataset": payload.get("dataset"),
        "index": payload.get("index"),
        "qid": payload.get("qid"),
        "question": payload.get("question"),
        "gold_answer": payload.get("gold_answer"),
        "status": payload.get("status", "ok"),
        "dag_valid": _dag_valid(payload),
        "dag_node_count": len(atomic_answers),
        "retrieval_mode": replay.get("retrieval_mode"),
        "old_final_answer": payload.get("old_final_answer"),
        "old_em": replay.get("old_em"),
        "old_f1": replay.get("old_f1"),
        "final_answer": final_answer.get("answer"),
        "em": replay.get("em"),
        "f1": replay.get("f1"),
        "changed": str(payload.get("old_final_answer") or "") != str(final_answer.get("answer") or ""),
        "source_output_dir": payload.get("source_output_dir"),
        "output_dir": payload.get("output_dir"),
        "hyperbranch_run_dir": payload.get("hyperbranch_run_dir"),
    }


def _dag_valid(payload: dict[str, Any]) -> Any:
    dag = (((payload.get("stages") or {}).get("depo") or {}).get("6_atomic_question_dag")) or {}
    if isinstance(dag, dict) and "valid" in dag:
        return dag.get("valid")
    return True


def _pipeline_markdown(payload: dict[str, Any]) -> str:
    final_answer = payload.get("final_answer") or {}
    replay = payload.get("replay") or {}
    atomic_answers = (((payload.get("stages") or {}).get("hyperbranch") or {}).get("atomic_answers")) or []
    dag_nodes = (((payload.get("stages") or {}).get("depo") or {}).get("6_atomic_question_dag") or {}).get("nodes")
    if not dag_nodes:
        dag_nodes = (((payload.get("stages") or {}).get("hyperbranch") or {}).get("dag_input")) or []
    lines = [
        f"# Cached Atomic Answer Replay #{payload.get('index')}",
        "",
        f"- Dataset: `{payload.get('dataset')}`",
        f"- Question: {payload.get('question')}",
        f"- Gold answer: {payload.get('gold_answer')}",
        f"- Source output: `{payload.get('source_output_dir')}`",
        f"- Question mode: `{replay.get('question_mode')}`",
        f"- Retrieval mode: `{replay.get('retrieval_mode')}`",
        f"- Old final answer: {payload.get('old_final_answer')}",
        f"- New final answer: {final_answer.get('answer')}",
        f"- Old F1: `{replay.get('old_f1')}`",
        f"- New F1: `{replay.get('f1')}`",
        "",
        "## Atomic DAG",
    ]
    for node in dag_nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id") or node.get("id")
        depends_on = node.get("dependencies") or node.get("depends_on") or []
        lines.append(f"- {node_id}: {node.get('question')}")
        lines.append(f"  - depends_on: {', '.join(depends_on) if depends_on else '(none)'}")
    lines.extend(["", "## Replayed Atomic Answers"])
    for answer in atomic_answers:
        lines.append(f"- {answer.get('node_id')}: {answer.get('answer')}")
        lines.append(f"  - question: {answer.get('question')}")
    lines.extend(["", "## Final Answer", str(final_answer.get("answer", ""))])
    return "\n".join(lines)


def _summary_header(
    source_run: Path,
    output_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> list[str]:
    return [
        "# Cached Atomic Answer Replay",
        "",
        f"- Source run: `{source_run}`",
        f"- Output: `{output_dir}`",
        f"- Selected questions: `{len(rows)}`",
        f"- Question mode: `{args.question_mode}`",
        f"- Retrieval mode: `{args.retrieval_mode}`",
        f"- Config: `{_config_path_for_replay(args, source_run, rows) if args.retrieval_mode == 'fact-query' else '(not used)'}`",
        f"- Top-k: `{args.top_k if args.top_k > 0 else 'all cached evidence'}`",
        "",
    ]


def _summary_question_lines(payload: dict[str, Any]) -> list[str]:
    replay = payload.get("replay") or {}
    final_answer = payload.get("final_answer") or {}
    return [
        f"## {payload.get('index')}. {payload.get('question')}",
        "",
        f"- Output: `{payload.get('output_dir')}`",
        f"- Old answer: {payload.get('old_final_answer')}",
        f"- New answer: {final_answer.get('answer')}",
        f"- Retrieval mode: `{replay.get('retrieval_mode')}`",
        f"- Old F1: `{replay.get('old_f1')}`",
        f"- New F1: `{replay.get('f1')}`",
        "",
    ]


def _summary_error_lines(payload: dict[str, Any]) -> list[str]:
    return [
        f"## {payload.get('index')}. {payload.get('question')}",
        "",
        f"- Output: `{payload.get('output_dir')}`",
        f"- Status: sample",
        f"- Error: `{payload.get('error_type')}: {payload.get('sample')}`",
        "",
    ]


def _error_payload(
    row: dict[str, Any],
    source_question_dir: Path,
    output_question_dir: Path,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "status": "sample",
        "method": METHOD,
        "index": row.get("index"),
        "qid": row.get("qid"),
        "dataset": row.get("dataset"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "source_output_dir": str(source_question_dir),
        "output_dir": str(output_question_dir),
        "error_type": type(exc).__name__,
        "sample": str(exc),
        "traceback": traceback.format_exc(),
    }


def _error_manifest_item(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": METHOD,
        "dataset": payload.get("dataset"),
        "index": payload.get("index"),
        "qid": payload.get("qid"),
        "question": payload.get("question"),
        "gold_answer": payload.get("gold_answer"),
        "status": "sample",
        "error_type": payload.get("error_type"),
        "sample": payload.get("sample"),
        "source_output_dir": payload.get("source_output_dir"),
        "output_dir": payload.get("output_dir"),
    }


def _error_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Cached Replay Error #{payload.get('index')}",
            "",
            f"- Question: {payload.get('question')}",
            f"- Error type: `{payload.get('error_type')}`",
            "",
            "```text",
            str(payload.get("sample") or ""),
            "```",
            "",
        ]
    )


def _replay_method(args: argparse.Namespace) -> str:
    return FACT_QUERY_METHOD if args.retrieval_mode == "fact-query" else METHOD


def _config_path_for_replay(args: argparse.Namespace, source_run: Path, rows: list[dict[str, Any]]) -> Path:
    if args.config:
        return _resolve_path(args.config)
    dataset = ""
    if rows:
        dataset = str(rows[0].get("dataset") or "").strip()
    if not dataset:
        dataset = source_run.parent.name
    config_path = PROJECT_ROOT / "configs" / f"{dataset}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not infer config path for dataset {dataset!r}; pass --config explicitly."
        )
    return config_path.resolve()


def _replay_config_text(args: argparse.Namespace, row: dict[str, Any]) -> str:
    if args.retrieval_mode != "fact-query":
        return ""
    if args.config:
        return str(_resolve_path(args.config))
    dataset = str(row.get("dataset") or "").strip()
    return str((PROJECT_ROOT / "configs" / f"{dataset}.yaml").resolve()) if dataset else ""


def _select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = _parse_ids(args.ids)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status", "ok")) not in {"ok", "success"}:
            continue
        index = int(row.get("index", 0) or 0)
        if wanted and index not in wanted:
            continue
        if not wanted:
            if index < args.start:
                continue
            if args.end is not None and index > args.end:
                continue
        if args.f1_zero_only:
            _, old_f1 = _score(row.get("gold_answer"), row.get("final_answer"))
            if old_f1 != 0.0:
                continue
        selected.append(row)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def _parse_ids(raw: str) -> set[int]:
    wanted: set[int] = set()
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            left, right = text.split("-", 1)
            start, end = int(left), int(right)
            wanted.update(range(start, end + 1))
        else:
            wanted.add(int(text))
    return wanted


def _append_eval(records: list[dict[str, Any]], manifest_item: dict[str, Any]) -> None:
    if manifest_item.get("em") is None or manifest_item.get("f1") is None:
        return
    records.append(manifest_item)


def _maybe_print_eval(records: list[dict[str, Any]], eval_every: int, *, force: bool = False) -> None:
    if not records:
        return
    if not force and eval_every > 0 and len(records) % eval_every != 0:
        return
    em = sum(float(item.get("em") or 0.0) for item in records) / len(records)
    f1 = sum(float(item.get("f1") or 0.0) for item in records) / len(records)
    changed = sum(1 for item in records if item.get("changed"))
    print(f"[eval] n={len(records)} EM={em:.4f} F1={f1:.4f} changed={changed}")


def _score(gold: Any, predicted: Any) -> tuple[float | None, float | None]:
    if gold is None:
        return None, None
    gold_text = str(gold)
    pred_text = str(predicted or "")
    return cal_em([[gold_text]], [pred_text]), cal_f1([[gold_text]], [pred_text])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_optional_json(path: Path, default: Any | None = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return default
    return json.loads(text)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(payload) + "\n", encoding="utf-8")


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _output_dir(args: argparse.Namespace, source_run: Path, rows: list[dict[str, Any]]) -> Path:
    if args.output_dir:
        return _resolve_path(args.output_dir)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = str(rows[0].get("dataset") or source_run.parent.name or "dataset")
    return _resolve_path(args.output_root) / dataset / run_id


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _clean_fact_query_payload(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return _clean_fact_query_text(value.get("fact_query"))


def _clean_fact_query_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _close_logger(logger: Any) -> None:
    for handler in list(getattr(logger, "handlers", [])):
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
