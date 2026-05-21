from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from .config import load_config
from .depo_adapter import generate_atomic_dag_with_depo
from .logging_utils import TraceStore, configure_logging, create_run_dir
from .pipeline import HyperBranchPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomic DAG execution with three-view hyperedge retrieval.")
    parser.add_argument("--question", help="Question to answer.")
    parser.add_argument("--question-file", help="Optional file containing the question.")
    parser.add_argument("--question-index", type=int, default=0, help="When --question-file is a JSON list, select this item index.")
    parser.add_argument("--dag", help="Optional JSON file containing an atomic question DAG, usually emitted by DEPO.")
    parser.add_argument("--config", default="configs/agriculture.yaml", help="Path to YAML config.")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock reasoning service and local hash embeddings.")
    parser.add_argument("--atomic-only", action="store_true", help="Do not call DEPO when --dag is omitted; run the question as one atomic node.")
    parser.add_argument("--depo-corenlp-url", default="http://localhost:9000", help="CoreNLP endpoint used by DEPO.")
    parser.add_argument("--depo-corenlp-memory", default="4G", help="Java heap memory for DEPO's managed CoreNLP server.")
    parser.add_argument("--depo-corenlp-home", help="Optional Stanford CoreNLP directory for DEPO.")
    parser.add_argument("--depo-corenlp-timeout-ms", type=int, default=60000, help="DEPO CoreNLP timeout in milliseconds.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed console logs instead of the concise summary view.")
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Write failure artifacts and return success so batch runs can continue.",
    )
    args = parser.parse_args()

    dag_payload = _load_json_file(Path(args.dag)) if args.dag else None
    question = _resolve_question(args.question, args.question_file, args.question_index, dag_payload)
    project_root = Path.cwd()
    config = load_config(Path(args.config), project_root)
    if args.mock_llm:
        config.llm.use_mock = True

    run_dir = create_run_dir(config.runtime.base_run_dir, question)
    logger = configure_logging(run_dir, config.runtime.log_level, verbose_console=args.verbose)
    trace_store = TraceStore(run_dir)

    try:
        if dag_payload is None and not args.atomic_only:
            if config.llm.use_mock:
                logger.info("Mock LLM is enabled; running the question as a single atomic node without DEPO.")
            else:
                logger.info("Generating atomic question DAG with DEPO.")
                dag_payload = generate_atomic_dag_with_depo(
                    question,
                    config,
                    corenlp_url=args.depo_corenlp_url,
                    corenlp_memory=args.depo_corenlp_memory,
                    corenlp_home=args.depo_corenlp_home,
                    corenlp_timeout_ms=args.depo_corenlp_timeout_ms,
                )
                trace_store.save_artifact("artifacts/depo_dag_raw.json", dag_payload)
        pipeline = HyperBranchPipeline(config=config, run_dir=run_dir, logger=logger, trace_store=trace_store)
        result = pipeline.run(question, dag_payload=dag_payload)
    except Exception as exc:
        logger.error("Pipeline failed for question: %s", question)
        trace_store.save_artifact(
            "artifacts/error.json",
            {
                "question": question,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "config": {
                    "mock_llm": config.llm.use_mock,
                    "model": config.llm.model,
                    "embedding_model": config.llm.embedding_model,
                },
            },
        )
        print("status=failed")
        print(f"error={type(exc).__name__}: {exc}")
        print(f"run_dir={run_dir}")
        exit_code = 0 if args.allow_failure else 1
    else:
        print(result["final_answer"]["answer"])
        print("status=success")
        print(f"run_dir={result['run_dir']}")
        exit_code = 0
    finally:
        _close_logger(logger)

    return exit_code


def _resolve_question(
    question: str | None,
    question_file: str | None,
    question_index: int,
    dag_payload: object | None = None,
) -> str:
    if question:
        return question.strip()
    if question_file:
        return _load_question_from_file(Path(question_file), question_index)
    dag_question = _extract_question_from_dag_payload(dag_payload)
    if dag_question:
        return dag_question
    raise SystemExit("Either --question or --question-file is required.")


def _load_json_file(path: Path) -> object:
    if not path.exists():
        raise SystemExit(f"DAG file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"DAG path is not a file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"DAG file is not valid JSON: {path}: {exc}") from exc


def _load_question_from_file(path: Path, question_index: int) -> str:
    raw_text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() != ".json":
        return raw_text

    payload = json.loads(raw_text)
    if isinstance(payload, dict):
        return _extract_question_field(payload, path)
    if isinstance(payload, list):
        if not payload:
            raise SystemExit(f"Question file is an empty JSON list: {path}")
        if question_index < 0 or question_index >= len(payload):
            raise SystemExit(
                f"question_index {question_index} is out of range for {path}; valid range is 0..{len(payload) - 1}"
            )
        item = payload[question_index]
        if not isinstance(item, dict):
            raise SystemExit(f"Expected JSON object at index {question_index} in {path}, got {type(item).__name__}")
        return _extract_question_field(item, path, question_index)
    raise SystemExit(f"Unsupported JSON question file format in {path}: expected object or list of objects.")


def _extract_question_field(payload: dict, path: Path, question_index: int | None = None) -> str:
    question = payload.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    location = f"{path}[{question_index}]" if question_index is not None else str(path)
    raise SystemExit(f"Missing non-empty 'question' field in {location}.")


def _extract_question_from_dag_payload(payload: object | None) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("original_question", "question"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _close_logger(logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
