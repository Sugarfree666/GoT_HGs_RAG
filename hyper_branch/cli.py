"""从提供的 DAG 或单个原子问题运行 HyperBranch 的命令行入口。"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from .config import load_config
from .logging_utils import TraceStore, configure_logging, create_run_dir
from .pipeline import HyperBranchPipeline


def main() -> int:
    """解析命令行输入，运行管线，并将异常写入可复核产物。"""

    parser = argparse.ArgumentParser(description="Execute DEPO atomic DAG with three-view HyperBranch hyperedge retrieval.")
    parser.add_argument("--question", help="Question to answer.")
    parser.add_argument("--question-file", help="Optional file containing the question.")
    parser.add_argument("--question-index", type=int, default=0, help="When --question-file is a JSON list, select this item index.")
    parser.add_argument("--dag", help="Optional JSON file containing an atomic question DAG, usually emitted by DEPO.")
    parser.add_argument("--config", default="configs/agriculture.yaml", help="Path to YAML config.")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock reasoning service and local hash embeddings.")
    parser.add_argument("--atomic-only", action="store_true", help="Run the question as one atomic node when --dag is omitted.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed console logs instead of the concise summary view.")
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Write failure artifacts and return success so batch runs can continue.",
    )
    args = parser.parse_args()

    # 有 DAG 时执行 DEPO 分解；显式 --atomic-only 才允许直接把原问题作为单节点。
    dag_payload = _load_json_file(Path(args.dag)) if args.dag else None
    question = _resolve_question(args.question, args.question_file, args.question_index, dag_payload)
    project_root = Path.cwd()
    config = load_config(Path(args.config), project_root)
    if args.mock_llm:
        config.llm.use_mock = True
    if dag_payload is None and not args.atomic_only and not config.llm.use_mock:
        parser.error("--dag is required for DEPO decomposition; use --atomic-only to run without a DAG.")

    run_dir = create_run_dir(config.runtime.base_run_dir, question)
    logger = configure_logging(run_dir, config.runtime.log_level, verbose_console=args.verbose)
    trace_store = TraceStore(run_dir)

    try:
        if dag_payload is None:
            if config.llm.use_mock:
                logger.info("Mock LLM is enabled; running the question as a single atomic node without DEPO.")
            else:
                logger.info("No DAG supplied; running the question as a single atomic node.")
        pipeline = HyperBranchPipeline(config=config, run_dir=run_dir, logger=logger, trace_store=trace_store)
        result = pipeline.run(question, dag_payload=dag_payload)
    except Exception as exc:
        logger.error("Pipeline failed for question: %s", question)
        trace_store.save_artifact(
            "artifacts/sample.json",
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
        print(f"sample={type(exc).__name__}: {exc}")
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
    """按命令行文本、问题文件、DAG 内问题的优先级确定输入问题。"""

    if question:
        return question.strip()
    if question_file:
        return _load_question_from_file(Path(question_file), question_index)
    dag_question = _extract_question_from_dag_payload(dag_payload)
    if dag_question:
        return dag_question
    raise SystemExit("Either --question or --question-file is required.")


def _load_json_file(path: Path) -> object:
    """读取并校验一个 JSON DAG 文件。"""

    if not path.exists():
        raise SystemExit(f"DAG file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"DAG path is not a file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"DAG file is not valid JSON: {path}: {exc}") from exc


def _load_question_from_file(path: Path, question_index: int) -> str:
    """从纯文本或 JSON 问题文件读取指定问题。"""

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
    """从兼容的 DAG 顶层字段中提取原问题。"""

    if not isinstance(payload, dict):
        return ""
    for key in ("original_question", "question"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _close_logger(logger) -> None:
    """关闭并移除本次运行创建的日志处理器。"""

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
