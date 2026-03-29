from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .logging_utils import TraceStore, configure_logging, create_run_dir
from .pipeline import GoTHyperPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-of-Thoughts over Knowledge Hypergraphs for multi-hop RAG.")
    parser.add_argument("--question", help="Question to answer.")
    parser.add_argument("--question-file", help="Optional file containing the question.")
    parser.add_argument("--config", default="configs/agriculture.yaml", help="Path to YAML config.")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock reasoning service and local hash embeddings.")
    args = parser.parse_args()

    question = _resolve_question(args.question, args.question_file)
    project_root = Path.cwd()
    config = load_config(Path(args.config), project_root)
    if args.mock_llm:
        config.llm.use_mock = True

    run_dir = create_run_dir(config.runtime.base_run_dir, question)
    logger = configure_logging(run_dir, config.runtime.log_level)
    trace_store = TraceStore(run_dir)

    pipeline = GoTHyperPipeline(config=config, run_dir=run_dir, logger=logger, trace_store=trace_store)
    result = pipeline.run(question)

    print(result["final_answer"]["answer"])
    print(f"run_dir={result['run_dir']}")


def _resolve_question(question: str | None, question_file: str | None) -> str:
    if question:
        return question.strip()
    if question_file:
        return Path(question_file).read_text(encoding="utf-8").strip()
    raise SystemExit("Either --question or --question-file is required.")


if __name__ == "__main__":
    main()
