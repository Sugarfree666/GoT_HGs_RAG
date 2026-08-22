"""Run a DEPO atomic DAG through HyperBranch."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .pipeline import HyperBranchPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute a DEPO atomic DAG with HyperBranch retrieval.")
    parser.add_argument("--dag", required=True, help="JSON file produced by the DEPO adapter.")
    parser.add_argument("--question", help="Original question; defaults to the DAG question field.")
    parser.add_argument("--config", default="configs/agriculture.yaml")
    args = parser.parse_args()

    dag = json.loads(Path(args.dag).read_text(encoding="utf-8-sig"))
    question = args.question or str(dag["question"]).strip()
    config = load_config(Path(args.config), Path.cwd())
    pipeline = HyperBranchPipeline(config, logging.getLogger("hyper_branch"))
    result = pipeline.run(question, dag, dag.get("original_question_entities"))
    print(result["final_answer"]["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
