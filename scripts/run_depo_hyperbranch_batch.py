from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (DEPO_ROOT, PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import run_depo_decomposition_batch as depo_batch
from hyper_branch.config import load_config
from hyper_branch.pipeline import HyperBranchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a DEPO DAG and answer it with HyperBranch."
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/.")
    parser.add_argument("--questions-file", help="Specific questions file path.")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--questions-root", default="questions")
    parser.add_argument("--config", help="HyperBranch YAML config.")
    parser.add_argument("--output-root", default="runs/depo_hyperbranch")
    parser.add_argument("--run-id", help="Run id under output-root/dataset/.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Skip questions with result.json.")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--hyperbranch-llm-model")
    parser.add_argument("--embedding-model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.environ["OPENAI_API_KEY"]
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    from hanlp_sdp_parser import HanLPSDPParser
    from llm_client import LLMClient
    from pipeline import run_depo

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
    parser = HanLPSDPParser()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _repo_path(args.output_root)

    for questions_file in depo_batch._resolve_question_files(args):
        dataset = depo_batch._dataset_name(questions_file)
        config_path = _config_path_for_dataset(args, dataset)
        output_dir = output_root / dataset / run_id
        items = depo_batch._slice_items(
            depo_batch._read_question_items(questions_file),
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
        print(f"Running {dataset}: {len(items)} question(s), output={output_dir}")
        hyperbranch_runner: _ReusableHyperBranchRunner | None = None

        for offset, item in enumerate(items, start=1):
            question_dir = output_dir / depo_batch._question_dir_name(
                item["index"], item.get("qid"), item["question"]
            )
            result_path = question_dir / "result.json"
            if args.resume and result_path.exists():
                print(f"[skip] {dataset} #{item['index']} {item['question']}")
                continue

            question_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run {offset}/{len(items)}] {dataset} #{item['index']} {item['question']}")
            decomposition = run_depo(item["question"], parser, llm_client)
            dag = decomposition["atomic_question_dag"]
            if hyperbranch_runner is None:
                hyperbranch_runner = _ReusableHyperBranchRunner(config_path, args)
            hyperbranch_result = hyperbranch_runner.run(
                question=item["question"],
                dag=dag,
                original_question_entities=decomposition["preprocess_result"].entities,
            )
            result = _result_payload(item, dag, hyperbranch_result)
            _write_json(result_path, result)
            print(
                f"[ok]  {dataset} #{item['index']} "
                f"answer={result['final_answer'].get('answer', '')!r}"
            )

    return 0


class _ReusableHyperBranchRunner:
    def __init__(self, config_path: Path, args: argparse.Namespace) -> None:
        config = load_config(config_path, PROJECT_ROOT)
        if args.hyperbranch_llm_model:
            config.llm.model = args.hyperbranch_llm_model
        else:
            config.llm.model = args.llm_model
        if args.embedding_model:
            config.llm.embedding_model = args.embedding_model
        self.pipeline = HyperBranchPipeline(config)

    def run(
        self,
        *,
        question: str,
        dag: dict[str, Any],
        original_question_entities: list[str],
    ) -> dict[str, Any]:
        return self.pipeline.run(question, dag, original_question_entities)


def _result_payload(
    item: dict[str, Any],
    dag: dict[str, Any],
    hyperbranch_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "question": item["question"],
        "gold_answer": item.get("answer"),
        "dag": dag,
        "atomic_answers": hyperbranch_result["atomic_answers"],
        "final_answer": hyperbranch_result["final_answer"],
    }


def _config_path_for_dataset(args: argparse.Namespace, dataset: str) -> Path:
    path = _repo_path(args.config) if args.config else PROJECT_ROOT / "configs" / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"HyperBranch config not found: {path}")
    return path


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(depo_batch._jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
