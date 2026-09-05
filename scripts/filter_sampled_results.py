"""Extract sampled-question evaluation results from completed full-dataset runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter full-run evaluation results to a reproducible sampled subset."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--run-id", default="93_1000_1")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be >= 1")

    for dataset in args.datasets:
        _filter_dataset(
            dataset=dataset,
            run_id=args.run_id,
            count=args.count,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    return 0


def _filter_dataset(
    dataset: str,
    run_id: str,
    count: int,
    seed: int,
    overwrite: bool,
) -> None:
    questions_dir = PROJECT_ROOT / "questions" / dataset
    source_questions_file = questions_dir / "questions.json"
    sampled_questions_file = questions_dir / f"questions_{count}_seed_{seed}.json"
    source_result_file = (
        PROJECT_ROOT
        / "eval"
        / "results"
        / "depo_hyperbranch"
        / dataset
        / run_id
        / "test_result.json"
    )
    output_dir = source_result_file.parent / f"sample_{count}_seed_{seed}"
    output_result_file = output_dir / "test_result.json"
    output_score_file = output_dir / "test_score.json"

    for path in (source_questions_file, sampled_questions_file, source_result_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
        )

    source_questions = _load_json_array(source_questions_file)
    sampled_questions = _load_json_array(sampled_questions_file)
    sampled_indices = _resolve_sampled_indices(
        source_questions=source_questions,
        sampled_questions=sampled_questions,
        count=count,
        seed=seed,
        sampled_questions_file=sampled_questions_file,
    )
    source_results = _load_json_array(source_result_file)
    results_by_index = _index_results(source_results, source_result_file)

    missing_indices = [index for index in sampled_indices if index not in results_by_index]
    if missing_indices:
        raise ValueError(
            f"{source_result_file} is missing sampled indices: {missing_indices}"
        )

    sampled_results = [results_by_index[index] for index in sampled_indices]
    _validate_questions(sampled_results, source_questions, sampled_indices, source_result_file)
    summary = _summarize(sampled_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_result_file.write_text(
        json.dumps(sampled_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_score_file.write_text(
        json.dumps(
            {
                "meta": {
                    "question_file": str(sampled_questions_file.resolve()),
                    "source_question_file": str(source_questions_file.resolve()),
                    "source_result_file": str(source_result_file.resolve()),
                    "sample_count": count,
                    "sample_seed": seed,
                    "indices": sampled_indices,
                    "metrics": ["em", "f1"],
                },
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{dataset}: total={summary['counts']['total']} "
        f"EM={summary['overall']['em']:.6f} F1={summary['overall']['f1']:.6f} "
        f"-> {output_dir}"
    )


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return payload


def _resolve_sampled_indices(
    source_questions: list[dict[str, Any]],
    sampled_questions: list[dict[str, Any]],
    count: int,
    seed: int,
    sampled_questions_file: Path,
) -> list[int]:
    if count > len(source_questions):
        raise ValueError(
            f"Cannot sample {count} questions: only {len(source_questions)} source questions available."
        )
    if len(sampled_questions) != count:
        raise ValueError(
            f"Expected {count} records in {sampled_questions_file}, found {len(sampled_questions)}."
        )

    zero_based_indices = random.Random(seed).sample(range(len(source_questions)), count)
    expected_questions = [source_questions[index] for index in zero_based_indices]
    if sampled_questions != expected_questions:
        raise ValueError(
            f"{sampled_questions_file} does not match the seed-{seed} sample from its source file."
        )
    return [index + 1 for index in zero_based_indices]


def _index_results(
    source_results: list[dict[str, Any]], source_result_file: Path
) -> dict[int, dict[str, Any]]:
    results_by_index: dict[int, dict[str, Any]] = {}
    for result in source_results:
        index = result.get("index")
        if not isinstance(index, int):
            raise ValueError(f"Result without an integer index in {source_result_file}")
        if index in results_by_index:
            raise ValueError(f"Duplicate result index {index} in {source_result_file}")
        results_by_index[index] = result
    return results_by_index


def _validate_questions(
    sampled_results: list[dict[str, Any]],
    source_questions: list[dict[str, Any]],
    sampled_indices: list[int],
    source_result_file: Path,
) -> None:
    for result, index in zip(sampled_results, sampled_indices):
        expected_question = str(source_questions[index - 1].get("question", "")).strip()
        result_question = str(result.get("question", "")).strip()
        if result_question != expected_question:
            raise ValueError(
                f"Question mismatch at index {index} between source questions and {source_result_file}"
            )


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    for metric in ("em", "f1"):
        if any(not isinstance(result.get(metric), (int, float)) for result in results):
            raise ValueError(f"All filtered results must contain numeric '{metric}' values")

    return {
        "counts": {"total": len(results), "completed": len(results), "missing": 0},
        "overall": {
            "em": mean(float(result["em"]) for result in results),
            "f1": mean(float(result["f1"]) for result in results),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
