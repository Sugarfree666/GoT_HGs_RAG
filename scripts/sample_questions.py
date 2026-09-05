"""Randomly sample question sets for ablation experiments.

The script samples without replacement and preserves each source record's schema
for downstream compatibility.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample questions without replacement for ablation experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset directory names under questions/ (default: all supported datasets).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Questions to sample from each dataset (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed used independently for every dataset (default: 42).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be >= 1")

    for dataset in args.datasets:
        _sample_dataset(
            dataset=dataset,
            count=args.count,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    return 0


def _sample_dataset(dataset: str, count: int, seed: int, overwrite: bool) -> None:
    dataset_dir = PROJECT_ROOT / "questions" / dataset
    input_file = dataset_dir / "questions.json"
    output_file = dataset_dir / f"questions_{count}_seed_{seed}.json"

    if not input_file.is_file():
        raise FileNotFoundError(f"Question file not found: {input_file}")
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}. Use --overwrite to replace it."
        )

    questions = _load_questions(input_file)
    if count > len(questions):
        raise ValueError(
            f"Cannot sample {count} questions from {dataset}: only {len(questions)} available."
        )

    # A separate generator keeps every dataset reproducible and independent.
    sampled_questions = random.Random(seed).sample(questions, count)
    output_file.write_text(
        json.dumps(sampled_questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{dataset}: {len(sampled_questions)} questions -> {output_file}")


def _load_questions(input_file: Path) -> list[dict[str, Any]]:
    questions = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON array in {input_file}")
    if not all(isinstance(question, dict) for question in questions):
        raise ValueError(f"Expected every question in {input_file} to be a JSON object")
    return questions


if __name__ == "__main__":
    raise SystemExit(main())
