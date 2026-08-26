from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval import cal_em, cal_f1
from hyper_branch.pipeline import HyperBranchPipeline


DATASET = "2wikimultihopqa"
SOURCE_RUN = PROJECT_ROOT / "runs" / "depo_hyperbranch" / DATASET / "730_1000"
OUTPUT_RUN = PROJECT_ROOT / "runs" / "depo_hyperbranch" / DATASET / "retrieval_from_730_1000"


def main() -> int:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / f"{DATASET}.yaml").read_text(encoding="utf-8")
    )
    pipeline = HyperBranchPipeline(
        PROJECT_ROOT / config["dataset_root"],
        top_k=config["top_k"],
        model=config["model"],
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    source_dirs = sorted(SOURCE_RUN.glob("[0-9][0-9][0-9][0-9][0-9]_*"))[:100]
    for source_dir in source_dirs:
        output_dir = OUTPUT_RUN / source_dir.name
        result_file = output_dir / "result.json"
        if result_file.exists():
            continue

        saved_dag = json.loads((source_dir / "hyperbranch_dag.json").read_text(encoding="utf-8"))
        baseline = json.loads(
            (source_dir / "hyperbranch_run" / "artifacts" / "final_answer.json").read_text(
                encoding="utf-8"
            )
        )
        source_record = json.loads((source_dir / "pipeline.json").read_text(encoding="utf-8"))

        result = pipeline.run(
            saved_dag["question"],
            {"nodes": saved_dag["nodes"]},
            saved_dag["original_question_entities"],
        )
        gold_answer = source_record["gold_answer"]
        baseline_answer = baseline["answer"]
        answer = result["final_answer"]["answer"]
        record = {
            "index": int(source_dir.name[:5]),
            "question": saved_dag["question"],
            "gold_answer": gold_answer,
            "baseline_answer": baseline_answer,
            "answer": answer,
            "baseline_em": cal_em([[gold_answer]], [baseline_answer]),
            "baseline_f1": cal_f1([[gold_answer]], [baseline_answer]),
            "em": cal_em([[gold_answer]], [answer]),
            "f1": cal_f1([[gold_answer]], [answer]),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps({**record, **result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"#{record['index']:05d}: {answer}")

    records = [
        json.loads((OUTPUT_RUN / source_dir.name / "result.json").read_text(encoding="utf-8"))
        for source_dir in source_dirs
    ]
    summary = {
        "source_run": str(SOURCE_RUN),
        "question_count": len(records),
        "baseline": {
            "em": mean(record["baseline_em"] for record in records),
            "f1": mean(record["baseline_f1"] for record in records),
        },
        "retrieval": {
            "em": mean(record["em"] for record in records),
            "f1": mean(record["f1"] for record in records),
        },
        "change": {
            "em": mean(record["em"] - record["baseline_em"] for record in records),
            "f1": mean(record["f1"] - record["baseline_f1"] for record in records),
        },
        "changed_answer_count": sum(
            record["answer"] != record["baseline_answer"] for record in records
        ),
    }
    OUTPUT_RUN.mkdir(parents=True, exist_ok=True)
    (OUTPUT_RUN / "comparison.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
