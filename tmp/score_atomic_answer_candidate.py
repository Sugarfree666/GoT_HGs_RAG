import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.eval import cal_em, cal_f1


def main() -> None:
    for dataset in ("musique", "hotpotqa", "2wikimultihopqa"):
        questions = json.loads((ROOT / "questions" / dataset / "questions.json").read_text(encoding="utf-8"))[199:249]
        run_root = ROOT / "runs" / "depo_hyperbranch" / dataset
        baseline = []
        candidate = []
        for index, item in enumerate(questions, start=200):
            gold = item["answer"]
            old_answer = json.loads((run_root / "reuse_730_dag_200_250_prompt_v2" / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))["nodes"][-1]["answer"]
            new_answer = json.loads((run_root / "atomic_answer_candidate_200_250" / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))["answer"]
            baseline.append({"index": index, "gold": gold, "answer": old_answer, "em": cal_em([[gold]], [old_answer]), "f1": cal_f1([[gold]], [old_answer])})
            candidate.append({"index": index, "gold": gold, "answer": new_answer, "em": cal_em([[gold]], [new_answer]), "f1": cal_f1([[gold]], [new_answer])})
        for name, rows in (("baseline", baseline), ("candidate", candidate)):
            print(json.dumps({
                "dataset": dataset, "run": name,
                "em": sum(row["em"] for row in rows) / len(rows),
                "f1": sum(row["f1"] for row in rows) / len(rows),
                "correct": [row["index"] for row in rows if row["em"] == 1.0],
            }, ensure_ascii=False))
        for old, new in zip(baseline, candidate):
            if old["em"] != new["em"]:
                print(json.dumps({
                    "dataset": dataset, "index": old["index"], "gold": old["gold"],
                    "before": old["answer"], "after": new["answer"],
                    "before_em": old["em"], "after_em": new["em"],
                }, ensure_ascii=False))


if __name__ == "__main__":
    main()
