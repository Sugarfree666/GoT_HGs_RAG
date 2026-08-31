import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.eval import cal_em, cal_f1

dataset = "2wikimultihopqa"
questions = json.loads((ROOT / "questions" / dataset / "questions.json").read_text(encoding="utf-8"))[199:249]
root = ROOT / "runs" / "depo_hyperbranch" / dataset
for run, answer_of in (
    ("baseline", lambda index: json.loads((root / "reuse_730_dag_200_250_prompt_v2" / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))["nodes"][-1]["answer"]),
    ("candidate2", lambda index: json.loads((root / "atomic_answer_candidate2_200_250" / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))["answer"]),
):
    rows = []
    for index, item in enumerate(questions, start=200):
        answer = answer_of(index)
        gold = item["answer"]
        rows.append({"index": index, "gold": gold, "answer": answer, "em": cal_em([[gold]], [answer]), "f1": cal_f1([[gold]], [answer])})
    print(json.dumps({"run": run, "em": sum(row["em"] for row in rows) / 50, "f1": sum(row["f1"] for row in rows) / 50, "correct": [row["index"] for row in rows if row["em"] == 1.0]}, ensure_ascii=False))
    if run == "baseline":
        baseline = rows
    else:
        for old, new in zip(baseline, rows):
            if old["em"] != new["em"]:
                print(json.dumps({"index": old["index"], "gold": old["gold"], "before": old["answer"], "after": new["answer"], "before_em": old["em"], "after_em": new["em"]}, ensure_ascii=False))
