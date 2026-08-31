import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))
from eval.eval import cal_em

for dataset in ("musique", "hotpotqa", "2wikimultihopqa"):
    questions = json.loads((ROOT / "questions" / dataset / "questions.json").read_text(encoding="utf-8"))[199:249]
    runs = ROOT / "runs" / "depo_hyperbranch" / dataset
    for index, item in enumerate(questions, start=200):
        result = json.loads((runs / "reuse_730_dag_200_250_prompt_v2" / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))
        current = result["nodes"][-1]
        old_path = next((runs / "730_1000").glob(f"{index:05d}_*")) / "pipeline.json"
        old_answer = json.loads(old_path.read_text(encoding="utf-8"))["final_answer"]["answer"]
        gold = item["answer"]
        current_em = cal_em([[gold]], [current["answer"]])
        old_em = cal_em([[gold]], [old_answer])
        if current_em != old_em:
            deps = []
            dag_path = next((runs / "730_1000").glob(f"{index:05d}_*")) / "hyperbranch_dag.json"
            dag = json.loads(dag_path.read_text(encoding="utf-8"))
            depends_on = {node["id"]: node.get("depends_on", []) for node in dag["nodes"]}
            nodes_by_id = {node["id"]: node for node in result["nodes"]}
            for node_id in depends_on[current["id"]]:
                dep = nodes_by_id[node_id]
                deps.append({"id": node_id, "question": dep["rewritten_question"], "answer": dep["answer"]})
            evidence = [
                edge["hyperedge_text"]
                for block in current["evidence_blocks"][:5]
                for edge in block["hyperedges"][:2]
            ][:8]
            print(json.dumps({
                "dataset": dataset, "index": index, "gold": gold,
                "current": current["answer"], "old": old_answer,
                "atomic_question": current["rewritten_question"],
                "dependencies": deps, "evidence": evidence,
            }, ensure_ascii=False))
