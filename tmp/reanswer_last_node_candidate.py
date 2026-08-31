from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hyper_branch.client import OpenAIClient

PROMPT = (ROOT / "tmp" / "atomic_answer_candidate.md").read_text(encoding="utf-8").strip()


def main() -> None:
    dataset = sys.argv[1]
    config = yaml.safe_load((ROOT / "configs" / f"{dataset}.yaml").read_text(encoding="utf-8"))
    client = OpenAIClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    source_dir = ROOT / "runs" / "depo_hyperbranch" / dataset / "reuse_730_dag_200_250_prompt_v2"
    output_dir = ROOT / "runs" / "depo_hyperbranch" / dataset / "atomic_answer_candidate_200_250"
    dag_dir = ROOT / "runs" / "depo_hyperbranch" / dataset / "730_1000"
    for index in range(200, 250):
        output_file = output_dir / f"{index:05d}" / "result.json"
        if output_file.exists():
            print(f"{dataset} #{index}: skipped", flush=True)
            continue
        source = json.loads((source_dir / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))
        source_nodes = {node["id"]: node for node in source["nodes"]}
        final_node = source["nodes"][-1]
        dag = json.loads((next(dag_dir.glob(f"{index:05d}_*")) / "hyperbranch_dag.json").read_text(encoding="utf-8"))
        dag_nodes = {node["id"]: node for node in dag["nodes"]}
        dependencies = [
            {
                "node_id": source_nodes[node_id]["id"],
                "question": source_nodes[node_id]["rewritten_question"],
                "entities": source_nodes[node_id]["entities"],
                "evidence_blocks": source_nodes[node_id]["evidence_blocks"],
                "answer": source_nodes[node_id]["answer"],
            }
            for node_id in dag_nodes[final_node["id"]].get("depends_on", [])
        ]
        question = json.loads((ROOT / "questions" / dataset / "questions.json").read_text(encoding="utf-8"))[index - 1]["question"]
        try:
            answer = str(
                client.chat_json(
                    PROMPT,
                    json.dumps(
                        {
                            "original_question": question,
                            "atomic_question": final_node["rewritten_question"],
                            "dependency_context": dependencies,
                            "evidence_blocks": final_node["evidence_blocks"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    max_tokens=900,
                )["answer"]
            ).strip()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(
                json.dumps(
                    {
                        "source_run": "reuse_730_dag_200_250_prompt_v2",
                        "node_id": final_node["id"],
                        "atomic_question": final_node["rewritten_question"],
                        "dependency_context": dependencies,
                        "evidence_blocks": final_node["evidence_blocks"],
                        "answer": answer,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{dataset} #{index}: {answer}", flush=True)
        except Exception as exc:
            print(f"{dataset} #{index} failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
