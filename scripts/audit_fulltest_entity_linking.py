"""Record current entity-linking decisions for a saved full-test run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from hyper_branch.client import OpenAIClient
from hyper_branch.database import HypergraphDatabase, _lookup_key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-run", default="full_test1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8"))
    client = OpenAIClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model=config["model"],
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    database = HypergraphDatabase(ROOT / config["dataset_root"])
    source_dir = ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.source_run
    questions = json.loads((ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8"))
    output_path = ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.run_id / "entity_links.json"
    records = []

    for index in range(args.start, args.end):
        result = json.loads((source_dir / f"{index:05d}" / "result.json").read_text(encoding="utf-8"))
        stages = [("original", "", questions[index - 1]["question"], result["topic_entities"])]
        stages.extend(("atomic", node["id"], node["rewritten_question"], node["entities"]) for node in result["nodes"])
        for stage, node_id, question, mentions in stages:
            for mention in mentions:
                exact_ids = database._entity_ids_by_name.get(_lookup_key(mention), [])
                if exact_ids:
                    linked_entity, method, score = exact_ids[0], "exact", 1.0
                else:
                    linked_entity, score = database.entity_vectors.query(client.embed_text(mention), 1)[0]
                    method = "vector" if score >= 0.5 else "unlinked"
                    if method == "unlinked":
                        linked_entity = ""
                records.append(
                    {
                        "index": index,
                        "stage": stage,
                        "node_id": node_id,
                        "question": question,
                        "mention": mention,
                        "linked_entity": linked_entity,
                        "linked_entities": exact_ids or ([linked_entity] if linked_entity else []),
                        "method": method,
                        "score": score,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.dataset}: {len(records)} links -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
