"""DEPO Step5：由恢复后的语义路径生成原子问题 DAG。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm_client import LLMClient


ATOMIC_QUESTION_DAG_PROMPT = (
    Path(__file__).resolve().parents[1] / "prompts" / "depo_atomic_question_dag.md"
).read_text(encoding="utf-8").strip()


def generate_atomic_question_dag(
    llm_client: LLMClient,
    original_question: str,
    question_entities: list[str],
    question_structure: list[list[str]],
) -> dict[str, Any]:
    """调用 Step5，并将依赖关系展开为 DAG 边。"""

    entities: list[str] = []
    for entity in question_entities:
        text = str(entity).strip()
        if text and text not in entities:
            entities.append(text)
    structure = [
        " -- ".join(node.strip() for node in branch if node.strip())
        for branch in question_structure
    ]
    payload = llm_client.chat_json(
        ATOMIC_QUESTION_DAG_PROMPT,
        json.dumps(
            {
                "original_question": original_question,
                "question_entities": entities,
                "question_structure": [branch for branch in structure if branch],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    nodes = [
        {
            "id": item["id"],
            "question": item["question"],
            "depends_on": item["depends_on"],
        }
        for item in payload["atomic_questions"]
    ]
    parent_ids = {
        dependency
        for node in nodes
        for dependency in node["depends_on"]
    }
    return {
        "nodes": nodes,
        "edges": [
            {"source": dependency, "target": node["id"]}
            for node in nodes
            for dependency in node["depends_on"]
        ],
        "leaf_node_ids": [node["id"] for node in nodes if node["id"] not in parent_ids],
    }


def restore_paths(paths: list[list[str]], mask_mapping: dict[str, str]) -> list[list[str]]:
    """将 Step4 路径中的实体占位符还原为原始实体。"""

    return [
        [
            re.sub(
                r"\bENTITY[A-Z0-9]*\b",
                lambda match: mask_mapping[match.group(0)],
                token,
            )
            for token in path
        ]
        for path in paths
    ]
