"""DEPO Step5：由恢复后的语义路径生成原子问题 DAG。"""

from __future__ import annotations

import re
from typing import Any

from llm_client import LLMClient
from prompts import ATOMIC_QUESTION_DAG_SYSTEM, build_atomic_question_dag_prompt


def generate_atomic_question_dag(
    llm_client: LLMClient,
    original_question: str,
    question_entities: list[str],
    question_structure: list[list[str]],
) -> dict[str, Any]:
    """调用 Step5，并将依赖关系展开为 DAG 边。"""

    payload = llm_client.chat_json(
        ATOMIC_QUESTION_DAG_SYSTEM,
        build_atomic_question_dag_prompt(
            original_question,
            question_entities,
            question_structure,
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
