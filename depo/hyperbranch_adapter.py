"""将持久化的 DEPO Step5 输出转换为 HyperBranch 所需的 DAG 输入结构。"""


from typing import Any


def build_hyperbranch_dag_payload(decomposition_payload: dict[str, Any]) -> dict[str, Any]:
    """将已校验的 DEPO Step5 DAG 适配为 HyperBranch 输入结构。"""

    dag = ((decomposition_payload.get("stages") or {}).get("6_atomic_question_dag")) or {}
    if not isinstance(dag, dict):
        raise ValueError("DEPO decomposition does not contain stages.6_atomic_question_dag.")
    if not dag.get("valid"):
        errors = dag.get("validation_errors") or []
        raise ValueError(f"DEPO atomic DAG is invalid: {errors}")
    nodes = dag.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("DEPO atomic DAG does not contain any nodes.")

    # 同一份显式实体既供 DAG 记录，也供 HyperBranch 构建原问题共享候选池。
    topic_entities = explicit_entity_texts(decomposition_payload)
    return {
        "question": decomposition_payload.get("question", ""),
        "topic_entities": topic_entities,
        "original_question_entities": topic_entities,
        "nodes": nodes,
        "edges": dag.get("edges") or [],
        "leaf_node_ids": dag.get("leaf_node_ids") or [],
        "source": "depo_stages.6_atomic_question_dag",
    }


def explicit_entity_texts(decomposition_payload: dict[str, Any]) -> list[str]:
    """按 DEPO 提取顺序返回去重后的显式实体文本。"""

    explicit = ((decomposition_payload.get("stages") or {}).get("1_explicit_entities")) or {}
    entities = explicit.get("entities") if isinstance(explicit, dict) else []
    entity_items = entities if isinstance(entities, list) else []
    texts: list[str] = []
    seen: set[str] = set()
    for item in entity_items:
        raw_text = item.get("text") if isinstance(item, dict) else item
        text = str(raw_text or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            texts.append(text)
    return texts
