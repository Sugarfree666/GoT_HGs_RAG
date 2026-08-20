"""HyperBranch 顶层编排：加载数据、组装服务、执行 DAG 并保存产物。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .atomic import (
    AtomicDagExecutor,
    AtomicHyperedgeRetriever,
    AtomicQuestionAnalyzer,
)
from .config import Config
from .data.loaders import HypergraphDatasetLoader
from .llm import LocalHashEmbeddingClient, MockAtomicLLMService, OpenAIAtomicLLMService, OpenAICompatibleClient, PromptManager
from .logging_utils import TraceStore


class HyperBranchPipeline:
    """面向一个数据集、配置和运行目录的可复用应用门面。"""

    def __init__(
        self,
        config: Config,
        run_dir: Path,
        logger: logging.Logger,
        trace_store: TraceStore,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.logger = logger
        self.trace_store = trace_store

        # 数据集只加载一次，确保所有原子节点使用同一份图快照。
        loader = HypergraphDatasetLoader(config.dataset, logger)
        self.dataset = loader.load()
        self.trace_store.save_artifact("dataset_summary.json", self.dataset.summary)

        prompts = PromptManager(config.prompts.directory)
        # Mock 模式让测试和本地结构调试不依赖外部服务。
        if config.llm.use_mock:
            self.embedder = LocalHashEmbeddingClient()
            self.llm_service = MockAtomicLLMService()
        else:
            client = OpenAICompatibleClient(config.llm, trace_store=trace_store)
            self.embedder = client
            self.llm_service = OpenAIAtomicLLMService(client=client, prompts=prompts)

        # 节点执行顺序由执行器负责；这些协作组件在单题内不保存状态。
        analyzer = AtomicQuestionAnalyzer(llm_service=self.llm_service)
        retriever = AtomicHyperedgeRetriever(
            dataset=self.dataset,
            embedder=self.embedder,
            config=config.retrieval,
            logger=logger,
        )
        self.retriever = retriever
        self.executor = AtomicDagExecutor(
            analyzer=analyzer,
            retriever=retriever,
            llm_service=self.llm_service,
            logger=logger,
        )

    def run(
        self,
        question: str,
        dag_payload: Any | None = None,
        original_question_entities: list[str] | None = None,
    ) -> dict[str, Any]:
        """执行一份 DEPO 提供的 DAG，并持久化精简且可检查的运行产物。"""

        self.logger.info("Starting HyperBranch pipeline for question: %s", question)
        if dag_payload is None:
            self.logger.info(
                "No DAG payload supplied; treating the question as a single atomic node for compatibility."
            )
        result = self.executor.run(
            original_question=question,
            dag_payload=dag_payload,
            original_question_entities=original_question_entities,
        )
        artifacts = result.artifacts
        self.trace_store.save_artifact("artifacts/dag_input.json", artifacts["dag_input"])
        self.trace_store.save_artifact("artifacts/dag_repair.json", artifacts["dag_repair"])
        self.trace_store.save_artifact("artifacts/original_question_analysis.json", artifacts["original_question_analysis"])
        self.trace_store.save_artifact("artifacts/atomic_question_analyses.json", artifacts["atomic_question_analyses"])
        self.trace_store.save_artifact("artifacts/final_answer.json", result.final_answer)

        # 完整调用和图细节保留在 trace 中；面向复核的产物保持精简。
        compact_shared_pool = _compact_candidate_pool(artifacts["shared_candidate_pool_final"])
        compact_retrieval = [
            _compact_retrieval_record(item)
            for item in artifacts["atomic_retrieval"]
            if isinstance(item, dict)
        ]
        compact_answers = [
            _compact_atomic_answer(item)
            for item in artifacts["atomic_answers"]
            if isinstance(item, dict)
        ]
        self.trace_store.save_artifact("artifacts/shared_candidate_pool.json", compact_shared_pool)
        self.trace_store.save_artifact("artifacts/atomic_retrieval.json", compact_retrieval)
        self.trace_store.save_artifact("artifacts/atomic_answers.json", compact_answers)
        payload = {
            "original_question": result.original_question,
            "atomic_results": compact_answers,
            "final_answer": dict(result.final_answer),
            "artifacts": {
                "dag_input": artifacts["dag_input"],
                "atomic_answers": compact_answers,
                "final_answer": dict(result.final_answer),
            },
        }
        payload["run_dir"] = str(self.run_dir)
        self.logger.info("Pipeline finished. Artifacts saved under %s", self.run_dir)
        return payload


def _compact_candidate_pool(payload: dict[str, Any]) -> dict[str, Any]:
    """保留候选池来源，同时从摘要中省略过大的标识符列表。"""

    return {
        "method": payload.get("method"),
        "primary_anchor_mention": payload.get("primary_anchor_mention"),
        "linked_entity_id": payload.get("linked_entity_id"),
        "anchor_match": payload.get("anchor_match") or {},
        "anchor_mentions": list(payload.get("anchor_mentions") or []),
        "linked_entities": list(payload.get("linked_entities") or []),
        "anchor_matches": list(payload.get("anchor_matches") or []),
        "unlinked_anchor_mentions": list(payload.get("unlinked_anchor_mentions") or []),
        "adjacent_hyperedge_count": len(payload.get("adjacent_hyperedge_ids") or []),
        "expansion_entity_count": len(payload.get("expansion_entity_ids") or []),
        "second_hop_hyperedge_count": len(payload.get("second_hop_hyperedge_ids") or []),
        "candidate_hyperedge_count": len(payload.get("candidate_hyperedge_ids") or []),
        "insufficient_reason": payload.get("insufficient_reason", ""),
        "fallback_reason": payload.get("fallback_reason", ""),
    }


def _compact_retrieval_record(payload: dict[str, Any]) -> dict[str, Any]:
    """生成逐节点检索产物，用于审计答案证据。"""

    answerer_evidence = payload.get("answerer_evidence") or payload.get("top_evidence") or []
    return {
        "method": payload.get("method"),
        "node_id": payload.get("node_id"),
        "original_question": payload.get("original_question"),
        "resolved_question": payload.get("resolved_question"),
        "retrieval_question": payload.get("retrieval_question"),
        "hyperedge_retrieval_query": payload.get("hyperedge_retrieval_query"),
        "dependency_question_rewrite": payload.get("dependency_question_rewrite") or {},
        "dependency_replacements": list(payload.get("dependency_replacements") or []),
        "dependency_answers_used": list(payload.get("dependency_answers_used") or []),
        "unresolved_dependency": list(payload.get("unresolved_dependency") or []),
        "active_ancestor_node_ids": list(payload.get("active_ancestor_node_ids") or []),
        "primary_anchor_mention": payload.get("primary_anchor_mention"),
        "linked_entity_id": payload.get("linked_entity_id"),
        "anchor_match": payload.get("anchor_match") or {},
        "anchor_mentions": list(payload.get("anchor_mentions") or []),
        "linked_entities": list(payload.get("linked_entities") or []),
        "anchor_matches": list(payload.get("anchor_matches") or []),
        "unlinked_anchor_mentions": list(payload.get("unlinked_anchor_mentions") or []),
        "adjacent_hyperedge_count": len(payload.get("adjacent_hyperedge_ids") or []),
        "expansion_entity_count": len(payload.get("expansion_entity_ids") or []),
        "second_hop_hyperedge_count": len(payload.get("second_hop_hyperedge_ids") or []),
        "candidate_hyperedge_count": len(payload.get("candidate_hyperedge_ids") or []),
        "shared_candidate_hyperedge_count": len(payload.get("shared_candidate_hyperedge_ids") or []),
        "local_candidate_hyperedge_count": len(payload.get("local_candidate_hyperedge_ids") or []),
        "top_hyperedges": [dict(item) for item in payload.get("top_hyperedges") or [] if isinstance(item, dict)],
        "answerer_evidence": [
            _compact_evidence(item)
            for item in answerer_evidence
            if isinstance(item, dict)
        ],
        "insufficient_reason": payload.get("insufficient_reason", ""),
        "local_insufficient_reason": payload.get("local_insufficient_reason", ""),
        "shared_insufficient_reason": payload.get("shared_insufficient_reason", ""),
        "fallback_reason": payload.get("fallback_reason", ""),
        "atomic_answer": _compact_atomic_answer(payload.get("atomic_answer") or {}),
    }


def _compact_atomic_answer(payload: dict[str, Any]) -> dict[str, Any]:
    """保留连接 DAG 节点、证据和依赖项的答案字段。"""

    return {
        "node_id": payload.get("node_id"),
        "question": payload.get("question"),
        "analysis": payload.get("analysis") or {},
        "answer": payload.get("answer", ""),
        "reasoning_summary": payload.get("reasoning_summary", ""),
        "used_dependencies": list(payload.get("used_dependencies") or []),
        "used_hyperedge_ids": list(payload.get("used_hyperedge_ids") or []),
        "insufficient": bool(payload.get("insufficient", False)),
    }


def _compact_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """将证据记录精简为答案相关文本、实体、分数和来源。"""

    score_breakdown = payload.get("score_breakdown") or {}
    compact_score_breakdown = {
        key: score_breakdown.get(key)
        for key in (
            "selection_source",
            "semantic_rank",
            "primary_anchor_mention",
            "via_first_hyperedge_ids",
        )
        if key in score_breakdown
    }
    entity_records = []
    for raw_record in payload.get("entity_records") or []:
        if not isinstance(raw_record, dict):
            continue
        entity_records.append(
            {
                "entity_id": raw_record.get("entity_id"),
                "label": raw_record.get("label"),
                "entity_type": raw_record.get("entity_type"),
                "description": raw_record.get("description", ""),
            }
        )
    return {
        "hyperedge_id": payload.get("hyperedge_id"),
        "hyperedge_text": payload.get("hyperedge_text", ""),
        "branch_support": list(payload.get("branch_support") or []),
        "anchor_score": payload.get("anchor_score", 0.0),
        "relation_score": payload.get("relation_score", 0.0),
        "semantic_score": payload.get("semantic_score", 0.0),
        "fusion_score": payload.get("fusion_score", 0.0),
        "entity_ids": list(payload.get("entity_ids") or []),
        "entity_records": entity_records,
        "chunk_ids": list(payload.get("chunk_ids") or []),
        "chunk_texts": list(payload.get("chunk_texts") or []),
        "rank": payload.get("rank"),
        "score_breakdown": compact_score_breakdown,
    }
