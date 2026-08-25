"""Execute a DEPO atomic-question DAG over HyperBranch evidence."""

from __future__ import annotations

import re
from typing import Any

from ..llm.service import OpenAIAtomicLLMService
from .models import (
    AtomicAnswerResult,
    AtomicQuestionNode,
    DagExecutionResult,
    FusedHyperedgeCandidate,
)
from .retriever import AtomicHyperedgeRetriever, LocalHyperedgeRetrievalResult


class AtomicDagExecutor:
    """Execute DEPO's atomic questions in dependency order."""

    def __init__(
        self,
        retriever: AtomicHyperedgeRetriever,
        llm_service: OpenAIAtomicLLMService,
    ) -> None:
        self.retriever = retriever
        self.llm_service = llm_service

    def run(
        self,
        original_question: str,
        dag_payload: dict[str, Any],
        original_question_entities: list[str],
    ) -> DagExecutionResult:
        #将字典对象转换成原子问题节点对象
        nodes = [
            AtomicQuestionNode(item["id"], item["question"], item["depends_on"])
            for item in dag_payload["nodes"]
        ]
        #进行拓扑排序
        order = self.topological_sort(nodes)
        #对原问题建立一个共享候选池
        shared_pool = self.retriever.build_candidate_pool(
            entities=original_question_entities,
        )
        #保存已回答原子问题的结果
        results_by_id: dict[str, AtomicAnswerResult] = {}
        #保存每个原子问题检索到的局部候选超边池
        local_pools: dict[str, LocalHyperedgeRetrievalResult] = {}
        #整理查找表记录依赖哪些节点
        dependencies_by_id = {node.node_id: node.dependencies for node in order}
        #记录节点在拓扑排序中的位置
        positions = {node.node_id: index for index, node in enumerate(order)}

        for node in order:
            dependency_answers = [
                {
                    "node_id": results_by_id[dependency_id].node_id,
                    "question": results_by_id[dependency_id].question,
                    "answer": results_by_id[dependency_id].answer,
                }
                for dependency_id in node.dependencies
            ]
            #重写问题
            question, dependency_entities = self._rewrite_dependency_question(
                node.question,
                dependency_answers,
            )
            #复用原问题中仍出现在当前问题里的实体，并加入实际替换的依赖答案
            entities = [
                entity
                for entity in original_question_entities
                if entity in question
            ]
            entities.extend(
                answer for answer in dependency_entities if answer not in entities
            )
            #得到原子问题的超边候选池
            local_pool = self.retriever.build_candidate_pool(
                entities=entities,
            )
            #合并祖先问题和原问题候选池
            shared_pool_for_node = self._active_shared_candidate_pool(
                shared_pool,
                self._ancestor_ids(node.node_id, dependencies_by_id, positions),
                local_pools,
            )
            #合并全部候选池
            retrieval = self.retriever.merge_candidate_pools(
                shared_pool=shared_pool_for_node,
                local_pool=local_pool,
            )
            #排序
            retrieval = self.retriever.rank_candidate_pool(retrieval, question=question)
            answer = self.llm_service.answer_atomic_question(
                atomic_question=question,
                dependency_answers=dependency_answers,
                evidence=self._evidence_payload(retrieval.evidence),
                original_question=original_question,
            )["answer"]
            results_by_id[node.node_id] = AtomicAnswerResult(
                node_id=node.node_id,
                question=question,
                answer=answer,
            )
            #供后续节点集成
            local_pools[node.node_id] = local_pool
        #按照拓扑排序来组织格式
        atomic_results = [results_by_id[node.node_id] for node in order]
        return DagExecutionResult(
            atomic_results=atomic_results,
            final_answer={"answer": atomic_results[-1].answer},
        )

    @staticmethod
    def topological_sort(nodes: list[AtomicQuestionNode]) -> list[AtomicQuestionNode]:
        """Order atomic questions so that every dependency is answered first."""
        #得到by_id={"q1": AtomicQuestionNode(q1)}
        by_id = {node.node_id: node for node in nodes}
        #建立依赖图，一个空字典
        dependents = {node.node_id: [] for node in nodes}
        #计算入度
        indegree = {node.node_id: len(node.dependencies) for node in nodes}
        #建立依赖关系
        for node in nodes:
            for dependency_id in node.dependencies:
                dependents[dependency_id].append(node.node_id)
        #找到没有依赖的节点
        ready = [node.node_id for node in nodes if not indegree[node.node_id]]
        #创建拓扑排序结果列表
        order: list[AtomicQuestionNode] = []
        #只要还存在入度为0的节点
        while ready:
            node_id = ready.pop(0)
            order.append(by_id[node_id])
            #更新度数
            for dependent_id in dependents[node_id]:
                indegree[dependent_id] -= 1
                #如果依赖全部解决
                if not indegree[dependent_id]:
                    ready.append(dependent_id)
        return order

    @staticmethod
    def _rewrite_dependency_question(
        question: str,
        dependency_answers: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        """Replace qN answer references with already computed answers."""
        rewritten = question
        #保存替换进去的答案实体
        anchors: list[str] = []
        for dependency in dependency_answers:
            #获取答案
            answer = str(dependency["answer"]).strip()
            pattern = rf"\b{re.escape(dependency['node_id'])}(?:\s+answer|['\u2019]s\s+answer)\b"
            #查找位置
            matches = list(re.finditer(pattern, rewritten, flags=re.IGNORECASE))
            #倒序替换
            for match in reversed(matches):
                rewritten = f"{rewritten[:match.start()]}{answer}{rewritten[match.end():]}"
            if matches and answer not in anchors:
                anchors.append(answer)
        return rewritten, anchors

    #用于整理结构化证据给LLM
    def _evidence_payload(
        self,
        evidence: list[FusedHyperedgeCandidate],
    ) -> dict[str, list[dict[str, Any]]]:
        """Merge retrieved hyperedges into the evidence blocks given to the LLM."""

        blocks: list[dict[str, Any]] = []
        blocks_by_chunk_id: dict[str, dict[str, Any]] = {}
        for rank, candidate in enumerate(evidence, start=1):
            hyperedge = {
                "hyperedge_id": f"H{rank}",
                "hyperedge_text": candidate.hyperedge_text,
            }
            if candidate.first_hop_hyperedge_ids:
                hyperedge["first_hop_hyperedge_text"] = (
                    self.retriever.dataset.graph.describe_hyperedge(
                        candidate.first_hop_hyperedge_ids[0]
                    )["hyperedge_text"]
                )
            for chunk_id, text in _candidate_chunks(candidate) or [(f"__{rank}", "")]:
                block = blocks_by_chunk_id.get(chunk_id)
                if block is None:
                    title, _, body = text.partition("\n")
                    block = {
                        "chunk_id": f"C{len(blocks) + 1}",
                        "title": title.strip(),
                        "text": body.strip() or title.strip(),
                        "hyperedges": [],
                    }
                    blocks_by_chunk_id[chunk_id] = block
                    blocks.append(block)
                block["hyperedges"].append(hyperedge)
        return {"evidence_blocks": blocks}

    def _active_shared_candidate_pool(
        self,
        original_pool: LocalHyperedgeRetrievalResult,
        ancestor_ids: list[str],
        local_pools: dict[str, LocalHyperedgeRetrievalResult],
    ) -> LocalHyperedgeRetrievalResult:
        pool = original_pool
        #合并祖先节点证据
        for node_id in ancestor_ids:
            pool = self.retriever.merge_candidate_pools(
                shared_pool=pool,
                local_pool=local_pools[node_id],
            )
        return pool

    @staticmethod
    #找到DAG 中某个节点的所有祖先节点，并按照执行顺序排序。
    def _ancestor_ids(
        node_id: str,
        dependencies_by_id: dict[str, list[str]],
        #拓扑排序后的位置
        positions: dict[str, int],
    ) -> list[str]:
        ancestors: set[str] = set()
        #深度优先搜索
        def visit(current_id: str) -> None:
            for dependency_id in dependencies_by_id[current_id]:
                if dependency_id not in ancestors:
                    ancestors.add(dependency_id)
                    visit(dependency_id)

        visit(node_id)
        return sorted(ancestors, key=positions.__getitem__)


def _candidate_chunks(candidate: FusedHyperedgeCandidate) -> list[tuple[str, str]]:
    return list(dict.fromkeys(zip(candidate.chunk_ids, candidate.chunk_texts)))
