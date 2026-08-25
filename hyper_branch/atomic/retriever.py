"""Retrieve local hypergraph evidence for each atomic question."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..utils import normalize_label
from .models import FusedHyperedgeCandidate


@dataclass(slots=True)
class LocalHyperedgeRetrievalResult:
    """候选超边、二跳路径和最终证据。"""
    #一次局部检索结果
    #候选超边id
    candidate_hyperedge_ids: list[str] = field(default_factory=list)
    #二跳超边来自哪个一跳超边
    first_hops_by_candidate: dict[str, list[str]] = field(default_factory=dict)
    #证据
    evidence: list[FusedHyperedgeCandidate] = field(default_factory=list)


class AtomicHyperedgeRetriever:
    """实体链接、两跳超图扩展及候选超边排序。"""
    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        #从已加载的知识超图中找实体
        entity_ids = [
            node_id
            for node_id, node in self.dataset.graph.nodes.items()
            if node.role == "entity"
        ]
        #建立实体查找表：实体名字 → entity id，用于快速匹配
        self._entity_lookup = self._build_entity_lookup(entity_ids)

    def build_candidate_pool(
        self,
        *,
        entities: list[str],
    ) -> LocalHyperedgeRetrievalResult:
        """从问题实体出发，收集一跳和二跳超边。"""
        #保存已经链接到超图中的实体 ID。
        anchor_ids: list[str] = []
        for mention in entities:
            #实体链接，获得实体id
            entity_id = self.link_anchor_entity(mention=mention)
            #如果链接成功并且之前没有连接
            if entity_id is not None and entity_id not in anchor_ids:
                anchor_ids.append(entity_id)
        #初始化候选超边容器
        candidate_ids: list[str] = []
        #二跳超边对应的一跳来源。
        first_hops_by_candidate: dict[str, list[str]] = {}
        #遍历每个被连接的实体
        for entity_id in anchor_ids:
            """local_ids：该锚点扩展得到的全部候选超边 ID，包括一跳和二跳超边。
                local_first_hops：记录每个候选超边经过的第一跳超边。"""
            local_ids, local_first_hops = self._expand_anchor_neighborhood(entity_id)
            for hyperedge_id in local_ids:
                #不同实体可能扩展统一超边，去重
                if hyperedge_id not in candidate_ids:
                    candidate_ids.append(hyperedge_id)
                #拿到当前候选超边的第一跳超边
                paths = first_hops_by_candidate.setdefault(hyperedge_id, [])
                #将该超边的第一跳超边加入path
                for first_hop_id in local_first_hops[hyperedge_id]:
                    if first_hop_id not in paths:
                        paths.append(first_hop_id)

        return LocalHyperedgeRetrievalResult(
            #返回所有超边id和所有一跳超边id
            candidate_hyperedge_ids=candidate_ids,
            first_hops_by_candidate=first_hops_by_candidate,
        )

    def merge_candidate_pools(
        self,
        *,
        shared_pool: LocalHyperedgeRetrievalResult,
        local_pool: LocalHyperedgeRetrievalResult,
    ) -> LocalHyperedgeRetrievalResult:
        """融合原问题、祖先问题与当前问题的候选超边。"""
        # 初始化两个容器
        candidate_ids: list[str] = []
        first_hops_by_candidate: dict[str, list[str]] = {}
        #遍历每一个候选池
        for pool in (shared_pool, local_pool):
            #遍历每一个超边
            for hyperedge_id in pool.candidate_hyperedge_ids:
                if hyperedge_id not in candidate_ids:
                    candidate_ids.append(hyperedge_id)
                #获取一跳超边id
                paths = first_hops_by_candidate.setdefault(hyperedge_id, [])
                for first_hop_id in pool.first_hops_by_candidate[hyperedge_id]:
                    if first_hop_id not in paths:
                        paths.append(first_hop_id)

        return LocalHyperedgeRetrievalResult(
            candidate_hyperedge_ids=candidate_ids,
            first_hops_by_candidate=first_hops_by_candidate,
        )

    def rank_candidate_pool(
        self,
        result: LocalHyperedgeRetrievalResult,
        *,
        question: str,
    ) -> LocalHyperedgeRetrievalResult:
        """仅在局部候选池内按问题与超边的向量相似度排序。"""
        if not result.candidate_hyperedge_ids:
            return result
        #对问题生成 embedding
        question_vector = self.embedder.embed_texts([question])[0]
        #计算问题与超边相似度
        scores = self.dataset.hyperedge_store.similarities(
            np.asarray(question_vector, dtype=np.float32),
            result.candidate_hyperedge_ids,
        )
        #排序候选超边
        ranked_ids = sorted(
            result.candidate_hyperedge_ids,
            key=lambda hyperedge_id: (-scores.get(hyperedge_id, 0.0), hyperedge_id),
        )
        #取top-k超边
        result.evidence = [
            #转换格式，输入超边，转化成FusedHyperedgeCandidate对象
            self._evidence_from_hyperedge(hyperedge_id, result.first_hops_by_candidate[hyperedge_id])
            for hyperedge_id in ranked_ids[: self.config.local_hyperedge_top_k]
        ]
        return result

    def link_anchor_entity(self, *, mention: str) -> str | None:
        """优先精确匹配；无精确匹配时使用实体名称向量库。"""
        exact_ids = self._entity_lookup.get(self._entity_lookup_key(mention), [])
        if len(exact_ids) == 1:
            return exact_ids[0]
        if exact_ids:
            return None

        mention_vector = self.embedder.embed_texts([mention])[0]
        matches = self.dataset.entity_store.query(
            mention_vector,
            top_k=self.config.entity_link_vector_top_k,
        )
        for match in matches:
            if match.score < self.config.entity_link_vector_min_score:
                break
            matched_ids = self._entity_lookup.get(self._entity_lookup_key(match.label), [])
            if matched_ids:
                return matched_ids[0]
        return None

    def _expand_anchor_neighborhood(
        self,
        anchor_entity_id: str,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """收集一个锚点的一跳超边，并通过关联实体扩展二跳超边。"""
        #找一跳超边，entity_hyperedge_ids作用是找一个实体链接了哪些超边
        first_hop_ids = list(
            dict.fromkeys(self.dataset.graph.entity_hyperedge_ids(anchor_entity_id))
        )
        #初始化
        candidate_ids = list(first_hop_ids)
        first_hops_by_candidate = {hyperedge_id: [] for hyperedge_id in first_hop_ids}

        #遍历每一个一跳超边
        for first_hop_id in first_hop_ids:
            #获取超边实体
            bridge_entity_ids = list(
                self.dataset.graph.hyperedge_entity_ids(first_hop_id)
            )
            #从chunk中补充实体
            for chunk_id in self.dataset.graph.hyperedge_chunk_ids(first_hop_id):
                bridge_entity_ids.extend(
                    node_id
                    for node_id in self.dataset.graph.source_to_nodes[chunk_id]
                    if self.dataset.graph.nodes[node_id].role == "entity"
                )

            for bridge_entity_id in dict.fromkeys(bridge_entity_ids):
                #跳过自己
                if bridge_entity_id == anchor_entity_id:
                    continue
                #继续每个实体获取一跳超边
                for second_hop_id in self.dataset.graph.entity_hyperedge_ids(
                    bridge_entity_id
                ):
                    #去重，如果存在则不加入
                    if second_hop_id in first_hop_ids:
                        continue
                    if second_hop_id not in candidate_ids:
                        candidate_ids.append(second_hop_id)
                    paths = first_hops_by_candidate.setdefault(second_hop_id, [])
                    if first_hop_id not in paths:
                        paths.append(first_hop_id)

        return candidate_ids, first_hops_by_candidate

    def _evidence_from_hyperedge(
        self,
        hyperedge_id: str,
        first_hop_ids: list[str],
    ) -> FusedHyperedgeCandidate:
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        chunk_ids = list(dict.fromkeys(description["chunk_ids"]))
        return FusedHyperedgeCandidate(
            hyperedge_id=hyperedge_id,
            hyperedge_text=normalize_label(description["hyperedge_text"]),
            chunk_ids=chunk_ids,
            chunk_texts=[self.dataset.get_chunk_text(chunk_id) for chunk_id in chunk_ids],
            first_hop_hyperedge_ids=first_hop_ids,
        )

    #将实体名称转化成统一格式
    @staticmethod
    def _build_entity_lookup(entity_ids: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for entity_id in entity_ids:
            key = AtomicHyperedgeRetriever._entity_lookup_key(entity_id)
            if entity_id not in lookup[key]:
                lookup[key].append(entity_id)
        return lookup

    @staticmethod
    def _entity_lookup_key(text: str) -> str:
        return unicodedata.normalize("NFKC", normalize_label(text)).casefold()
