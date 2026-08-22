"""围绕实体锚点、结合向量检索，为原子问题获取局部超图证据。"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..models import VectorMatch
from ..utils import normalize_label
from .models import AtomicQuestionAnalysis, FusedHyperedgeCandidate

_LOCAL_RETRIEVAL_METHOD = "two_hop_multi_anchor_topk"
_SHARED_RETRIEVAL_METHOD = "shared_original_question_augmented_topk"

@dataclass(slots=True)
class AnchorEntityMatch:
    """一条查询提及与图实体的链接结果及其匹配证据。"""

    query_index: int
    query_entity: str
    entity_id: str
    match_type: str
    link_score: float
    vector_score: float | None = None
    candidate_rank: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "query_entity": self.query_entity,
            "matched_entity": normalize_label(self.entity_id),
            "matched_entity_id": self.entity_id,
            "match_type": self.match_type,
            "link_score": self.link_score,
            "vector_score": self.vector_score,
            "candidate_rank": self.candidate_rank,
        }


@dataclass(slots=True)
class LocalHyperedgeRetrievalResult:
    """候选池及其来源、不足原因、排序结果和作答证据。"""

    method: str = _LOCAL_RETRIEVAL_METHOD
    primary_anchor_mention: str = ""
    linked_entity_id: str = ""
    anchor_match: dict[str, Any] = field(default_factory=dict)
    anchor_mentions: list[str] = field(default_factory=list)
    linked_entities: list[dict[str, Any]] = field(default_factory=list)
    anchor_matches: list[dict[str, Any]] = field(default_factory=list)
    unlinked_anchor_mentions: list[str] = field(default_factory=list)
    adjacent_hyperedge_ids: list[str] = field(default_factory=list)
    expansion_entity_ids: list[str] = field(default_factory=list)
    second_hop_hyperedge_ids: list[str] = field(default_factory=list)
    candidate_hyperedge_ids: list[str] = field(default_factory=list)
    shared_candidate_hyperedge_ids: list[str] = field(default_factory=list)
    local_candidate_hyperedge_ids: list[str] = field(default_factory=list)
    candidate_sources: list[dict[str, Any]] = field(default_factory=list)
    top_hyperedges: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[FusedHyperedgeCandidate] = field(default_factory=list)
    insufficient_reason: str = ""
    local_insufficient_reason: str = ""
    shared_insufficient_reason: str = ""

    @property
    def insufficient(self) -> bool:
        return bool(self.insufficient_reason) or not self.evidence

class AtomicHyperedgeRetriever:
    """围绕实体锚点构建局部候选，再按问题相似度排序超边。"""

    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self._entity_ids = [
            node_id for node_id, node in self.dataset.graph.nodes.items() if getattr(node, "role", "") == "entity"
        ]
        self._entity_lookup = self._build_entity_lookup(self._entity_ids)

    def build_original_question_candidate_pool(
        self,
        *,
        question: str,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str = "",
    ) -> LocalHyperedgeRetrievalResult:
        """在执行 DAG 前，从原问题构建共享检索上下文。"""

        result = self._build_anchor_candidate_pool(
            analysis=analysis,
            primary_anchor_mention=primary_anchor_mention,
            method=_SHARED_RETRIEVAL_METHOD,
            pool_source="original_question_shared_pool",
        )
        result.shared_candidate_hyperedge_ids = list(result.candidate_hyperedge_ids)
        result.shared_insufficient_reason = result.insufficient_reason
        return result

    def build_atomic_candidate_pool(
        self,
        *,
        question: str,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str,
    ) -> LocalHyperedgeRetrievalResult:
        """根据节点已解析的问题和主锚点构建局部候选。"""

        result = self._build_anchor_candidate_pool(
            analysis=analysis,
            primary_anchor_mention=primary_anchor_mention,
            method=_LOCAL_RETRIEVAL_METHOD,
            pool_source="atomic_node_local_pool",
        )
        result.local_candidate_hyperedge_ids = list(result.candidate_hyperedge_ids)
        result.local_insufficient_reason = result.insufficient_reason
        return result

    def _build_anchor_candidate_pool(
        self,
        *,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str,
        method: str,
        pool_source: str,
    ) -> LocalHyperedgeRetrievalResult:
        """链接锚点提及、扩展图邻域，并记录每个候选的来源。"""

        anchor_mentions = self._anchor_mentions(primary_anchor_mention, analysis)
        primary_mention = anchor_mentions[0] if anchor_mentions else ""
        result = LocalHyperedgeRetrievalResult(
            method=method,
            primary_anchor_mention=primary_mention,
            anchor_mentions=list(anchor_mentions),
        )
        if not anchor_mentions:
            result.insufficient_reason = "missing_primary_anchor"
            return result

        linked_matches: list[AnchorEntityMatch] = []
        seen_linked_entity_ids: set[str] = set()
        for query_index, mention in enumerate(anchor_mentions):
            # 保留未链接提及，便于在结果中明确报告实体链接失败。
            match = self.link_anchor_entity(mention=mention, query_index=query_index)
            if match is None:
                result.unlinked_anchor_mentions.append(mention)
                continue
            if match.entity_id in seen_linked_entity_ids:
                continue
            seen_linked_entity_ids.add(match.entity_id)
            linked_matches.append(match)

        if not linked_matches:
            result.insufficient_reason = "unlinked_primary_anchor"
            return result

        expansion_entity_ids = {match.entity_id for match in linked_matches}
        result.linked_entity_id = linked_matches[0].entity_id
        result.anchor_match = linked_matches[0].to_metadata()
        result.anchor_matches = [match.to_metadata() for match in linked_matches]
        result.linked_entities = [
            {
                "mention": match.query_entity,
                "entity_id": match.entity_id,
                "match_type": match.match_type,
                "link_score": match.link_score,
                "used_for_expansion": match.entity_id in expansion_entity_ids,
            }
            for match in linked_matches
        ]

        candidate_pool = self._multi_anchor_candidate_pool(linked_matches)
        result.adjacent_hyperedge_ids = list(candidate_pool["adjacent_hyperedge_ids"])
        result.expansion_entity_ids = list(candidate_pool["expansion_entity_ids"])
        result.second_hop_hyperedge_ids = list(candidate_pool["second_hop_hyperedge_ids"])
        result.candidate_hyperedge_ids = list(candidate_pool["candidate_hyperedge_ids"])
        result.candidate_sources = [dict(item) for item in candidate_pool["candidate_sources"]]
        self._tag_candidate_pool_sources(result, pool_source)
        if not result.candidate_hyperedge_ids:
            result.insufficient_reason = (
                "primary_anchor_has_no_adjacent_hyperedges"
                if not result.adjacent_hyperedge_ids
                else "no_local_candidate_hyperedges"
            )
            return result

        return result

    def merge_candidate_pools(
        self,
        *,
        shared_pool: LocalHyperedgeRetrievalResult,
        local_pool: LocalHyperedgeRetrievalResult,
    ) -> LocalHyperedgeRetrievalResult:
        """融合原问题和节点局部候选，同时不丢失其来源。"""

        result = LocalHyperedgeRetrievalResult(
            method=_SHARED_RETRIEVAL_METHOD,
            primary_anchor_mention=local_pool.primary_anchor_mention,
            linked_entity_id=local_pool.linked_entity_id,
            anchor_match=dict(local_pool.anchor_match),
            anchor_mentions=list(local_pool.anchor_mentions),
            linked_entities=[dict(item) for item in local_pool.linked_entities],
            anchor_matches=[dict(item) for item in local_pool.anchor_matches],
            unlinked_anchor_mentions=list(local_pool.unlinked_anchor_mentions),
            adjacent_hyperedge_ids=list(local_pool.adjacent_hyperedge_ids),
            expansion_entity_ids=_dedupe_strings(
                [*shared_pool.expansion_entity_ids, *local_pool.expansion_entity_ids]
            ),
            second_hop_hyperedge_ids=_dedupe_strings(
                [*shared_pool.second_hop_hyperedge_ids, *local_pool.second_hop_hyperedge_ids]
            ),
            shared_candidate_hyperedge_ids=list(shared_pool.candidate_hyperedge_ids),
            local_candidate_hyperedge_ids=list(local_pool.candidate_hyperedge_ids),
            local_insufficient_reason=local_pool.insufficient_reason,
            shared_insufficient_reason=shared_pool.insufficient_reason,
        )
        candidate_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}
        for pool in (shared_pool, local_pool):
            # 保持首次出现顺序，使排序并列时仍可复现。
            for hyperedge_id in pool.candidate_hyperedge_ids:
                if hyperedge_id not in candidate_ids:
                    candidate_ids.append(hyperedge_id)
            for source in pool.candidate_sources:
                self._merge_candidate_source(source_by_id, dict(source))

        result.candidate_hyperedge_ids = candidate_ids
        result.candidate_sources = [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids if hyperedge_id in source_by_id]
        if not result.candidate_hyperedge_ids:
            result.insufficient_reason = local_pool.insufficient_reason or shared_pool.insufficient_reason
        return result

    def rank_candidate_pool(
        self,
        result: LocalHyperedgeRetrievalResult,
        *,
        question: str,
    ) -> LocalHyperedgeRetrievalResult:
        """对融合候选池进行向量排序，并将 Top 超边转换为可作答证据。"""

        source_by_id: dict[str, dict[str, Any]] = {}
        for source in result.candidate_sources:
            self._merge_candidate_source(source_by_id, dict(source))
        # 相似度只对已被局部图检索接纳的候选排序。
        scores = self._hyperedge_similarity_scores(question, result.candidate_hyperedge_ids)
        ranked = [
            {
                "hyperedge_id": hyperedge_id,
                "semantic_score": float(scores.get(hyperedge_id, 0.0)),
                "rank": 0,
                "hop": int(source_by_id.get(hyperedge_id, {}).get("hop", 1)),
                "via_entity_ids": list(source_by_id.get(hyperedge_id, {}).get("via_entity_ids", [])),
                "via_first_hyperedge_ids": list(source_by_id.get(hyperedge_id, {}).get("via_first_hyperedge_ids", [])),
                "expansion_sources": list(source_by_id.get(hyperedge_id, {}).get("expansion_sources", [])),
                "via_chunk_ids": list(source_by_id.get(hyperedge_id, {}).get("via_chunk_ids", [])),
                "pool_sources": list(source_by_id.get(hyperedge_id, {}).get("pool_sources", [])),
            }
            for hyperedge_id in result.candidate_hyperedge_ids
        ]
        ranked.sort(key=lambda item: (-float(item["semantic_score"]), str(item["hyperedge_id"])))
        top_k = max(0, int(self.config.local_hyperedge_top_k))
        selected = ranked[:top_k]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        result.top_hyperedges = selected
        result.evidence = [
            self._evidence_from_hyperedge(
                hyperedge_id=str(item["hyperedge_id"]),
                semantic_score=float(item["semantic_score"]),
                rank=int(item["rank"]),
                primary_anchor_mention=result.primary_anchor_mention,
                primary_anchor_entity_id=result.linked_entity_id,
                candidate_source=source_by_id.get(str(item["hyperedge_id"]), {}),
                selection_source=result.method,
            )
            for item in selected
        ]
        result.evidence = [item for item in result.evidence if item.hyperedge_id]
        if not result.evidence and not result.insufficient_reason:
            result.insufficient_reason = "no_valid_local_evidence"
        return result

    @staticmethod
    def _tag_candidate_pool_sources(result: LocalHyperedgeRetrievalResult, pool_source: str) -> None:
        if not pool_source:
            return
        for source in result.candidate_sources:
            pool_sources = source.setdefault("pool_sources", [])
            if pool_source not in pool_sources:
                pool_sources.append(pool_source)

    @staticmethod
    def _anchor_mentions(primary_anchor_mention: str, analysis: AtomicQuestionAnalysis) -> list[str]:
        raw_mentions = [primary_anchor_mention, *analysis.entities]
        mentions: list[str] = []
        seen: set[str] = set()
        for raw_mention in raw_mentions:
            mention = normalize_label(str(raw_mention or "").strip())
            key = mention.lower()
            if mention and key not in seen:
                seen.add(key)
                mentions.append(mention)
        return mentions

    def link_anchor_entity(
        self,
        *,
        mention: str,
        query_index: int,
    ) -> AnchorEntityMatch | None:
        """按精确名称或达到阈值的向量相似度链接单条实体提及。"""

        matches = self._resolve_anchor_entity_matches(
            entity=mention,
            query_index=query_index,
        )
        if not matches:
            return None
        matches.sort(
            key=lambda item: (
                -float(item.link_score),
                int(item.candidate_rank if item.candidate_rank is not None else 0),
                str(item.entity_id),
            )
        )
        return matches[0]

    def _adjacent_hyperedge_ids(self, entity_id: str) -> list[str]:
        return _dedupe_strings([str(item) for item in self.dataset.graph.entity_hyperedge_ids(entity_id)])

    def _multi_anchor_candidate_pool(self, matches: list[AnchorEntityMatch]) -> dict[str, Any]:
        adjacent_ids: list[str] = []
        expansion_entity_ids: list[str] = []
        second_hop_ids: list[str] = []
        candidate_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}

        for match in matches:
            first_hop_ids = self._adjacent_hyperedge_ids(match.entity_id)
            adjacent_ids.extend(hyperedge_id for hyperedge_id in first_hop_ids if hyperedge_id not in adjacent_ids)
            candidate_pool = self._local_candidate_pool(match.entity_id, first_hop_ids)
            expansion_entity_ids.extend(
                entity_id
                for entity_id in candidate_pool["expansion_entity_ids"]
                if entity_id not in expansion_entity_ids
            )
            second_hop_ids.extend(
                hyperedge_id
                for hyperedge_id in candidate_pool["second_hop_hyperedge_ids"]
                if hyperedge_id not in second_hop_ids
            )
            for hyperedge_id in candidate_pool["candidate_hyperedge_ids"]:
                if hyperedge_id not in candidate_ids:
                    candidate_ids.append(hyperedge_id)
            for source in candidate_pool["candidate_sources"]:
                enriched = dict(source)
                enriched["anchor_mentions"] = [match.query_entity]
                enriched["anchor_entity_ids"] = [match.entity_id]
                enriched["anchor_query_indices"] = [match.query_index]
                self._merge_candidate_source(source_by_id, enriched)

        return {
            "adjacent_hyperedge_ids": adjacent_ids,
            "expansion_entity_ids": expansion_entity_ids,
            "second_hop_hyperedge_ids": second_hop_ids,
            "candidate_hyperedge_ids": candidate_ids,
            "candidate_sources": [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids],
        }

    def _local_candidate_pool(self, primary_anchor_entity_id: str, first_hop_ids: list[str]) -> dict[str, Any]:
        candidate_ids = _dedupe_strings(list(first_hop_ids))
        expansion_entity_ids: list[str] = []
        second_hop_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}
        for hyperedge_id in first_hop_ids:
            source_by_id[hyperedge_id] = {
                "hyperedge_id": hyperedge_id,
                "hop": 1,
                "via_entity_ids": [primary_anchor_entity_id],
                "via_first_hyperedge_ids": [],
            }

        max_hops = max(1, int(self.config.local_hyperedge_hops))
        if max_hops >= 2:
            for first_hop_id in first_hop_ids:
                for entity_id in self._hyperedge_entity_ids(first_hop_id):
                    if entity_id == primary_anchor_entity_id:
                        continue
                    self._add_second_hop_candidates(
                        entity_id=entity_id,
                        first_hop_id=first_hop_id,
                        first_hop_ids=first_hop_ids,
                        candidate_ids=candidate_ids,
                        expansion_entity_ids=expansion_entity_ids,
                        second_hop_ids=second_hop_ids,
                        source_by_id=source_by_id,
                        expansion_source="hyperedge_entity",
                    )
                for entity_id in self._chunk_entity_ids_for_hyperedge(first_hop_id):
                    if entity_id == primary_anchor_entity_id:
                        continue
                    self._add_second_hop_candidates(
                        entity_id=entity_id,
                        first_hop_id=first_hop_id,
                        first_hop_ids=first_hop_ids,
                        candidate_ids=candidate_ids,
                        expansion_entity_ids=expansion_entity_ids,
                        second_hop_ids=second_hop_ids,
                        source_by_id=source_by_id,
                        expansion_source="chunk_entity",
                        via_chunk_ids=self._hyperedge_chunk_ids(first_hop_id),
                    )

        return {
            "expansion_entity_ids": expansion_entity_ids,
            "second_hop_hyperedge_ids": second_hop_ids,
            "candidate_hyperedge_ids": candidate_ids,
            "candidate_sources": [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids],
        }

    def _add_second_hop_candidates(
        self,
        *,
        entity_id: str,
        first_hop_id: str,
        first_hop_ids: list[str],
        candidate_ids: list[str],
        expansion_entity_ids: list[str],
        second_hop_ids: list[str],
        source_by_id: dict[str, dict[str, Any]],
        expansion_source: str,
        via_chunk_ids: list[str] | None = None,
    ) -> None:
        if entity_id not in expansion_entity_ids:
            expansion_entity_ids.append(entity_id)
        for second_hop_id in self._adjacent_hyperedge_ids(entity_id):
            if second_hop_id in first_hop_ids:
                continue
            if second_hop_id not in candidate_ids:
                candidate_ids.append(second_hop_id)
            if second_hop_id not in second_hop_ids:
                second_hop_ids.append(second_hop_id)
            source = source_by_id.setdefault(
                second_hop_id,
                {
                    "hyperedge_id": second_hop_id,
                    "hop": 2,
                    "via_entity_ids": [],
                    "via_first_hyperedge_ids": [],
                    "expansion_sources": [],
                    "via_chunk_ids": [],
                },
            )
            if int(source.get("hop", 2)) > 1:
                source["hop"] = 2
            if entity_id not in source["via_entity_ids"]:
                source["via_entity_ids"].append(entity_id)
            if first_hop_id not in source["via_first_hyperedge_ids"]:
                source["via_first_hyperedge_ids"].append(first_hop_id)
            if expansion_source not in source["expansion_sources"]:
                source["expansion_sources"].append(expansion_source)
            for chunk_id in via_chunk_ids or []:
                if chunk_id not in source["via_chunk_ids"]:
                    source["via_chunk_ids"].append(chunk_id)

    @staticmethod
    def _merge_candidate_source(source_by_id: dict[str, dict[str, Any]], source: dict[str, Any]) -> None:
        hyperedge_id = str(source.get("hyperedge_id", "") or "")
        if not hyperedge_id:
            return
        existing = source_by_id.setdefault(
            hyperedge_id,
            {
                "hyperedge_id": hyperedge_id,
                "hop": int(source.get("hop", 1) or 1),
                "via_entity_ids": [],
                "via_first_hyperedge_ids": [],
                "expansion_sources": [],
                "via_chunk_ids": [],
                "anchor_mentions": [],
                "anchor_entity_ids": [],
                "anchor_query_indices": [],
                "pool_sources": [],
            },
        )
        existing["hop"] = min(int(existing.get("hop", 1) or 1), int(source.get("hop", 1) or 1))
        for key in (
            "via_entity_ids",
            "via_first_hyperedge_ids",
            "expansion_sources",
            "via_chunk_ids",
            "anchor_mentions",
            "anchor_entity_ids",
            "pool_sources",
        ):
            for value in source.get(key, []):
                if value not in existing[key]:
                    existing[key].append(value)
        for value in source.get("anchor_query_indices", []):
            if value not in existing["anchor_query_indices"]:
                existing["anchor_query_indices"].append(value)

    def _hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        return _dedupe_strings([str(item) for item in self.dataset.graph.hyperedge_entity_ids(hyperedge_id)])

    def _hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        return _dedupe_strings([str(item) for item in self.dataset.graph.hyperedge_chunk_ids(hyperedge_id)])

    def _chunk_entity_ids(self, chunk_id: str) -> list[str]:
        source_to_nodes = self.dataset.graph.source_to_nodes
        nodes = self.dataset.graph.nodes
        entity_ids: list[str] = []
        for node_id in source_to_nodes.get(chunk_id, []):
            entity_id = str(node_id)
            if nodes[entity_id].role != "entity":
                continue
            if entity_id not in entity_ids:
                entity_ids.append(entity_id)
        return entity_ids

    def _chunk_entity_ids_for_hyperedge(self, hyperedge_id: str) -> list[str]:
        entity_ids: list[str] = []
        for chunk_id in self._hyperedge_chunk_ids(hyperedge_id):
            for entity_id in self._chunk_entity_ids(chunk_id):
                if entity_id not in entity_ids:
                    entity_ids.append(entity_id)
        return entity_ids

    def _hyperedge_similarity_scores(self, query: str, hyperedge_ids: list[str]) -> dict[str, float]:
        if not hyperedge_ids:
            return {}
        vector = self.embedder.embed_texts(
            [query], stage="atomic_local_hyperedge_retrieval"
        )[0]
        scores = self.dataset.hyperedge_store.similarities(
            np.asarray(vector, dtype=np.float32),
            hyperedge_ids,
        )
        return {hyperedge_id: float(scores.get(hyperedge_id, 0.0)) for hyperedge_id in hyperedge_ids}

    def _evidence_from_hyperedge(
        self,
        *,
        hyperedge_id: str,
        semantic_score: float,
        rank: int,
        primary_anchor_mention: str,
        primary_anchor_entity_id: str,
        candidate_source: dict[str, Any],
        selection_source: str = _LOCAL_RETRIEVAL_METHOD,
    ) -> FusedHyperedgeCandidate:
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        hyperedge_text = normalize_label(str(description.get("hyperedge_text") or hyperedge_id))
        entity_ids = _dedupe_strings([str(item) for item in description.get("entity_ids", [])])
        chunk_ids = _dedupe_strings([str(item) for item in description.get("chunk_ids", [])])
        chunk_texts = [self.dataset.get_chunk_text(chunk_id) for chunk_id in chunk_ids]
        entity_records = [self._entity_payload(entity_id) for entity_id in entity_ids]
        candidate_hop = int(candidate_source.get("hop", 1) or 1)
        return FusedHyperedgeCandidate(
            hyperedge_id=hyperedge_id,
            hyperedge_text=hyperedge_text,
            branch_support={"local_primary_anchor", f"hop{candidate_hop}"},
            semantic_score=float(semantic_score),
            entity_ids=entity_ids,
            entity_records=entity_records,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            evidence_texts=[text for text in [hyperedge_text, *chunk_texts] if text],
            rank=int(rank),
            score_breakdown={
                "selection_source": selection_source,
                "semantic_rank": int(rank),
                "semantic_score": float(semantic_score),
                "primary_anchor_mention": primary_anchor_mention,
                "primary_anchor_entity_id": primary_anchor_entity_id,
                "candidate_hop": candidate_hop,
                "via_entity_ids": list(candidate_source.get("via_entity_ids", [])),
                "via_first_hyperedge_ids": list(candidate_source.get("via_first_hyperedge_ids", [])),
                "expansion_sources": list(candidate_source.get("expansion_sources", [])),
                "via_chunk_ids": list(candidate_source.get("via_chunk_ids", [])),
                "pool_sources": list(candidate_source.get("pool_sources", [])),
                "anchor_mentions": list(candidate_source.get("anchor_mentions", [])),
                "anchor_entity_ids": list(candidate_source.get("anchor_entity_ids", [])),
                "anchor_query_indices": list(candidate_source.get("anchor_query_indices", [])),
            },
        )

    def _entity_payload(self, entity_id: str) -> dict[str, Any]:
        node = self.dataset.graph.nodes[entity_id]
        payload = {
            "entity_id": entity_id,
            "label": normalize_label(entity_id),
            "entity_type": getattr(node, "entity_type", None),
            "description": str(getattr(node, "description", "") or ""),
        }
        if hasattr(node, "to_dict"):
            payload["metadata"] = node.to_dict()
        return payload

    def _resolve_anchor_entity_matches(
        self,
        entity: str,
        query_index: int,
    ) -> list[AnchorEntityMatch]:
        exact_candidates = self._entity_lookup_candidates(entity)
        if len(exact_candidates) == 1:
            return [self._anchor_match_from_candidate(exact_candidates[0], query_index=query_index, query_entity=entity)]
        # 规范化后仍对应多个名称时，不使用向量模型猜测其中一个。
        if exact_candidates:
            return []
        vector_candidates = self._anchor_entity_vector_candidates(entity)
        if not vector_candidates:
            return []
        return [self._anchor_match_from_candidate(vector_candidates[0], query_index=query_index, query_entity=entity)]

    def _entity_lookup_candidates(self, entity: str) -> list[dict[str, Any]]:
        return self._entity_lookup_candidates_for_label(entity)

    def _entity_lookup_candidates_for_label(self, label: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for key in _lookup_keys_from_variants([label]):
            for entity_id in self._entity_lookup.get(key, []):
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                candidate = {
                    "entity_id": entity_id,
                    "label": normalize_label(entity_id),
                    "link_score": 1.0,
                    "vector_score": 1.0,
                    "candidate_rank": 1,
                    "source_label": normalize_label(entity_id),
                    "source_item_id": entity_id,
                    "match_type": "exact",
                }
                candidates.append(candidate)
        return candidates

    def _anchor_entity_vector_candidates(self, entity: str) -> list[dict[str, Any]]:
        top_k = max(1, int(self.config.entity_link_vector_top_k))
        min_score = float(self.config.entity_link_vector_min_score)
        matches = self._query_entity_store(entity, top_k)
        for rank, match in enumerate(matches, start=1):
            score = float(match.score)
            if score < min_score:
                break
            entity_id = self._resolve_entity_id_from_vector_match(match)
            if not entity_id:
                continue
            return [
                {
                    "entity_id": entity_id,
                    "label": normalize_label(entity_id),
                    "link_score": score,
                    "vector_score": score,
                    "candidate_rank": rank,
                    "source_label": normalize_label(match.label),
                    "source_item_id": match.item_id,
                    "match_type": "vector",
                }
            ]
        return []

    @staticmethod
    def _anchor_match_from_candidate(
        candidate: dict[str, Any],
        *,
        query_index: int,
        query_entity: str,
    ) -> AnchorEntityMatch:
        link_score = float(candidate.get("link_score", candidate.get("vector_score", 0.0)) or 0.0)
        return AnchorEntityMatch(
            query_index=query_index,
            query_entity=query_entity,
            entity_id=str(candidate["entity_id"]),
            match_type=str(candidate.get("match_type", "entity_link")),
            link_score=link_score,
            vector_score=float(candidate.get("vector_score", link_score) or link_score),
            candidate_rank=int(candidate.get("candidate_rank", 0) or 0),
        )

    def _query_entity_store(self, entity: str, top_k: int) -> list[VectorMatch]:
        vector = self.embedder.embed_texts([entity], stage="atomic_anchor_entity_retrieval")[0]
        return list(self.dataset.entity_store.query(vector, top_k=top_k))

    def _resolve_entity_id_from_vector_match(self, match: VectorMatch) -> str | None:
        metadata = match.metadata if isinstance(match.metadata, dict) else {}
        raw_candidates = [
            match.label,
            match.item_id,
            metadata.get("entity_name"),
            metadata.get("__id__"),
            metadata.get("name"),
        ]
        for raw_candidate in raw_candidates:
            if raw_candidate is None:
                continue
            candidate = str(raw_candidate).strip()
            if not candidate:
                continue
            node = self.dataset.graph.nodes.get(candidate)
            if node is not None and getattr(node, "role", "") == "entity":
                return candidate
            mapped = self._entity_lookup.get(_entity_lookup_key(candidate), [])
            if mapped:
                return mapped[0]
        return None

    @staticmethod
    def _build_entity_lookup(values: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for value in values:
            for key in _lookup_keys_from_variants([value]):
                if key and value not in lookup[key]:
                    lookup[key].append(value)
        return lookup


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _lookup_keys_from_variants(variants: list[str]) -> list[str]:
    keys: list[str] = []
    for variant in variants:
        key = _entity_lookup_key(variant)
        if key and key not in keys:
            keys.append(key)
    return keys


def _entity_lookup_key(text: str) -> str:
    """仅做 Unicode 兼容规范化、展示标签清洗和大小写折叠。"""

    return unicodedata.normalize("NFKC", normalize_label(str(text or ""))).casefold()
