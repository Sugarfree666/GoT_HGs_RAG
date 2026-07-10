from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..llm.service import AtomicLLMService
from ..models import VectorMatch
from ..utils import normalize_label, short_text
from .models import AtomicQuestionAnalysis, FusedHyperedgeCandidate


_ANCHOR_ENTITY_TOP_K = 3
_ANCHOR_ENTITY_LLM_MIN_CONFIDENCE = 0.6


@dataclass(slots=True)
class AnchorEntityMatch:
    query_index: int
    query_entity: str
    entity_id: str
    match_type: str
    link_score: float
    vector_score: float | None = None
    llm_confidence: float | None = None
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
            "llm_confidence": self.llm_confidence,
            "candidate_rank": self.candidate_rank,
        }


@dataclass(slots=True)
class LocalHyperedgeRetrievalResult:
    primary_anchor_mention: str = ""
    linked_entity_id: str = ""
    anchor_match: dict[str, Any] = field(default_factory=dict)
    adjacent_hyperedge_ids: list[str] = field(default_factory=list)
    top_hyperedges: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[FusedHyperedgeCandidate] = field(default_factory=list)
    insufficient_reason: str = ""

    @property
    def insufficient(self) -> bool:
        return bool(self.insufficient_reason) or not self.evidence

    def to_artifact(self) -> dict[str, Any]:
        return {
            "primary_anchor_mention": self.primary_anchor_mention,
            "linked_entity_id": self.linked_entity_id,
            "anchor_match": dict(self.anchor_match),
            "adjacent_hyperedge_ids": list(self.adjacent_hyperedge_ids),
            "top_hyperedges": [dict(item) for item in self.top_hyperedges],
            "evidence": [item.to_dict() for item in self.evidence],
            "insufficient_reason": self.insufficient_reason,
        }


class AtomicHyperedgeRetriever:
    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
        llm_service: AtomicLLMService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self.llm_service = llm_service
        self.logger = logger or logging.getLogger(__name__)
        self._entity_ids = [
            node_id for node_id, node in self.dataset.graph.nodes.items() if getattr(node, "role", "") == "entity"
        ]
        self._entity_lookup = self._normalized_lookup(self._entity_ids)

    def retrieve_primary_anchor_local(
        self,
        *,
        question: str,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str,
    ) -> LocalHyperedgeRetrievalResult:
        mention = normalize_label(str(primary_anchor_mention or "").strip())
        result = LocalHyperedgeRetrievalResult(primary_anchor_mention=mention)
        if not mention:
            result.insufficient_reason = "missing_primary_anchor"
            return result

        match = self.link_primary_anchor(
            question=question,
            mention=mention,
            analysis=analysis,
        )
        if match is None:
            result.insufficient_reason = "unlinked_primary_anchor"
            return result

        result.linked_entity_id = match.entity_id
        result.anchor_match = match.to_metadata()
        adjacent_ids = self._adjacent_hyperedge_ids(match.entity_id)
        result.adjacent_hyperedge_ids = list(adjacent_ids)
        if not adjacent_ids:
            result.insufficient_reason = "primary_anchor_has_no_adjacent_hyperedges"
            return result

        scores = self._hyperedge_similarity_scores(question, adjacent_ids)
        ranked = [
            {
                "hyperedge_id": hyperedge_id,
                "semantic_score": float(scores.get(hyperedge_id, 0.0)),
                "rank": 0,
            }
            for hyperedge_id in adjacent_ids
        ]
        ranked.sort(key=lambda item: (-float(item["semantic_score"]), str(item["hyperedge_id"])))
        top_k = max(0, int(getattr(self.config, "local_hyperedge_top_k", 3)))
        selected = ranked[:top_k]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        result.top_hyperedges = selected
        result.evidence = [
            self._evidence_from_hyperedge(
                hyperedge_id=str(item["hyperedge_id"]),
                semantic_score=float(item["semantic_score"]),
                rank=int(item["rank"]),
                primary_anchor_mention=mention,
                primary_anchor_entity_id=match.entity_id,
            )
            for item in selected
        ]
        result.evidence = [item for item in result.evidence if item.hyperedge_id]
        if not result.evidence:
            result.insufficient_reason = "no_valid_local_evidence"
        return result

    def link_primary_anchor(
        self,
        *,
        question: str,
        mention: str,
        analysis: AtomicQuestionAnalysis,
    ) -> AnchorEntityMatch | None:
        matches = self._resolve_anchor_entity_matches(
            question=question,
            entity=mention,
            analysis=analysis,
            query_index=0,
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
        if not entity_id or not hasattr(self.dataset.graph, "entity_hyperedge_ids"):
            return []
        return _dedupe_strings([str(item) for item in self.dataset.graph.entity_hyperedge_ids(entity_id)])

    def _hyperedge_similarity_scores(self, query: str, hyperedge_ids: list[str]) -> dict[str, float]:
        if not query.strip() or not hyperedge_ids:
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        store = getattr(self.dataset, "hyperedge_store", None)
        if store is None or not hasattr(store, "similarities"):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        try:
            vectors = self.embedder.embed_texts([query], stage="atomic_local_hyperedge_retrieval")
            if not vectors:
                return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
            query_vector = np.asarray(vectors[0], dtype=np.float32)
            scores = dict(store.similarities(query_vector, list(hyperedge_ids)))
        except (TypeError, ValueError, AttributeError):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        return {hyperedge_id: float(scores.get(hyperedge_id, 0.0)) for hyperedge_id in hyperedge_ids}

    def _evidence_from_hyperedge(
        self,
        *,
        hyperedge_id: str,
        semantic_score: float,
        rank: int,
        primary_anchor_mention: str,
        primary_anchor_entity_id: str,
    ) -> FusedHyperedgeCandidate:
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        hyperedge_text = normalize_label(str(description.get("hyperedge_text") or hyperedge_id))
        entity_ids = _dedupe_strings([str(item) for item in description.get("entity_ids", [])])
        if not entity_ids and hasattr(self.dataset.graph, "hyperedge_entity_ids"):
            entity_ids = _dedupe_strings([str(item) for item in self.dataset.graph.hyperedge_entity_ids(hyperedge_id)])
        chunk_ids = _dedupe_strings([str(item) for item in description.get("chunk_ids", [])])
        chunk_texts = [self.dataset.get_chunk_text(chunk_id) for chunk_id in chunk_ids]
        entity_records = [self._entity_payload(entity_id) for entity_id in entity_ids]
        return FusedHyperedgeCandidate(
            hyperedge_id=hyperedge_id,
            hyperedge_text=hyperedge_text,
            branch_support={"local_primary_anchor"},
            semantic_score=float(semantic_score),
            fusion_score=float(semantic_score),
            entity_ids=entity_ids,
            entity_records=entity_records,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            evidence_texts=[text for text in [hyperedge_text, *chunk_texts] if text],
            rank=int(rank),
            score_breakdown={
                "selection_source": "single_hop_primary_anchor_top3",
                "semantic_rank": int(rank),
                "semantic_score": float(semantic_score),
                "primary_anchor_mention": primary_anchor_mention,
                "primary_anchor_entity_id": primary_anchor_entity_id,
            },
        )

    def _entity_payload(self, entity_id: str) -> dict[str, Any]:
        nodes = getattr(self.dataset.graph, "nodes", {})
        node = nodes.get(entity_id) if hasattr(nodes, "get") else None
        payload = {
            "entity_id": entity_id,
            "label": normalize_label(entity_id),
            "entity_type": getattr(node, "entity_type", None),
            "description": str(getattr(node, "description", "") or ""),
        }
        if node is not None and hasattr(node, "to_dict"):
            payload["metadata"] = node.to_dict()
        return payload

    def _resolve_anchor_entity_matches(
        self,
        question: str,
        entity: str,
        analysis: AtomicQuestionAnalysis,
        query_index: int,
    ) -> list[AnchorEntityMatch]:
        exact_entity_ids = self._resolve_entity_ids(entity)
        if exact_entity_ids:
            return [
                AnchorEntityMatch(
                    query_index=query_index,
                    query_entity=entity,
                    entity_id=entity_id,
                    match_type="exact",
                    link_score=1.0,
                )
                for entity_id in exact_entity_ids
            ]

        candidates = self._anchor_entity_vector_candidates(entity)
        if not candidates:
            return []

        selected = self._select_anchor_entity_with_llm(
            question=question,
            entity=entity,
            analysis=analysis,
            candidates=candidates,
        )
        if selected is None:
            return []

        return [
            AnchorEntityMatch(
                query_index=query_index,
                query_entity=entity,
                entity_id=str(selected["entity_id"]),
                match_type="vector_llm",
                link_score=float(selected["vector_score"]),
                vector_score=float(selected["vector_score"]),
                llm_confidence=float(selected["llm_confidence"]),
                candidate_rank=int(selected["candidate_rank"]),
            )
        ]

    def _resolve_entity_ids(self, entity: str) -> list[str]:
        normalized = normalize_label(entity).lower()
        if not normalized:
            return []
        exact = self._entity_lookup.get(normalized, [])
        return list(exact) if exact else []

    def _anchor_entity_vector_candidates(self, entity: str) -> list[dict[str, Any]]:
        matches = self._query_entity_store(entity, _ANCHOR_ENTITY_TOP_K)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, match in enumerate(matches, start=1):
            entity_id = self._resolve_entity_id_from_vector_match(match)
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            candidates.append(
                {
                    "entity_id": entity_id,
                    "label": normalize_label(entity_id),
                    "vector_score": float(match.score),
                    "candidate_rank": rank,
                    "source_label": normalize_label(match.label),
                    "source_item_id": match.item_id,
                }
            )
        return candidates

    def _query_entity_store(self, entity: str, top_k: int) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return []
        store = getattr(self.dataset, "entity_store", None)
        if store is None or not hasattr(store, "query"):
            return []
        try:
            vectors = self.embedder.embed_texts([entity], stage="atomic_anchor_entity_retrieval")
            if not vectors:
                return []
            return list(store.query(vectors[0], top_k=top_k))
        except (TypeError, ValueError):
            return []

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
            normalized = normalize_label(candidate).lower()
            mapped = self._entity_lookup.get(normalized, [])
            if mapped:
                return mapped[0]
        return None

    def _select_anchor_entity_with_llm(
        self,
        question: str,
        entity: str,
        analysis: AtomicQuestionAnalysis,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if self.llm_service is None or not hasattr(self.llm_service, "select_anchor_entity"):
            return None
        try:
            response = self.llm_service.select_anchor_entity(question, entity, analysis, candidates)
        except Exception as exc:  # pragma: no cover - defensive guard for external LLM failures
            self.logger.warning("Anchor entity selector failed for %s: %s", short_text(entity, 80), exc)
            return None

        if not isinstance(response, dict):
            return None
        selected_entity_id = str(response.get("selected_entity_id", "NONE")).strip()
        if not selected_entity_id or selected_entity_id.upper() == "NONE":
            return None

        try:
            confidence = float(response.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if confidence < _ANCHOR_ENTITY_LLM_MIN_CONFIDENCE:
            return None

        candidates_by_id = {str(candidate["entity_id"]): candidate for candidate in candidates}
        selected = candidates_by_id.get(selected_entity_id)
        if selected is None:
            return None

        return {
            "entity_id": selected["entity_id"],
            "vector_score": float(selected["vector_score"]),
            "llm_confidence": confidence,
            "candidate_rank": int(selected["candidate_rank"]),
        }

    @staticmethod
    def _normalized_lookup(values: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for value in values:
            normalized = normalize_label(value).lower()
            if normalized:
                lookup[normalized].append(value)
        return lookup


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result
