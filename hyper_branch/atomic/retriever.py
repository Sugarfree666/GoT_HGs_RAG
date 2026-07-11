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
_LOCAL_RETRIEVAL_METHOD = "two_hop_multi_anchor_topk"


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
    anchor_mentions: list[str] = field(default_factory=list)
    linked_entities: list[dict[str, Any]] = field(default_factory=list)
    anchor_matches: list[dict[str, Any]] = field(default_factory=list)
    unlinked_anchor_mentions: list[str] = field(default_factory=list)
    adjacent_hyperedge_ids: list[str] = field(default_factory=list)
    expansion_entity_ids: list[str] = field(default_factory=list)
    second_hop_hyperedge_ids: list[str] = field(default_factory=list)
    candidate_hyperedge_ids: list[str] = field(default_factory=list)
    candidate_sources: list[dict[str, Any]] = field(default_factory=list)
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
            "anchor_mentions": list(self.anchor_mentions),
            "linked_entities": [dict(item) for item in self.linked_entities],
            "anchor_matches": [dict(item) for item in self.anchor_matches],
            "unlinked_anchor_mentions": list(self.unlinked_anchor_mentions),
            "adjacent_hyperedge_ids": list(self.adjacent_hyperedge_ids),
            "expansion_entity_ids": list(self.expansion_entity_ids),
            "second_hop_hyperedge_ids": list(self.second_hop_hyperedge_ids),
            "candidate_hyperedge_ids": list(self.candidate_hyperedge_ids),
            "candidate_sources": [dict(item) for item in self.candidate_sources],
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
        anchor_mentions = self._anchor_mentions(primary_anchor_mention, analysis)
        primary_mention = anchor_mentions[0] if anchor_mentions else ""
        result = LocalHyperedgeRetrievalResult(
            primary_anchor_mention=primary_mention,
            anchor_mentions=list(anchor_mentions),
        )
        if not anchor_mentions:
            result.insufficient_reason = "missing_primary_anchor"
            return result

        linked_matches: list[AnchorEntityMatch] = []
        seen_linked_entity_ids: set[str] = set()
        for query_index, mention in enumerate(anchor_mentions):
            match = self.link_anchor_entity(
                question=question,
                mention=mention,
                analysis=analysis,
                query_index=query_index,
            )
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

        result.linked_entity_id = linked_matches[0].entity_id
        result.anchor_match = linked_matches[0].to_metadata()
        result.anchor_matches = [match.to_metadata() for match in linked_matches]
        result.linked_entities = [
            {
                "mention": match.query_entity,
                "entity_id": match.entity_id,
                "match_type": match.match_type,
                "link_score": match.link_score,
            }
            for match in linked_matches
        ]

        candidate_pool = self._multi_anchor_candidate_pool(linked_matches)
        result.adjacent_hyperedge_ids = list(candidate_pool["adjacent_hyperedge_ids"])
        if not result.adjacent_hyperedge_ids:
            result.insufficient_reason = "primary_anchor_has_no_adjacent_hyperedges"
            return result

        result.expansion_entity_ids = list(candidate_pool["expansion_entity_ids"])
        result.second_hop_hyperedge_ids = list(candidate_pool["second_hop_hyperedge_ids"])
        result.candidate_hyperedge_ids = list(candidate_pool["candidate_hyperedge_ids"])
        result.candidate_sources = [dict(item) for item in candidate_pool["candidate_sources"]]
        if not result.candidate_hyperedge_ids:
            result.insufficient_reason = "no_local_candidate_hyperedges"
            return result

        source_by_id = {
            str(item["hyperedge_id"]): item
            for item in result.candidate_sources
            if str(item.get("hyperedge_id", "")).strip()
        }
        scores = self._hyperedge_similarity_scores(question, result.candidate_hyperedge_ids)
        ranked = [
            {
                "hyperedge_id": hyperedge_id,
                "semantic_score": float(scores.get(hyperedge_id, 0.0)),
                "rank": 0,
                "hop": int(source_by_id.get(hyperedge_id, {}).get("hop", 1)),
                "via_entity_ids": list(source_by_id.get(hyperedge_id, {}).get("via_entity_ids", [])),
                "via_first_hyperedge_ids": list(source_by_id.get(hyperedge_id, {}).get("via_first_hyperedge_ids", [])),
            }
            for hyperedge_id in result.candidate_hyperedge_ids
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
                primary_anchor_mention=primary_mention,
                primary_anchor_entity_id=result.linked_entity_id,
                candidate_source=source_by_id.get(str(item["hyperedge_id"]), {}),
            )
            for item in selected
        ]
        result.evidence = [item for item in result.evidence if item.hyperedge_id]
        if not result.evidence:
            result.insufficient_reason = "no_valid_local_evidence"
        return result

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

    def link_primary_anchor(
        self,
        *,
        question: str,
        mention: str,
        analysis: AtomicQuestionAnalysis,
    ) -> AnchorEntityMatch | None:
        return self.link_anchor_entity(question=question, mention=mention, analysis=analysis, query_index=0)

    def link_anchor_entity(
        self,
        *,
        question: str,
        mention: str,
        analysis: AtomicQuestionAnalysis,
        query_index: int,
    ) -> AnchorEntityMatch | None:
        matches = self._resolve_anchor_entity_matches(
            question=question,
            entity=mention,
            analysis=analysis,
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
        if not entity_id or not hasattr(self.dataset.graph, "entity_hyperedge_ids"):
            return []
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

        max_hops = max(1, int(getattr(self.config, "local_hyperedge_hops", 2)))
        if max_hops >= 2:
            for first_hop_id in first_hop_ids:
                for entity_id in self._hyperedge_entity_ids(first_hop_id):
                    if entity_id == primary_anchor_entity_id:
                        continue
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
                            },
                        )
                        if int(source.get("hop", 2)) > 1:
                            source["hop"] = 2
                        if entity_id not in source["via_entity_ids"]:
                            source["via_entity_ids"].append(entity_id)
                        if first_hop_id not in source["via_first_hyperedge_ids"]:
                            source["via_first_hyperedge_ids"].append(first_hop_id)

        return {
            "expansion_entity_ids": expansion_entity_ids,
            "second_hop_hyperedge_ids": second_hop_ids,
            "candidate_hyperedge_ids": candidate_ids,
            "candidate_sources": [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids],
        }

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
                "anchor_mentions": [],
                "anchor_entity_ids": [],
                "anchor_query_indices": [],
            },
        )
        existing["hop"] = min(int(existing.get("hop", 1) or 1), int(source.get("hop", 1) or 1))
        for key in ("via_entity_ids", "via_first_hyperedge_ids", "anchor_mentions", "anchor_entity_ids"):
            for value in source.get(key, []):
                if value not in existing[key]:
                    existing[key].append(value)
        for value in source.get("anchor_query_indices", []):
            if value not in existing["anchor_query_indices"]:
                existing["anchor_query_indices"].append(value)

    def _hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        if hasattr(self.dataset.graph, "hyperedge_entity_ids"):
            entity_ids = self.dataset.graph.hyperedge_entity_ids(hyperedge_id)
            if entity_ids:
                return _dedupe_strings([str(item) for item in entity_ids])
        if hasattr(self.dataset.graph, "describe_hyperedge"):
            description = self.dataset.graph.describe_hyperedge(hyperedge_id)
            return _dedupe_strings([str(item) for item in description.get("entity_ids", [])])
        return []

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
        candidate_source: dict[str, Any],
    ) -> FusedHyperedgeCandidate:
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        hyperedge_text = normalize_label(str(description.get("hyperedge_text") or hyperedge_id))
        entity_ids = _dedupe_strings([str(item) for item in description.get("entity_ids", [])])
        if not entity_ids and hasattr(self.dataset.graph, "hyperedge_entity_ids"):
            entity_ids = _dedupe_strings([str(item) for item in self.dataset.graph.hyperedge_entity_ids(hyperedge_id)])
        chunk_ids = _dedupe_strings([str(item) for item in description.get("chunk_ids", [])])
        chunk_texts = [self.dataset.get_chunk_text(chunk_id) for chunk_id in chunk_ids]
        entity_records = [self._entity_payload(entity_id) for entity_id in entity_ids]
        candidate_hop = int(candidate_source.get("hop", 1) or 1)
        return FusedHyperedgeCandidate(
            hyperedge_id=hyperedge_id,
            hyperedge_text=hyperedge_text,
            branch_support={"local_primary_anchor", f"hop{candidate_hop}"},
            semantic_score=float(semantic_score),
            fusion_score=float(semantic_score),
            entity_ids=entity_ids,
            entity_records=entity_records,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            evidence_texts=[text for text in [hyperedge_text, *chunk_texts] if text],
            rank=int(rank),
            score_breakdown={
                "selection_source": _LOCAL_RETRIEVAL_METHOD,
                "semantic_rank": int(rank),
                "semantic_score": float(semantic_score),
                "primary_anchor_mention": primary_anchor_mention,
                "primary_anchor_entity_id": primary_anchor_entity_id,
                "candidate_hop": candidate_hop,
                "via_entity_ids": list(candidate_source.get("via_entity_ids", [])),
                "via_first_hyperedge_ids": list(candidate_source.get("via_first_hyperedge_ids", [])),
                "anchor_mentions": list(candidate_source.get("anchor_mentions", [])),
                "anchor_entity_ids": list(candidate_source.get("anchor_entity_ids", [])),
                "anchor_query_indices": list(candidate_source.get("anchor_query_indices", [])),
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
