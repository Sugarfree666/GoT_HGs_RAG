from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..llm.service import AtomicLLMService
from ..models import VectorMatch
from ..utils import lexical_overlap_score, normalize_label, short_text
from .models import AtomicQuestionAnalysis, BranchHit


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
        self._hyperedge_ids = [
            node_id for node_id, node in self.dataset.graph.nodes.items() if getattr(node, "role", "") == "hyperedge"
        ]
        self._entity_lookup = self._normalized_lookup(self._entity_ids)
        self._hyperedge_lookup = self._normalized_lookup(self._hyperedge_ids)
        self._chunk_to_hyperedge_ids = self._build_chunk_to_hyperedge_index()
        self._chunk_ids = self._collect_chunk_ids()
        self._chunk_lookup = self._normalized_lookup(self._chunk_ids)

    def retrieve(self, question: str, analysis: AtomicQuestionAnalysis) -> list[BranchHit]:
        hits = [
            *self.retrieve_anchor_branch(question, analysis),
            *self.retrieve_relation_branch(analysis),
            *self.retrieve_semantic_branch(question),
        ]
        self.logger.info(
            "Atomic retrieval returned %s hits for question=%s",
            len(hits),
            short_text(question, 90),
        )
        return hits

    def retrieve_anchor_branch(self, question: str, analysis: AtomicQuestionAnalysis) -> list[BranchHit]:
        if not analysis.entities:
            return []

        query_entities = [
            (index, normalize_label(str(entity)))
            for index, entity in enumerate(analysis.entities)
            if normalize_label(str(entity))
        ]
        if not query_entities:
            return []

        hyperedge_scores: dict[str, dict[int, float]] = defaultdict(dict)
        hyperedge_matches: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        max_per_entity = getattr(self.config, "max_anchor_hyperedges_per_entity", None)

        for query_index, entity in query_entities:
            matches = self._resolve_anchor_entity_matches(
                question=question,
                entity=entity,
                analysis=analysis,
                query_index=query_index,
            )
            for match in matches:
                hyperedge_ids = self.dataset.graph.entity_hyperedge_ids(match.entity_id)
                if isinstance(max_per_entity, int) and max_per_entity > 0:
                    hyperedge_ids = hyperedge_ids[:max_per_entity]
                for hyperedge_id in hyperedge_ids:
                    current_score = hyperedge_scores[hyperedge_id].get(query_index)
                    if current_score is None or match.link_score > current_score:
                        hyperedge_scores[hyperedge_id][query_index] = match.link_score
                        hyperedge_matches[hyperedge_id][query_index] = match.to_metadata()

        denominator = len(query_entities)
        hits: list[BranchHit] = []
        for hyperedge_id, scores_by_query in hyperedge_scores.items():
            raw_score = sum(scores_by_query.values()) / max(denominator, 1)
            anchor_matches = [
                hyperedge_matches[hyperedge_id][query_index]
                for query_index in sorted(hyperedge_matches[hyperedge_id])
            ]
            hits.append(
                self._hit_from_hyperedge(
                    hyperedge_id=hyperedge_id,
                    branch="anchor",
                    raw_score=raw_score,
                    extra_metadata={"anchor_matches": anchor_matches},
                )
            )
        return hits

    def retrieve_relation_branch(self, analysis: AtomicQuestionAnalysis) -> list[BranchHit]:
        query = analysis.relation_query or " ".join(analysis.relations)
        top_k = int(getattr(self.config, "relation_top_k", 10))
        return self._vector_or_lexical_hyperedge_hits(query=query, branch="relation", top_k=top_k)

    def retrieve_semantic_branch(self, question: str) -> list[BranchHit]:
        query = str(question or "").strip()
        top_k = int(getattr(self.config, "semantic_chunk_top_k", getattr(self.config, "semantic_top_k", 10)))
        if not query or top_k <= 0:
            return []

        matches = self._query_chunk_store(query, top_k)
        hits_by_hyperedge: dict[str, dict[str, Any]] = {}
        max_per_chunk = getattr(self.config, "max_semantic_hyperedges_per_chunk", 20)
        for match in matches:
            chunk_id = self._resolve_chunk_id_from_vector_match(match)
            if not chunk_id:
                continue
            hyperedge_ids = list(self._chunk_to_hyperedge_ids.get(chunk_id, []))
            if isinstance(max_per_chunk, int) and max_per_chunk > 0:
                hyperedge_ids = hyperedge_ids[:max_per_chunk]
            chunk_score = float(match.score)
            chunk_text = self.dataset.get_chunk_text(chunk_id)
            for hyperedge_id in hyperedge_ids:
                payload = hits_by_hyperedge.setdefault(
                    hyperedge_id,
                    {
                        "raw_score": chunk_score,
                        "matched_chunk_ids": [],
                        "matched_chunk_scores": {},
                        "matched_chunk_texts": [],
                    },
                )
                payload["raw_score"] = max(float(payload["raw_score"]), chunk_score)
                if chunk_id not in payload["matched_chunk_ids"]:
                    payload["matched_chunk_ids"].append(chunk_id)
                scores = payload["matched_chunk_scores"]
                scores[chunk_id] = max(float(scores.get(chunk_id, 0.0)), chunk_score)
                if chunk_text and chunk_text not in payload["matched_chunk_texts"]:
                    payload["matched_chunk_texts"].append(chunk_text)

        hits: list[BranchHit] = []
        for hyperedge_id, payload in hits_by_hyperedge.items():
            matched_chunk_ids = [str(item) for item in payload["matched_chunk_ids"]]
            matched_chunk_texts = [str(item) for item in payload["matched_chunk_texts"]]
            description = self.dataset.graph.describe_hyperedge(hyperedge_id)
            default_chunk_ids = [str(item) for item in description.get("chunk_ids", [])]
            default_hyperedge_text = normalize_label(str(description.get("hyperedge_text") or hyperedge_id))
            default_evidence = self._evidence_texts(default_hyperedge_text, default_chunk_ids)
            evidence_texts = self._dedupe([*matched_chunk_texts, *default_evidence])
            chunk_ids = self._dedupe([*matched_chunk_ids, *default_chunk_ids])
            hits.append(
                self._hit_from_hyperedge(
                    hyperedge_id=hyperedge_id,
                    branch="semantic",
                    raw_score=float(payload["raw_score"]),
                    extra_metadata={
                        "semantic_source": "chunk_store",
                        "matched_chunk_ids": matched_chunk_ids,
                        "matched_chunk_scores": dict(payload["matched_chunk_scores"]),
                        "matched_chunk_texts": matched_chunk_texts,
                        "evidence_texts": evidence_texts,
                    },
                    chunk_ids_override=chunk_ids,
                )
            )
        return hits

    def _anchor_hyperedges_for_entity(self, entity: str) -> list[str]:
        entity_ids = self._resolve_entity_ids(entity)
        seen: set[str] = set()
        hyperedge_ids: list[str] = []
        for entity_id in entity_ids:
            for hyperedge_id in self.dataset.graph.entity_hyperedge_ids(entity_id):
                if hyperedge_id not in seen:
                    seen.add(hyperedge_id)
                    hyperedge_ids.append(hyperedge_id)
        return hyperedge_ids

    def _resolve_entity_ids(self, entity: str) -> list[str]:
        normalized = normalize_label(entity).lower()
        if not normalized:
            return []

        exact = self._entity_lookup.get(normalized, [])
        if exact:
            return list(exact)

        return []

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

    def _anchor_entity_vector_candidates(self, entity: str) -> list[dict[str, Any]]:
        matches = self._query_entity_store(entity, int(getattr(self.config, "anchor_entity_top_k", 3)))
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
        min_confidence = float(getattr(self.config, "anchor_entity_llm_min_confidence", 0.6))
        if confidence < min_confidence:
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

    def _vector_or_lexical_hyperedge_hits(self, query: str, branch: str, top_k: int) -> list[BranchHit]:
        query = str(query or "").strip()
        if not query or top_k <= 0:
            return []

        matches = self._query_hyperedge_store(query, top_k)
        if not matches:
            matches = self._lexical_hyperedge_matches(query, top_k)

        hits: list[BranchHit] = []
        seen: set[str] = set()
        for match in matches[:top_k]:
            hyperedge_id = self._resolve_hyperedge_id(match)
            if not hyperedge_id or hyperedge_id in seen:
                continue
            seen.add(hyperedge_id)
            hit = self._hit_from_hyperedge(
                hyperedge_id=hyperedge_id,
                branch=branch,
                raw_score=float(match.score),
                match_metadata=dict(match.metadata),
            )
            hits.append(hit)
        return hits

    def _query_hyperedge_store(self, query: str, top_k: int) -> list[VectorMatch]:
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return []
        store = getattr(self.dataset, "hyperedge_store", None)
        if store is None or not hasattr(store, "query"):
            return []
        vectors = self.embedder.embed_texts([query], stage="atomic_hyperedge_retrieval")
        if not vectors:
            return []
        return list(store.query(vectors[0], top_k=top_k))

    def _query_chunk_store(self, query: str, top_k: int) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return []
        store = getattr(self.dataset, "chunk_store", None)
        if store is None or not hasattr(store, "query"):
            return []
        try:
            vectors = self.embedder.embed_texts([query], stage="atomic_chunk_semantic_retrieval")
            if not vectors:
                return []
            return list(store.query(vectors[0], top_k=top_k))
        except (TypeError, ValueError):
            return []

    def _resolve_chunk_id_from_vector_match(self, match: VectorMatch) -> str | None:
        metadata = match.metadata if isinstance(match.metadata, dict) else {}
        raw_candidates = [
            match.item_id,
            match.label,
            metadata.get("__id__"),
            metadata.get("chunk_id"),
            metadata.get("id"),
        ]
        for raw_candidate in raw_candidates:
            if raw_candidate is None:
                continue
            candidate = str(raw_candidate).strip()
            if not candidate:
                continue
            if candidate in self._chunk_ids:
                return candidate
            normalized = normalize_label(candidate).lower()
            mapped = self._chunk_lookup.get(normalized, [])
            if mapped:
                return mapped[0]
        return None

    def _lexical_hyperedge_matches(self, query: str, top_k: int) -> list[VectorMatch]:
        scored: list[VectorMatch] = []
        for hyperedge_id in self._hyperedge_ids:
            score = lexical_overlap_score([query], normalize_label(hyperedge_id))
            if score <= 0:
                continue
            scored.append(
                VectorMatch(
                    item_id=hyperedge_id,
                    label=hyperedge_id,
                    score=score,
                    metadata={"source": "lexical_fallback"},
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _resolve_hyperedge_id(self, match: VectorMatch) -> str:
        for candidate in (match.label, match.item_id):
            if candidate in self.dataset.graph.nodes:
                return candidate
            normalized = normalize_label(candidate).lower()
            mapped = self._hyperedge_lookup.get(normalized, [])
            if mapped:
                return mapped[0]
        return match.label or match.item_id

    def _hit_from_hyperedge(
        self,
        hyperedge_id: str,
        branch: str,
        raw_score: float,
        match_metadata: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        chunk_ids_override: list[str] | None = None,
    ) -> BranchHit:
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        hyperedge_text = normalize_label(str(description.get("hyperedge_text") or hyperedge_id))
        entity_ids = [str(item) for item in description.get("entity_ids", [])]
        chunk_ids = list(chunk_ids_override) if chunk_ids_override is not None else [str(item) for item in description.get("chunk_ids", [])]
        evidence_texts = self._evidence_texts(hyperedge_text, chunk_ids)
        metadata = {
            "evidence_texts": evidence_texts,
            "relation_texts": self._relation_texts_for_hyperedge(hyperedge_id, hyperedge_text),
        }
        if match_metadata:
            metadata["vector_metadata"] = match_metadata
        if extra_metadata:
            metadata.update(extra_metadata)
        return BranchHit(
            hyperedge_id=hyperedge_id,
            branch=branch,  # type: ignore[arg-type]
            raw_score=float(raw_score),
            hyperedge_text=hyperedge_text,
            entity_ids=entity_ids,
            chunk_ids=chunk_ids,
            metadata=metadata,
        )

    def _evidence_texts(self, hyperedge_text: str, chunk_ids: list[str]) -> list[str]:
        texts = [hyperedge_text] if hyperedge_text else []
        for chunk_id in chunk_ids[:3]:
            chunk_text = self.dataset.get_chunk_text(chunk_id)
            if chunk_text:
                texts.append(short_text(chunk_text, 900))
        return self._dedupe(texts)

    def _relation_texts_for_hyperedge(self, hyperedge_id: str, hyperedge_text: str) -> list[str]:
        texts = [hyperedge_text]
        node = self.dataset.graph.nodes.get(hyperedge_id)
        description = getattr(node, "description", None)
        if description:
            texts.append(str(description))

        adjacency = getattr(self.dataset.graph, "adjacency", {})
        edges = getattr(self.dataset.graph, "edges", {})
        for edge_id in adjacency.get(hyperedge_id, []):
            edge = edges.get(edge_id)
            if edge is None:
                continue
            role = normalize_label(str(getattr(edge, "role", "") or ""))
            if role:
                texts.append(role)
            other_id = getattr(edge, "target", "") if getattr(edge, "source", "") == hyperedge_id else getattr(edge, "source", "")
            if other_id:
                texts.append(normalize_label(str(other_id)))
        return self._dedupe(texts)

    @staticmethod
    def _normalized_lookup(values: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for value in values:
            normalized = normalize_label(value).lower()
            if normalized:
                lookup[normalized].append(value)
        return lookup

    def _build_chunk_to_hyperedge_index(self) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for hyperedge_id in self._hyperedge_ids:
            chunk_ids = self._hyperedge_chunk_ids(hyperedge_id)
            for chunk_id in chunk_ids:
                if hyperedge_id not in lookup[chunk_id]:
                    lookup[chunk_id].append(hyperedge_id)
        return dict(lookup)

    def _collect_chunk_ids(self) -> list[str]:
        seen: set[str] = set()
        chunk_ids: list[str] = []
        text_chunks = getattr(self.dataset, "text_chunks", {})
        if isinstance(text_chunks, dict):
            for chunk_id in text_chunks:
                text = str(chunk_id)
                if text and text not in seen:
                    seen.add(text)
                    chunk_ids.append(text)
        for values in self._chunk_to_hyperedge_ids:
            if values and values not in seen:
                seen.add(values)
                chunk_ids.append(values)
        return chunk_ids

    def _hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        graph = self.dataset.graph
        if hasattr(graph, "hyperedge_chunk_ids"):
            return [str(item) for item in graph.hyperedge_chunk_ids(hyperedge_id)]
        description = graph.describe_hyperedge(hyperedge_id)
        return [str(item) for item in description.get("chunk_ids", [])]

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            text = normalize_label(str(value).strip())
            if text and text not in deduped:
                deduped.append(text)
        return deduped
