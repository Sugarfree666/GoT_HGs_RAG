from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..utils import lexical_overlap_score, normalize_label
from .models import AtomicQuestionAnalysis, BranchHit, FusedHyperedgeCandidate


DEFAULT_FUSION_WEIGHTS = {"anchor": 0.4, "relation": 0.4, "semantic": 0.2}


class AtomicEvidenceFusion:
    def __init__(
        self,
        config: RetrievalConfig | None = None,
        embedder: Any | None = None,
        hyperedge_store: Any | None = None,
        chunk_store: Any | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.hyperedge_store = hyperedge_store
        self.chunk_store = chunk_store
        self.weights = dict(DEFAULT_FUSION_WEIGHTS)
        if config is not None:
            self.weights.update(
                {
                    "anchor": float(getattr(config, "anchor_weight", self.weights["anchor"])),
                    "relation": float(getattr(config, "relation_weight", self.weights["relation"])),
                    "semantic": float(getattr(config, "semantic_weight", self.weights["semantic"])),
                }
            )

    def fuse(
        self,
        question: str,
        analysis: AtomicQuestionAnalysis,
        hits: list[BranchHit],
        top_k: int | None = None,
    ) -> list[FusedHyperedgeCandidate]:
        grouped = self._group_hits(hits)
        scored_items: list[tuple[FusedHyperedgeCandidate, list[BranchHit]]] = []
        for hyperedge_id, group_hits in grouped.items():
            candidate = self._candidate_from_hits(hyperedge_id, group_hits)
            scored_items.append((candidate, group_hits))

        self._score_candidates(question, analysis, scored_items)

        candidates: list[FusedHyperedgeCandidate] = []
        for candidate, _ in scored_items:
            candidate.fusion_score = (
                (self.weights["anchor"] * candidate.anchor_score)
                + (self.weights["relation"] * candidate.relation_score)
                + (self.weights["semantic"] * candidate.semantic_score)
            )
            candidate.score_breakdown = {
                "A": round(candidate.anchor_score, 6),
                "R": round(candidate.relation_score, 6),
                "S": round(candidate.semantic_score, 6),
                "anchor_weight": self.weights["anchor"],
                "relation_weight": self.weights["relation"],
                "semantic_weight": self.weights["semantic"],
                "fusion_score": round(candidate.fusion_score, 6),
            }
            candidates.append(candidate)

        candidates.sort(key=lambda item: (item.fusion_score, len(item.branch_support)), reverse=True)
        limit = top_k
        if limit is None and self.config is not None:
            limit = int(getattr(self.config, "evidence_top_k", 5))
        if limit is None:
            limit = 5
        return candidates[:limit]

    def anchor_score(
        self,
        entities: list[str],
        candidate: FusedHyperedgeCandidate,
        hits: list[BranchHit] | None = None,
    ) -> float:
        query_entities = [
            (index, normalize_label(str(entity)).lower())
            for index, entity in enumerate(entities)
            if normalize_label(str(entity))
        ]
        if not query_entities:
            return 0.0
        anchor_matches = self._anchor_matches_from_hits(hits or [])
        hyperedge_entities = {normalize_label(entity_id).lower() for entity_id in candidate.entity_ids}
        hyperedge_entity_ids = {str(entity_id) for entity_id in candidate.entity_ids}
        if anchor_matches:
            matched_scores: dict[int, float] = {}
            valid_query_indices = {index for index, _ in query_entities}
            for match in anchor_matches:
                query_index = self._anchor_match_query_index(match, query_entities)
                if query_index is None or query_index not in valid_query_indices:
                    continue
                matched_entity_id = str(match.get("matched_entity_id") or match.get("matched_entity") or "").strip()
                if not matched_entity_id:
                    continue
                normalized_entity_id = normalize_label(matched_entity_id).lower()
                if matched_entity_id not in hyperedge_entity_ids and normalized_entity_id not in hyperedge_entities:
                    continue
                link_score = _safe_float(match.get("link_score"), 0.0)
                matched_scores[query_index] = max(matched_scores.get(query_index, 0.0), link_score)
            return _bound(sum(matched_scores.values()) / max(len(query_entities), 1))

        matched = sum(1 for _, entity in query_entities if entity in hyperedge_entities)
        return _bound(matched / max(len(query_entities), 1))

    def relation_score(
        self,
        analysis: AtomicQuestionAnalysis,
        candidate: FusedHyperedgeCandidate,
        hits: list[BranchHit],
    ) -> float:
        query_texts = [analysis.relation_query, *analysis.relations]
        query_texts = [text for text in query_texts if str(text).strip()]
        candidate_texts = self._relation_texts(candidate, hits)
        if not query_texts:
            return 0.0
        if not candidate_texts:
            candidate_texts = [candidate.hyperedge_text]
        return _bound(self._max_text_similarity(query_texts, candidate_texts, stage="atomic_relation_candidate_scoring"))

    def semantic_score(
        self,
        question: str,
        candidate: FusedHyperedgeCandidate,
        hits: list[BranchHit],
    ) -> float:
        del hits
        return _bound(self._lexical_semantic_score(question, candidate))

    def _group_hits(self, hits: list[BranchHit]) -> dict[str, list[BranchHit]]:
        grouped: dict[str, list[BranchHit]] = defaultdict(list)
        for hit in hits:
            if hit.hyperedge_id:
                grouped[hit.hyperedge_id].append(hit)
        return dict(grouped)

    def _candidate_from_hits(self, hyperedge_id: str, hits: list[BranchHit]) -> FusedHyperedgeCandidate:
        first = hits[0]
        branch_support = {hit.branch for hit in hits}
        entity_ids: list[str] = []
        chunk_ids: list[str] = []
        evidence_texts: list[str] = []
        for hit in hits:
            _append_unique(entity_ids, hit.entity_ids)
            _append_unique(chunk_ids, hit.chunk_ids)
            matched_chunk_ids = hit.metadata.get("matched_chunk_ids", [])
            if isinstance(matched_chunk_ids, list):
                _append_unique(chunk_ids, [str(item) for item in matched_chunk_ids])
            metadata_evidence = hit.metadata.get("evidence_texts", [])
            if isinstance(metadata_evidence, list):
                _append_unique(evidence_texts, [str(item) for item in metadata_evidence])
            elif metadata_evidence:
                _append_unique(evidence_texts, [str(metadata_evidence)])
        return FusedHyperedgeCandidate(
            hyperedge_id=hyperedge_id,
            hyperedge_text=first.hyperedge_text,
            branch_support=set(branch_support),
            entity_ids=entity_ids,
            chunk_ids=chunk_ids,
            evidence_texts=evidence_texts,
        )

    def _relation_texts(self, candidate: FusedHyperedgeCandidate, hits: list[BranchHit]) -> list[str]:
        texts = [candidate.hyperedge_text]
        for hit in hits:
            relation_texts = hit.metadata.get("relation_texts", [])
            if isinstance(relation_texts, list):
                texts.extend(str(item) for item in relation_texts)
            elif relation_texts:
                texts.append(str(relation_texts))
        return _dedupe(texts)

    def _max_text_similarity(self, query_texts: list[str], candidate_texts: list[str], stage: str) -> float:
        del stage
        if not query_texts or not candidate_texts:
            return 0.0
        return max(
            lexical_overlap_score([query_text], candidate_text)
            for query_text in query_texts
            for candidate_text in candidate_texts
        )

    def _score_candidates(
        self,
        question: str,
        analysis: AtomicQuestionAnalysis,
        scored_items: list[tuple[FusedHyperedgeCandidate, list[BranchHit]]],
    ) -> None:
        relation_query = normalize_label(analysis.relation_query or " ".join(analysis.relations)).strip()
        relation_query_texts = _dedupe([relation_query, *analysis.relations])
        relation_texts_by_candidate: dict[str, list[str]] = {}

        for candidate, hits in scored_items:
            relation_texts = self._relation_texts(candidate, hits)
            if not relation_texts:
                relation_texts = [candidate.hyperedge_text]
            relation_texts_by_candidate[candidate.hyperedge_id] = relation_texts

        candidate_ids = [candidate.hyperedge_id for candidate, _ in scored_items]
        relation_vector_scores = self._hyperedge_vector_scores(
            query=relation_query,
            candidate_ids=candidate_ids,
            stage="atomic_relation_candidate_scoring",
        )
        chunk_ids = _dedupe([chunk_id for candidate, _ in scored_items for chunk_id in candidate.chunk_ids])
        chunk_vector_scores = self._chunk_vector_scores(
            query=question,
            chunk_ids=chunk_ids,
            stage="atomic_semantic_candidate_scoring",
        )

        for candidate, hits in scored_items:
            candidate.anchor_score = self.anchor_score(analysis.entities, candidate, hits)

            if not relation_query_texts:
                candidate.relation_score = 0.0
            elif relation_vector_scores is not None:
                candidate.relation_score = _bound(relation_vector_scores.get(candidate.hyperedge_id, 0.0))
            else:
                relation_texts = relation_texts_by_candidate.get(candidate.hyperedge_id, [])
                candidate.relation_score = _bound(self._lexical_relation_score(relation_query_texts, relation_texts))

            if chunk_vector_scores is not None and candidate.chunk_ids:
                candidate.semantic_score = _bound(
                    max((chunk_vector_scores.get(chunk_id, 0.0) for chunk_id in candidate.chunk_ids), default=0.0)
                )
            else:
                candidate.semantic_score = _bound(self._lexical_semantic_score(question, candidate))

    def _hyperedge_vector_scores(
        self,
        query: str,
        candidate_ids: list[str],
        stage: str,
    ) -> dict[str, float] | None:
        """Embed the query online once and score candidates with stored vectors."""
        if not query.strip():
            return {}
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return None
        if self.hyperedge_store is None or not hasattr(self.hyperedge_store, "similarities"):
            return None
        try:
            query_vectors = self.embedder.embed_texts([query], stage=stage)
            if not query_vectors:
                return None
            query_vector = np.asarray(query_vectors[0], dtype=np.float32)
            return dict(self.hyperedge_store.similarities(query_vector, _dedupe(candidate_ids)))
        except (TypeError, ValueError):
            return None

    def _chunk_vector_scores(
        self,
        query: str,
        chunk_ids: list[str],
        stage: str,
    ) -> dict[str, float] | None:
        if not query.strip():
            return {}
        if not chunk_ids:
            return None
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return None
        if self.chunk_store is None or not hasattr(self.chunk_store, "similarities"):
            return None
        try:
            query_vectors = self.embedder.embed_texts([query], stage=stage)
            if not query_vectors:
                return None
            query_vector = np.asarray(query_vectors[0], dtype=np.float32)
            return dict(self.chunk_store.similarities(query_vector, _dedupe(chunk_ids)))
        except (TypeError, ValueError):
            return None

    def _lexical_relation_score(self, query_texts: list[str], relation_texts: list[str]) -> float:
        return max(
            (
                lexical_overlap_score([query_text], candidate_text)
                for query_text in query_texts
                for candidate_text in relation_texts
            ),
            default=0.0,
        )

    def _lexical_semantic_score(self, question: str, candidate: FusedHyperedgeCandidate) -> float:
        if not question.strip():
            return 0.0
        evidence_texts = candidate.evidence_texts or [candidate.hyperedge_text]
        return max((lexical_overlap_score([question], text) for text in evidence_texts), default=0.0)

    def _anchor_matches_from_hits(self, hits: list[BranchHit]) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for hit in hits:
            raw_matches = hit.metadata.get("anchor_matches", [])
            if isinstance(raw_matches, list):
                matches.extend(item for item in raw_matches if isinstance(item, dict))
        return matches

    def _anchor_match_query_index(
        self,
        match: dict[str, Any],
        query_entities: list[tuple[int, str]],
    ) -> int | None:
        raw_index = match.get("query_index")
        try:
            query_index = int(raw_index)
        except (TypeError, ValueError):
            query_index = None
        if query_index is not None and any(index == query_index for index, _ in query_entities):
            return query_index

        query_entity = normalize_label(str(match.get("query_entity", ""))).lower()
        if not query_entity:
            return None
        for index, normalized_entity in query_entities:
            if query_entity == normalized_entity:
                return index
        return None


def _append_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        text = str(value).strip()
        if text and text not in target:
            target.append(text)


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = normalize_label(str(value).strip())
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _bound(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
