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
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.hyperedge_store = hyperedge_store
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
            candidate.anchor_score = self.anchor_score(analysis.entities, candidate)
            scored_items.append((candidate, group_hits))

        self._score_relation_and_semantic_candidates(question, analysis, scored_items)

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

        candidates.sort(key=lambda item: (len(item.branch_support), item.fusion_score), reverse=True)
        limit = top_k
        if limit is None and self.config is not None:
            limit = int(getattr(self.config, "evidence_top_k", 5))
        if limit is None:
            limit = 5
        return candidates[:limit]

    def anchor_score(self, entities: list[str], candidate: FusedHyperedgeCandidate) -> float:
        query_entities = [normalize_label(entity).lower() for entity in entities if normalize_label(entity)]
        if not query_entities:
            return 0.0
        hyperedge_entities = {normalize_label(entity_id).lower() for entity_id in candidate.entity_ids}
        hyperedge_text = normalize_label(candidate.hyperedge_text).lower()
        matched = 0
        for entity in query_entities:
            if entity in hyperedge_entities or entity in hyperedge_text:
                matched += 1
        return matched / max(len(query_entities), 1)

    def relation_score(
        self,
        analysis: AtomicQuestionAnalysis,
        candidate: FusedHyperedgeCandidate,
        hits: list[BranchHit],
    ) -> float:
        raw_relation = max((hit.raw_score for hit in hits if hit.branch == "relation"), default=0.0)
        query_texts = [analysis.relation_query, *analysis.relations]
        query_texts = [text for text in query_texts if str(text).strip()]
        candidate_texts = self._relation_texts(candidate, hits)
        if not query_texts:
            return _bound(raw_relation)
        if not candidate_texts:
            candidate_texts = [candidate.hyperedge_text]
        computed = self._max_text_similarity(query_texts, candidate_texts, stage="atomic_relation_similarity")
        return _bound(max(raw_relation, computed))

    def semantic_score(
        self,
        question: str,
        candidate: FusedHyperedgeCandidate,
        hits: list[BranchHit],
    ) -> float:
        raw_semantic = max((hit.raw_score for hit in hits if hit.branch == "semantic"), default=0.0)
        if raw_semantic:
            return _bound(raw_semantic)
        candidate_text = " ".join([candidate.hyperedge_text, *candidate.evidence_texts]).strip()
        return _bound(self._max_text_similarity([question], [candidate_text], stage="atomic_semantic_similarity"))

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

    def _score_relation_and_semantic_candidates(
        self,
        question: str,
        analysis: AtomicQuestionAnalysis,
        scored_items: list[tuple[FusedHyperedgeCandidate, list[BranchHit]]],
    ) -> None:
        relation_query = normalize_label(analysis.relation_query or " ".join(analysis.relations)).strip()
        relation_query_texts = _dedupe([relation_query, *analysis.relations])
        relation_texts_by_candidate: dict[str, list[str]] = {}
        semantic_text_by_candidate: dict[str, str] = {}

        for candidate, hits in scored_items:
            relation_texts = self._relation_texts(candidate, hits)
            if not relation_texts:
                relation_texts = [candidate.hyperedge_text]
            relation_texts_by_candidate[candidate.hyperedge_id] = relation_texts
            semantic_text_by_candidate[candidate.hyperedge_id] = " ".join(
                [candidate.hyperedge_text, *candidate.evidence_texts]
            ).strip()

        candidate_ids = [candidate.hyperedge_id for candidate, _ in scored_items]
        needs_relation_vector = bool(relation_query) and any(
            _branch_raw_score(hits, "relation") is None for _, hits in scored_items
        )
        needs_semantic_vector = bool(question.strip()) and any(
            _branch_raw_score(hits, "semantic") is None for _, hits in scored_items
        )
        relation_vector_scores = self._vector_store_scores(
            query=relation_query,
            candidate_ids=candidate_ids,
            stage="atomic_relation_similarity",
            enabled=needs_relation_vector,
        )
        semantic_vector_scores = self._vector_store_scores(
            query=question,
            candidate_ids=candidate_ids,
            stage="atomic_semantic_similarity",
            enabled=needs_semantic_vector,
        )

        for candidate, hits in scored_items:
            raw_relation = _branch_raw_score(hits, "relation")
            if raw_relation is not None:
                candidate.relation_score = _bound(raw_relation)
            elif relation_query_texts:
                relation_texts = relation_texts_by_candidate.get(candidate.hyperedge_id, [])
                lexical_relation = max(
                    (
                        lexical_overlap_score([query_text], candidate_text)
                        for query_text in relation_query_texts
                        for candidate_text in relation_texts
                    ),
                    default=0.0,
                )
                vector_relation = relation_vector_scores.get(candidate.hyperedge_id, 0.0)
                candidate.relation_score = _bound(max(lexical_relation, vector_relation))
            else:
                candidate.relation_score = 0.0

            raw_semantic = _branch_raw_score(hits, "semantic")
            if raw_semantic is not None:
                candidate.semantic_score = _bound(raw_semantic)
            else:
                semantic_text = semantic_text_by_candidate.get(candidate.hyperedge_id, "")
                lexical_semantic = lexical_overlap_score([question], semantic_text)
                vector_semantic = semantic_vector_scores.get(candidate.hyperedge_id, 0.0)
                candidate.semantic_score = _bound(max(lexical_semantic, vector_semantic))

    def _vector_store_scores(
        self,
        query: str,
        candidate_ids: list[str],
        stage: str,
        *,
        enabled: bool,
    ) -> dict[str, float]:
        """Embed the query online once and score candidates with stored vectors."""
        if not enabled or not query.strip():
            return {}
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return {}
        if self.hyperedge_store is None or not hasattr(self.hyperedge_store, "similarities"):
            return {}
        try:
            query_vectors = self.embedder.embed_texts([query], stage=stage)
            if not query_vectors:
                return {}
            query_vector = np.asarray(query_vectors[0], dtype=np.float32)
            return dict(self.hyperedge_store.similarities(query_vector, _dedupe(candidate_ids)))
        except (TypeError, ValueError):
            return {}


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


def _branch_raw_score(hits: list[BranchHit], branch: str) -> float | None:
    scores = [float(hit.raw_score) for hit in hits if hit.branch == branch]
    if not scores:
        return None
    return max(scores)


def _bound(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
