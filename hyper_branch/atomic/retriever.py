from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..models import VectorMatch
from ..utils import lexical_overlap_score, normalize_label, short_text
from .models import AtomicQuestionAnalysis, BranchHit


class AtomicHyperedgeRetriever:
    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._entity_ids = [
            node_id for node_id, node in self.dataset.graph.nodes.items() if getattr(node, "role", "") == "entity"
        ]
        self._hyperedge_ids = [
            node_id for node_id, node in self.dataset.graph.nodes.items() if getattr(node, "role", "") == "hyperedge"
        ]
        self._entity_lookup = self._normalized_lookup(self._entity_ids)
        self._hyperedge_lookup = self._normalized_lookup(self._hyperedge_ids)

    def retrieve(self, question: str, analysis: AtomicQuestionAnalysis) -> list[BranchHit]:
        hits = [
            *self.retrieve_anchor_branch(analysis),
            *self.retrieve_relation_branch(analysis),
            *self.retrieve_semantic_branch(question),
        ]
        self.logger.info(
            "Atomic retrieval returned %s hits for question=%s",
            len(hits),
            short_text(question, 90),
        )
        return hits

    def retrieve_anchor_branch(self, analysis: AtomicQuestionAnalysis) -> list[BranchHit]:
        if not analysis.entities:
            return []

        hits: dict[str, BranchHit] = {}
        max_per_entity = getattr(self.config, "max_anchor_hyperedges_per_entity", None)
        for entity in analysis.entities:
            hyperedge_ids = self._anchor_hyperedges_for_entity(entity)
            if isinstance(max_per_entity, int) and max_per_entity > 0:
                hyperedge_ids = hyperedge_ids[:max_per_entity]
            for hyperedge_id in hyperedge_ids:
                hit = self._hit_from_hyperedge(
                    hyperedge_id=hyperedge_id,
                    branch="anchor",
                    raw_score=self._anchor_raw_score(analysis.entities, hyperedge_id),
                )
                hits[hyperedge_id] = hit
        return list(hits.values())

    def retrieve_relation_branch(self, analysis: AtomicQuestionAnalysis) -> list[BranchHit]:
        query = analysis.relation_query or " ".join(analysis.relations)
        top_k = int(getattr(self.config, "relation_top_k", 10))
        return self._vector_or_lexical_hyperedge_hits(query=query, branch="relation", top_k=top_k)

    def retrieve_semantic_branch(self, question: str) -> list[BranchHit]:
        top_k = int(getattr(self.config, "semantic_top_k", 10))
        return self._vector_or_lexical_hyperedge_hits(query=question, branch="semantic", top_k=top_k)

    def _anchor_hyperedges_for_entity(self, entity: str) -> list[str]:
        entity_ids = self._resolve_entity_ids(entity)
        seen: set[str] = set()
        hyperedge_ids: list[str] = []
        for entity_id in entity_ids:
            for hyperedge_id in self.dataset.graph.entity_hyperedge_ids(entity_id):
                if hyperedge_id not in seen:
                    seen.add(hyperedge_id)
                    hyperedge_ids.append(hyperedge_id)

        if hyperedge_ids:
            return hyperedge_ids

        normalized_entity = normalize_label(entity).lower()
        for hyperedge_id in self._hyperedge_ids:
            if normalized_entity and normalized_entity in normalize_label(hyperedge_id).lower():
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

        matches: list[str] = []
        for entity_id in self._entity_ids:
            candidate = normalize_label(entity_id).lower()
            if normalized in candidate or candidate in normalized:
                matches.append(entity_id)
        return matches

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
    ) -> BranchHit:
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        hyperedge_text = normalize_label(str(description.get("hyperedge_text") or hyperedge_id))
        entity_ids = [str(item) for item in description.get("entity_ids", [])]
        chunk_ids = [str(item) for item in description.get("chunk_ids", [])]
        evidence_texts = self._evidence_texts(hyperedge_text, chunk_ids)
        metadata = {
            "evidence_texts": evidence_texts,
            "relation_texts": self._relation_texts_for_hyperedge(hyperedge_id, hyperedge_text),
        }
        if match_metadata:
            metadata["vector_metadata"] = match_metadata
        return BranchHit(
            hyperedge_id=hyperedge_id,
            branch=branch,  # type: ignore[arg-type]
            raw_score=float(raw_score),
            hyperedge_text=hyperedge_text,
            entity_ids=entity_ids,
            chunk_ids=chunk_ids,
            metadata=metadata,
        )

    def _anchor_raw_score(self, entities: list[str], hyperedge_id: str) -> float:
        normalized_entities = [normalize_label(entity).lower() for entity in entities if normalize_label(entity)]
        if not normalized_entities:
            return 0.0
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        entity_ids = [normalize_label(str(item)).lower() for item in description.get("entity_ids", [])]
        hyperedge_text = normalize_label(str(description.get("hyperedge_text") or hyperedge_id)).lower()
        matched = 0
        for entity in normalized_entities:
            if entity in entity_ids or entity in hyperedge_text:
                matched += 1
        return matched / max(len(normalized_entities), 1)

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

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            text = normalize_label(str(value).strip())
            if text and text not in deduped:
                deduped.append(text)
        return deduped
