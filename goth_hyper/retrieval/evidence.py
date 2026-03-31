from __future__ import annotations

import logging
from typing import Any

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..models import EvidenceItem, ThoughtState


class EvidenceRetriever:
    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
        logger: logging.Logger,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self.logger = logger

    def retrieve(self, thought: ThoughtState) -> list[EvidenceItem]:
        query_texts = [thought.content]
        for anchor in thought.grounding.anchor_texts[:2]:
            if anchor and anchor not in query_texts:
                query_texts.append(anchor)
        query_vectors = self.embedder.embed_texts(query_texts, stage="evidence_retrieval")

        scored_chunks: dict[str, float] = {}
        chunk_support: dict[str, set[str]] = {}
        matched_node_ids: set[str] = set()

        for query_text, query_vector in zip(query_texts, query_vectors, strict=True):
            chunk_matches = self.dataset.chunk_store.query(query_vector, top_k=self.config.chunk_top_k)
            entity_matches = self.dataset.entity_store.query(query_vector, top_k=self.config.entity_top_k)
            hyperedge_matches = self.dataset.hyperedge_store.query(query_vector, top_k=self.config.hyperedge_top_k)

            for match in chunk_matches:
                scored_chunks[match.item_id] = max(scored_chunks.get(match.item_id, float("-inf")), match.score)
                chunk_support.setdefault(match.item_id, set()).add(f"chunk:{query_text}")

            for match in entity_matches + hyperedge_matches:
                node_id = match.label
                graph_node = self.dataset.graph.nodes.get(node_id)
                if graph_node is None:
                    continue
                matched_node_ids.add(node_id)
                bonus = 0.06 if graph_node.role == "entity" else 0.08
                for chunk_id in graph_node.source_ids:
                    if chunk_id not in self.dataset.chunk_store.id_to_index:
                        continue
                    scored_chunks[chunk_id] = max(scored_chunks.get(chunk_id, float("-inf")), match.score + bonus)
                    chunk_support.setdefault(chunk_id, set()).add(f"{graph_node.role}:{query_text}")

        for chunk_id in thought.grounding.chunk_ids:
            if chunk_id in self.dataset.chunk_store.id_to_index:
                scored_chunks[chunk_id] = max(scored_chunks.get(chunk_id, float("-inf")), 0.05)
                chunk_support.setdefault(chunk_id, set()).add("grounding:existing-chunk")

        ranked_ids = sorted(scored_chunks, key=lambda chunk_id: scored_chunks[chunk_id], reverse=True)[: self.config.evidence_keep]
        evidence_items: list[EvidenceItem] = []
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            content = self.dataset.get_chunk_text(chunk_id)
            if not content:
                continue
            source_node_ids = [
                node_id
                for node_id in self.dataset.graph.source_to_nodes.get(chunk_id, [])
                if node_id in matched_node_ids
            ]
            if not source_node_ids:
                source_node_ids = self.dataset.graph.source_to_nodes.get(chunk_id, [])[:8]
            source_edge_ids = self.dataset.graph.source_to_edges.get(chunk_id, [])[:12]
            evidence_items.append(
                EvidenceItem(
                    evidence_id=f"ev-{thought.thought_id}-{rank}",
                    chunk_id=chunk_id,
                    content=content,
                    score=float(scored_chunks[chunk_id]),
                    source_node_ids=source_node_ids,
                    source_edge_ids=source_edge_ids,
                    notes=[
                        "global-hypergraph-retrieval",
                        f"retrieved-for:{thought.thought_id}",
                        *sorted(chunk_support.get(chunk_id, set())),
                    ],
                )
            )
        self.logger.info("Retrieved %s evidence items for thought %s", len(evidence_items), thought.thought_id)
        return evidence_items
