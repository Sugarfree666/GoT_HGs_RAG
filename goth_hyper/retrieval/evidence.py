from __future__ import annotations

import logging
from typing import Any

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..models import EvidenceItem, ExtractedSubgraph, ThoughtState


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

    def retrieve(self, thought: ThoughtState, subgraph: ExtractedSubgraph) -> list[EvidenceItem]:
        query_vector = self.embedder.embed_texts([thought.content], stage="evidence_retrieval")[0]
        local_ids = {chunk_id for chunk_id in subgraph.source_chunk_ids if chunk_id in self.dataset.chunk_store.id_to_index}
        local_matches = self.dataset.chunk_store.query(query_vector, top_k=self.config.evidence_top_k_local, allowed_ids=local_ids)
        global_matches = self.dataset.chunk_store.query(query_vector, top_k=self.config.evidence_top_k_global)

        scored_chunks: dict[str, float] = {}
        for match in global_matches:
            scored_chunks[match.item_id] = max(scored_chunks.get(match.item_id, float("-inf")), match.score)
        for match in local_matches:
            scored_chunks[match.item_id] = max(scored_chunks.get(match.item_id, float("-inf")), match.score + 0.08)
        for chunk_id in thought.grounding.chunk_ids:
            if chunk_id in self.dataset.chunk_store.id_to_index:
                scored_chunks[chunk_id] = max(scored_chunks.get(chunk_id, float("-inf")), 0.05)

        ranked_ids = sorted(scored_chunks, key=lambda chunk_id: scored_chunks[chunk_id], reverse=True)[: self.config.evidence_keep]
        evidence_items: list[EvidenceItem] = []
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            content = self.dataset.get_chunk_text(chunk_id)
            if not content:
                continue
            evidence_items.append(
                EvidenceItem(
                    evidence_id=f"ev-{thought.thought_id}-{rank}",
                    chunk_id=chunk_id,
                    content=content,
                    score=float(scored_chunks[chunk_id]),
                    source_node_ids=[
                        node_id
                        for node_id in self.dataset.graph.source_to_nodes.get(chunk_id, [])
                        if node_id in subgraph.node_ids
                    ],
                    source_edge_ids=[
                        edge_id
                        for edge_id in self.dataset.graph.source_to_edges.get(chunk_id, [])
                        if edge_id in subgraph.edge_ids
                    ],
                    notes=[
                        "local-subgraph" if chunk_id in local_ids else "global-retrieval",
                        f"retrieved-for:{thought.thought_id}",
                    ],
                )
            )
        self.logger.info("Retrieved %s evidence items for thought %s", len(evidence_items), thought.thought_id)
        return evidence_items
