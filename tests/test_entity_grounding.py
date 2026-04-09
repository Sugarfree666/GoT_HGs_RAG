from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from hyper_branch.config import RetrievalConfig
from hyper_branch.models import GraphNode, VectorMatch, TaskFrame
from hyper_branch.retrieval.evidence import EvidenceRetriever
from hyper_branch.utils import normalize_label


class TextEmbedder:
    def embed_texts(self, texts: list[str], stage: str) -> list[str]:
        del stage
        return [normalize_label(text) for text in texts]


class MappingStore:
    def __init__(self, matches_by_query: dict[str, list[VectorMatch]], label_field: str) -> None:
        self.matches_by_query = {
            normalize_label(query): list(matches)
            for query, matches in matches_by_query.items()
        }
        labels: list[str] = []
        for matches in self.matches_by_query.values():
            for match in matches:
                if match.label not in labels:
                    labels.append(match.label)
        self.rows = [{label_field: label} for label in labels]
        self.row_ids = [f"row-{index}" for index, _ in enumerate(labels)]
        self._row_id_by_label = {label: row_id for label, row_id in zip(labels, self.row_ids, strict=True)}
        self.id_to_index = {row_id: index for index, row_id in enumerate(self.row_ids)}

    def query(self, vector: str, top_k: int, allowed_ids: set[str] | None = None) -> list[VectorMatch]:
        del allowed_ids
        query = normalize_label(str(vector))
        matches = self.matches_by_query.get(query, [])
        cloned: list[VectorMatch] = []
        for match in matches[:top_k]:
            cloned.append(
                VectorMatch(
                    item_id=self._row_id_by_label.get(match.label, match.item_id),
                    label=match.label,
                    score=match.score,
                    metadata=dict(match.metadata),
                )
            )
        return cloned

    def similarity(self, query_vector: str, row_id: str) -> float:
        del query_vector, row_id
        return 0.0

    def _label_for_row(self, row: dict[str, object], fallback: str) -> str:
        return str(row.get("entity_name") or row.get("hyperedge_name") or fallback)


class GroundingGraph:
    def __init__(self) -> None:
        self.distribution_hyperedge = '<hyperedge>"western North Carolina supports local food distribution."'
        self.asheville_hyperedge = '<hyperedge>"Asheville participates in the local food network."'
        self.nodes = {
            '"ROBUST DISTRIBUTION NETWORK"': GraphNode(
                node_id='"ROBUST DISTRIBUTION NETWORK"',
                role="entity",
                source_ids=["chunk-1"],
            ),
            '"REGION"': GraphNode(
                node_id='"REGION"',
                role="entity",
                source_ids=["chunk-2"],
            ),
            '"WESTERN NORTH CAROLINA"': GraphNode(
                node_id='"WESTERN NORTH CAROLINA"',
                role="entity",
                source_ids=["chunk-1"],
            ),
            '"ASHEVILLE"': GraphNode(
                node_id='"ASHEVILLE"',
                role="entity",
                source_ids=["chunk-3"],
            ),
            self.distribution_hyperedge: GraphNode(
                node_id=self.distribution_hyperedge,
                role="hyperedge",
                source_ids=["chunk-1"],
            ),
            self.asheville_hyperedge: GraphNode(
                node_id=self.asheville_hyperedge,
                role="hyperedge",
                source_ids=["chunk-3"],
            ),
        }
        self.adjacency = {
            self.distribution_hyperedge: [],
            self.asheville_hyperedge: [],
        }

    def expand_from_entities(self, entity_ids: list[str]) -> list[str]:
        expanded: list[str] = []
        for entity_id in entity_ids:
            if entity_id in {'"ROBUST DISTRIBUTION NETWORK"', '"WESTERN NORTH CAROLINA"'}:
                expanded.append(self.distribution_hyperedge)
            if entity_id == '"ASHEVILLE"':
                expanded.append(self.asheville_hyperedge)
        return expanded

    def hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        mapping = {
            self.distribution_hyperedge: ['"WESTERN NORTH CAROLINA"', '"ROBUST DISTRIBUTION NETWORK"'],
            self.asheville_hyperedge: ['"ASHEVILLE"'],
        }
        return mapping.get(hyperedge_id, [])

    def hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        mapping = {
            self.distribution_hyperedge: ["chunk-1"],
            self.asheville_hyperedge: ["chunk-3"],
        }
        return mapping.get(hyperedge_id, [])


class EntityGroundingTest(unittest.TestCase):
    def _build_retriever(self, entity_matches_by_query: dict[str, list[VectorMatch]]) -> EvidenceRetriever:
        logger = logging.getLogger("test.entity_grounding")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        dataset = SimpleNamespace(
            entity_store=MappingStore(entity_matches_by_query, label_field="entity_name"),
            hyperedge_store=MappingStore({}, label_field="hyperedge_name"),
            graph=GroundingGraph(),
            get_chunk_text=lambda chunk_id: {
                "chunk-1": "western North Carolina supports a robust local food distribution network.",
                "chunk-2": "The question asks for a region, but this node is too generic to anchor search.",
                "chunk-3": "Asheville participates in the local food network.",
            }.get(chunk_id, ""),
        )
        return EvidenceRetriever(
            dataset=dataset,
            embedder=TextEmbedder(),
            config=RetrievalConfig(
                entity_top_k=3,
                topic_entity_link_top_k=1,
                topic_entity_link_threshold=0.6,
                hyperedge_top_k=1,
                branch_candidate_pool=2,
            ),
            logger=logger,
        )

    def test_anchor_task_frame_filters_generic_graph_entities(self) -> None:
        retriever = self._build_retriever(
            {
                "distribution network": [
                    VectorMatch(
                        item_id="row-rdn",
                        label='"ROBUST DISTRIBUTION NETWORK"',
                        score=0.91,
                    )
                ],
            }
        )
        task_frame = TaskFrame.from_payload(
            "What region is known for its robust distribution network for local food?",
            {
                "topic_entities": ["distribution network", "region"],
                "anchors": ["distribution network", "region"],
                "answer_type_hint": "location",
                "relation_intent": "find the location linked to the food distribution clue",
                "hard_constraints": ["answer the region, not the institution"],
            },
        )

        payload = retriever.anchor_task_frame(task_frame.question, task_frame)

        self.assertEqual(task_frame.topic_entities, ["ROBUST DISTRIBUTION NETWORK"])
        self.assertEqual(task_frame.anchors, ["ROBUST DISTRIBUTION NETWORK"])
        self.assertEqual(payload["initial_entity_ids"], ['"ROBUST DISTRIBUTION NETWORK"'])
        self.assertEqual(payload["initial_hyperedge_ids"], [retriever.dataset.graph.distribution_hyperedge])
        self.assertEqual(task_frame.checklist["anchors"][0].text, "ROBUST DISTRIBUTION NETWORK")
        self.assertIn('"REGION"', task_frame.metadata["entity_grounding"]["filtered_non_discriminative"])
        self.assertFalse(task_frame.metadata["entity_grounding"]["used_question_fallback"])

    def test_anchor_task_frame_uses_question_fallback_when_topics_do_not_link(self) -> None:
        question = "What region is known for its robust distribution network for local food?"
        retriever = self._build_retriever(
            {
                question: [
                    VectorMatch(
                        item_id="row-wnc",
                        label='"WESTERN NORTH CAROLINA"',
                        score=0.84,
                    )
                ],
            }
        )
        task_frame = TaskFrame.from_payload(
            question,
            {
                "topic_entities": ["imagined cooperative"],
                "anchors": ["imagined cooperative"],
                "answer_type_hint": "location",
                "relation_intent": "find the location linked to the food distribution clue",
                "hard_constraints": ["answer the region, not the institution"],
            },
        )

        payload = retriever.anchor_task_frame(question, task_frame)

        self.assertEqual(task_frame.topic_entities, ["WESTERN NORTH CAROLINA"])
        self.assertEqual(payload["initial_entity_ids"], ['"WESTERN NORTH CAROLINA"'])
        self.assertTrue(task_frame.metadata["entity_grounding"]["used_question_fallback"])
        self.assertEqual(
            task_frame.metadata["entity_grounding"]["seed_traces"][-1]["stage"],
            "question_fallback",
        )

    def test_anchor_task_frame_keeps_single_token_named_entity(self) -> None:
        retriever = self._build_retriever({})
        task_frame = TaskFrame.from_payload(
            "What role does Asheville play in local food distribution?",
            {
                "topic_entities": ["Asheville"],
                "anchors": ["Asheville"],
                "answer_type_hint": "location",
                "relation_intent": "find the place connected to local food distribution",
                "hard_constraints": [],
            },
        )

        payload = retriever.anchor_task_frame(task_frame.question, task_frame)

        self.assertEqual(task_frame.topic_entities, ["ASHEVILLE"])
        self.assertEqual(payload["initial_entity_ids"], ['"ASHEVILLE"'])
        self.assertEqual(payload["initial_hyperedge_ids"], [retriever.dataset.graph.asheville_hyperedge])


if __name__ == "__main__":
    unittest.main()
