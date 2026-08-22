from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from hyper_branch.atomic.executor import AtomicDagExecutor, DagCycleError
from hyper_branch.atomic.models import AtomicQuestionAnalysis, AtomicQuestionNode
from hyper_branch.atomic.retriever import AtomicHyperedgeRetriever
from hyper_branch.config import RetrievalConfig
from hyper_branch.models import GraphNode, VectorMatch


class AtomicPipelineTest(unittest.TestCase):
    def test_two_hop_anchor_retrieval_and_semantic_ranking(self) -> None:
        retriever = _retriever(
            graph=LocalGraph(
                entity_edges={"A": ["H1"], "B": ["H2"]},
                hyperedge_entities={"H1": ["A", "B"], "H2": ["B"]},
            ),
            scores={"H1": 0.2, "H2": 0.9},
        )

        pool = retriever.build_atomic_candidate_pool(
            question="What is connected to A?",
            analysis=AtomicQuestionAnalysis(entities=["A"]),
            primary_anchor_mention="A",
        )
        ranked = retriever.rank_candidate_pool(pool, question="What is connected to A?")

        self.assertEqual(pool.adjacent_hyperedge_ids, ["H1"])
        self.assertEqual(pool.second_hop_hyperedge_ids, ["H2"])
        self.assertEqual([item.hyperedge_id for item in ranked.evidence], ["H2", "H1"])

    def test_vector_entity_linking_is_used_when_no_exact_name_exists(self) -> None:
        store = EntityStore([VectorMatch("A", "A", 0.8, {"entity_name": "A"})])
        retriever = _retriever(
            graph=LocalGraph(entity_edges={"A": ["H1"]}, hyperedge_entities={"H1": ["A"]}),
            scores={"H1": 1.0},
            entity_store=store,
        )

        pool = retriever.build_atomic_candidate_pool(
            question="Question about alias",
            analysis=AtomicQuestionAnalysis(entities=["alias"]),
            primary_anchor_mention="alias",
        )

        self.assertEqual(pool.linked_entity_id, "A")
        self.assertEqual(pool.anchor_match["match_type"], "vector")
        self.assertEqual(store.calls, 1)

    def test_missing_anchor_reports_insufficient_evidence_without_global_fallback(self) -> None:
        retriever = _retriever(
            graph=LocalGraph(entity_edges={"A": ["H1"]}, hyperedge_entities={"H1": ["A"]}),
            scores={"H1": 1.0},
        )

        pool = retriever.build_atomic_candidate_pool(
            question="No named entity",
            analysis=AtomicQuestionAnalysis(),
            primary_anchor_mention="",
        )

        self.assertEqual(pool.insufficient_reason, "missing_primary_anchor")
        self.assertEqual(pool.candidate_hyperedge_ids, [])

    def test_executor_uses_validated_dag_dependency_rewrite_and_shared_pool(self) -> None:
        graph = LocalGraph(
            entity_edges={"A": ["H1"], "B": ["H2"]},
            hyperedge_entities={"H1": ["A", "B"], "H2": ["B"]},
            hyperedge_chunks={"H1": ["C1"], "H2": ["C2"]},
            chunk_texts={"C1": "A is linked to B", "C2": "B was recorded in Place"},
        )
        llm = RecordingLLMService(["B", "Place"])
        executor = AtomicDagExecutor(
            analyzer=QuestionAnalyzer({
                "Who is linked to A?": ["A"],
                "Where was B recorded?": ["B"],
            }),
            retriever=_retriever(graph=graph, scores={"H1": 0.8, "H2": 0.9}),
            llm_service=llm,
        )
        dag = {
            "nodes": [
                {"id": "q1", "question": "Who is linked to A?", "depends_on": [], "operation": "lookup"},
                {"id": "q2", "question": "Where was q1's answer recorded?", "depends_on": ["q1"], "operation": "lookup"},
            ]
        }

        result = executor.run("Where was the entity linked to A recorded?", dag, ["A"])

        self.assertEqual([item.answer for item in result.atomic_results], ["B", "Place"])
        self.assertEqual(result.atomic_results[1].question, "Where was B recorded?")
        self.assertEqual(result.final_answer["answer"], "Place")
        self.assertEqual(result.atomic_results[1].used_hyperedge_ids, ["H2"])
        self.assertEqual(llm.answer_calls[1]["dependency_answers"][0]["answer"], "B")

    def test_executor_rejects_invalid_dag_instead_of_repairing_it(self) -> None:
        executor = AtomicDagExecutor(
            analyzer=QuestionAnalyzer({}),
            retriever=_retriever(
                graph=LocalGraph(entity_edges={"A": ["H1"]}, hyperedge_entities={"H1": ["A"]}),
                scores={"H1": 1.0},
            ),
            llm_service=RecordingLLMService(),
        )
        dag = {
            "nodes": [
                {"id": "q1", "question": "First", "depends_on": [], "operation": "lookup"},
                {"id": "q2", "question": "Second", "depends_on": [], "operation": "lookup"},
            ]
        }

        with self.assertRaisesRegex(ValueError, "one final terminal leaf"):
            executor.run("Question", dag)

    def test_topological_sort_rejects_cycles(self) -> None:
        nodes = [
            AtomicQuestionNode("q1", "one", ["q2"]),
            AtomicQuestionNode("q2", "two", ["q1"]),
        ]
        with self.assertRaises(DagCycleError):
            AtomicDagExecutor.topological_sort(nodes)


class QuestionAnalyzer:
    def __init__(self, entities_by_question: dict[str, list[str]]) -> None:
        self.entities_by_question = entities_by_question

    def analyze(self, question: str, dependency_answers: list[dict[str, object]]) -> AtomicQuestionAnalysis:
        return AtomicQuestionAnalysis(entities=self.entities_by_question.get(question, []))


class RecordingLLMService:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = list(answers or [])
        self.answer_calls: list[dict[str, object]] = []

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, object],
        dependency_answers: list[dict[str, object]],
        evidence: dict[str, list[dict[str, object]]],
        original_question: str = "",
    ) -> dict[str, str]:
        self.answer_calls.append(
            {
                "atomic_question": atomic_question,
                "answer_contract": answer_contract,
                "dependency_answers": dependency_answers,
                "evidence": evidence,
                "original_question": original_question,
            }
        )
        return {"answer": self.answers.pop(0) if self.answers else "INSUFFICIENT_EVIDENCE"}


class LocalGraph:
    def __init__(
        self,
        *,
        entity_edges: dict[str, list[str]],
        hyperedge_entities: dict[str, list[str]],
        hyperedge_chunks: dict[str, list[str]] | None = None,
        chunk_texts: dict[str, str] | None = None,
    ) -> None:
        self.entity_edges = entity_edges
        self.hyperedge_entities = hyperedge_entities
        self.hyperedge_chunks = hyperedge_chunks or {}
        self.chunk_texts = chunk_texts or {}
        entity_ids = set(entity_edges)
        entity_ids.update(entity for values in hyperedge_entities.values() for entity in values)
        self.nodes = {
            entity_id: GraphNode(entity_id, "entity", description=f"{entity_id} description")
            for entity_id in entity_ids
        }
        self.nodes.update(
            {
                hyperedge_id: GraphNode(hyperedge_id, "hyperedge", source_ids=self.hyperedge_chunks.get(hyperedge_id, []))
                for hyperedge_id in hyperedge_entities
            }
        )
        self.source_to_nodes = {
            chunk_id: self.hyperedge_entities.get(hyperedge_id, [])
            for hyperedge_id, chunk_ids in self.hyperedge_chunks.items()
            for chunk_id in chunk_ids
        }

    def entity_hyperedge_ids(self, entity_id: str) -> list[str]:
        return self.entity_edges.get(entity_id, [])

    def hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        return self.hyperedge_entities[hyperedge_id]

    def hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        return self.hyperedge_chunks.get(hyperedge_id, [])

    def describe_hyperedge(self, hyperedge_id: str) -> dict[str, object]:
        return {
            "hyperedge_id": hyperedge_id,
            "hyperedge_text": hyperedge_id,
            "entity_ids": self.hyperedge_entities[hyperedge_id],
            "chunk_ids": self.hyperedge_chunks.get(hyperedge_id, []),
        }


class ScoreStore:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def similarities(self, vector: np.ndarray, ids: list[str]) -> dict[str, float]:
        return {item_id: self.scores.get(item_id, 0.0) for item_id in ids}


class EntityStore:
    def __init__(self, matches: list[VectorMatch] | None = None) -> None:
        self.matches = matches or []
        self.calls = 0

    def query(self, vector: np.ndarray, top_k: int) -> list[VectorMatch]:
        self.calls += 1
        return self.matches[:top_k]


class Embedder:
    def embed_texts(self, texts: list[str], stage: str) -> list[np.ndarray]:
        return [np.ones(3, dtype=np.float32) for _ in texts]


def _retriever(
    *,
    graph: LocalGraph,
    scores: dict[str, float],
    entity_store: EntityStore | None = None,
) -> AtomicHyperedgeRetriever:
    dataset = SimpleNamespace(
        graph=graph,
        hyperedge_store=ScoreStore(scores),
        entity_store=entity_store or EntityStore(),
        get_chunk_text=lambda chunk_id: graph.chunk_texts.get(chunk_id, ""),
    )
    return AtomicHyperedgeRetriever(
        dataset=dataset,
        embedder=Embedder(),
        config=RetrievalConfig(local_hyperedge_top_k=3, local_hyperedge_hops=2),
    )
