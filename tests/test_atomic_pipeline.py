from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hyper_branch.atomic import (
    AtomicDagExecutor,
    AtomicHyperedgeRetriever,
    AtomicQuestionAnalysis,
    AtomicQuestionNode,
    DagCycleError,
    FinalAnswerComposer,
)
from hyper_branch.config import RetrievalConfig, load_config
from hyper_branch.llm import MockAtomicLLMService
from hyper_branch.models import GraphNode


class RetrievalConfigTest(unittest.TestCase):
    def test_load_config_uses_local_hyperedge_top_k(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                """
dataset:
  root: datasets/agriculture
runtime:
  base_run_dir: runs/test
retrieval:
  local_hyperedge_top_k: 4
llm:
  use_mock: true
prompts:
  dir: prompts
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path, project_root)

        self.assertEqual(config.retrieval.local_hyperedge_top_k, 4)

    def test_retrieval_config_defaults_to_top3(self) -> None:
        self.assertEqual(RetrievalConfig().local_hyperedge_top_k, 3)


class SingleHopAtomicExecutorTest(unittest.TestCase):
    def test_dag_runs_topologically_and_rewrites_dependency_question(self) -> None:
        graph = LocalGraph(
            entity_edges={
                "B Boy": ["H_PERFORMER"],
                "Meek Mill": ["H_PERFORMER", "H_DETAINED"],
                "Other Artist": ["H_OTHER"],
            },
            hyperedge_entities={
                "H_PERFORMER": ["B Boy", "Meek Mill"],
                "H_DETAINED": ["Meek Mill", "Police Station"],
                "H_OTHER": ["Other Artist", "Global Place"],
            },
            hyperedge_texts={
                "H_PERFORMER": "B Boy was performed by Meek Mill.",
                "H_DETAINED": "Meek Mill was detained at Police Station.",
                "H_OTHER": "Other Artist was detained at Global Place.",
            },
            hyperedge_chunks={
                "H_PERFORMER": ["C_PERFORMER"],
                "H_DETAINED": ["C_DETAINED"],
                "H_OTHER": ["C_OTHER"],
            },
            chunk_texts={
                "C_PERFORMER": "B Boy was performed by Meek Mill in the source.",
                "C_DETAINED": "Meek Mill was detained at Police Station in the source.",
                "C_OTHER": "Other Artist was detained elsewhere.",
            },
        )
        llm = MockAtomicLLMService(
            answer_responses=[
                {
                    "answer": "Meek Mill",
                    "confidence": 0.9,
                    "reasoning_summary": "B Boy was performed by Meek Mill.",
                    "used_hyperedge_ids": ["H_PERFORMER"],
                    "insufficient": False,
                },
                {
                    "answer": "Police Station",
                    "confidence": 0.9,
                    "reasoning_summary": "Meek Mill was detained at Police Station.",
                    "used_hyperedge_ids": ["H_DETAINED"],
                    "insufficient": False,
                },
            ]
        )
        executor = _executor(
            graph=graph,
            scores={"H_PERFORMER": 0.7, "H_DETAINED": 0.95, "H_OTHER": 1.0},
            analyzer=QuestionAnalyzer(
                {
                    "Who performed the song B Boy?": AtomicQuestionAnalysis(entities=["B Boy"], answer_type="person"),
                    "Where was Meek Mill detained?": AtomicQuestionAnalysis(entities=[], answer_type="place"),
                }
            ),
            llm=llm,
        )
        dag = {
            "nodes": [
                {"node_id": "q2", "question": "Where was q1's answer detained?", "dependencies": ["q1"]},
                {"node_id": "q1", "question": "Who performed the song B Boy?", "dependencies": []},
            ]
        }

        result = executor.run("Where was the performer of B Boy detained?", dag)

        self.assertEqual(result.artifacts["execution_order"], ["q1", "q2"])
        self.assertEqual([item.question for item in result.atomic_results], ["Who performed the song B Boy?", "Where was Meek Mill detained?"])
        self.assertEqual(result.atomic_results[1].answer, "Police Station")
        self.assertEqual(len(llm.answer_calls), 2)
        self.assertEqual(llm.answer_calls[1]["atomic_question"], "Where was Meek Mill detained?")
        self.assertEqual(llm.answer_calls[1]["dependency_answers"][0]["answer"], "Meek Mill")
        self.assertFalse(hasattr(llm, "route_reasoning_paths"))
        self.assertFalse(hasattr(llm, "answer_atomic_question_from_paths"))

        second_retrieval = result.artifacts["atomic_retrieval"][1]
        self.assertEqual(second_retrieval["method"], "single_hop_primary_anchor_top3")
        self.assertEqual(second_retrieval["primary_anchor_mention"], "Meek Mill")
        self.assertEqual(second_retrieval["linked_entity_id"], "Meek Mill")
        self.assertEqual(second_retrieval["adjacent_hyperedge_ids"], ["H_PERFORMER", "H_DETAINED"])
        self.assertEqual([item["hyperedge_id"] for item in second_retrieval["top_hyperedges"]], ["H_DETAINED", "H_PERFORMER"])

    def test_retrieval_is_primary_anchor_local_top3_and_stably_sorted(self) -> None:
        graph = LocalGraph(
            entity_edges={
                "Anchor": ["H1", "H2", "H3", "H4"],
                "Other": ["H_GLOBAL"],
            },
            hyperedge_entities={
                "H1": ["Anchor", "A1"],
                "H2": ["Anchor", "A2"],
                "H3": ["Anchor", "A3"],
                "H4": ["Anchor", "A4"],
                "H_GLOBAL": ["Other", "Wrong"],
            },
            hyperedge_chunks={
                "H1": ["C1"],
                "H2": ["C2"],
                "H3": ["C3"],
                "H4": ["C4", "C4B"],
                "H_GLOBAL": ["CG"],
            },
            chunk_texts={
                "C1": "chunk one",
                "C2": "chunk two",
                "C3": "chunk three",
                "C4": "full chunk four text " * 30,
                "C4B": "second full chunk for four",
                "CG": "global chunk",
            },
        )
        store = ScoreHyperedgeStore({"H1": 0.1, "H2": 0.7, "H3": 0.7, "H4": 0.9, "H_GLOBAL": 1.0})
        retriever = AtomicHyperedgeRetriever(
            dataset=_dataset(graph, store),
            embedder=CountingEmbedder(),
            config=RetrievalConfig(local_hyperedge_top_k=3),
            llm_service=MockAtomicLLMService(),
            logger=logging.getLogger("test.local_retriever"),
        )

        result = retriever.retrieve_primary_anchor_local(
            question="Question about Anchor",
            analysis=AtomicQuestionAnalysis(entities=["Anchor"]),
            primary_anchor_mention="Anchor",
        )

        self.assertEqual(store.calls, [["H1", "H2", "H3", "H4"]])
        self.assertEqual([item["hyperedge_id"] for item in result.top_hyperedges], ["H4", "H2", "H3"])
        self.assertEqual([item.rank for item in result.evidence], [1, 2, 3])
        self.assertNotIn("H_GLOBAL", [item.hyperedge_id for item in result.evidence])
        first = result.evidence[0].to_dict()
        self.assertEqual(first["hyperedge_id"], "H4")
        self.assertEqual(first["entity_ids"], ["Anchor", "A4"])
        self.assertEqual(first["entity_records"][0]["entity_id"], "Anchor")
        self.assertEqual(first["chunk_ids"], ["C4", "C4B"])
        self.assertEqual(first["chunk_texts"][0], "full chunk four text " * 30)

    def test_answerer_receives_complete_top3_evidence_once(self) -> None:
        graph = LocalGraph(
            entity_edges={"Subject": ["H1", "H2", "H3", "H4"]},
            hyperedge_entities={
                "H1": ["Subject", "Answer One"],
                "H2": ["Subject", "Answer Two"],
                "H3": ["Subject", "Answer Three"],
                "H4": ["Subject", "Answer Four"],
            },
            hyperedge_chunks={"H1": ["C1"], "H2": ["C2"], "H3": ["C3"], "H4": ["C4"]},
            chunk_texts={"C1": "one", "C2": "two", "C3": "three", "C4": "four"},
        )
        llm = MockAtomicLLMService(
            answer_responses=[
                {
                    "answer": "Answer Two",
                    "confidence": 0.8,
                    "reasoning_summary": "Selected from evidence.",
                    "used_hyperedge_ids": ["H2"],
                    "insufficient": False,
                }
            ]
        )
        executor = _executor(
            graph=graph,
            scores={"H1": 0.1, "H2": 0.9, "H3": 0.7, "H4": 0.5},
            analyzer=QuestionAnalyzer({"Who is linked to Subject?": AtomicQuestionAnalysis(entities=["Subject"])}),
            llm=llm,
        )

        result = executor.run("Who is linked to Subject?")

        self.assertEqual(len(llm.answer_calls), 1)
        evidence = llm.answer_calls[0]["evidence"]
        self.assertEqual([item["hyperedge_id"] for item in evidence], ["H2", "H3", "H4"])
        self.assertIn("entity_records", evidence[0])
        self.assertIn("chunk_texts", evidence[0])
        self.assertEqual(result.atomic_results[0].used_hyperedge_ids, ["H2"])

    def test_missing_anchor_and_missing_evidence_return_insufficient_without_answerer(self) -> None:
        no_anchor_llm = MockAtomicLLMService()
        no_anchor_executor = _executor(
            graph=LocalGraph(entity_edges={}, hyperedge_entities={}),
            scores={},
            analyzer=QuestionAnalyzer({"Question?": AtomicQuestionAnalysis(entities=[])}),
            llm=no_anchor_llm,
        )

        no_anchor = no_anchor_executor.run("Question?")

        self.assertEqual(no_anchor.atomic_results[0].answer, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(no_anchor.atomic_results[0].confidence, 0.0)
        self.assertEqual(no_anchor_llm.answer_calls, [])
        self.assertEqual(no_anchor.artifacts["atomic_retrieval"][0]["insufficient_reason"], "missing_primary_anchor")

        no_edges_llm = MockAtomicLLMService()
        no_edges_executor = _executor(
            graph=LocalGraph(entity_edges={"Isolated": []}, hyperedge_entities={}),
            scores={},
            analyzer=QuestionAnalyzer({"Question?": AtomicQuestionAnalysis(entities=["Isolated"])}),
            llm=no_edges_llm,
        )

        no_edges = no_edges_executor.run("Question?")

        self.assertEqual(no_edges.atomic_results[0].answer, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(no_edges_llm.answer_calls, [])
        self.assertEqual(no_edges.artifacts["atomic_retrieval"][0]["insufficient_reason"], "primary_anchor_has_no_adjacent_hyperedges")

    def test_topological_sort_rejects_cycles(self) -> None:
        nodes = [
            AtomicQuestionNode(node_id="q1", question="one", dependencies=["q2"]),
            AtomicQuestionNode(node_id="q2", question="two", dependencies=["q1"]),
        ]

        with self.assertRaises(DagCycleError):
            AtomicDagExecutor.topological_sort(nodes)


class QuestionAnalyzer:
    def __init__(self, responses: dict[str, AtomicQuestionAnalysis]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def analyze(self, atomic_question: str, dependency_answers=None) -> AtomicQuestionAnalysis:
        del dependency_answers
        self.calls.append(atomic_question)
        return self.responses.get(atomic_question, AtomicQuestionAnalysis())


class StaticComposer(FinalAnswerComposer):
    def __init__(self) -> None:
        super().__init__(llm_service=None)


class LocalGraph:
    def __init__(
        self,
        *,
        entity_edges: dict[str, list[str]],
        hyperedge_entities: dict[str, list[str]],
        hyperedge_texts: dict[str, str] | None = None,
        hyperedge_chunks: dict[str, list[str]] | None = None,
        chunk_texts: dict[str, str] | None = None,
    ) -> None:
        self.entity_edges = {entity_id: list(ids) for entity_id, ids in entity_edges.items()}
        self.hyperedge_entities = {hyperedge_id: list(ids) for hyperedge_id, ids in hyperedge_entities.items()}
        self.hyperedge_texts = dict(hyperedge_texts or {})
        self.hyperedge_chunks = {hyperedge_id: list(ids) for hyperedge_id, ids in (hyperedge_chunks or {}).items()}
        self.chunk_texts = dict(chunk_texts or {})
        entity_ids = set(self.entity_edges)
        for values in self.hyperedge_entities.values():
            entity_ids.update(values)
        self.nodes = {
            entity_id: GraphNode(node_id=entity_id, role="entity", entity_type="entity", description=f"{entity_id} description")
            for entity_id in entity_ids
        }
        self.nodes.update(
            {
                hyperedge_id: GraphNode(
                    node_id=hyperedge_id,
                    role="hyperedge",
                    source_ids=list(self.hyperedge_chunks.get(hyperedge_id, [])),
                    description=self.hyperedge_texts.get(hyperedge_id, hyperedge_id),
                )
                for hyperedge_id in self.hyperedge_entities
            }
        )

    def entity_hyperedge_ids(self, entity_id: str) -> list[str]:
        return list(self.entity_edges.get(entity_id, []))

    def hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        return list(self.hyperedge_entities.get(hyperedge_id, []))

    def describe_hyperedge(self, hyperedge_id: str) -> dict[str, object]:
        chunk_ids = self.hyperedge_chunks.get(hyperedge_id, [])
        return {
            "hyperedge_id": hyperedge_id,
            "hyperedge_text": self.hyperedge_texts.get(hyperedge_id, hyperedge_id),
            "entity_ids": list(self.hyperedge_entities.get(hyperedge_id, [])),
            "chunk_ids": list(chunk_ids),
        }


class ScoreHyperedgeStore:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = dict(scores)
        self.calls: list[list[str]] = []

    def similarities(self, query_vector, row_ids: list[str]) -> dict[str, float]:
        del query_vector
        self.calls.append(list(row_ids))
        return {row_id: float(self.scores.get(row_id, 0.0)) for row_id in row_ids}


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def embed_texts(self, texts: list[str], stage: str | None = None):
        self.calls.append((list(texts), stage))
        return [np.ones(3, dtype=np.float32) for _ in texts]


def _dataset(graph: LocalGraph, store: ScoreHyperedgeStore):
    return SimpleNamespace(
        graph=graph,
        hyperedge_store=store,
        entity_store=None,
        chunk_store=None,
        text_chunks={chunk_id: {"content": text} for chunk_id, text in graph.chunk_texts.items()},
        full_docs={},
        summary={},
        get_chunk_text=lambda chunk_id: graph.chunk_texts.get(chunk_id, ""),
    )


def _executor(
    *,
    graph: LocalGraph,
    scores: dict[str, float],
    analyzer: QuestionAnalyzer,
    llm: MockAtomicLLMService,
) -> AtomicDagExecutor:
    retriever = AtomicHyperedgeRetriever(
        dataset=_dataset(graph, ScoreHyperedgeStore(scores)),
        embedder=CountingEmbedder(),
        config=RetrievalConfig(local_hyperedge_top_k=3),
        llm_service=llm,
        logger=logging.getLogger("test.single_hop_executor"),
    )
    return AtomicDagExecutor(
        analyzer=analyzer,  # type: ignore[arg-type]
        retriever=retriever,
        composer=StaticComposer(),
        llm_service=llm,
        logger=logging.getLogger("test.single_hop_executor"),
    )


if __name__ == "__main__":
    unittest.main()
