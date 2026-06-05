from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hyper_branch.atomic import (
    AtomicAnswerResult,
    AtomicDagExecutor,
    AtomicEvidenceFusion,
    AtomicHyperedgeRetriever,
    AtomicQuestionAnalysis,
    AtomicQuestionNode,
    BranchHit,
    DagCycleError,
    FinalAnswerComposer,
    FusedHyperedgeCandidate,
)
from hyper_branch.config import RetrievalConfig, load_config
from hyper_branch.data.vector_store import VectorStore
from hyper_branch.llm import MockAtomicLLMService
from hyper_branch.logging_utils import TraceStore, configure_logging, create_run_dir
from hyper_branch.models import GraphNode, VectorMatch
from hyper_branch.pipeline import HyperBranchPipeline
from hyper_branch.utils import normalize_label
from tests.agriculture_fixture import ensure_agriculture_fixture


class AtomicDagAdapterTest(unittest.TestCase):
    def test_depo_entity_inputs_are_not_dependencies_when_depends_on_is_empty(self) -> None:
        dag = {
            "nodes": [
                {
                    "id": "q1",
                    "question": "What is the release date of Aas Ka Panchhi?",
                    "inputs": ["Aas Ka Panchhi"],
                    "output": "X1",
                    "depends_on": [],
                },
                {
                    "id": "q2",
                    "question": "What is the release date of Phoolwari?",
                    "inputs": ["Phoolwari"],
                    "output": "X2",
                    "depends_on": [],
                },
                {
                    "id": "q3",
                    "question": "Which film was released first?",
                    "inputs": ["X1", "X2"],
                    "output": "FINAL",
                    "depends_on": ["q1", "q2"],
                },
            ],
            "edges": [
                {"source": "q1", "target": "q3", "variable": "X1"},
                {"source": "q2", "target": "q3", "variable": "X2"},
            ],
            "variable_to_question": {"X1": "q1", "X2": "q2", "FINAL": "q3"},
        }

        nodes = AtomicDagExecutor.normalize_dag_payload(dag)
        by_id = {node.node_id: node for node in nodes}
        order = AtomicDagExecutor.topological_sort(nodes)

        self.assertEqual(by_id["q1"].dependencies, [])
        self.assertEqual(by_id["q2"].dependencies, [])
        self.assertEqual(by_id["q3"].dependencies, ["q1", "q2"])
        self.assertEqual([node.node_id for node in order], ["q1", "q2", "q3"])

    def test_inputs_can_derive_dependencies_only_from_known_nodes_or_variables(self) -> None:
        dag = {
            "nodes": [
                {"id": "q1", "question": "What is X?", "output": "X1"},
                {"id": "q2", "question": "Use X.", "inputs": ["X1", "literal entity"]},
            ],
            "variable_to_question": {"X1": "q1"},
        }

        nodes = AtomicDagExecutor.normalize_dag_payload(dag)
        by_id = {node.node_id: node for node in nodes}

        self.assertEqual(by_id["q1"].dependencies, [])
        self.assertEqual(by_id["q2"].dependencies, ["q1"])

    def test_minimal_dag_metadata_is_not_nested(self) -> None:
        dag = {
            "nodes": [
                {"node_id": "q1", "question": "First?", "dependencies": []},
                {
                    "node_id": "q2",
                    "question": "Compare?",
                    "dependencies": ["q1"],
                    "metadata": {
                        "operator": "COMPARE_LESS",
                        "candidates": [{"label": "A", "source_node_id": "q1"}],
                    },
                },
            ]
        }

        nodes = AtomicDagExecutor.normalize_dag_payload(dag)
        by_id = {node.node_id: node for node in nodes}

        self.assertEqual(by_id["q2"].metadata["operator"], "COMPARE_LESS")
        self.assertEqual(by_id["q2"].metadata["candidates"], [{"label": "A", "source_node_id": "q1"}])
        self.assertNotIn("metadata", by_id["q2"].metadata)

    def test_topological_sort_respects_dependencies(self) -> None:
        nodes = [
            AtomicQuestionNode(node_id="c", question="third", dependencies=["b"]),
            AtomicQuestionNode(node_id="a", question="first"),
            AtomicQuestionNode(node_id="b", question="second", dependencies=["a"]),
        ]

        order = AtomicDagExecutor.topological_sort(nodes)

        self.assertEqual([node.node_id for node in order], ["a", "b", "c"])

    def test_topological_sort_rejects_cycle(self) -> None:
        nodes = [
            AtomicQuestionNode(node_id="a", question="first", dependencies=["b"]),
            AtomicQuestionNode(node_id="b", question="second", dependencies=["a"]),
        ]

        with self.assertRaises(DagCycleError):
            AtomicDagExecutor.topological_sort(nodes)

    def test_dependency_answer_rewrites_retrieval_question_when_confident_entity(self) -> None:
        analyzer = RecordingAnalyzer()
        retriever = RecordingRetriever()
        fusion = RecordingFusion()
        executor = AtomicDagExecutor(
            analyzer=analyzer,
            retriever=retriever,
            fusion=fusion,
            composer=StaticComposer(),
            llm_service=DependencyRewriteLLM({"q1": ("Ermengarde of Tours", 0.85), "q2": ("20 March 851", 0.9)}),
        )
        dag = {
            "nodes": [
                {"node_id": "q1", "question": "Who is the mother of Lothair II?", "dependencies": []},
                {"node_id": "q2", "question": "When did the mother of Lothair II die?", "dependencies": ["q1"]},
            ]
        }

        result = executor.run("When did Lothair II's mother die?", dag)

        self.assertEqual(analyzer.questions[1], "When did Ermengarde of Tours die?")
        self.assertEqual(retriever.questions[1], "When did Ermengarde of Tours die?")
        self.assertEqual(fusion.questions[1], "When did Ermengarde of Tours die?")
        self.assertEqual(result.atomic_results[1].question, "When did the mother of Lothair II die?")
        rewrite = result.artifacts["atomic_question_analyses"][1]["dependency_question_rewrite"]
        self.assertTrue(rewrite["whether_rewritten"])
        self.assertEqual(rewrite["replacement_span"], "the mother of Lothair II")
        self.assertEqual(rewrite["replacement_answer"], "Ermengarde of Tours")
        self.assertEqual(rewrite["retrieval_question"], "When did Ermengarde of Tours die?")

    def test_dependency_answer_does_not_rewrite_when_low_confidence_or_non_entity(self) -> None:
        analyzer = RecordingAnalyzer()
        executor = AtomicDagExecutor(
            analyzer=analyzer,
            retriever=RecordingRetriever(),
            fusion=RecordingFusion(),
            composer=StaticComposer(),
            llm_service=DependencyRewriteLLM({"q1": ("20 March 851", 0.95), "q2": ("done", 0.9)}),
        )
        dag = {
            "nodes": [
                {"node_id": "q1", "question": "When did Lothair II die?", "dependencies": []},
                {"node_id": "q2", "question": "Where is Lothair II buried?", "dependencies": ["q1"]},
            ]
        }

        result = executor.run("Where is Lothair II buried?", dag)

        self.assertEqual(analyzer.questions[1], "Where is Lothair II buried?")
        rewrite = result.artifacts["atomic_question_analyses"][1]["dependency_question_rewrite"]
        self.assertFalse(rewrite["whether_rewritten"])
        self.assertEqual(rewrite["retrieval_question"], "Where is Lothair II buried?")


class RecordingAnalyzer:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def analyze(self, atomic_question: str, dependency_answers=None) -> AtomicQuestionAnalysis:
        del dependency_answers
        self.questions.append(atomic_question)
        answer_type = "person" if atomic_question.lower().startswith("who ") else "date"
        return AtomicQuestionAnalysis(
            entities=["Lothair II"] if "Lothair II" in atomic_question else ["Ermengarde of Tours"],
            relations=["relation"],
            relation_query=atomic_question,
            answer_type=answer_type,
        )


class RecordingRetriever:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def retrieve(self, question: str, analysis: AtomicQuestionAnalysis) -> list[BranchHit]:
        del analysis
        self.questions.append(question)
        return []


class RecordingFusion:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def fuse(
        self,
        question: str,
        analysis: AtomicQuestionAnalysis,
        branch_hits: list[BranchHit],
    ) -> list[FusedHyperedgeCandidate]:
        del analysis, branch_hits
        self.questions.append(question)
        return []


class StaticComposer:
    def compose(self, original_question, atomic_results, dag_nodes=None):
        del original_question, dag_nodes
        return {"answer": atomic_results[-1].answer if atomic_results else ""}


class DependencyRewriteLLM:
    def __init__(self, answers_by_node_id: dict[str, tuple[str, float]]) -> None:
        self.answers_by_node_id = answers_by_node_id

    def analyze_atomic_question(self, atomic_question, dependency_answers):
        raise NotImplementedError

    def answer_atomic_question(self, atomic_question, dependency_answers, evidence):
        del dependency_answers, evidence
        if "mother of Lothair" in atomic_question:
            answer, confidence = self.answers_by_node_id["q1"]
        elif "Lothair II die" in atomic_question:
            answer, confidence = self.answers_by_node_id["q1"]
        else:
            answer, confidence = self.answers_by_node_id.get("q2", ("INSUFFICIENT_EVIDENCE", 0.0))
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_summary": "test answer",
            "used_hyperedge_ids": [],
            "insufficient": answer == "INSUFFICIENT_EVIDENCE",
        }

    def compose_final_answer(self, original_question, dag_nodes, atomic_results):
        raise NotImplementedError

    def finalize_answer_span(self, original_question, synthesis_candidate):
        raise NotImplementedError


class AtomicFusionTest(unittest.TestCase):
    def test_branch_hits_merge_by_hyperedge_id(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(
            entities=["Entity A"],
            relations=["relation"],
            relation_query="relation",
            answer_type="entity",
        )
        hits = [
            BranchHit("h1", "anchor", 1.0, "Entity A relation", entity_ids=["Entity A"], chunk_ids=["c1"]),
            BranchHit("h1", "relation", 0.5, "Entity A relation", entity_ids=["Entity B"], chunk_ids=["c2"]),
        ]

        candidates = fusion.fuse("question", analysis, hits, top_k=5)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].branch_support, {"anchor", "relation"})
        self.assertEqual(candidates[0].entity_ids, ["Entity A", "Entity B"])
        self.assertEqual(candidates[0].chunk_ids, ["c1", "c2"])

    def test_fusion_score_precedes_branch_support_in_sorting(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(relation_query="no lexical overlap")
        hits = [
            BranchHit("double", "anchor", 0.0, "nothing useful"),
            BranchHit("double", "relation", 0.0, "nothing useful"),
            BranchHit("single", "semantic", 0.0, "no lexical overlap"),
        ]

        candidates = fusion.fuse("question", analysis, hits, top_k=5)

        self.assertEqual(candidates[0].hyperedge_id, "single")
        self.assertEqual(len(candidates[1].branch_support), 2)
        self.assertGreater(candidates[0].fusion_score, candidates[1].fusion_score)

    def test_fusion_score_formula(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(
            entities=["Entity A"],
            relations=["graduated from"],
            relation_query="graduated from university",
        )
        hits = [
            BranchHit("h1", "anchor", 0.2, "Entity A graduated from university", entity_ids=["Entity A"]),
            BranchHit(
                "h1",
                "relation",
                0.1,
                "Entity A graduated from university",
                metadata={"relation_texts": ["graduated from university"]},
            ),
            BranchHit("h1", "semantic", 0.05, "Entity A graduated from university"),
        ]

        candidate = fusion.fuse("Who graduated from university?", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.anchor_score, 1.0)
        self.assertAlmostEqual(candidate.relation_score, 1.0)
        self.assertAlmostEqual(candidate.semantic_score, 1.0)
        self.assertAlmostEqual(candidate.fusion_score, 1.0)

    def test_fusion_does_not_reuse_branch_raw_scores(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["Apple"], relations=["directed by"], relation_query="directed by")
        hits = [
            BranchHit(
                "h1",
                "anchor",
                0.99,
                "no useful text",
                entity_ids=["Apple tree"],
            ),
            BranchHit("h1", "relation", 0.99, "no useful text"),
            BranchHit("h1", "semantic", 0.99, "no useful text"),
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.anchor_score, 0.0)
        self.assertAlmostEqual(candidate.relation_score, 0.0)
        self.assertAlmostEqual(candidate.semantic_score, 0.0)

    def test_anchor_score_uses_anchor_matches_instead_of_raw_score(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["US", "China"])
        hits = [
            BranchHit(
                "h1",
                "anchor",
                0.1,
                "United States and China relation",
                entity_ids=["United States", "China"],
                metadata={
                    "anchor_matches": [
                        {
                            "query_index": 0,
                            "query_entity": "US",
                            "matched_entity_id": "United States",
                            "match_type": "vector_llm",
                            "link_score": 0.8,
                        },
                        {
                            "query_index": 1,
                            "query_entity": "China",
                            "matched_entity_id": "China",
                            "match_type": "exact",
                            "link_score": 1.0,
                        },
                    ]
                },
            )
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.anchor_score, 0.9)

    def test_anchor_score_does_not_use_substring_or_hyperedge_text(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["Apple"])
        hits = [
            BranchHit(
                "h1",
                "relation",
                0.9,
                "Apple appears in text.",
                entity_ids=["Apple tree"],
            )
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.anchor_score, 0.0)

    def test_relation_score_uses_hyperedge_store_similarities_not_raw_score(self) -> None:
        embedder = CountingEmbedder()
        store = SimilarityStore({"h1": 0.42})
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=embedder, hyperedge_store=store)
        analysis = AtomicQuestionAnalysis(relations=["directed by"], relation_query="directed by")
        hits = [
            BranchHit(
                hyperedge_id="h1",
                branch="relation",
                raw_score=0.9,
                hyperedge_text="unrelated",
            )
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.relation_score, 0.42)
        self.assertEqual(embedder.calls[0], ("atomic_relation_candidate_scoring", ["directed by"]))
        self.assertEqual(store.calls, [1])

    def test_semantic_score_uses_chunk_store_similarities_not_raw_score(self) -> None:
        embedder = CountingEmbedder()
        chunk_store = SimilarityStore({"chunk-1": 0.4})
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=embedder, chunk_store=chunk_store)
        analysis = AtomicQuestionAnalysis()
        hits = [
            BranchHit("h1", "semantic", 0.9, "unrelated", chunk_ids=["chunk-1"]),
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.semantic_score, 0.4)
        self.assertEqual(embedder.calls[0], ("atomic_semantic_candidate_scoring", ["question"]))
        self.assertEqual(chunk_store.calls, [1])

    def test_fusion_uses_lexical_fallback_when_vector_stores_are_unavailable(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None, hyperedge_store=None, chunk_store=None)
        analysis = AtomicQuestionAnalysis(relation_query="directed by")
        hits = [
            BranchHit("missing", "anchor", 1.0, "Entity A directed by Person B", entity_ids=["Entity A"]),
        ]

        candidate = fusion.fuse("Who directed Entity A?", analysis, hits, top_k=5)[0]

        self.assertGreater(candidate.relation_score, 0.0)
        self.assertGreater(candidate.semantic_score, 0.0)

    def test_vector_store_batch_similarities_resolve_ids_and_labels(self) -> None:
        store = VectorStore(
            name="test",
            rows=[
                {"__id__": "row-1", "hyperedge_name": "Hyperedge One"},
                {"__id__": "row-2", "hyperedge_name": "Hyperedge Two"},
            ],
            matrix=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            label_fields=("hyperedge_name",),
        )

        scores = store.similarities(np.asarray([1.0, 0.0], dtype=np.float32), ["row-1", "Hyperedge Two", "missing"])

        self.assertAlmostEqual(scores["row-1"], 1.0)
        self.assertAlmostEqual(scores["Hyperedge Two"], 0.0)
        self.assertNotIn("missing", scores)


class TextEmbedder:
    def embed_texts(self, texts: list[str], stage: str) -> list[str]:
        del stage
        return [normalize_label(text) for text in texts]


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def embed_texts(self, texts: list[str], stage: str) -> list[np.ndarray]:
        self.calls.append((stage, list(texts)))
        return [np.asarray([len(text), sum(ord(char) for char in text) % 97, 1.0], dtype=np.float32) for text in texts]


class SimilarityStore:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[int] = []

    def similarities(self, query_vector: np.ndarray, row_ids: list[str]) -> dict[str, float]:
        del query_vector
        self.calls.append(len(row_ids))
        return {row_id: self.scores[row_id] for row_id in row_ids if row_id in self.scores}


class MappingHyperedgeStore:
    def __init__(self, matches_by_query: dict[str, list[VectorMatch]]) -> None:
        self.matches_by_query = {normalize_label(key): list(value) for key, value in matches_by_query.items()}
        self.calls: list[tuple[str, int]] = []

    def query(self, vector: str, top_k: int) -> list[VectorMatch]:
        query = normalize_label(vector)
        self.calls.append((query, top_k))
        return self.matches_by_query.get(query, [])[:top_k]


class FailingHyperedgeStore(MappingHyperedgeStore):
    def __init__(self) -> None:
        super().__init__({})

    def query(self, vector: str, top_k: int) -> list[VectorMatch]:
        del vector, top_k
        raise AssertionError("semantic branch must not query hyperedge_store")


class MappingChunkStore:
    def __init__(self, matches_by_query: dict[str, list[VectorMatch]], scores: dict[str, float] | None = None) -> None:
        self.matches_by_query = {normalize_label(key): list(value) for key, value in matches_by_query.items()}
        self.scores = scores or {}
        self.calls: list[tuple[str, int]] = []
        self.similarity_calls: list[int] = []

    def query(self, vector: str, top_k: int) -> list[VectorMatch]:
        query = normalize_label(vector)
        self.calls.append((query, top_k))
        return self.matches_by_query.get(query, [])[:top_k]

    def similarities(self, query_vector: np.ndarray, row_ids: list[str]) -> dict[str, float]:
        del query_vector
        self.similarity_calls.append(len(row_ids))
        return {row_id: self.scores[row_id] for row_id in row_ids if row_id in self.scores}


class MappingEntityStore:
    def __init__(self, matches_by_query: dict[str, list[VectorMatch]]) -> None:
        self.matches_by_query = {normalize_label(key): list(value) for key, value in matches_by_query.items()}
        self.calls: list[tuple[str, int]] = []

    def query(self, vector: str, top_k: int) -> list[VectorMatch]:
        query = normalize_label(vector)
        self.calls.append((query, top_k))
        return self.matches_by_query.get(query, [])[:top_k]


class AnchorGraph:
    def __init__(self, hyperedge_entities: dict[str, list[str]]) -> None:
        entity_ids = sorted({entity_id for entity_list in hyperedge_entities.values() for entity_id in entity_list})
        self.nodes = {
            **{entity_id: GraphNode(node_id=entity_id, role="entity") for entity_id in entity_ids},
            **{hyperedge_id: GraphNode(node_id=hyperedge_id, role="hyperedge") for hyperedge_id in hyperedge_entities},
        }
        self._hyperedge_entities = {hyperedge_id: list(entity_ids) for hyperedge_id, entity_ids in hyperedge_entities.items()}
        self._entity_hyperedges: dict[str, list[str]] = {entity_id: [] for entity_id in entity_ids}
        for hyperedge_id, hyperedge_entity_ids in hyperedge_entities.items():
            for entity_id in hyperedge_entity_ids:
                self._entity_hyperedges.setdefault(entity_id, []).append(hyperedge_id)

    def entity_hyperedge_ids(self, entity_id: str) -> list[str]:
        return list(self._entity_hyperedges.get(entity_id, []))

    def describe_hyperedge(self, hyperedge_id: str) -> dict[str, object]:
        return {
            "hyperedge_id": hyperedge_id,
            "hyperedge_text": hyperedge_id,
            "entity_ids": self._hyperedge_entities.get(hyperedge_id, []),
            "chunk_ids": [],
        }


class ChunkGraph:
    def __init__(
        self,
        hyperedge_chunks: dict[str, list[str]],
        hyperedge_entities: dict[str, list[str]] | None = None,
    ) -> None:
        hyperedge_entities = hyperedge_entities or {hyperedge_id: [] for hyperedge_id in hyperedge_chunks}
        entity_ids = sorted({entity_id for values in hyperedge_entities.values() for entity_id in values})
        self.nodes = {
            **{entity_id: GraphNode(node_id=entity_id, role="entity") for entity_id in entity_ids},
            **{
                hyperedge_id: GraphNode(node_id=hyperedge_id, role="hyperedge", source_ids=list(chunk_ids))
                for hyperedge_id, chunk_ids in hyperedge_chunks.items()
            },
        }
        self._hyperedge_chunks = {hyperedge_id: list(chunk_ids) for hyperedge_id, chunk_ids in hyperedge_chunks.items()}
        self._hyperedge_entities = {
            hyperedge_id: list(hyperedge_entities.get(hyperedge_id, [])) for hyperedge_id in hyperedge_chunks
        }

    def entity_hyperedge_ids(self, entity_id: str) -> list[str]:
        return [
            hyperedge_id
            for hyperedge_id, entity_ids in self._hyperedge_entities.items()
            if entity_id in entity_ids
        ]

    def hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        return list(self._hyperedge_chunks.get(hyperedge_id, []))

    def describe_hyperedge(self, hyperedge_id: str) -> dict[str, object]:
        return {
            "hyperedge_id": hyperedge_id,
            "hyperedge_text": hyperedge_id,
            "entity_ids": self._hyperedge_entities.get(hyperedge_id, []),
            "chunk_ids": self._hyperedge_chunks.get(hyperedge_id, []),
        }


class RetrieverGraph:
    def __init__(self, hyperedge_ids: list[str]) -> None:
        self.nodes = {
            '"ENTITY A"': GraphNode(node_id='"ENTITY A"', role="entity"),
            **{hyperedge_id: GraphNode(node_id=hyperedge_id, role="hyperedge", source_ids=[f"chunk-{index}"]) for index, hyperedge_id in enumerate(hyperedge_ids)},
        }

    def entity_hyperedge_ids(self, entity_id: str) -> list[str]:
        if entity_id == '"ENTITY A"':
            return list(self.hyperedge_ids)
        return []

    @property
    def hyperedge_ids(self) -> list[str]:
        return [node_id for node_id, node in self.nodes.items() if node.role == "hyperedge"]

    def describe_hyperedge(self, hyperedge_id: str) -> dict[str, object]:
        return {
            "hyperedge_id": hyperedge_id,
            "hyperedge_text": hyperedge_id,
            "entity_ids": ['"ENTITY A"'],
            "chunk_ids": [self.nodes[hyperedge_id].source_ids[0]],
        }


class AtomicRetrieverTest(unittest.TestCase):
    def test_semantic_branch_uses_chunk_store_and_maps_chunks_to_hyperedges(self) -> None:
        chunk_store = MappingChunkStore(
            {
                "semantic question": [
                    VectorMatch(item_id="chunk-1", label="chunk-1", score=0.77, metadata={"__id__": "chunk-1"})
                ]
            }
        )
        dataset = SimpleNamespace(
            graph=ChunkGraph({"H1": ["chunk-1"]}, {"H1": ["Entity A"]}),
            chunk_store=chunk_store,
            hyperedge_store=FailingHyperedgeStore(),
            text_chunks={"chunk-1": {"content": "matched chunk text"}},
            get_chunk_text=lambda chunk_id: {"chunk-1": "matched chunk text"}.get(chunk_id, ""),
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=TextEmbedder(),
            config=RetrievalConfig(semantic_chunk_top_k=10),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve_semantic_branch("semantic question")

        self.assertEqual([hit.hyperedge_id for hit in hits], ["H1"])
        self.assertEqual(hits[0].branch, "semantic")
        self.assertAlmostEqual(hits[0].raw_score, 0.77)
        self.assertEqual(hits[0].metadata["semantic_source"], "chunk_store")
        self.assertEqual(hits[0].metadata["matched_chunk_ids"], ["chunk-1"])
        self.assertEqual(hits[0].metadata["matched_chunk_scores"], {"chunk-1": 0.77})
        self.assertEqual(hits[0].metadata["matched_chunk_texts"], ["matched chunk text"])
        self.assertEqual(chunk_store.calls, [("semantic question", 10)])

    def test_semantic_branch_limits_chunk_to_hyperedge_expansion(self) -> None:
        chunk_store = MappingChunkStore(
            {
                "semantic question": [
                    VectorMatch(item_id="chunk-1", label="chunk-1", score=0.5, metadata={"chunk_id": "chunk-1"})
                ]
            }
        )
        dataset = SimpleNamespace(
            graph=ChunkGraph({f"H{index}": ["chunk-1"] for index in range(5)}),
            chunk_store=chunk_store,
            hyperedge_store=FailingHyperedgeStore(),
            text_chunks={"chunk-1": {"content": "matched chunk text"}},
            get_chunk_text=lambda chunk_id: "matched chunk text",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=TextEmbedder(),
            config=RetrievalConfig(semantic_chunk_top_k=10, max_semantic_hyperedges_per_chunk=2),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve_semantic_branch("semantic question")

        self.assertEqual([hit.hyperedge_id for hit in hits], ["H0", "H1"])

    def test_anchor_entity_substring_match_is_removed(self) -> None:
        dataset = SimpleNamespace(
            graph=AnchorGraph({"h_apple_tree": ["Apple tree"]}),
            hyperedge_store=MappingHyperedgeStore({}),
            get_chunk_text=lambda chunk_id: "",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=None,
            config=RetrievalConfig(),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve_anchor_branch("What is Apple?", AtomicQuestionAnalysis(entities=["Apple"]))

        self.assertEqual(hits, [])

    def test_anchor_hyperedge_id_substring_fallback_is_removed(self) -> None:
        dataset = SimpleNamespace(
            graph=AnchorGraph({"Apple facts hyperedge": ["Banana"]}),
            hyperedge_store=MappingHyperedgeStore({}),
            get_chunk_text=lambda chunk_id: "",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=None,
            config=RetrievalConfig(),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve_anchor_branch("What is Apple?", AtomicQuestionAnalysis(entities=["Apple"]))

        self.assertEqual(hits, [])

    def test_anchor_exact_match_raw_score_is_one(self) -> None:
        dataset = SimpleNamespace(
            graph=AnchorGraph({"h_china": ["China"]}),
            hyperedge_store=MappingHyperedgeStore({}),
            get_chunk_text=lambda chunk_id: "",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=None,
            config=RetrievalConfig(),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve_anchor_branch("What is China?", AtomicQuestionAnalysis(entities=["China"]))

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].hyperedge_id, "h_china")
        self.assertAlmostEqual(hits[0].raw_score, 1.0)

    def test_anchor_vector_llm_match_uses_vector_score_in_raw_score(self) -> None:
        entity_store = MappingEntityStore(
            {
                "US": [
                    VectorMatch(
                        item_id="entity-row-us",
                        label="United States",
                        score=0.8,
                        metadata={"entity_name": "United States"},
                    )
                ]
            }
        )
        dataset = SimpleNamespace(
            graph=AnchorGraph(
                {
                    "h_both": ["United States", "China"],
                    "h_us": ["United States"],
                }
            ),
            entity_store=entity_store,
            hyperedge_store=MappingHyperedgeStore({}),
            get_chunk_text=lambda chunk_id: "",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=TextEmbedder(),
            config=RetrievalConfig(anchor_entity_top_k=3),
            llm_service=MockAtomicLLMService(),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve_anchor_branch(
            "How are US and China related?",
            AtomicQuestionAnalysis(entities=["US", "China"]),
        )
        scores = {hit.hyperedge_id: hit.raw_score for hit in hits}

        self.assertAlmostEqual(scores["h_both"], 0.9)
        self.assertAlmostEqual(scores["h_us"], 0.4)
        self.assertEqual(entity_store.calls, [("US", 3)])

    def test_fusion_does_not_reuse_anchor_raw_score_without_grounding(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["Apple"])
        hits = [
            BranchHit(
                hyperedge_id="h1",
                branch="anchor",
                raw_score=0.73,
                hyperedge_text="A relation about a company.",
                entity_ids=["Some Entity"],
            )
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.anchor_score, 0.0)

    def test_non_anchor_anchor_score_uses_only_exact_entity_id_coverage(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["Apple"], relation_query="contains Apple")
        hits = [
            BranchHit(
                hyperedge_id="h1",
                branch="relation",
                raw_score=0.9,
                hyperedge_text="Apple appears in text.",
                entity_ids=["Apple tree"],
            )
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.anchor_score, 0.0)

    def test_retrieve_runs_anchor_relation_and_semantic_branches(self) -> None:
        hyperedge_ids = ["anchor-hit", "relation-hit", "semantic-hit"]
        hyperedge_store = MappingHyperedgeStore(
            {
                "relation query": [VectorMatch(item_id="relation-hit", label="relation-hit", score=0.8)],
            }
        )
        chunk_store = MappingChunkStore(
            {
                "semantic question": [
                    VectorMatch(item_id="chunk-2", label="chunk-2", score=0.7, metadata={"__id__": "chunk-2"})
                ]
            }
        )
        dataset = SimpleNamespace(
            graph=RetrieverGraph(hyperedge_ids),
            hyperedge_store=hyperedge_store,
            chunk_store=chunk_store,
            text_chunks={f"chunk-{index}": {"content": f"text for chunk-{index}"} for index in range(3)},
            get_chunk_text=lambda chunk_id: f"text for {chunk_id}",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=TextEmbedder(),
            config=RetrievalConfig(relation_top_k=10, semantic_top_k=10),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve(
            "semantic question",
            AtomicQuestionAnalysis(entities=["ENTITY A"], relation_query="relation query"),
        )

        self.assertEqual({hit.branch for hit in hits}, {"anchor", "relation", "semantic"})

    def test_relation_and_semantic_branches_use_top_10(self) -> None:
        hyperedge_ids = [f"h{index}" for index in range(12)]
        relation_matches = [VectorMatch(item_id=item, label=item, score=1.0 - (index * 0.01)) for index, item in enumerate(hyperedge_ids)]
        semantic_matches = [
            VectorMatch(item_id=f"chunk-{index}", label=f"chunk-{index}", score=0.9 - (index * 0.01), metadata={"__id__": f"chunk-{index}"})
            for index in range(12)
        ]
        hyperedge_store = MappingHyperedgeStore(
            {
                "relation query": relation_matches,
            }
        )
        chunk_store = MappingChunkStore({"semantic question": semantic_matches})
        dataset = SimpleNamespace(
            graph=RetrieverGraph(hyperedge_ids),
            hyperedge_store=hyperedge_store,
            chunk_store=chunk_store,
            text_chunks={f"chunk-{index}": {"content": f"text for chunk-{index}"} for index in range(12)},
            get_chunk_text=lambda chunk_id: f"text for {chunk_id}",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=TextEmbedder(),
            config=RetrievalConfig(relation_top_k=10, semantic_chunk_top_k=10),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        relation_hits = retriever.retrieve_relation_branch(AtomicQuestionAnalysis(relation_query="relation query"))
        semantic_hits = retriever.retrieve_semantic_branch("semantic question")

        self.assertEqual(len(relation_hits), 10)
        self.assertEqual(len(semantic_hits), 10)
        self.assertEqual(hyperedge_store.calls, [("relation query", 10)])
        self.assertEqual(chunk_store.calls, [("semantic question", 10)])


class FinalAnswerComposerTest(unittest.TestCase):
    def test_final_synthesis_uses_two_llm_stages(self) -> None:
        llm = TwoStageLLM()
        composer = FinalAnswerComposer(llm)
        result = AtomicAnswerResult(
            node_id="q1",
            question="Was A or B born first?",
            analysis=AtomicQuestionAnalysis(),
            evidence=[
                FusedHyperedgeCandidate(
                    hyperedge_id="h1",
                    hyperedge_text="A was born in 1900. B was born in 1910.",
                    branch_support={"semantic"},
                )
            ],
            answer="A was born first in 1900.",
            confidence=0.9,
            reasoning_summary="A predates B.",
            used_hyperedge_ids=["h1"],
        )

        payload = composer.compose(
            "Was A or B born first?",
            [result],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="Was A or B born first?")],
        )

        self.assertEqual(llm.calls, ["compose", "span"])
        self.assertEqual(payload["candidate_answer"], "A was born first in 1900.")
        self.assertEqual(payload["answer"], "A")
        self.assertEqual(payload["atomic_answer_trace"][0]["node_id"], "q1")


class TwoStageLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze_atomic_question(self, atomic_question, dependency_answers):
        raise NotImplementedError

    def answer_atomic_question(self, atomic_question, dependency_answers, evidence):
        raise NotImplementedError

    def compose_final_answer(self, original_question, dag_nodes, atomic_results):
        self.calls.append("compose")
        self.compose_payload = {
            "original_question": original_question,
            "dag_nodes": dag_nodes,
            "atomic_results": atomic_results,
        }
        return {
            "candidate_answer": "A was born first in 1900.",
            "reasoning_summary": "A has the earlier birth date.",
            "confidence": 0.9,
            "atomic_answer_trace": [
                {
                    "node_id": "q1",
                    "question": "Was A or B born first?",
                    "answer": "A was born first in 1900.",
                    "used_hyperedge_ids": ["h1"],
                }
            ],
            "remaining_gaps": [],
        }

    def finalize_answer_span(self, original_question, synthesis_candidate):
        self.calls.append("span")
        self.span_payload = {
            "original_question": original_question,
            "synthesis_candidate": synthesis_candidate,
        }
        return {
            "answer": "A",
            "confidence": 0.9,
            "answer_span_reasoning": "The selected candidate is A.",
        }


class AtomicPipelineSmokeTest(unittest.TestCase):
    def test_mock_pipeline_runs_minimal_dag(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        ensure_agriculture_fixture(project_root)
        config = load_config(project_root / "configs" / "agriculture.yaml", project_root)
        config.llm.use_mock = True

        question = "How can urban farms build community support?"
        dag = {
            "nodes": [
                {
                    "id": "q1",
                    "question": "How can urban farms build community support?",
                    "dependencies": [],
                }
            ]
        }
        run_dir = create_run_dir(config.runtime.base_run_dir, "test atomic smoke run")
        logger = configure_logging(run_dir, config.runtime.log_level)
        trace_store = TraceStore(run_dir)

        pipeline = HyperBranchPipeline(config=config, run_dir=run_dir, logger=logger, trace_store=trace_store)
        result = pipeline.run(question, dag_payload=dag)

        self.assertEqual(result["original_question"], question)
        self.assertTrue(result["atomic_results"])
        self.assertIn("analysis", result["atomic_results"][0])
        self.assertIn("evidence", result["atomic_results"][0])
        self.assertTrue(result["final_answer"]["answer"])
        self.assertTrue((run_dir / "artifacts" / "dag_input.json").exists())
        self.assertTrue((run_dir / "artifacts" / "atomic_question_analyses.json").exists())
        self.assertTrue((run_dir / "artifacts" / "atomic_retrieval.json").exists())
        self.assertTrue((run_dir / "artifacts" / "atomic_answers.json").exists())
        self.assertTrue((run_dir / "artifacts" / "final_answer.json").exists())
        self.assertFalse((run_dir / "artifacts" / "task_frame.json").exists())
        self.assertFalse((run_dir / "artifacts" / "thought_graph.json").exists())
        self.assertFalse((run_dir / "artifacts" / "evidence_subgraph.json").exists())
        self.assertFalse((run_dir / "artifacts" / "llm_evidence_view.json").exists())

    def test_pipeline_source_does_not_import_old_controller(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "hyper_branch" / "pipeline.py").read_text(encoding="utf-8")
        forbidden = [
            "Thought" + "Controller",
            "Thought" + "Scorer",
            "TaskFrame" + "Builder",
            "TaskFrame" + "Registry",
            "Evidence" + "Retriever",
        ]

        self.assertFalse(any(item in source for item in forbidden))


if __name__ == "__main__":
    unittest.main()
