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
from hyper_branch.logging_utils import TraceStore, configure_logging, create_run_dir
from hyper_branch.models import GraphNode, VectorMatch
from hyper_branch.pipeline import HyperBranchPipeline
from hyper_branch.utils import normalize_label


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

    def test_branch_support_layering_precedes_fusion_score(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(relation_query="no lexical overlap")
        hits = [
            BranchHit("double", "anchor", 0.0, "nothing useful"),
            BranchHit("double", "relation", 0.0, "nothing useful"),
            BranchHit("single", "semantic", 1.0, "high semantic score"),
        ]

        candidates = fusion.fuse("question", analysis, hits, top_k=5)

        self.assertEqual(candidates[0].hyperedge_id, "double")
        self.assertEqual(len(candidates[0].branch_support), 2)
        self.assertGreater(candidates[1].fusion_score, candidates[0].fusion_score)

    def test_fusion_score_formula(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(
            entities=["Entity A"],
            relations=["graduated from"],
            relation_query="graduated from university",
        )
        hits = [
            BranchHit("h1", "anchor", 1.0, "Entity A evidence", entity_ids=["Entity A"]),
            BranchHit(
                "h1",
                "relation",
                0.5,
                "Entity A evidence",
                metadata={"relation_texts": ["unrelated"]},
            ),
            BranchHit("h1", "semantic", 0.25, "Entity A evidence"),
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.anchor_score, 1.0)
        self.assertAlmostEqual(candidate.relation_score, 0.5)
        self.assertAlmostEqual(candidate.semantic_score, 0.25)
        self.assertAlmostEqual(candidate.fusion_score, 0.65)

    def test_anchor_score_uses_simplified_entity_hit_fraction(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["Demis Hassabis", "University College London"])
        hits = [
            BranchHit(
                "h1",
                "anchor",
                1.0,
                "Demis Hassabis graduated from a university.",
                entity_ids=['"DEMIS HASSABIS"'],
            )
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertAlmostEqual(candidate.anchor_score, 0.5)

    def test_fusion_reuses_hyperedge_vectors_without_embedding_candidate_texts(self) -> None:
        embedder = CountingEmbedder()
        store = SimilarityStore({f"h{index}": 0.9 - (index * 0.001) for index in range(233)})
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=embedder, hyperedge_store=store)
        analysis = AtomicQuestionAnalysis(
            entities=["Entity A"],
            relations=["directed by"],
            relation_query="a film was directed by a person",
        )
        hits = [
            BranchHit(
                hyperedge_id=f"h{index}",
                branch="anchor",
                raw_score=1.0,
                hyperedge_text=f"Entity A directed by Person {index}",
                entity_ids=["Entity A", f"Person {index}"],
            )
            for index in range(233)
        ]

        fusion.fuse("Who directed Entity A?", analysis, hits, top_k=5)

        embedded_texts = [text for _, texts in embedder.calls for text in texts]

        self.assertEqual(embedded_texts, ["a film was directed by a person", "Who directed Entity A?"])
        self.assertFalse(any(text.startswith("Entity A directed by Person") for text in embedded_texts))
        self.assertEqual(store.calls, [233, 233])

    def test_fusion_does_not_rescore_when_relation_and_semantic_raw_scores_exist(self) -> None:
        embedder = CountingEmbedder()
        store = SimilarityStore({"h1": 0.9})
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=embedder, hyperedge_store=store)
        analysis = AtomicQuestionAnalysis(relation_query="query")
        hits = [
            BranchHit("h1", "anchor", 1.0, "Entity A relation", entity_ids=["Entity A"]),
            BranchHit("h1", "relation", 0.7, "Entity A relation"),
            BranchHit("h1", "semantic", 0.6, "Entity A relation"),
        ]

        candidate = fusion.fuse("question", analysis, hits, top_k=5)[0]

        self.assertEqual(embedder.calls, [])
        self.assertEqual(store.calls, [])
        self.assertAlmostEqual(candidate.relation_score, 0.7)
        self.assertAlmostEqual(candidate.semantic_score, 0.6)

    def test_fusion_uses_lexical_fallback_when_candidate_vector_is_missing(self) -> None:
        embedder = CountingEmbedder()
        store = SimilarityStore({})
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=embedder, hyperedge_store=store)
        analysis = AtomicQuestionAnalysis(relation_query="directed by")
        hits = [
            BranchHit("missing", "anchor", 1.0, "Entity A directed by Person B", entity_ids=["Entity A"]),
        ]

        candidate = fusion.fuse("Who directed Entity A?", analysis, hits, top_k=5)[0]

        self.assertEqual([text for _, texts in embedder.calls for text in texts], ["directed by", "Who directed Entity A?"])
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
    def test_retrieve_runs_anchor_relation_and_semantic_branches(self) -> None:
        hyperedge_ids = ["anchor-hit", "relation-hit", "semantic-hit"]
        store = MappingHyperedgeStore(
            {
                "relation query": [VectorMatch(item_id="relation-hit", label="relation-hit", score=0.8)],
                "semantic question": [VectorMatch(item_id="semantic-hit", label="semantic-hit", score=0.7)],
            }
        )
        dataset = SimpleNamespace(
            graph=RetrieverGraph(hyperedge_ids),
            hyperedge_store=store,
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
        semantic_matches = [VectorMatch(item_id=item, label=item, score=0.9 - (index * 0.01)) for index, item in enumerate(reversed(hyperedge_ids))]
        store = MappingHyperedgeStore(
            {
                "relation query": relation_matches,
                "semantic question": semantic_matches,
            }
        )
        dataset = SimpleNamespace(
            graph=RetrieverGraph(hyperedge_ids),
            hyperedge_store=store,
            get_chunk_text=lambda chunk_id: f"text for {chunk_id}",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=TextEmbedder(),
            config=RetrievalConfig(relation_top_k=10, semantic_top_k=10),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        relation_hits = retriever.retrieve_relation_branch(AtomicQuestionAnalysis(relation_query="relation query"))
        semantic_hits = retriever.retrieve_semantic_branch("semantic question")

        self.assertEqual(len(relation_hits), 10)
        self.assertEqual(len(semantic_hits), 10)
        self.assertEqual(store.calls, [("relation query", 10), ("semantic question", 10)])


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
