from __future__ import annotations

import json
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
    HypergraphReasoningPath,
    RoutedHypergraphWalker,
)
from hyper_branch.atomic.dependency_rewrite import resolve_dependency_question
from hyper_branch.config import RetrievalConfig, load_config
from hyper_branch.data.vector_store import VectorStore
from hyper_branch.llm import MockAtomicLLMService
from hyper_branch.logging_utils import TraceStore, configure_logging, create_run_dir
from hyper_branch.models import GraphNode, VectorMatch
from hyper_branch.pipeline import HyperBranchPipeline
from hyper_branch.utils import normalize_label
from tests.agriculture_fixture import ensure_agriculture_fixture


class AtomicDagAdapterTest(unittest.TestCase):
    def test_dependency_variable_rewrite_apostrophe_answer(self) -> None:
        rewrite = resolve_dependency_question(
            "When did q1's answer die?",
            [{"node_id": "q1", "answer": "Ermengarde of Tours"}],
        )

        self.assertEqual(rewrite.retrieval_question, "When did Ermengarde of Tours die?")
        self.assertEqual(rewrite.primary_anchor_entities, ["Ermengarde of Tours"])
        self.assertTrue(rewrite.whether_rewritten)

    def test_dependency_variable_rewrite_curly_apostrophe_answer(self) -> None:
        rewrite = resolve_dependency_question(
            "When did q1\u2019s answer die?",
            [{"node_id": "q1", "answer": "Ermengarde of Tours"}],
        )

        self.assertEqual(rewrite.retrieval_question, "When did Ermengarde of Tours die?")

    def test_dependency_variable_rewrite_braced_answer(self) -> None:
        rewrite = resolve_dependency_question(
            "When did {q1.answer} die?",
            [{"node_id": "q1", "answer": "Ermengarde of Tours"}],
        )

        self.assertEqual(rewrite.retrieval_question, "When did Ermengarde of Tours die?")

    def test_dependency_variable_rewrite_multiple_dependencies(self) -> None:
        rewrite = resolve_dependency_question(
            "Which is older, q1's answer or q2's answer?",
            [
                {"node_id": "q1", "answer": "A"},
                {"node_id": "q2", "answer": "B"},
            ],
        )

        self.assertEqual(rewrite.retrieval_question, "Which is older, A or B?")
        self.assertEqual(rewrite.primary_anchor_entities, ["A", "B"])

    def test_dependency_variable_rewrite_missing_answer_records_unresolved(self) -> None:
        rewrite = resolve_dependency_question(
            "When did q1's answer die?",
            [{"node_id": "q1", "answer": ""}],
        )

        self.assertEqual(rewrite.retrieval_question, "When did q1's answer die?")
        self.assertFalse(rewrite.whether_rewritten)
        self.assertEqual(rewrite.unresolved_dependencies[0]["node_id"], "q1")

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
        self.assertEqual(result.atomic_results[1].question, "When did Ermengarde of Tours die?")
        rewrite = result.artifacts["atomic_question_analyses"][1]["dependency_question_rewrite"]
        self.assertTrue(rewrite["whether_rewritten"])
        self.assertEqual(rewrite["replacement_span"], "the mother of Lothair II")
        self.assertEqual(rewrite["replacement_answer"], "Ermengarde of Tours")
        self.assertEqual(rewrite["retrieval_question"], "When did Ermengarde of Tours die?")
        self.assertEqual(result.artifacts["atomic_question_analyses"][1]["original_question"], "When did the mother of Lothair II die?")
        self.assertEqual(result.artifacts["atomic_question_analyses"][1]["resolved_question"], "When did Ermengarde of Tours die?")

    def test_executor_resolves_dependency_answer_variables_before_all_atomic_stages(self) -> None:
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
                {"node_id": "q2", "question": "When did q1's answer die?", "dependencies": ["q1"]},
            ]
        }

        result = executor.run("When did Lothair II's mother die?", dag)

        self.assertEqual(analyzer.questions[1], "When did Ermengarde of Tours die?")
        self.assertEqual(retriever.questions[1], "When did Ermengarde of Tours die?")
        self.assertEqual(fusion.questions[1], "When did Ermengarde of Tours die?")
        self.assertEqual(result.atomic_results[1].question, "When did Ermengarde of Tours die?")
        artifact = result.artifacts["atomic_question_analyses"][1]
        self.assertEqual(artifact["original_question"], "When did q1's answer die?")
        self.assertEqual(artifact["resolved_question"], "When did Ermengarde of Tours die?")
        self.assertEqual(artifact["primary_anchor_entities"], ["Ermengarde of Tours"])

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


class RetrievalConfigTest(unittest.TestCase):
    def test_load_config_uses_public_branch_and_evidence_top_k_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            config_path = project_root / "config.yaml"
            config_path.write_text(
                """
dataset:
  root: dataset
runtime:
  base_run_dir: runs
retrieval:
  walk_top_k: 9
  branch_top_k: 7
  evidence_top_k: 3
llm:
  use_mock: true
prompts:
  dir: prompts
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path, project_root)

        self.assertEqual(config.retrieval.branch_top_k, 7)
        self.assertEqual(config.retrieval.evidence_top_k, 3)
        self.assertEqual(config.retrieval.walk_top_k, 9)
        self.assertFalse(hasattr(config.retrieval, "anchor_weight"))
        self.assertFalse(hasattr(config.retrieval, "relation_weight"))
        self.assertFalse(hasattr(config.retrieval, "semantic_weight"))

    def test_load_config_tolerates_legacy_branch_top_k_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            config_path = project_root / "config.yaml"
            config_path.write_text(
                """
dataset:
  root: dataset
runtime:
  base_run_dir: runs
retrieval:
  relation_top_k: 4
  semantic_top_k: 6
  semantic_chunk_top_k: 5
  evidence_top_k: 2
  anchor_weight: 0.4
  relation_weight: 0.4
  semantic_weight: 0.2
llm:
  use_mock: true
prompts:
  dir: prompts
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path, project_root)

        self.assertEqual(config.retrieval.branch_top_k, 6)
        self.assertEqual(config.retrieval.evidence_top_k, 2)
        self.assertEqual(config.retrieval.walk_top_k, 5)


class RoutedHypergraphWalkerTest(unittest.TestCase):
    def test_local_semantic_top_k_uses_only_adjacent_hyperedges_and_stable_sort(self) -> None:
        graph = WalkGraph(
            entity_edges={
                "E": ["H1", "H2", "H3", "H4", "H5", "H6"],
                "Z": ["H_GLOBAL"],
            },
            hyperedge_entities={
                "H1": ["E", "A"],
                "H2": ["E", "B"],
                "H3": ["E", "C"],
                "H4": ["E", "D"],
                "H5": ["E", "F"],
                "H6": ["E", "G"],
                "H_GLOBAL": ["Z", "A"],
            },
        )
        store = WalkHyperedgeStore(
            {
                "H1": 0.5,
                "H2": 0.8,
                "H3": 0.8,
                "H4": 0.9,
                "H5": 0.2,
                "H6": 0.1,
                "H_GLOBAL": 1.0,
            }
        )
        embedder = CountingEmbedder()
        walker = FixedAnchorWalker(
            _walk_dataset(graph, store),
            embedder,
            RetrievalConfig(walk_top_k=5),
            anchors=["E"],
        )

        top = walker.local_semantic_top_hyperedges("question about E", "E")

        self.assertEqual([item["hyperedge_id"] for item in top], ["H4", "H2", "H3", "H1", "H5"])
        self.assertNotIn("H_GLOBAL", {item["hyperedge_id"] for item in top})
        self.assertEqual(store.calls, [["H1", "H2", "H3", "H4", "H5", "H6"]])
        self.assertEqual(len(embedder.calls), 1)

        graph.entity_edges["A"] = ["H7", "H8"]
        graph.hyperedge_entities["H7"] = ["A", "K"]
        graph.hyperedge_entities["H8"] = ["A", "L"]
        store.scores.update({"H7": 0.3, "H8": 0.4})

        short_top = walker.local_semantic_top_hyperedges("new query", "A")

        self.assertEqual([item["hyperedge_id"] for item in short_top], ["H8", "H7"])

    def test_path_construction_keeps_complete_hyperedges_and_preserves_chunks(self) -> None:
        graph = WalkGraph(
            entity_edges={
                "E": ["H1"],
                "A": ["H1", "H2"],
            },
            hyperedge_entities={
                "H1": ["E", "A", "B"],
                "H2": ["A", "E", "C"],
            },
            hyperedge_chunks={
                "H1": ["C1"],
                "H2": ["C2"],
            },
            chunk_texts={
                "C1": "E connects to A and B.",
                "C2": "A connects to C.",
            },
        )
        walker = FixedAnchorWalker(
            _walk_dataset(graph, WalkHyperedgeStore({"H1": 0.9, "H2": 0.8})),
            CountingEmbedder(),
            RetrievalConfig(walk_top_k=5),
            anchors=["E"],
        )
        root = HypergraphReasoningPath(
            path_id=walker._path_id(["E"], []),
            anchor_entity_id="E",
            entity_ids=["E"],
            hyperedge_ids=[],
            steps=[],
            hop_count=0,
        )

        one_hop, _, _, _ = walker._expand_frontier(atomic_question="question", frontier=[root], hop=1)

        self.assertEqual(len(one_hop), 1)
        path_h1 = one_hop[0]
        self.assertEqual(path_h1.entity_ids, ["E", "A", "B"])
        self.assertEqual(path_h1.expand_from_entity_ids, ["A", "B"])
        self.assertEqual(path_h1.hyperedge_ids, ["H1"])
        self.assertEqual(path_h1.steps[0].entity_ids, ["E", "A", "B"])
        self.assertEqual(path_h1.steps[0].to_entity_ids, ["A", "B"])
        self.assertEqual(path_h1.steps[0].chunk_ids, ["C1"])
        self.assertEqual(path_h1.steps[0].chunk_texts, ["E connects to A and B."])
        path_h1_payload = walker.path_payload(path_h1)
        self.assertIsNone(path_h1_payload["current_tail_entity"])
        self.assertEqual(path_h1_payload["frontier_entity_ids"], ["A", "B"])
        self.assertEqual(path_h1_payload["hyperedges"][0]["to_entity_id"], "")
        self.assertEqual(path_h1_payload["hyperedges"][0]["to_entity_ids"], ["A", "B"])

        two_hop, _, _, _ = walker._expand_frontier(atomic_question="question", frontier=[path_h1], hop=2)

        self.assertEqual(len(two_hop), 1)
        self.assertEqual(two_hop[0].entity_ids, ["E", "A", "B", "C"])
        self.assertEqual(two_hop[0].hyperedge_ids, ["H1", "H2"])
        self.assertEqual(two_hop[0].expand_from_entity_ids, ["C"])
        self.assertEqual(two_hop[0].steps[1].chunk_ids, ["C2"])
        self.assertEqual(two_hop[0].path_id, walker._path_id(["E", "A", "B", "C"], ["H1", "H2"]))

    def test_first_hop_answer_stops_without_answer_generation_inside_walker(self) -> None:
        graph = WalkGraph(
            entity_edges={"E": ["H1", "H2"], "A": ["H3"]},
            hyperedge_entities={"H1": ["E", "A"], "H2": ["E", "X"], "H3": ["A", "B"]},
        )
        walker = FixedAnchorWalker(
            _walk_dataset(graph, WalkHyperedgeStore({"H1": 1.0, "H2": 0.9, "H3": 0.8})),
            CountingEmbedder(),
            RetrievalConfig(walk_top_k=5),
            anchors=["E"],
            llm_service=MockAtomicLLMService(
                route_responses=[
                    {
                        "labels": [
                            {
                                "path_id": walker_path_id(["E", "A"], ["H1"]),
                                "label": "ANSWER",
                                "answer_entity_ids": ["A"],
                                "reason": "H1 answers.",
                            },
                            {
                                "path_id": walker_path_id(["E", "X"], ["H2"]),
                                "label": "EXPAND",
                                "answer_entity_ids": [],
                                "reason": "H2 is a prefix.",
                            },
                        ]
                    }
                ],
                path_answer_responses=[{"answer": "should not be used"}],
            ),
        )

        result = walker.run_atomic_walk("Who is connected to E?", AtomicQuestionAnalysis(entities=["E"]), [], node_id="q1")

        self.assertEqual(result.evidence_mode, "routed_answer")
        self.assertEqual([path.path_id for path in result.selected_paths], [walker_path_id(["E", "A"], ["H1"])])
        self.assertEqual(len(walker.llm_service.route_calls), 1)
        self.assertEqual(walker.llm_service.path_answer_calls, [])

    def test_first_hop_expand_only_drives_second_hop_and_drop_does_not_expand(self) -> None:
        graph = WalkGraph(
            entity_edges={"E": ["H1", "H2"], "A": ["H1", "H3"], "X": ["H2", "H4"]},
            hyperedge_entities={"H1": ["E", "A"], "H2": ["E", "X"], "H3": ["A", "B"], "H4": ["X", "Y"]},
        )
        llm = MockAtomicLLMService(
            route_responses=[
                {
                    "labels": [
                        {
                            "path_id": walker_path_id(["E", "A"], ["H1"]),
                            "label": "EXPAND",
                            "answer_entity_ids": [],
                            "reason": "A may lead to answer.",
                        },
                        {
                            "path_id": walker_path_id(["E", "X"], ["H2"]),
                            "label": "DROP",
                            "answer_entity_ids": [],
                            "reason": "X is irrelevant.",
                        },
                    ]
                },
                {
                    "labels": [
                        {
                            "path_id": walker_path_id(["E", "A", "B"], ["H1", "H3"]),
                            "label": "ANSWER",
                            "answer_entity_ids": ["B"],
                            "reason": "Two-hop path answers.",
                        }
                    ]
                },
            ]
        )
        walker = FixedAnchorWalker(
            _walk_dataset(graph, WalkHyperedgeStore({"H1": 1.0, "H2": 0.9, "H3": 0.8, "H4": 0.99})),
            CountingEmbedder(),
            RetrievalConfig(walk_top_k=5),
            anchors=["E"],
            llm_service=llm,
        )

        result = walker.run_atomic_walk("Where is the answer for E?", AtomicQuestionAnalysis(entities=["E"]), [], node_id="q1")

        self.assertEqual(result.evidence_mode, "routed_answer")
        self.assertEqual([path.path_id for path in result.selected_paths], [walker_path_id(["E", "A", "B"], ["H1", "H3"])])
        self.assertEqual(len(llm.route_calls), 2)
        second_hop_ids = {
            path["path_id"]
            for path in llm.route_calls[1]["candidate_paths"]
        }
        self.assertEqual(second_hop_ids, {walker_path_id(["E", "A", "B"], ["H1", "H3"])})
        self.assertNotIn(walker_path_id(["E", "X", "Y"], ["H2", "H4"]), second_hop_ids)

    def test_second_hop_expand_fallback_and_all_drop_states(self) -> None:
        graph = WalkGraph(
            entity_edges={"E": ["H1"], "A": ["H1", "H2"]},
            hyperedge_entities={"H1": ["E", "A"], "H2": ["A", "B"]},
        )

        fallback_walker = FixedAnchorWalker(
            _walk_dataset(graph, WalkHyperedgeStore({"H1": 1.0, "H2": 0.8})),
            CountingEmbedder(),
            RetrievalConfig(walk_top_k=5),
            anchors=["E"],
            llm_service=MockAtomicLLMService(
                route_responses=[
                    {"labels": [{"path_id": walker_path_id(["E", "A"], ["H1"]), "label": "EXPAND", "answer_entity_ids": [], "reason": ""}]},
                    {"labels": [{"path_id": walker_path_id(["E", "A", "B"], ["H1", "H2"]), "label": "EXPAND", "answer_entity_ids": [], "reason": ""}]},
                ]
            ),
        )

        fallback = fallback_walker.run_atomic_walk("Find B from E", AtomicQuestionAnalysis(entities=["E"]), [], node_id="q1")

        self.assertEqual(fallback.evidence_mode, "second_hop_expand_fallback")
        self.assertFalse(fallback.insufficient)
        self.assertEqual(fallback.selected_paths[0].path_id, walker_path_id(["E", "A", "B"], ["H1", "H2"]))

        drop_walker = FixedAnchorWalker(
            _walk_dataset(graph, WalkHyperedgeStore({"H1": 1.0, "H2": 0.8})),
            CountingEmbedder(),
            RetrievalConfig(walk_top_k=5),
            anchors=["E"],
            llm_service=MockAtomicLLMService(
                route_responses=[
                    {"labels": [{"path_id": walker_path_id(["E", "A"], ["H1"]), "label": "EXPAND", "answer_entity_ids": [], "reason": ""}]},
                    {"labels": [{"path_id": walker_path_id(["E", "A", "B"], ["H1", "H2"]), "label": "DROP", "answer_entity_ids": [], "reason": ""}]},
                ]
            ),
        )

        dropped = drop_walker.run_atomic_walk("Find B from E", AtomicQuestionAnalysis(entities=["E"]), [], node_id="q1")

        self.assertTrue(dropped.insufficient)
        self.assertEqual(dropped.evidence_mode, "insufficient")
        self.assertEqual(dropped.selected_paths, [])

    def test_router_validation_falls_back_to_expand_and_records_errors(self) -> None:
        graph = WalkGraph(
            entity_edges={"E": ["H1", "H2"]},
            hyperedge_entities={"H1": ["E", "A"], "H2": ["E", "B"]},
        )
        walker = FixedAnchorWalker(
            _walk_dataset(graph, WalkHyperedgeStore({"H1": 1.0, "H2": 0.9})),
            CountingEmbedder(),
            RetrievalConfig(walk_top_k=5),
            anchors=["E"],
        )
        root = HypergraphReasoningPath(
            path_id=walker_path_id(["E"], []),
            anchor_entity_id="E",
            entity_ids=["E"],
            hyperedge_ids=[],
            steps=[],
            hop_count=0,
        )
        candidates, _, _, _ = walker._expand_frontier(atomic_question="question", frontier=[root], hop=1)

        routed, labels, errors = walker._apply_router_labels(
            candidates,
            {
                "labels": [
                    {"path_id": walker_path_id(["E", "A"], ["H1"]), "label": "BAD", "answer_entity_ids": [], "reason": "bad"},
                    {"path_id": walker_path_id(["E", "B"], ["H2"]), "label": "ANSWER", "answer_entity_ids": ["OUTSIDE"], "reason": "bad"},
                    {"path_id": "UNKNOWN", "label": "DROP", "answer_entity_ids": [], "reason": "bad"},
                ]
            },
        )

        self.assertEqual({path.label for path in routed}, {"EXPAND"})
        self.assertEqual({item["label"] for item in labels}, {"EXPAND"})
        self.assertGreaterEqual(len(errors), 3)
        self.assertTrue(any(error["error"] == "answer_entity_outside_path" for error in errors))

    def test_executor_uses_dependency_answer_as_downstream_anchor_with_walker(self) -> None:
        graph = WalkGraph(
            entity_edges={"B Boy": ["H_PERFORMER"], "Meek Mill": ["H_PERFORMER", "H_DETAINED"]},
            hyperedge_entities={
                "H_PERFORMER": ["B Boy", "Meek Mill"],
                "H_DETAINED": ["Meek Mill", "Police Station"],
            },
            hyperedge_texts={
                "H_PERFORMER": "B Boy was performed by Meek Mill.",
                "H_DETAINED": "Meek Mill was detained at Police Station.",
            },
        )
        llm = MockAtomicLLMService(
            route_responses=[
                {
                    "labels": [
                        {
                            "path_id": walker_path_id(["B Boy", "Meek Mill"], ["H_PERFORMER"]),
                            "label": "ANSWER",
                            "answer_entity_ids": ["Meek Mill"],
                            "reason": "Performer path answers.",
                        }
                    ]
                },
                {
                    "labels": [
                        {
                            "path_id": walker_path_id(["Meek Mill", "Police Station"], ["H_DETAINED"]),
                            "label": "ANSWER",
                            "answer_entity_ids": ["Police Station"],
                            "reason": "Detention path answers.",
                        }
                    ]
                },
            ],
            path_answer_responses=[
                {
                    "answer": "Meek Mill",
                    "confidence": 0.9,
                    "reasoning_summary": "B Boy was performed by Meek Mill.",
                    "used_path_ids": [walker_path_id(["B Boy", "Meek Mill"], ["H_PERFORMER"])],
                    "used_hyperedge_ids": ["H_PERFORMER"],
                    "insufficient": False,
                },
                {
                    "answer": "Police Station",
                    "confidence": 0.9,
                    "reasoning_summary": "Meek Mill was detained at Police Station.",
                    "used_path_ids": [walker_path_id(["Meek Mill", "Police Station"], ["H_DETAINED"])],
                    "used_hyperedge_ids": ["H_DETAINED"],
                    "insufficient": False,
                },
            ],
        )
        walker = FixedAnchorWalker(
            _walk_dataset(graph, WalkHyperedgeStore({"H_PERFORMER": 1.0, "H_DETAINED": 1.0})),
            CountingEmbedder(),
            RetrievalConfig(walk_top_k=5),
            anchors=[],
            llm_service=llm,
        )
        executor = AtomicDagExecutor(
            analyzer=DependencyAnchorAnalyzer(),
            retriever=None,
            fusion=None,
            composer=StaticComposer(),
            llm_service=llm,
            walker=walker,
        )
        dag = {
            "nodes": [
                {"node_id": "q1", "question": "Who performed the song B Boy?", "dependencies": []},
                {"node_id": "q2", "question": "Where was q1's answer detained?", "dependencies": ["q1"]},
            ]
        }

        result = executor.run("Where was the performer of B Boy detained?", dag)

        self.assertEqual(result.atomic_results[1].question, "Where was Meek Mill detained?")
        self.assertEqual(result.atomic_results[1].answer, "Police Station")
        self.assertEqual(result.artifacts["atomic_question_analyses"][1]["primary_anchor_entities"], ["Meek Mill"])
        self.assertEqual(result.artifacts["atomic_retrieval"][1]["resolved_anchor_entity_ids"], ["Meek Mill"])
        self.assertEqual(len(llm.path_answer_calls), 2)

    def test_path_answerer_receives_prompt_contract_payload(self) -> None:
        graph = WalkGraph(
            entity_edges={"Subject": ["H_ANSWER"]},
            hyperedge_entities={"H_ANSWER": ["Subject", "Answer Entity"]},
            hyperedge_texts={"H_ANSWER": "Subject was connected to Answer Entity."},
            hyperedge_chunks={"H_ANSWER": ["C_ANSWER"]},
            chunk_texts={"C_ANSWER": "Subject was connected to Answer Entity in the source chunk."},
        )
        llm = MockAtomicLLMService(
            route_responses=[
                {
                    "labels": [
                        {
                            "path_id": walker_path_id(["Subject", "Answer Entity"], ["H_ANSWER"]),
                            "label": "ANSWER",
                            "answer_entity_ids": ["Answer Entity"],
                            "reason": "The path answers.",
                        }
                    ]
                }
            ],
            path_answer_responses=[
                {
                    "answer": "Answer Entity",
                    "confidence": 0.91,
                    "reasoning_summary": "The selected path supports Answer Entity.",
                    "used_path_ids": [walker_path_id(["Subject", "Answer Entity"], ["H_ANSWER"])],
                    "used_hyperedge_ids": ["H_ANSWER"],
                    "insufficient": False,
                }
            ],
        )
        walker = FixedAnchorWalker(
            _walk_dataset(graph, WalkHyperedgeStore({"H_ANSWER": 1.0})),
            CountingEmbedder(),
            RetrievalConfig(walk_top_k=5),
            anchors=["Subject"],
            llm_service=llm,
        )
        executor = AtomicDagExecutor(
            analyzer=StaticAnalyzer(["Subject"]),
            retriever=None,
            fusion=None,
            composer=StaticComposer(),
            llm_service=llm,
            walker=walker,
        )

        result = executor.run("Who is connected to Subject?", None)

        self.assertEqual(result.atomic_results[0].answer, "Answer Entity")
        path_payload = llm.path_answer_calls[0]["paths"][0]
        self.assertEqual(path_payload["label"], "ANSWER")
        self.assertEqual(path_payload["answer_entity_ids"], ["Answer Entity"])
        self.assertIn("entity_path", path_payload)
        answer_entity = next(entity for entity in path_payload["entity_path"] if entity["entity_id"] == "Answer Entity")
        self.assertEqual(answer_entity["label"], "Answer Entity")
        self.assertIn("entity_type", answer_entity)
        self.assertIn("description", answer_entity)
        hyperedge = path_payload["hyperedges"][0]
        self.assertEqual(hyperedge["hyperedge_id"], "H_ANSWER")
        self.assertEqual(hyperedge["chunk_ids"], ["C_ANSWER"])
        self.assertEqual(hyperedge["chunk_texts"], ["Subject was connected to Answer Entity in the source chunk."])
        self.assertEqual(hyperedge["to_entity_id"], "Answer Entity")
        self.assertEqual(hyperedge["to_entity_ids"], ["Answer Entity"])
        self.assertEqual(path_payload["frontier_entity_ids"], ["Answer Entity"])


class WalkGraph:
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
        self.hyperedge_chunks = {
            hyperedge_id: list(chunk_ids)
            for hyperedge_id, chunk_ids in (hyperedge_chunks or {}).items()
        }
        self.chunk_texts = dict(chunk_texts or {})
        entity_ids = set(self.entity_edges)
        for values in self.hyperedge_entities.values():
            entity_ids.update(values)
        self.nodes = {
            entity_id: GraphNode(node_id=entity_id, role="entity", entity_type="entity", description=entity_id)
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
        chunk_ids = self.hyperedge_chunks.get(hyperedge_id, [f"chunk-{hyperedge_id}"])
        return {
            "hyperedge_id": hyperedge_id,
            "hyperedge_text": self.hyperedge_texts.get(hyperedge_id, hyperedge_id),
            "entity_ids": list(self.hyperedge_entities.get(hyperedge_id, [])),
            "chunk_ids": list(chunk_ids),
        }


class WalkHyperedgeStore:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = dict(scores)
        self.calls: list[list[str]] = []

    def similarities(self, query_vector, row_ids: list[str]) -> list[tuple[str, float]]:
        del query_vector
        self.calls.append(list(row_ids))
        return [(row_id, float(self.scores.get(row_id, 0.0))) for row_id in row_ids]


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def embed_texts(self, texts: list[str], stage: str | None = None):
        self.calls.append((list(texts), stage))
        return [np.ones(3, dtype=np.float32) for _ in texts]


class FixedAnchorWalker(RoutedHypergraphWalker):
    def __init__(self, dataset, embedder, config, *, anchors: list[str], llm_service=None) -> None:
        self.fixed_anchors = list(anchors)
        super().__init__(
            dataset=dataset,
            embedder=embedder,
            config=config,
            llm_service=llm_service,
            logger=logging.getLogger("test.routed_walker"),
        )

    def _resolve_anchor_entities(self, question: str, analysis: AtomicQuestionAnalysis) -> list[dict[str, object]]:
        del question
        source_entities = self.fixed_anchors or list(analysis.entities)
        anchors: list[dict[str, object]] = []
        for index, entity_id in enumerate(source_entities):
            if not self.dataset.graph.entity_hyperedge_ids(entity_id):
                continue
            anchors.append(
                {
                    "query_index": index,
                    "mention": entity_id,
                    "entity_id": entity_id,
                    "label": entity_id,
                    "match_type": "test_exact",
                    "link_score": 1.0,
                    "vector_score": 1.0,
                    "llm_confidence": 0.0,
                }
            )
        return anchors


class DependencyAnchorAnalyzer:
    def analyze(self, atomic_question: str, dependency_answers=None) -> AtomicQuestionAnalysis:
        del dependency_answers
        if "B Boy" in atomic_question:
            return AtomicQuestionAnalysis(entities=["B Boy"], relations=["performed"], answer_type="person")
        if "Meek Mill" in atomic_question:
            return AtomicQuestionAnalysis(entities=[], relations=["detained"], answer_type="place")
        return AtomicQuestionAnalysis()


class StaticAnalyzer:
    def __init__(self, entities: list[str]) -> None:
        self.entities = list(entities)

    def analyze(self, atomic_question: str, dependency_answers=None) -> AtomicQuestionAnalysis:
        del atomic_question, dependency_answers
        return AtomicQuestionAnalysis(entities=list(self.entities), relations=["relation"], answer_type="entity")


def _walk_dataset(graph: WalkGraph, store: WalkHyperedgeStore):
    return SimpleNamespace(
        graph=graph,
        hyperedge_store=store,
        entity_store=None,
        chunk_store=None,
        text_chunks={chunk_id: {"content": text} for chunk_id, text in graph.chunk_texts.items()},
        full_docs={},
        summary={},
        get_chunk_text=lambda chunk_id: graph.chunk_texts.get(chunk_id, f"text for {chunk_id}"),
    )


def walker_path_id(entity_ids: list[str], hyperedge_ids: list[str]) -> str:
    import hashlib

    signature = "|".join([*entity_ids, "=>", *hyperedge_ids])
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:14]
    return f"p{len(hyperedge_ids)}_{digest}"


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

    def test_consensus_bucket_order_prioritizes_multi_branch_support(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["Entity A"], relations=["relation"], relation_query="relation")
        hits = [
            BranchHit("ar", "anchor", 1.0, "Entity A relation", entity_ids=["Entity A"]),
            BranchHit("ar", "relation", 1.0, "Entity A relation"),
            BranchHit("rs", "relation", 1.0, "Entity A relation"),
            BranchHit("rs", "semantic", 1.0, "Entity A relation"),
            BranchHit("as", "anchor", 1.0, "Entity A relation", entity_ids=["Entity A"]),
            BranchHit("as", "semantic", 1.0, "Entity A relation"),
            BranchHit("ars", "anchor", 1.0, "Entity A relation", entity_ids=["Entity A"]),
            BranchHit("ars", "relation", 1.0, "Entity A relation"),
            BranchHit("ars", "semantic", 1.0, "Entity A relation"),
        ]

        candidates = fusion.fuse("question relation", analysis, hits, top_k=4)

        self.assertEqual([candidate.hyperedge_id for candidate in candidates], ["ars", "ar", "as", "rs"])
        self.assertEqual([candidate.score_breakdown["selection_bucket"] for candidate in candidates], ["A_R_S", "A_R", "A_S", "R_S"])

    def test_anchor_residual_completion_ignores_single_non_anchor_branches(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["Entity A"], relations=["born"], relation_query="born")
        hits = [
            BranchHit("rs", "relation", 1.0, "Entity A born in city"),
            BranchHit("rs", "semantic", 1.0, "Entity A born in city"),
            BranchHit("a_low", "anchor", 1.0, "Entity A unrelated", entity_ids=["Entity A"]),
            BranchHit("a_high", "anchor", 1.0, "Entity A born in city", entity_ids=["Entity A"]),
            BranchHit("relation_only", "relation", 1.0, "Entity A born in city"),
            BranchHit("semantic_only", "semantic", 1.0, "Entity A born in city"),
        ]

        candidates = fusion.fuse("Who was born in city?", analysis, hits, top_k=5)

        self.assertEqual([candidate.hyperedge_id for candidate in candidates], ["rs", "a_high", "a_low"])
        self.assertNotIn("relation_only", {candidate.hyperedge_id for candidate in candidates})
        self.assertNotIn("semantic_only", {candidate.hyperedge_id for candidate in candidates})
        self.assertEqual(candidates[1].score_breakdown["selection_bucket"], "A_residual")
        self.assertEqual(candidates[2].score_breakdown["selection_bucket"], "A_residual")

    def test_score_breakdown_uses_selection_metadata_not_weights(self) -> None:
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
        self.assertEqual(candidate.score_breakdown["selection_bucket"], "A_R_S")
        self.assertEqual(candidate.score_breakdown["branch_support_count"], 3)
        self.assertNotIn("anchor_weight", candidate.score_breakdown)
        self.assertNotIn("relation_weight", candidate.score_breakdown)
        self.assertNotIn("semantic_weight", candidate.score_breakdown)

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
                "anchor",
                0.9,
                "Apple appears in text.",
                entity_ids=["Apple tree"],
            ),
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
                branch="semantic",
                raw_score=0.9,
                hyperedge_text="unrelated",
            ),
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
            BranchHit("h1", "anchor", 0.9, "unrelated", entity_ids=["Entity A"]),
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


class RelationGraph:
    def __init__(self, relation_texts_by_hyperedge: dict[str, str]) -> None:
        self.nodes = {
            hyperedge_id: GraphNode(
                node_id=hyperedge_id,
                role="hyperedge",
                description=relation_text,
            )
            for hyperedge_id, relation_text in relation_texts_by_hyperedge.items()
        }
        self.adjacency: dict[str, list[str]] = {}
        self.edges: dict[str, object] = {}
        self._relation_texts_by_hyperedge = dict(relation_texts_by_hyperedge)

    def entity_hyperedge_ids(self, entity_id: str) -> list[str]:
        del entity_id
        return []

    def describe_hyperedge(self, hyperedge_id: str) -> dict[str, object]:
        return {
            "hyperedge_id": hyperedge_id,
            "hyperedge_text": self._relation_texts_by_hyperedge.get(hyperedge_id, hyperedge_id),
            "entity_ids": [],
            "chunk_ids": [],
        }


def _relation_retriever(
    *,
    relation_texts_by_hyperedge: dict[str, str],
    matches_by_query: dict[str, list[VectorMatch]],
    branch_top_k: int = 10,
) -> AtomicHyperedgeRetriever:
    dataset = SimpleNamespace(
        graph=RelationGraph(relation_texts_by_hyperedge),
        hyperedge_store=MappingHyperedgeStore(matches_by_query),
        chunk_store=MappingChunkStore({}),
        text_chunks={},
        get_chunk_text=lambda chunk_id: "",
    )
    return AtomicHyperedgeRetriever(
        dataset=dataset,
        embedder=TextEmbedder(),
        config=RetrievalConfig(branch_top_k=branch_top_k),
        logger=logging.getLogger("test.atomic_retriever"),
    )


class AtomicRetrieverTest(unittest.TestCase):
    def test_atomic_analysis_prompt_defines_relation_query_as_signature(self) -> None:
        prompt = Path("prompts/atomic_question_analysis.md").read_text(encoding="utf-8")

        self.assertIn("relation_query is a compact predicate signature", prompt)
        self.assertIn("relation_query is not a natural-language question", prompt)
        self.assertIn("Do not include question words", prompt)
        self.assertIn("mother parent female parent", prompt)
        self.assertIn("date of death death date died on", prompt)

    def test_relation_signature_seeds_use_relations_before_relation_query(self) -> None:
        retriever = _relation_retriever(
            relation_texts_by_hyperedge={},
            matches_by_query={},
        )

        seeds = retriever._relation_signature_seeds(
            AtomicQuestionAnalysis(
                relations=["mother", "parent"],
                relation_query="who was the mother of a historical figure",
            )
        )

        self.assertEqual(seeds, ["mother", "parent"])

    def test_relation_signature_seeds_deduplicate_preserving_order(self) -> None:
        retriever = _relation_retriever(
            relation_texts_by_hyperedge={},
            matches_by_query={},
        )

        seeds = retriever._relation_signature_seeds(
            AtomicQuestionAnalysis(
                relations=["mother", "mother ", "Parent", "parent", "female parent"],
                relation_query="mother parent female parent",
            )
        )

        self.assertEqual(seeds, ["mother", "Parent", "female parent"])

    def test_relation_branch_retrieves_from_multiple_relation_seeds(self) -> None:
        retriever = _relation_retriever(
            relation_texts_by_hyperedge={
                "H_DEATH": "date of death",
                "H_MOTHER": "mother parent",
            },
            matches_by_query={
                "date of death": [VectorMatch(item_id="H_DEATH", label="H_DEATH", score=0.4)],
                "mother": [VectorMatch(item_id="H_MOTHER", label="H_MOTHER", score=0.3)],
            },
        )

        hits = retriever.retrieve_relation_branch(
            AtomicQuestionAnalysis(relations=["date of death", "mother"], relation_query="ignored query")
        )

        self.assertEqual({hit.hyperedge_id for hit in hits}, {"H_DEATH", "H_MOTHER"})
        self.assertEqual({hit.branch for hit in hits}, {"relation"})
        self.assertEqual(retriever.dataset.hyperedge_store.calls, [("date of death", 10), ("mother", 10)])

    def test_relation_branch_ranks_date_of_death_relation_text_above_weak_vector_match(self) -> None:
        retriever = _relation_retriever(
            relation_texts_by_hyperedge={
                "H_WEAK": "film release date",
                "H_DEATH": "date of death death date died on",
            },
            matches_by_query={
                "date of death": [
                    VectorMatch(item_id="H_WEAK", label="H_WEAK", score=0.99),
                    VectorMatch(item_id="H_DEATH", label="H_DEATH", score=0.2),
                ]
            },
        )

        hits = retriever.retrieve_relation_branch(
            AtomicQuestionAnalysis(relations=["date of death"], relation_query="date of death death date died on")
        )

        self.assertEqual(hits[0].hyperedge_id, "H_DEATH")
        self.assertEqual(hits[0].branch, "relation")
        self.assertEqual(hits[0].metadata["best_relation_seed"], "date of death")
        self.assertGreater(hits[0].metadata["lexical_relation_score"], hits[1].metadata["lexical_relation_score"])

    def test_relation_branch_ranks_mother_parent_relation_text_above_unrelated_candidate(self) -> None:
        retriever = _relation_retriever(
            relation_texts_by_hyperedge={
                "H_UNRELATED": "director film director",
                "H_PARENT": "mother parent female parent",
            },
            matches_by_query={
                "mother": [VectorMatch(item_id="H_UNRELATED", label="H_UNRELATED", score=0.99)],
                "parent": [VectorMatch(item_id="H_PARENT", label="H_PARENT", score=0.1)],
            },
        )

        hits = retriever.retrieve_relation_branch(
            AtomicQuestionAnalysis(relations=["mother", "parent"], relation_query="mother parent female parent")
        )

        self.assertEqual(hits[0].hyperedge_id, "H_PARENT")
        self.assertEqual(hits[0].branch, "relation")
        self.assertEqual(hits[0].metadata["relation_retrieval_mode"], "signature")

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
            config=RetrievalConfig(branch_top_k=10),
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

    def test_semantic_branch_caps_final_hyperedges_by_branch_top_k(self) -> None:
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
            config=RetrievalConfig(branch_top_k=2),
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

    def test_anchor_branch_caps_final_hyperedges_by_branch_top_k(self) -> None:
        dataset = SimpleNamespace(
            graph=AnchorGraph({"h3": ["China"], "h1": ["China"], "h2": ["China"]}),
            hyperedge_store=MappingHyperedgeStore({}),
            get_chunk_text=lambda chunk_id: "",
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=None,
            config=RetrievalConfig(branch_top_k=2),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve_anchor_branch("What is China?", AtomicQuestionAnalysis(entities=["China"]))

        self.assertEqual([hit.hyperedge_id for hit in hits], ["h1", "h2"])

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
            config=RetrievalConfig(),
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

    def test_anchor_score_uses_only_exact_entity_id_coverage(self) -> None:
        fusion = AtomicEvidenceFusion(config=RetrievalConfig(), embedder=None)
        analysis = AtomicQuestionAnalysis(entities=["Apple"], relation_query="contains Apple")
        hits = [
            BranchHit(
                hyperedge_id="h1",
                branch="anchor",
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
            config=RetrievalConfig(branch_top_k=10),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        hits = retriever.retrieve(
            "semantic question",
            AtomicQuestionAnalysis(entities=["ENTITY A"], relation_query="relation query"),
        )

        self.assertEqual({hit.branch for hit in hits}, {"anchor", "relation", "semantic"})

    def test_relation_and_semantic_branches_use_branch_top_k(self) -> None:
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
            config=RetrievalConfig(branch_top_k=10),
            logger=logging.getLogger("test.atomic_retriever"),
        )

        relation_hits = retriever.retrieve_relation_branch(AtomicQuestionAnalysis(relation_query="relation query"))
        semantic_hits = retriever.retrieve_semantic_branch("semantic question")

        self.assertEqual(len(relation_hits), 10)
        self.assertEqual(len(semantic_hits), 10)
        self.assertEqual(hyperedge_store.calls, [("relation query", 10)])
        self.assertEqual(chunk_store.calls, [("semantic question", 10)])


class FinalAnswerComposerTest(unittest.TestCase):
    def test_final_synthesis_uses_single_llm_stage(self) -> None:
        llm = TwoStageLLM()
        composer = FinalAnswerComposer(llm)
        result = AtomicAnswerResult(
            node_id="q1",
            question="Was Alpha or Beta born first?",
            analysis=AtomicQuestionAnalysis(),
            evidence=[
                FusedHyperedgeCandidate(
                    hyperedge_id="h1",
                    hyperedge_text="A was born in 1900. B was born in 1910.",
                    branch_support={"semantic"},
                )
            ],
            answer="Alpha was born first in 1900.",
            confidence=0.9,
            reasoning_summary="A predates B.",
            used_hyperedge_ids=["h1"],
        )

        payload = composer.compose(
            "Was Alpha or Beta born first?",
            [result],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="Was Alpha or Beta born first?")],
        )

        self.assertEqual(llm.calls, ["compose"])
        self.assertEqual(payload["candidate_answer"], "Alpha was born first in 1900.")
        self.assertEqual(payload["answer"], "Alpha")
        self.assertEqual(payload["atomic_answer_trace"][0]["node_id"], "q1")
        self.assertFalse(hasattr(llm, "span_payload"))

    def test_final_answer_payload_schema_is_backward_compatible(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("Stockholm"))

        payload = composer.compose(
            "What is the place of birth of Olof Palme?",
            [_atomic_result("q1", "What is the place of birth of Olof Palme?", "Stockholm, Sweden")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="What is the place of birth of Olof Palme?")],
        )

        for key in (
            "answer",
            "candidate_answer",
            "reasoning_summary",
            "answer_span_reasoning",
            "confidence",
            "atomic_answer_trace",
            "remaining_gaps",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["semantic_answer"], "Stockholm")
        self.assertIsNone(payload["judgment"])

    def test_single_stage_canonicalizes_location_spans(self) -> None:
        composer = FinalAnswerComposer(RuleBasedFinalLLM())

        cases = [
            (
                "What is the place of birth of Olof Palme?",
                "Stockholm, Sweden",
                "Stockholm",
            ),
            (
                "Where did Isaac Schwartz die?",
                "Siversky, near Saint Petersburg, Russian Federation",
                "Siversky",
            ),
        ]
        for question, atomic_answer, expected in cases:
            with self.subTest(question=question):
                payload = composer.compose(
                    question,
                    [_atomic_result("q1", question, atomic_answer)],
                    dag_nodes=[AtomicQuestionNode(node_id="q1", question=question)],
                )
                self.assertEqual(payload["answer"], expected)

    def test_single_stage_canonicalizes_institution_alias(self) -> None:
        composer = FinalAnswerComposer(RuleBasedFinalLLM())

        payload = composer.compose(
            "Where does John Tyndall work?",
            [_atomic_result("q1", "Where does John Tyndall work?", "Royal Institution of Great Britain")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="Where does John Tyndall work?")],
        )

        self.assertEqual(payload["answer"], "Royal Institution")

    def test_postprocess_canonicalizes_raw_single_stage_spans(self) -> None:
        cases = [
            (
                "What is the place of birth of Olof Palme?",
                "Stockholm, Sweden",
                "Stockholm",
            ),
            (
                "Where did Isaac Schwartz die?",
                "Siversky, near Saint Petersburg, Russian Federation",
                "Siversky",
            ),
            (
                "Where was Corina from?",
                "Manhattan, New York",
                "Manhattan",
            ),
            (
                "Where does John Tyndall work?",
                "Royal Institution of Great Britain",
                "Royal Institution",
            ),
            (
                "When was Nicholas the Small born?",
                "1322/1327",
                "1322",
            ),
            (
                "What nationality was the composer?",
                "Austria",
                "Austria",
            ),
            (
                "Which song was released first, Where Does It Hurt? or Other Song?",
                "Does It Hurt",
                "Where Does It Hurt",
            ),
        ]
        for question, raw_answer, expected in cases:
            with self.subTest(question=question):
                composer = FinalAnswerComposer(StaticFinalLLM(raw_answer))
                payload = composer.compose(
                    question,
                    [_atomic_result("q1", question, raw_answer)],
                    dag_nodes=[AtomicQuestionNode(node_id="q1", question=question)],
                )
                self.assertEqual(payload["answer"], expected)

    def test_postprocess_preserves_country_surface_for_nationality_questions(self) -> None:
        cases = [
            ("What nationality is Beatrice I's husband?", "Germany", "German"),
            ("What nationality is the performer?", "France", "French"),
        ]
        for question, atomic_answer, llm_answer in cases:
            with self.subTest(question=question):
                composer = FinalAnswerComposer(StaticFinalLLM(llm_answer))
                payload = composer.compose(
                    question,
                    [_atomic_result("q1", question, atomic_answer)],
                    dag_nodes=[AtomicQuestionNode(node_id="q1", question=question)],
                )
                self.assertEqual(payload["answer"], atomic_answer)
                self.assertTrue(payload["deterministic_terminal_surface_preserved"])

    def test_postprocess_preserves_supported_terminal_surface_for_unseen_alias(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("John Smith"))

        payload = composer.compose(
            "Who founded Example Institute?",
            [_atomic_result("q1", "Who founded Example Institute?", "Jonathan Smith")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="Who founded Example Institute?")],
        )

        self.assertEqual(payload["answer"], "Jonathan Smith")
        self.assertEqual(payload["candidate_answer"], "Jonathan Smith")
        self.assertTrue(payload["deterministic_terminal_surface_preserved"])

    def test_postprocess_does_not_create_unseen_country_or_demonym_surface(self) -> None:
        cases = [
            ("What nationality is Beatrice I's husband?", "Germany"),
            ("What nationality is the performer?", "France"),
        ]
        for question, raw_answer in cases:
            with self.subTest(question=question):
                composer = FinalAnswerComposer(StaticFinalLLM(raw_answer))
                payload = composer.compose(
                    question,
                    [_atomic_result("q1", question, raw_answer)],
                    dag_nodes=[AtomicQuestionNode(node_id="q1", question=question)],
                )
                self.assertEqual(payload["answer"], raw_answer)
                self.assertNotIn("deterministic_nationality_normalized", payload)

    def test_final_synthesis_corrects_born_first_from_atomic_dates(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("El Tonto"))
        dag_nodes = [
            AtomicQuestionNode(node_id="q1", question="Who is the director of El Tonto?"),
            AtomicQuestionNode(node_id="q2", question="Who is the director of The Heart Of Doreon?"),
            AtomicQuestionNode(node_id="q3", question="When was q1's answer born?", dependencies=["q1"]),
            AtomicQuestionNode(node_id="q4", question="When was q2's answer born?", dependencies=["q2"]),
        ]
        results = [
            _atomic_result("q1", "Who is the director of El Tonto?", "Charlie Day"),
            _atomic_result("q2", "Who is the director of The Heart Of Doreon?", "Robert North Bradbury"),
            _atomic_result("q3", "When was Charlie Day born?", "February 9, 1976", dependencies=["q1"]),
            _atomic_result("q4", "When was Robert North Bradbury born?", "March 23, 1886", dependencies=["q2"]),
        ]

        payload = composer.compose(
            "Which film whose director was born first, El Tonto or The Heart Of Doreon?",
            results,
            dag_nodes=dag_nodes,
        )

        self.assertEqual(payload["answer"], "The Heart Of Doreon")
        self.assertEqual(payload["deterministic_final_correction"]["selected_node_id"], "q4")

    def test_final_synthesis_corrects_older_from_birth_years(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("Airheads"))
        dag_nodes = [
            AtomicQuestionNode(node_id="q1", question="Who is the director of Airheads?"),
            AtomicQuestionNode(node_id="q2", question="Who is the director of Return To Cabin By The Lake?"),
            AtomicQuestionNode(node_id="q3", question="When was q1's answer born?", dependencies=["q1"]),
            AtomicQuestionNode(node_id="q4", question="When was q2's answer born?", dependencies=["q2"]),
        ]
        results = [
            _atomic_result("q1", "Who is the director of Airheads?", "Michael Lehmann"),
            _atomic_result("q2", "Who is the director of Return To Cabin By The Lake?", "Po-Chih Leong"),
            _atomic_result("q3", "When was Michael Lehmann born?", "1957", dependencies=["q1"]),
            _atomic_result("q4", "When was Po-Chih Leong born?", "1939", dependencies=["q2"]),
        ]

        payload = composer.compose(
            "Which film has the director who is older, Airheads or Return To Cabin By The Lake?",
            results,
            dag_nodes=dag_nodes,
        )

        self.assertEqual(payload["answer"], "Return To Cabin By The Lake")

    def test_final_span_selects_minimal_candidate_name(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("Phoolwari was released first in 1946."))

        payload = composer.compose(
            "Which film was released first, Aas Ka Panchhi or Phoolwari?",
            [_atomic_result("q1", "When was Phoolwari released?", "1946")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="When was Phoolwari released?")],
        )

        self.assertEqual(payload["answer"], "Phoolwari")

    def test_final_synthesis_same_question_compares_terminal_branch_answers(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("yes"))
        dag_nodes = [
            AtomicQuestionNode(node_id="q1", question="Who is the director of Film A?"),
            AtomicQuestionNode(node_id="q2", question="What is the nationality of q1's answer?", dependencies=["q1"]),
            AtomicQuestionNode(node_id="q3", question="Who is the director of Film B?"),
            AtomicQuestionNode(node_id="q4", question="What is the nationality of q3's answer?", dependencies=["q3"]),
        ]
        results = [
            _atomic_result("q1", "Who is the director of Film A?", "Director A"),
            _atomic_result("q2", "What is the nationality of Director A?", "American", dependencies=["q1"]),
            _atomic_result("q3", "Who is the director of Film B?", "Director B"),
            _atomic_result("q4", "What is the nationality of Director B?", "Canadian", dependencies=["q3"]),
        ]

        payload = composer.compose(
            "Do director of Film A and director of Film B share the same nationality?",
            results,
            dag_nodes=dag_nodes,
        )

        self.assertEqual(payload["answer"], "no")

    def test_final_synthesis_overrides_terminal_no_for_shared_nationality_components(self) -> None:
        cases = [
            ("American", "Puerto Rican"),
            ("French", "French-Armenian"),
            ("Czech-American", "Romanian-American"),
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                composer = FinalAnswerComposer(StaticFinalLLM("no"))
                dag_nodes = [
                    AtomicQuestionNode(node_id="q1", question="What is the nationality of X?"),
                    AtomicQuestionNode(node_id="q2", question="What is the nationality of Y?"),
                    AtomicQuestionNode(
                        node_id="q3",
                        question="Based on q1's answer and q2's answer, do X and Y share the same nationality?",
                        dependencies=["q1", "q2"],
                    ),
                ]
                results = [
                    _atomic_result("q1", "What is the nationality of X?", left),
                    _atomic_result("q2", "What is the nationality of Y?", right),
                    _atomic_result(
                        "q3",
                        f"Based on {left} and {right}, do X and Y share the same nationality?",
                        "no",
                        dependencies=["q1", "q2"],
                    ),
                ]

                payload = composer.compose(
                    "Do X and Y share the same nationality?",
                    results,
                    dag_nodes=dag_nodes,
                )

                self.assertEqual(payload["answer"], "yes")
                self.assertIn("nationality component", payload["answer_span_reasoning"])

    def test_final_synthesis_uses_explicit_terminal_yes_no_answer(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("no"))
        dag_nodes = [
            AtomicQuestionNode(node_id="q1", question="Which country did Inside The Room originate from?"),
            AtomicQuestionNode(node_id="q2", question="Which country did Crude Set Drama originate from?"),
            AtomicQuestionNode(
                node_id="q3",
                question="Is q1's answer the same as q2's answer?",
                dependencies=["q1", "q2"],
            ),
        ]
        results = [
            _atomic_result("q1", "Which country did Inside The Room originate from?", "United Kingdom"),
            _atomic_result("q2", "Which country did Crude Set Drama originate from?", "United Kingdom"),
            _atomic_result("q3", "Is United Kingdom the same as United Kingdom?", "yes", dependencies=["q1", "q2"]),
        ]

        payload = composer.compose(
            "Did the movies Inside The Room and Crude Set Drama originate from the same country?",
            results,
            dag_nodes=dag_nodes,
        )

        self.assertEqual(payload["answer"], "yes")

    def test_final_synthesis_born_later_ignores_first_inside_candidate_title(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("The First Day Of Freedom"))
        dag_nodes = [
            AtomicQuestionNode(node_id="q1", question="Who is the director of The First Day Of Freedom?"),
            AtomicQuestionNode(node_id="q2", question="Who is the director of Malabimba - The Malicious Whore?"),
            AtomicQuestionNode(node_id="q3", question="When was q1's answer born?", dependencies=["q1"]),
            AtomicQuestionNode(node_id="q4", question="When was q2's answer born?", dependencies=["q2"]),
        ]
        results = [
            _atomic_result("q1", "Who is the director of The First Day Of Freedom?", "Aleksander Ford"),
            _atomic_result("q2", "Who is the director of Malabimba - The Malicious Whore?", "Andrea Bianchi"),
            _atomic_result("q3", "When was Aleksander Ford born?", "24 November 1908", dependencies=["q1"]),
            _atomic_result("q4", "When was Andrea Bianchi born?", "March 31, 1925", dependencies=["q2"]),
        ]

        payload = composer.compose(
            "Which film has the director who was born later, The First Day Of Freedom or Malabimba - The Malicious Whore?",
            results,
            dag_nodes=dag_nodes,
        )

        self.assertEqual(payload["answer"], "Malabimba - The Malicious Whore")

    def test_final_span_recovers_full_candidate_with_comma_title(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("Honor And Oh-Baby!"))

        payload = composer.compose(
            "Which film has the director who died later, Love, Honor And Oh-Baby! or I Cover The Underworld?",
            [_atomic_result("q1", "When did Charles Lamont die?", "September 12, 1993")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="When did Charles Lamont die?")],
        )

        self.assertEqual(payload["answer"], "Love, Honor And Oh-Baby!")

    def test_final_span_strips_question_auxiliary_and_recovers_accented_candidate(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("Was Juan Carlos Falc"))

        payload = composer.compose(
            "Was Angus Wagner or Juan Carlos Falcón born first?",
            [_atomic_result("q1", "When was Juan Carlos Falcón born?", "19 November 1979")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="When was Juan Carlos Falcón born?")],
        )

        self.assertEqual(payload["answer"], "Juan Carlos Falcón")

    def test_final_synthesis_candidate_label_does_not_include_question_auxiliary(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("Was Sammy Hagar"))
        dag_nodes = [
            AtomicQuestionNode(node_id="q1", question="When was Sammy Hagar born?"),
            AtomicQuestionNode(node_id="q2", question="When was Renaud Garcia-Fons born?"),
        ]
        results = [
            _atomic_result("q1", "When was Sammy Hagar born?", "October 13, 1947"),
            _atomic_result("q2", "When was Renaud Garcia-Fons born?", "December 24, 1962"),
        ]

        payload = composer.compose(
            "Was Sammy Hagar or Renaud Garcia-Fons born first?",
            results,
            dag_nodes=dag_nodes,
        )

        self.assertEqual(payload["answer"], "Sammy Hagar")

    def test_final_span_prefers_parenthetical_person_alias(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("María del Pilar Cordero"))

        payload = composer.compose(
            "Who is the spouse of the director of film My Three Merry Widows?",
            [_atomic_result("q1", "Who is the spouse of Fernando Cortés?", "María del Pilar Cordero (Mapy Cortés)")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="Who is the spouse of Fernando Cortés?")],
        )

        self.assertEqual(payload["answer"], "Mapy Cortés")

    def test_final_span_prefers_parenthetical_nationality_alias(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("Greek (Athenian)"))

        payload = composer.compose(
            "What nationality is Lamprocles's father?",
            [_atomic_result("q1", "What is the nationality of Socrates?", "Greek (Athenian)")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="What is the nationality of Socrates?")],
        )

        self.assertEqual(payload["answer"], "Athenian")

    def test_final_span_preserves_single_stage_nationality_label(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("German"))

        payload = composer.compose(
            "What nationality is Beatrice I's husband?",
            [_atomic_result("q1", "What is the nationality of Frederick Barbarossa?", "German")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="What is the nationality of Frederick Barbarossa?")],
        )

        self.assertEqual(payload["answer"], "German")

    def test_final_span_uses_primary_component_for_compound_nationality(self) -> None:
        composer = FinalAnswerComposer(StaticFinalLLM("Chinese American"))

        payload = composer.compose(
            "What nationality is the director of film Blood Street?",
            [_atomic_result("q1", "What is the nationality of Leo Fong?", "Chinese American")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="What is the nationality of Leo Fong?")],
        )

        self.assertEqual(payload["answer"], "Chinese American")

    def test_single_stage_candidate_selection_outputs_candidate_not_date(self) -> None:
        composer = FinalAnswerComposer(RuleBasedFinalLLM())
        dag_nodes = [
            AtomicQuestionNode(node_id="q1", question="When was A released?"),
            AtomicQuestionNode(node_id="q2", question="When was B released?"),
        ]
        results = [
            _atomic_result("q1", "When was A released?", "1950"),
            _atomic_result("q2", "When was B released?", "1940"),
        ]

        payload = composer.compose(
            "Which film was released first, A or B?",
            results,
            dag_nodes=dag_nodes,
        )

        self.assertEqual(payload["answer"], "B")
        self.assertNotEqual(payload["answer"], "1940")

    def test_single_stage_yes_no_judgment_outputs_yes_or_no(self) -> None:
        composer = FinalAnswerComposer(RuleBasedFinalLLM())
        dag_nodes = [
            AtomicQuestionNode(node_id="q1", question="What country is X from?"),
            AtomicQuestionNode(node_id="q2", question="What country is Y from?"),
        ]
        results = [
            _atomic_result("q1", "What country is X from?", "France"),
            _atomic_result("q2", "What country is Y from?", "Germany"),
        ]

        payload = composer.compose(
            "Are X and Y from the same country?",
            results,
            dag_nodes=dag_nodes,
        )

        self.assertEqual(payload["answer"], "no")
        self.assertEqual(payload["judgment"], "no")
        self.assertNotIn(payload["answer"], {"France", "Germany"})

    def test_single_stage_nationality_exact_same_is_conservative(self) -> None:
        composer = FinalAnswerComposer(RuleBasedFinalLLM())

        cases = [
            ("American", "Puerto Rican"),
            ("French", "French-Armenian"),
            ("Czech-American", "Romanian-American"),
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                payload = composer.compose(
                    "Do X and Y have the exact same nationality?",
                    [
                        _atomic_result("q1", "What is the nationality of X?", left),
                        _atomic_result("q2", "What is the nationality of Y?", right),
                    ],
                    dag_nodes=[
                        AtomicQuestionNode(node_id="q1", question="What is the nationality of X?"),
                        AtomicQuestionNode(node_id="q2", question="What is the nationality of Y?"),
                    ],
                )
                self.assertEqual(payload["answer"], "no")

    def test_single_stage_nationality_component_and_what_nationality_semantics(self) -> None:
        composer = FinalAnswerComposer(RuleBasedFinalLLM())

        component_payload = composer.compose(
            "Do X and Y share any nationality component?",
            [
                _atomic_result("q1", "What is the nationality of X?", "Czech-American"),
                _atomic_result("q2", "What is the nationality of Y?", "Romanian-American"),
            ],
            dag_nodes=[
                AtomicQuestionNode(node_id="q1", question="What is the nationality of X?"),
                AtomicQuestionNode(node_id="q2", question="What is the nationality of Y?"),
            ],
        )
        self.assertEqual(component_payload["answer"], "yes")

        nationality_payload = composer.compose(
            "What nationality is Henri Verneuil?",
            [_atomic_result("q1", "What nationality is Henri Verneuil?", "French-Armenian")],
            dag_nodes=[AtomicQuestionNode(node_id="q1", question="What nationality is Henri Verneuil?")],
        )
        self.assertEqual(nationality_payload["answer"], "French-Armenian")


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
            "answer": "Alpha was born first in 1900.",
            "candidate_answer": "Alpha was born first in 1900.",
            "semantic_answer": "Alpha has the earlier birth date.",
            "judgment": None,
            "reasoning_summary": "Alpha has the earlier birth date.",
            "answer_span_reasoning": "Single-stage test resolver returned the selected answer.",
            "confidence": 0.9,
            "atomic_answer_trace": [
                {
                    "node_id": "q1",
                    "question": "Was Alpha or Beta born first?",
                    "answer": "Alpha was born first in 1900.",
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


class RuleBasedFinalLLM:
    def analyze_atomic_question(self, atomic_question, dependency_answers):
        raise NotImplementedError

    def answer_atomic_question(self, atomic_question, dependency_answers, evidence):
        raise NotImplementedError

    def compose_final_answer(self, original_question, dag_nodes, atomic_results):
        del dag_nodes
        answers = [str(item.get("answer", "") or "").strip() for item in atomic_results]
        answer = self._resolve(original_question, answers)
        judgment = answer if answer in {"yes", "no"} else None
        return {
            "answer": answer,
            "candidate_answer": answer,
            "semantic_answer": answer,
            "judgment": judgment,
            "reasoning_summary": "Rule-based single-stage test resolver.",
            "answer_span_reasoning": "Rule-based test resolver returns the canonical final answer.",
            "confidence": 0.9 if answer != "INSUFFICIENT_EVIDENCE" else 0.0,
            "atomic_answer_trace": [
                {
                    "node_id": item.get("node_id", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "used_hyperedge_ids": list(item.get("used_hyperedge_ids", [])),
                }
                for item in atomic_results
            ],
            "remaining_gaps": [],
        }

    def finalize_answer_span(self, original_question, synthesis_candidate):
        raise AssertionError("finalize_answer_span should not be called in the single-stage pipeline")

    def _resolve(self, original_question: str, answers: list[str]) -> str:
        question = original_question.lower()
        usable = [answer for answer in answers if answer and answer != "INSUFFICIENT_EVIDENCE"]
        if not usable:
            return "INSUFFICIENT_EVIDENCE"
        if "royal institution of great britain" in usable[-1].lower():
            return "Royal Institution"
        if "siversky, near" in usable[-1].lower():
            return "Siversky"
        if "," in usable[-1] and any(term in question for term in ("birth", "born", "death", "die", "place")):
            return usable[-1].split(",", 1)[0].strip()
        if "released first" in question or "released earlier" in question:
            candidates = [part.strip(" ?") for part in original_question.rstrip("?").split(",", 1)[-1].split(" or ")]
            years = [int(answer) for answer in usable if answer.isdigit()]
            if len(candidates) >= 2 and len(years) >= 2:
                return candidates[0] if years[0] < years[1] else candidates[1]
        if "same country" in question or "exact same nationality" in question:
            if len(usable) >= 2:
                return "yes" if _normalize_test_label(usable[-2]) == _normalize_test_label(usable[-1]) else "no"
        if "share any nationality component" in question:
            if len(usable) >= 2:
                left = set(_nationality_components(usable[-2]))
                right = set(_nationality_components(usable[-1]))
                return "yes" if left & right else "no"
        if "what nationality" in question:
            return usable[-1]
        return usable[-1]


class StaticFinalLLM:
    def __init__(self, final_answer: str) -> None:
        self.final_answer = final_answer

    def analyze_atomic_question(self, atomic_question, dependency_answers):
        raise NotImplementedError

    def answer_atomic_question(self, atomic_question, dependency_answers, evidence):
        raise NotImplementedError

    def compose_final_answer(self, original_question, dag_nodes, atomic_results):
        return {
            "answer": self.final_answer,
            "candidate_answer": self.final_answer,
            "semantic_answer": self.final_answer,
            "judgment": self.final_answer if self.final_answer in {"yes", "no"} else None,
            "reasoning_summary": "static candidate",
            "answer_span_reasoning": "Static single-stage final resolver mirrors the selected answer.",
            "confidence": 0.6,
            "atomic_answer_trace": [
                {
                    "node_id": item.get("node_id", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "used_hyperedge_ids": item.get("used_hyperedge_ids", []),
                }
                for item in atomic_results
            ],
            "remaining_gaps": [],
        }

    def finalize_answer_span(self, original_question, synthesis_candidate):
        raise AssertionError("finalize_answer_span should not be called in the single-stage pipeline")


def _normalize_test_label(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _nationality_components(value: str) -> list[str]:
    return [
        item.strip().lower()
        for item in value.replace("-", " ").split()
        if item.strip()
    ]


def _atomic_result(
    node_id: str,
    question: str,
    answer: str,
    *,
    dependencies: list[str] | None = None,
) -> AtomicAnswerResult:
    return AtomicAnswerResult(
        node_id=node_id,
        question=question,
        analysis=AtomicQuestionAnalysis(),
        evidence=[],
        answer=answer,
        confidence=0.9,
        reasoning_summary="test atomic answer",
        used_dependencies=dependencies or [],
        used_hyperedge_ids=[],
    )


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
        final_artifact = json.loads((run_dir / "artifacts" / "final_answer.json").read_text(encoding="utf-8"))
        for key in (
            "answer",
            "candidate_answer",
            "reasoning_summary",
            "answer_span_reasoning",
            "confidence",
            "atomic_answer_trace",
            "remaining_gaps",
        ):
            self.assertIn(key, final_artifact)
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
