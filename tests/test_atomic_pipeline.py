from __future__ import annotations

import json
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
    FusedHyperedgeCandidate,
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
  local_hyperedge_hops: 2
llm:
  use_mock: true
prompts:
  dir: prompts
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path, project_root)

        self.assertEqual(config.retrieval.local_hyperedge_top_k, 4)
        self.assertEqual(config.retrieval.local_hyperedge_hops, 2)

    def test_retrieval_config_defaults_to_top3_two_hop(self) -> None:
        self.assertEqual(RetrievalConfig().local_hyperedge_top_k, 3)
        self.assertEqual(RetrievalConfig().local_hyperedge_hops, 2)

    def test_atomic_answer_prompt_allows_only_answer_key(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        prompt = (project_root / "prompts" / "atomic_answer.md").read_text(encoding="utf-8")

        self.assertIn('"answer": "..."', prompt)
        self.assertIn('"answer": "INSUFFICIENT_EVIDENCE"', prompt)
        self.assertIn('The only allowed key is "answer".', prompt)
        self.assertNotIn("answer_type", prompt)
        self.assertNotIn('"confidence"', prompt)
        self.assertNotIn('"reasoning_summary"', prompt)
        self.assertNotIn('"used_evidence_ids"', prompt)
        self.assertNotIn('"insufficient"', prompt)

    def test_mock_answer_service_returns_only_answer_key(self) -> None:
        llm = MockAtomicLLMService(
            answer_responses=[
                {
                    "answer": "Answer",
                    "confidence": 0.4,
                    "reasoning_summary": "ignored",
                    "used_evidence_ids": ["E1"],
                    "insufficient": False,
                }
            ]
        )

        response = llm.answer_atomic_question(
            atomic_question="Question?",
            answer_contract={},
            dependency_answers=[],
            evidence=[],
        )

        self.assertEqual(response, {"answer": "Answer"})


class LocalTwoHopAtomicExecutorTest(unittest.TestCase):
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
                    "used_evidence_ids": ["E2"],
                    "insufficient": False,
                },
                {
                    "answer": "Police Station",
                    "confidence": 0.9,
                    "reasoning_summary": "Meek Mill was detained at Police Station.",
                    "used_evidence_ids": ["E1"],
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
        self.assertEqual(result.final_answer["answer"], result.atomic_results[-1].answer)
        self.assertEqual(result.final_answer["source_node_id"], "q2")
        self.assertFalse(result.final_answer["insufficient"])
        self.assertEqual(len(llm.answer_calls), 2)
        self.assertEqual(llm.answer_calls[1]["atomic_question"], "Where was Meek Mill detained?")
        self.assertEqual(
            llm.answer_calls[1]["answer_contract"],
            {"output_format": "short answer only"},
        )
        self.assertEqual(llm.answer_calls[1]["dependency_answers"][0]["answer"], "Meek Mill")
        self.assertFalse(llm.answer_calls[1]["dependency_answers"][0]["insufficient"])
        self.assertEqual(
            set(llm.answer_calls[1]["dependency_answers"][0]),
            {"node_id", "question", "resolved_question", "answer", "insufficient"},
        )
        self.assertNotIn("answer_type", llm.answer_calls[1]["answer_contract"])
        self.assertNotIn("answer_type", llm.answer_calls[1]["dependency_answers"][0])
        self.assertNotIn("confidence", llm.answer_calls[1]["dependency_answers"][0])
        self.assertNotIn("evidence_summary", llm.answer_calls[1]["dependency_answers"][0])
        self.assertNotIn("used_hyperedge_ids", llm.answer_calls[1]["dependency_answers"][0])
        self.assertNotIn("reasoning_summary", llm.answer_calls[1]["dependency_answers"][0])
        self.assertFalse(hasattr(llm, "compose_final_answer"))
        self.assertFalse(hasattr(llm, "finalize_answer_span"))
        self.assertFalse(hasattr(llm, "route_reasoning_paths"))
        self.assertFalse(hasattr(llm, "answer_atomic_question_from_paths"))

        second_retrieval = result.artifacts["atomic_retrieval"][1]
        self.assertEqual(second_retrieval["method"], "two_hop_multi_anchor_topk")
        self.assertEqual(second_retrieval["primary_anchor_mention"], "Meek Mill")
        self.assertEqual(second_retrieval["linked_entity_id"], "Meek Mill")
        self.assertEqual(second_retrieval["adjacent_hyperedge_ids"], ["H_PERFORMER", "H_DETAINED"])
        self.assertEqual(second_retrieval["candidate_hyperedge_ids"], ["H_PERFORMER", "H_DETAINED"])
        self.assertEqual([item["hyperedge_id"] for item in second_retrieval["top_hyperedges"]], ["H_DETAINED", "H_PERFORMER"])

    def test_dependency_rewrite_injects_natural_language_role_reference_for_retrieval(self) -> None:
        graph = LocalGraph(
            entity_edges={
                "Changed It": ["H_SONG"],
                "Nicki Minaj": ["H_SONG", "H_BIRTH"],
                "Lil Wayne": ["H_SONG", "H_OTHER"],
            },
            hyperedge_entities={
                "H_SONG": ["Changed It", "Nicki Minaj", "Lil Wayne"],
                "H_BIRTH": ["Nicki Minaj", "Port of Spain"],
                "H_OTHER": ["Lil Wayne", "New Orleans"],
            },
            hyperedge_texts={
                "H_SONG": "Changed It is a song by Nicki Minaj and Lil Wayne.",
                "H_BIRTH": "Nicki Minaj was born in Port of Spain.",
                "H_OTHER": "Lil Wayne was born in New Orleans.",
            },
            hyperedge_chunks={
                "H_SONG": ["C_SONG"],
                "H_BIRTH": ["C_BIRTH"],
                "H_OTHER": ["C_OTHER"],
            },
            chunk_texts={
                "C_SONG": "Changed It is a song by Nicki Minaj and Lil Wayne.",
                "C_BIRTH": "Nicki Minaj was born in Port of Spain, Trinidad and Tobago.",
                "C_OTHER": "Lil Wayne was born in New Orleans.",
            },
        )
        llm = MockAtomicLLMService(
            answer_responses=[
                {
                    "answer": "Nicki Minaj and Lil Wayne",
                    "confidence": 0.95,
                    "reasoning_summary": "The song evidence names both performers.",
                    "used_evidence_ids": ["E3"],
                    "insufficient": False,
                },
                {
                    "answer": "Port of Spain",
                    "confidence": 0.9,
                    "reasoning_summary": "Nicki Minaj was born in Port of Spain.",
                    "used_evidence_ids": ["E1"],
                    "insufficient": False,
                },
            ]
        )
        executor = _executor(
            graph=graph,
            scores={"H_SONG": 0.4, "H_BIRTH": 0.95, "H_OTHER": 0.7},
            analyzer=QuestionAnalyzer(
                {
                    "Who performed the song Changed It?": AtomicQuestionAnalysis(entities=["Changed It"], answer_type="person"),
                    "What is the place of birth of Nicki Minaj and Lil Wayne?": AtomicQuestionAnalysis(
                        entities=["Nicki Minaj", "Lil Wayne"],
                        answer_type="place",
                    ),
                }
            ),
            llm=llm,
        )
        dag = {
            "nodes": [
                {"node_id": "q1", "question": "Who performed the song Changed It?"},
                {
                    "node_id": "q2",
                    "question": "What is the place of birth of the performer of song Changed It?",
                    "dependencies": ["q1"],
                },
            ]
        }

        result = executor.run("What is the place of birth of the performer of song Changed It?", dag)

        self.assertEqual(result.atomic_results[1].question, "What is the place of birth of Nicki Minaj and Lil Wayne?")
        second_retrieval = result.artifacts["atomic_retrieval"][1]
        self.assertEqual(second_retrieval["primary_anchor_mention"], "Nicki Minaj")
        self.assertEqual(second_retrieval["linked_entity_id"], "Nicki Minaj")
        self.assertEqual(second_retrieval["dependency_replacements"][0]["replacement_span"], "the performer of song Changed It")
        self.assertEqual([item["hyperedge_id"] for item in second_retrieval["top_hyperedges"][:2]], ["H_BIRTH", "H_OTHER"])
        self.assertEqual(result.final_answer["answer"], "Port of Spain")

    def test_dependency_rewrite_injects_generic_role_reference_for_retrieval(self) -> None:
        graph = LocalGraph(
            entity_edges={
                "Talk About A Stranger": ["H_FILM"],
                "David Bradley": ["H_FILM", "H_WORK"],
            },
            hyperedge_entities={
                "H_FILM": ["Talk About A Stranger", "David Bradley"],
                "H_WORK": ["David Bradley", "UCLA"],
            },
            hyperedge_texts={
                "H_FILM": "Talk About a Stranger was directed by David Bradley.",
                "H_WORK": "David Bradley worked at UCLA.",
            },
            hyperedge_chunks={"H_FILM": ["C_FILM"], "H_WORK": ["C_WORK"]},
            chunk_texts={
                "C_FILM": "Talk About a Stranger was directed by David Bradley.",
                "C_WORK": "David Bradley worked at UCLA.",
            },
        )
        llm = MockAtomicLLMService(
            answer_responses=[
                {
                    "answer": "David Bradley",
                    "confidence": 0.95,
                    "reasoning_summary": "The film evidence names David Bradley.",
                    "used_evidence_ids": ["E2"],
                    "insufficient": False,
                },
                {
                    "answer": "UCLA",
                    "confidence": 0.9,
                    "reasoning_summary": "David Bradley worked at UCLA.",
                    "used_evidence_ids": ["E1"],
                    "insufficient": False,
                },
            ]
        )
        executor = _executor(
            graph=graph,
            scores={"H_FILM": 0.6, "H_WORK": 0.95},
            analyzer=QuestionAnalyzer(
                {
                    "Who is the director of the film Talk About A Stranger?": AtomicQuestionAnalysis(
                        entities=["Talk About A Stranger"],
                        answer_type="person",
                    ),
                    "Where does David Bradley work?": AtomicQuestionAnalysis(entities=["David Bradley"], answer_type="organization"),
                }
            ),
            llm=llm,
        )
        dag = {
            "nodes": [
                {"node_id": "q1", "question": "Who is the director of the film Talk About A Stranger?"},
                {"node_id": "q2", "question": "Where does the director work?", "dependencies": ["q1"]},
            ]
        }

        result = executor.run("Where does the director of film Talk About A Stranger work at?", dag)

        self.assertEqual(result.atomic_results[1].question, "Where does David Bradley work?")
        second_retrieval = result.artifacts["atomic_retrieval"][1]
        self.assertEqual(second_retrieval["primary_anchor_mention"], "David Bradley")
        self.assertEqual(second_retrieval["linked_entity_id"], "David Bradley")
        self.assertEqual(second_retrieval["dependency_replacements"][0]["replacement_span"], "the director")
        self.assertEqual([item["hyperedge_id"] for item in second_retrieval["top_hyperedges"]], ["H_WORK", "H_FILM"])
        self.assertEqual(result.final_answer["answer"], "UCLA")

    def test_retrieval_uses_primary_anchor_two_hop_pool_top3_and_stable_sort(self) -> None:
        graph = LocalGraph(
            entity_edges={
                "Anchor": ["H1", "H2", "H3", "H4"],
                "A2": ["H2", "H_SECOND"],
                "Other": ["H_GLOBAL"],
            },
            hyperedge_entities={
                "H1": ["Anchor", "A1"],
                "H2": ["Anchor", "A2"],
                "H3": ["Anchor", "A3"],
                "H4": ["Anchor", "A4"],
                "H_SECOND": ["A2", "Second Hop Answer"],
                "H_GLOBAL": ["Other", "Wrong"],
            },
            hyperedge_chunks={
                "H1": ["C1"],
                "H2": ["C2"],
                "H3": ["C3"],
                "H4": ["C4", "C4B"],
                "H_SECOND": ["C_SECOND"],
                "H_GLOBAL": ["CG"],
            },
            chunk_texts={
                "C1": "chunk one",
                "C2": "chunk two",
                "C3": "chunk three",
                "C4": "full chunk four text " * 30,
                "C4B": "second full chunk for four",
                "C_SECOND": "second hop chunk",
                "CG": "global chunk",
            },
        )
        store = ScoreHyperedgeStore({"H1": 0.1, "H2": 0.7, "H3": 0.7, "H4": 0.9, "H_SECOND": 1.0, "H_GLOBAL": 0.99})
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

        self.assertEqual(store.calls, [["H1", "H2", "H3", "H4", "H_SECOND"]])
        self.assertEqual(result.adjacent_hyperedge_ids, ["H1", "H2", "H3", "H4"])
        self.assertEqual(result.expansion_entity_ids, ["A1", "A2", "A3", "A4"])
        self.assertEqual(result.second_hop_hyperedge_ids, ["H_SECOND"])
        self.assertEqual(result.candidate_hyperedge_ids, ["H1", "H2", "H3", "H4", "H_SECOND"])
        self.assertEqual([item["hyperedge_id"] for item in result.top_hyperedges], ["H_SECOND", "H4", "H2"])
        self.assertEqual([item.rank for item in result.evidence], [1, 2, 3])
        self.assertNotIn("H_GLOBAL", [item.hyperedge_id for item in result.evidence])
        first = result.evidence[0].to_dict()
        self.assertEqual(first["hyperedge_id"], "H_SECOND")
        self.assertEqual(first["entity_ids"], ["A2", "Second Hop Answer"])
        self.assertEqual(first["entity_records"][0]["entity_id"], "A2")
        self.assertEqual(first["chunk_ids"], ["C_SECOND"])
        self.assertEqual(first["chunk_texts"][0], "second hop chunk")
        self.assertEqual(first["score_breakdown"]["candidate_hop"], 2)
        self.assertEqual(first["score_breakdown"]["via_entity_ids"], ["A2"])

    def test_retrieval_expands_from_concrete_entities_in_first_hop_chunks(self) -> None:
        graph = LocalGraph(
            entity_edges={
                "Anchor": ["H_BRIDGE"],
                "Bridge Person": ["H_ANSWER"],
                "Generic Role": ["H_ROLE_NOISE"],
                "Generic Concept": ["H_CONCEPT_NOISE"],
            },
            hyperedge_entities={
                "H_BRIDGE": ["Anchor"],
                "H_ANSWER": ["Bridge Person", "Answer University"],
                "H_ROLE_NOISE": ["Generic Role", "Wrong Role Answer"],
                "H_CONCEPT_NOISE": ["Generic Concept", "Wrong Concept Answer"],
            },
            hyperedge_texts={
                "H_BRIDGE": "Anchor is related to a person named Bridge Person.",
                "H_ANSWER": "Bridge Person studied at Answer University.",
                "H_ROLE_NOISE": "A generic role points to the wrong answer.",
                "H_CONCEPT_NOISE": "A generic concept points to the wrong answer.",
            },
            hyperedge_chunks={"H_BRIDGE": ["C_BRIDGE"], "H_ANSWER": ["C_ANSWER"]},
            chunk_texts={
                "C_BRIDGE": "Anchor was the child of Bridge Person. Generic Role and Generic Concept are labels.",
                "C_ANSWER": "Bridge Person studied at Answer University.",
            },
            chunk_entities={"C_BRIDGE": ["Anchor", "Bridge Person", "Generic Role", "Generic Concept"]},
            entity_types={
                "Anchor": "PERSON",
                "Bridge Person": "PERSON",
                "Answer University": "ORGANIZATION",
                "Generic Role": "ROLE",
                "Generic Concept": "CONCEPT",
            },
        )
        store = ScoreHyperedgeStore(
            {
                "H_BRIDGE": 0.1,
                "H_ANSWER": 0.9,
                "H_ROLE_NOISE": 1.0,
                "H_CONCEPT_NOISE": 1.0,
            }
        )
        retriever = AtomicHyperedgeRetriever(
            dataset=_dataset(graph, store),
            embedder=CountingEmbedder(),
            config=RetrievalConfig(local_hyperedge_top_k=3),
            llm_service=MockAtomicLLMService(),
            logger=logging.getLogger("test.chunk_entity_retriever"),
        )

        result = retriever.retrieve_primary_anchor_local(
            question="Where did Anchor's parent study?",
            analysis=AtomicQuestionAnalysis(entities=["Anchor"]),
            primary_anchor_mention="Anchor",
        )

        self.assertEqual(store.calls, [["H_BRIDGE", "H_ANSWER"]])
        self.assertEqual(result.adjacent_hyperedge_ids, ["H_BRIDGE"])
        self.assertEqual(result.expansion_entity_ids, ["Bridge Person"])
        self.assertEqual(result.candidate_hyperedge_ids, ["H_BRIDGE", "H_ANSWER"])
        self.assertNotIn("H_ROLE_NOISE", result.candidate_hyperedge_ids)
        self.assertNotIn("H_CONCEPT_NOISE", result.candidate_hyperedge_ids)
        self.assertEqual(result.top_hyperedges[0]["hyperedge_id"], "H_ANSWER")
        self.assertEqual(result.top_hyperedges[0]["expansion_sources"], ["chunk_entity"])
        self.assertEqual(result.top_hyperedges[0]["via_chunk_ids"], ["C_BRIDGE"])
        answer_source = next(item for item in result.candidate_sources if item["hyperedge_id"] == "H_ANSWER")
        self.assertEqual(answer_source["via_entity_ids"], ["Bridge Person"])
        self.assertEqual(answer_source["via_first_hyperedge_ids"], ["H_BRIDGE"])
        self.assertEqual(answer_source["expansion_sources"], ["chunk_entity"])
        self.assertEqual(answer_source["via_chunk_ids"], ["C_BRIDGE"])

    def test_retrieval_merges_two_hop_pool_from_all_detected_entities(self) -> None:
        graph = LocalGraph(
            entity_edges={
                "Messi": ["H_MSN"],
                "Copa del Rey": ["H_COMPARED"],
                "Barcelona": ["H_MSN"],
                "Neymar": ["H_MSN", "H_WORLD_CUP"],
                "Diego Maradona's goal of the century": ["H_COMPARED", "H_GOAL"],
            },
            hyperedge_entities={
                "H_MSN": ["Messi", "Neymar", "Barcelona"],
                "H_WORLD_CUP": ["Neymar", "World Cup"],
                "H_COMPARED": ["Copa del Rey", "Goal from Messi", "Diego Maradona's goal of the century", "Getafe"],
                "H_GOAL": ["Diego Maradona's goal of the century", "Diego Maradona"],
            },
            hyperedge_texts={
                "H_MSN": "Barcelona's attacking trio of Messi, Suarez and Neymar scored 122 goals.",
                "H_WORLD_CUP": "Neymar scored five goals in the 2014 World Cup.",
                "H_COMPARED": "A goal from Messi in the Copa del Rey brought comparison to Diego Maradona's goal of the century.",
                "H_GOAL": "Diego Maradona's goal of the century refers to Diego Maradona.",
            },
        )
        store = ScoreHyperedgeStore({"H_MSN": 0.2, "H_WORLD_CUP": 0.1, "H_COMPARED": 0.99, "H_GOAL": 0.8})
        retriever = AtomicHyperedgeRetriever(
            dataset=_dataset(graph, store),
            embedder=CountingEmbedder(),
            config=RetrievalConfig(local_hyperedge_top_k=3),
            llm_service=MockAtomicLLMService(),
            logger=logging.getLogger("test.multi_anchor_retriever"),
        )

        result = retriever.retrieve_primary_anchor_local(
            question="Who is the person that Messi's goals in Copa del Rey were compared to for getting signed by Barcelona?",
            analysis=AtomicQuestionAnalysis(entities=["Messi", "Copa del Rey", "Barcelona"], answer_type="person"),
            primary_anchor_mention="Messi",
        )

        self.assertEqual(result.primary_anchor_mention, "Messi")
        self.assertEqual([item["mention"] for item in result.linked_entities], ["Messi", "Copa del Rey", "Barcelona"])
        self.assertIn("H_COMPARED", result.candidate_hyperedge_ids)
        self.assertIn("H_GOAL", result.candidate_hyperedge_ids)
        self.assertEqual([item["hyperedge_id"] for item in result.top_hyperedges[:2]], ["H_COMPARED", "H_GOAL"])
        compared = next(item for item in result.evidence if item.hyperedge_id == "H_COMPARED")
        self.assertIn("Copa del Rey", compared.score_breakdown["anchor_mentions"])
        self.assertIn("Copa del Rey", compared.score_breakdown["anchor_entity_ids"])

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
                    "used_evidence_ids": ["E1", "E999"],
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
        payload_text = json.dumps(llm.answer_calls[0], ensure_ascii=False)
        self.assertEqual([item["evidence_id"] for item in evidence], ["E1", "E2", "E3"])
        self.assertEqual([item["hyperedge_text"] for item in evidence], ["H2", "H3", "H4"])
        self.assertEqual(evidence[0]["chunk_texts"], ["two"])
        self.assertNotIn("hyperedge_id", evidence[0])
        self.assertNotIn("entity_records", evidence[0])
        self.assertNotIn("score_breakdown", evidence[0])
        self.assertNotIn("chunk_ids", evidence[0])
        for forbidden in (
            "source_ids",
            "entity_records",
            "metadata",
            "score_breakdown",
            "chunk_ids",
            "via_entity_ids",
            "via_first_hyperedge_ids",
            "evidence_summary",
            "evidence_texts",
            "semantic_score",
            "fusion_score",
        ):
            self.assertNotIn(forbidden, payload_text)
        self.assertEqual(result.atomic_results[0].used_hyperedge_ids, ["H2"])
        self.assertEqual(result.final_answer["answer"], "Answer Two")
        self.assertEqual(result.final_answer["answer"], result.atomic_results[-1].answer)
        artifact_evidence = result.artifacts["atomic_retrieval"][0]["answerer_evidence"][0]
        self.assertIn("entity_records", artifact_evidence)
        self.assertIn("chunk_ids", artifact_evidence)
        self.assertIn("score_breakdown", artifact_evidence)

    def test_answerer_payload_deduplicates_repeated_chunk_texts_without_changing_artifact(self) -> None:
        graph = LocalGraph(
            entity_edges={"Subject": ["H1", "H2", "H3"]},
            hyperedge_entities={
                "H1": ["Subject", "Answer One"],
                "H2": ["Subject", "Answer Two"],
                "H3": ["Subject", "Answer Three"],
            },
            hyperedge_texts={
                "H1": "Subject is linked to Answer One.",
                "H2": "Subject is also linked to Answer Two.",
                "H3": "Subject is linked to Answer Three.",
            },
            hyperedge_chunks={"H1": ["C_SHARED"], "H2": ["C_SHARED"], "H3": ["C_UNIQUE"]},
            chunk_texts={
                "C_SHARED": "The same source chunk mentions Subject and two answers.",
                "C_UNIQUE": "A separate source chunk mentions Answer Three.",
            },
        )
        llm = MockAtomicLLMService(
            answer_responses=[
                {
                    "answer": "Answer One",
                    "confidence": 0.8,
                    "reasoning_summary": "Selected from E1.",
                    "used_evidence_ids": ["E1"],
                    "insufficient": False,
                }
            ]
        )
        executor = _executor(
            graph=graph,
            scores={"H1": 0.9, "H2": 0.8, "H3": 0.7},
            analyzer=QuestionAnalyzer({"Who is linked to Subject?": AtomicQuestionAnalysis(entities=["Subject"])}),
            llm=llm,
        )

        result = executor.run("Who is linked to Subject?")

        evidence = llm.answer_calls[0]["evidence"]
        all_chunk_texts = [text for item in evidence for text in item["chunk_texts"]]
        self.assertEqual(all_chunk_texts.count("The same source chunk mentions Subject and two answers."), 1)
        self.assertEqual(evidence[0]["chunk_texts"], ["The same source chunk mentions Subject and two answers."])
        self.assertEqual(evidence[1]["chunk_texts"], [])
        artifact_evidence = result.artifacts["atomic_retrieval"][0]["answerer_evidence"]
        self.assertEqual(artifact_evidence[0]["chunk_ids"], ["C_SHARED"])
        self.assertEqual(artifact_evidence[1]["chunk_ids"], ["C_SHARED"])
        self.assertEqual(artifact_evidence[0]["chunk_texts"], ["The same source chunk mentions Subject and two answers."])
        self.assertEqual(artifact_evidence[1]["chunk_texts"], ["The same source chunk mentions Subject and two answers."])

    def test_compact_answer_payload_is_much_smaller_than_full_candidate_dicts(self) -> None:
        long_description = "role metadata description " * 200
        chunk_text = "Perry Bhandal is a British film director, screenwriter, and producer. " * 40
        candidates = [
            FusedHyperedgeCandidate(
                hyperedge_id=f"H{i}",
                hyperedge_text=f"Hyperedge {i}: Perry Bhandal is British.",
                branch_support={"local_primary_anchor", "hop2"},
                semantic_score=0.9 - (i * 0.01),
                fusion_score=0.9 - (i * 0.01),
                entity_ids=["Perry Bhandal", "DIRECTOR", "PRODUCER"],
                entity_records=[
                    {
                        "entity_id": "DIRECTOR",
                        "label": "DIRECTOR",
                        "description": long_description,
                        "metadata": {"description": long_description, "source_ids": ["S1"]},
                    }
                ],
                chunk_ids=[f"C{i}"],
                chunk_texts=[chunk_text],
                evidence_texts=[f"Hyperedge {i}: Perry Bhandal is British.", chunk_text, long_description],
                rank=i + 1,
                score_breakdown={"semantic_score": 0.9, "via_entity_ids": ["Perry Bhandal"]},
            )
            for i in range(5)
        ]

        full_payload = {
            "atomic_question": "What country is Perry Bhandal from?",
            "dependency_answers": [
                {
                    "node_id": "q1",
                    "question": "Who directed Interview With A Hitman?",
                    "answer": "Perry Bhandal",
                    "confidence": 1.0,
                    "answer_type": "person",
                    "evidence_summary": [candidates[0].to_dict()],
                }
            ],
            "top_evidence": [item.to_dict() for item in candidates],
        }
        compact_evidence = AtomicDagExecutor._answer_evidence_payload(candidates)
        compact_payload = {
            "atomic_question": "What country is Perry Bhandal from?",
            "answer_contract": AtomicDagExecutor._answer_contract("What country is Perry Bhandal from?"),
            "dependency_answers": AtomicDagExecutor._answer_dependency_context(full_payload["dependency_answers"]),
            "evidence": compact_evidence,
        }

        full_size = len(json.dumps(full_payload, ensure_ascii=False))
        compact_size = len(json.dumps(compact_payload, ensure_ascii=False))
        self.assertNotIn("answer_type", json.dumps(compact_payload, ensure_ascii=False))
        self.assertLess(compact_size, full_size * 0.2)

    def test_missing_anchor_and_missing_evidence_still_call_answerer_once(self) -> None:
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
        self.assertTrue(no_anchor.atomic_results[0].insufficient)
        self.assertEqual(len(no_anchor_llm.answer_calls), 1)
        self.assertEqual(no_anchor_llm.answer_calls[0]["evidence"], [])
        self.assertEqual(no_anchor.artifacts["atomic_retrieval"][0]["insufficient_reason"], "missing_primary_anchor")
        self.assertEqual(no_anchor.final_answer["answer"], no_anchor.atomic_results[-1].answer)
        self.assertTrue(no_anchor.final_answer["insufficient"])

        no_edges_llm = MockAtomicLLMService()
        no_edges_executor = _executor(
            graph=LocalGraph(entity_edges={"Isolated": []}, hyperedge_entities={}),
            scores={},
            analyzer=QuestionAnalyzer({"Question?": AtomicQuestionAnalysis(entities=["Isolated"])}),
            llm=no_edges_llm,
        )

        no_edges = no_edges_executor.run("Question?")

        self.assertEqual(no_edges.atomic_results[0].answer, "INSUFFICIENT_EVIDENCE")
        self.assertTrue(no_edges.atomic_results[0].insufficient)
        self.assertEqual(len(no_edges_llm.answer_calls), 1)
        self.assertEqual(no_edges_llm.answer_calls[0]["evidence"], [])
        self.assertEqual(no_edges.artifacts["atomic_retrieval"][0]["insufficient_reason"], "primary_anchor_has_no_adjacent_hyperedges")
        self.assertEqual(no_edges.final_answer["answer"], no_edges.atomic_results[-1].answer)

    def test_dependency_only_comparison_node_calls_answerer_and_becomes_final_answer(self) -> None:
        graph = LocalGraph(
            entity_edges={"Film A": ["H_A"], "Film B": ["H_B"]},
            hyperedge_entities={
                "H_A": ["Film A", "1960"],
                "H_B": ["Film B", "1960"],
            },
            hyperedge_texts={
                "H_A": "Film A was released in 1960.",
                "H_B": "Film B was released in 1960.",
            },
            hyperedge_chunks={"H_A": ["C_A"], "H_B": ["C_B"]},
            chunk_texts={
                "C_A": "The original source says Film A was released in 1960.",
                "C_B": "The original source says Film B was released in 1960.",
            },
        )
        llm = MockAtomicLLMService(
            answer_responses=[
                {
                    "answer": "1960",
                    "confidence": 0.91,
                    "reasoning_summary": "Film A release year.",
                    "used_evidence_ids": ["E1"],
                    "insufficient": False,
                },
                {
                    "answer": "1960",
                    "confidence": 0.92,
                    "reasoning_summary": "Film B release year.",
                    "used_evidence_ids": ["E1"],
                    "insufficient": False,
                },
                {
                    "answer": "yes",
                    "confidence": 0.86,
                    "reasoning_summary": "Both dependency answers are 1960.",
                    "used_evidence_ids": [],
                    "insufficient": False,
                },
            ]
        )
        executor = _executor(
            graph=graph,
            scores={"H_A": 0.8, "H_B": 0.8},
            analyzer=QuestionAnalyzer(
                {
                    "When was Film A released?": AtomicQuestionAnalysis(entities=["Film A"], answer_type="date"),
                    "When was Film B released?": AtomicQuestionAnalysis(entities=["Film B"], answer_type="date"),
                }
            ),
            llm=llm,
        )
        dag = {
            "nodes": [
                {"node_id": "q1", "question": "When was Film A released?"},
                {"node_id": "q2", "question": "When was Film B released?"},
                {
                    "node_id": "q3",
                    "question": "Were q1's answer and q2's answer the same?",
                    "dependencies": ["q1", "q2"],
                },
            ]
        }

        result = executor.run("Were Film A and Film B released in the same year?", dag)

        self.assertEqual(len(llm.answer_calls), 3)
        self.assertEqual(llm.answer_calls[2]["evidence"], [])
        self.assertEqual([item["answer"] for item in llm.answer_calls[2]["dependency_answers"]], ["1960", "1960"])
        self.assertEqual(result.atomic_results[-1].answer, "yes")
        self.assertEqual(result.final_answer["answer"], "yes")
        self.assertEqual(result.final_answer["answer"], result.atomic_results[-1].answer)
        self.assertEqual(result.final_answer["used_hyperedge_ids"], [])
        self.assertFalse(result.final_answer["insufficient"])

    def test_dag_requires_unique_terminal_leaf(self) -> None:
        executor = _executor(
            graph=LocalGraph(entity_edges={}, hyperedge_entities={}),
            scores={},
            analyzer=QuestionAnalyzer({}),
            llm=MockAtomicLLMService(),
        )
        dag = {
            "nodes": [
                {"node_id": "q1", "question": "Question one?"},
                {"node_id": "q2", "question": "Question two?"},
            ]
        }

        with self.assertRaisesRegex(ValueError, "exactly one leaf"):
            executor.run("Original?", dag)

    def test_terminal_leaf_must_be_final_topological_node(self) -> None:
        nodes = [
            AtomicQuestionNode(node_id="q2", question="final", dependencies=["q1"]),
            AtomicQuestionNode(node_id="q1", question="dependency"),
        ]

        with self.assertRaisesRegex(ValueError, "terminal leaf must be the final"):
            AtomicDagExecutor.validate_terminal_leaf(nodes)

    def test_all_non_final_nodes_can_reach_terminal_node(self) -> None:
        nodes = [
            AtomicQuestionNode(node_id="q1", question="left"),
            AtomicQuestionNode(node_id="q2", question="right"),
            AtomicQuestionNode(node_id="q3", question="final", dependencies=["q1", "q2"]),
        ]

        AtomicDagExecutor.validate_terminal_leaf(nodes)

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


class LocalGraph:
    def __init__(
        self,
        *,
        entity_edges: dict[str, list[str]],
        hyperedge_entities: dict[str, list[str]],
        hyperedge_texts: dict[str, str] | None = None,
        hyperedge_chunks: dict[str, list[str]] | None = None,
        chunk_texts: dict[str, str] | None = None,
        chunk_entities: dict[str, list[str]] | None = None,
        entity_types: dict[str, str] | None = None,
    ) -> None:
        self.entity_edges = {entity_id: list(ids) for entity_id, ids in entity_edges.items()}
        self.hyperedge_entities = {hyperedge_id: list(ids) for hyperedge_id, ids in hyperedge_entities.items()}
        self.hyperedge_texts = dict(hyperedge_texts or {})
        self.hyperedge_chunks = {hyperedge_id: list(ids) for hyperedge_id, ids in (hyperedge_chunks or {}).items()}
        self.chunk_texts = dict(chunk_texts or {})
        self.chunk_entities = {chunk_id: list(ids) for chunk_id, ids in (chunk_entities or {}).items()}
        self.source_to_nodes = {chunk_id: list(ids) for chunk_id, ids in self.chunk_entities.items()}
        self.entity_types = dict(entity_types or {})
        entity_ids = set(self.entity_edges)
        for values in self.hyperedge_entities.values():
            entity_ids.update(values)
        for values in self.chunk_entities.values():
            entity_ids.update(values)
        self.nodes = {
            entity_id: GraphNode(
                node_id=entity_id,
                role="entity",
                entity_type=self.entity_types.get(entity_id, "entity"),
                description=f"{entity_id} description",
            )
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
        logger=logging.getLogger("test.two_hop_executor"),
    )
    return AtomicDagExecutor(
        analyzer=analyzer,  # type: ignore[arg-type]
        retriever=retriever,
        llm_service=llm,
        logger=logging.getLogger("test.two_hop_executor"),
    )


if __name__ == "__main__":
    unittest.main()
