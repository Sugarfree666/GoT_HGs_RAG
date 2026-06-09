from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from entity_path_pipeline import (  # noqa: E402
    EntityPathSemanticParser,
    build_path_set_candidates,
    build_selected_dependency_path_evidence,
    select_top_paths_by_entity,
)
from entity_path_projector import (  # noqa: E402
    enumerate_entity_origin_paths,
    extract_entity_start_nodes,
    parse_path_pruned_ast_payload,
    prune_terminal_glue_paths,
    undirected_graph_edge_payloads,
    validate_selected_entity_paths,
)
from graph_builder import GraphBuilder  # noqa: E402
from main import run_pipeline  # noqa: E402
from models import (  # noqa: E402
    AtomicQuestionDAG,
    CoreNLPToken,
    DependencyEdge,
    DependencyParse,
    EntityOriginPath,
    EntityStartNode,
    ExplicitEntity,
    ExplicitEntityResult,
    MaskMapping,
    MaskReplacement,
    MaskSpan,
    MaskSpanResult,
    PathSetCandidate,
    QuestionRecord,
    RestoredGraphNodeCandidate,
    ScoredEntityPath,
    SelectedEntityPath,
    SemanticNormalizationResult,
)
from path_projector import build_undirected_dependency_graph  # noqa: E402
from prompts import (  # noqa: E402
    CANDIDATE_NODES_SYSTEM,
    ENTITY_PATH_SCORING_SYSTEM,
    GROUNDED_ATOMIC_DAG_SYSTEM,
    PROBLEM_FRAME_SYSTEM,
    build_grounded_atomic_dag_prompt,
)
from subquestion_generator import SubquestionGenerator  # noqa: E402


class EntityOriginPipelineTest(unittest.TestCase):
    def test_entity_origin_paths_young_man_luther(self) -> None:
        question = "Who is the spouse of Young Man Luther's author?"
        dependency_parse = _dependency_parse(
            ["Who", "is", "the", "spouse", "of", "the", "author", "of", "BookA", "?"],
            [
                (1, 2, "cop"),
                (1, 4, "nsubj"),
                (1, 10, "punct"),
                (3, 4, "det"),
                (4, 7, "nmod:of"),
                (5, 7, "case"),
                (6, 7, "det"),
                (7, 9, "nmod:of"),
                (8, 9, "case"),
            ],
        )
        replacement = _mask_replacement("BookA", "Young Man Luther", "Book")
        restored_candidates = [_restored_candidate("9", "BookA", "Young Man Luther", "Book")]
        graph = build_undirected_dependency_graph(dependency_parse, restored_candidates)

        starts = extract_entity_start_nodes(graph, restored_candidates, replacement)
        paths = enumerate_entity_origin_paths(graph, starts)
        self.assertEqual([entity.text for entity in starts], ["Young Man Luther"])
        self.assertIn(["Young Man Luther", "author", "spouse", "Who", "?"], [path.nodes for path in paths])

        llm = FakeEntityPathLLM(
            desired_paths={"e1": ["Young Man Luther", "author", "spouse", "Who", "?"]},
            ast_payload={
                "nodes": [
                    {
                        "id": "young_man_luther",
                        "label": "Young Man Luther",
                        "kind": "entity",
                        "semantic_type": "Book",
                        "source_path_ids": ["e1_p1"],
                        "source_node_ids": ["9"],
                    },
                    {
                        "id": "author",
                        "label": "author",
                        "kind": "type_variable",
                        "semantic_type": "Role",
                        "source_path_ids": ["e1_p1"],
                        "source_node_ids": ["7"],
                    },
                    {
                        "id": "spouse",
                        "label": "spouse",
                        "kind": "type_variable",
                        "semantic_type": "Person",
                        "source_path_ids": ["e1_p1"],
                        "source_node_ids": ["4"],
                    },
                ],
                "edges": [
                    {
                        "source": "young_man_luther",
                        "target": "author",
                        "relation": "author of Young Man Luther",
                        "support_path_id": "e1_p1",
                        "support_node_ids": ["9", "7"],
                    },
                    {
                        "source": "author",
                        "target": "spouse",
                        "relation": "spouse of the author",
                        "support_path_id": "e1_p1",
                        "support_node_ids": ["7", "4"],
                    },
                ],
                "branch_terminals": {"e1": "spouse"},
            },
        )
        parser = EntityPathSemanticParser(llm)
        selected, _ = parser.select_entity_paths(
            original_question=question,
            restored_question=question,
            entity_start_nodes=starts,
            entity_origin_paths=paths,
        )
        semantic_ast, _ = parser.build_path_pruned_ast(
            original_question=question,
            restored_question=question,
            selected_entity_paths=selected,
            entity_origin_paths=paths,
            undirected_graph_edges=undirected_graph_edge_payloads(graph),
        )

        self.assertEqual(
            [(edge.source, edge.target, edge.relation_hint) for edge in semantic_ast.edges],
            [
                ("young_man_luther", "author", "author of Young Man Luther"),
                ("author", "spouse", "spouse of the author"),
            ],
        )
        dag = SubquestionGenerator(llm).generate_dag(question, semantic_ast)
        self.assertEqual(len(dag.nodes), 2)

    def test_entity_origin_paths_parallel_nationality(self) -> None:
        question = (
            "Do director of film Ten9Eight: Shoot For The Moon and director of film "
            "Sabotage (1936 Film) share the same nationality?"
        )
        dependency_parse = _dependency_parse(
            ["Do", "director", "of", "film", "FilmA", "and", "director", "of", "film", "FilmB", "share", "same", "nationality", "?"],
            [
                (2, 5, "nmod:of"),
                (2, 13, "nmod"),
                (7, 10, "nmod:of"),
                (7, 13, "nmod"),
                (11, 13, "obj"),
                (12, 13, "amod"),
            ],
        )
        replacement = _parallel_replacement()
        restored_candidates = [
            _restored_candidate("5", "FilmA", "Ten9Eight: Shoot For The Moon", "Film"),
            _restored_candidate("10", "FilmB", "Sabotage (1936 Film)", "Film"),
        ]
        graph = build_undirected_dependency_graph(dependency_parse, restored_candidates)
        starts = extract_entity_start_nodes(graph, restored_candidates, replacement)
        paths = enumerate_entity_origin_paths(graph, starts)
        self.assertEqual([entity.text for entity in starts], ["Ten9Eight: Shoot For The Moon", "Sabotage (1936 Film)"])

        llm = FakeEntityPathLLM(
            desired_paths={
                "e1": ["Ten9Eight: Shoot For The Moon", "director", "nationality"],
                "e2": ["Sabotage (1936 Film)", "director", "nationality"],
            },
            ast_payload={
                "nodes": [
                    {"id": "ten9eight_shoot_for_the_moon", "label": "Ten9Eight: Shoot For The Moon", "kind": "entity", "semantic_type": "Film", "source_path_ids": ["e1_p1"], "source_node_ids": ["5"]},
                    {"id": "director_r1", "label": "director", "kind": "type_variable", "semantic_type": "Person", "source_path_ids": ["e1_p1"], "source_node_ids": ["2"]},
                    {"id": "nationality_r1", "label": "nationality", "kind": "type_variable", "semantic_type": "Nationality", "source_path_ids": ["e1_p1"], "source_node_ids": ["13"]},
                    {"id": "sabotage_1936_film", "label": "Sabotage (1936 Film)", "kind": "entity", "semantic_type": "Film", "source_path_ids": ["e2_p1"], "source_node_ids": ["10"]},
                    {"id": "director_r2", "label": "director", "kind": "type_variable", "semantic_type": "Person", "source_path_ids": ["e2_p1"], "source_node_ids": ["7"]},
                    {"id": "nationality_r2", "label": "nationality", "kind": "type_variable", "semantic_type": "Nationality", "source_path_ids": ["e2_p1"], "source_node_ids": ["13"]},
                ],
                "edges": [
                    {"source": "ten9eight_shoot_for_the_moon", "target": "director_r1", "relation": "director of Ten9Eight: Shoot For The Moon", "support_path_id": "e1_p1", "support_node_ids": ["5", "2"]},
                    {"source": "director_r1", "target": "nationality_r1", "relation": "nationality of the director", "support_path_id": "e1_p1", "support_node_ids": ["2", "13"]},
                    {"source": "sabotage_1936_film", "target": "director_r2", "relation": "director of Sabotage (1936 Film)", "support_path_id": "e2_p1", "support_node_ids": ["10", "7"]},
                    {"source": "director_r2", "target": "nationality_r2", "relation": "nationality of the director", "support_path_id": "e2_p1", "support_node_ids": ["7", "13"]},
                ],
                "branch_terminals": {"e1": "nationality_r1", "e2": "nationality_r2"},
            },
        )
        parser = EntityPathSemanticParser(llm)
        selected, _ = parser.select_entity_paths(
            original_question=question,
            restored_question=question,
            entity_start_nodes=starts,
            entity_origin_paths=paths,
        )
        semantic_ast, _ = parser.build_path_pruned_ast(
            original_question=question,
            restored_question=question,
            selected_entity_paths=selected,
            entity_origin_paths=paths,
            undirected_graph_edges=undirected_graph_edge_payloads(graph),
        )

        self.assertEqual(
            [(edge.source, edge.target) for edge in semantic_ast.edges],
            [
                ("ten9eight_shoot_for_the_moon", "director_r1"),
                ("director_r1", "nationality_r1"),
                ("sabotage_1936_film", "director_r2"),
                ("director_r2", "nationality_r2"),
            ],
        )
        self.assertEqual(
            [(edge.source, edge.target) for edge in semantic_ast.edges],
            [
                ("ten9eight_shoot_for_the_moon", "director_r1"),
                ("director_r1", "nationality_r1"),
                ("sabotage_1936_film", "director_r2"),
                ("director_r2", "nationality_r2"),
            ],
        )

    def test_common_answer_paths_do_not_pass_through_other_entity(self) -> None:
        dependency_parse = _dependency_parse(
            ["What", "screenplay", "was", "worked", "on", "by", "both", "PersonA", "and", "PersonB", "?"],
            [
                (2, 1, "det"),
                (4, 2, "nsubj:pass"),
                (4, 3, "aux:pass"),
                (4, 5, "compound:prt"),
                (8, 6, "case"),
                (8, 7, "cc:preconj"),
                (4, 8, "obl:agent"),
                (10, 9, "cc"),
                (4, 10, "obl:agent"),
                (8, 10, "conj:and"),
                (4, 11, "punct"),
            ],
        )
        replacement = _two_person_replacement()
        restored_candidates = [
            _restored_candidate("8", "PersonA", "Edward Carfagno", "Person"),
            _restored_candidate("10", "PersonB", "Miklos Rozsa", "Person"),
        ]
        graph = build_undirected_dependency_graph(dependency_parse, restored_candidates)
        starts = extract_entity_start_nodes(graph, restored_candidates, replacement)
        paths = enumerate_entity_origin_paths(graph, starts)
        paths_by_entity = {
            entity_id: [path for path in paths if path.entity_id == entity_id]
            for entity_id in ("e1", "e2")
        }

        self.assertEqual(paths_by_entity["e1"][0].nodes, ["Edward Carfagno", "worked", "screenplay", "What"])
        self.assertEqual(paths_by_entity["e2"][0].nodes, ["Miklos Rozsa", "worked", "screenplay", "What"])

        crossing_e1 = next(
            path
            for path in paths_by_entity["e1"]
            if path.nodes == ["Edward Carfagno", "Miklos Rozsa", "worked", "screenplay", "What"]
        )
        clean_e2 = paths_by_entity["e2"][0]
        with self.assertRaisesRegex(ValueError, "passes through another entity start"):
            validate_selected_entity_paths(
                selected_paths=[
                    SelectedEntityPath(entity_id="e1", path_id=crossing_e1.path_id),
                    SelectedEntityPath(entity_id="e2", path_id=clean_e2.path_id),
                ],
                entity_starts=starts,
                entity_origin_paths=paths,
            )

    def test_path_scoring_keeps_top2_per_entity(self) -> None:
        entity = EntityStartNode(entity_id="e1", text="Entity A", graph_node_ids=["1"])
        paths = [
            _entity_path("e1_p1", "e1", ["Entity A", "weak"]),
            _entity_path("e1_p2", "e1", ["Entity A", "good"]),
            _entity_path("e1_p3", "e1", ["Entity A", "best"]),
            _entity_path("e1_p4", "e1", ["Entity A", "bad"]),
        ]
        llm = CandidateFlowLLM(
            path_scores=[
                {"entity_id": "e1", "path_id": "e1_p1", "score": 20, "valid": False},
                {"entity_id": "e1", "path_id": "e1_p2", "score": 88, "valid": True},
                {"entity_id": "e1", "path_id": "e1_p3", "score": 96, "valid": True},
                {"entity_id": "e1", "path_id": "e1_p4", "score": 60, "valid": True},
            ],
            ast_payloads_by_path_set={},
            best_candidate_id="",
        )
        parser = EntityPathSemanticParser(llm)
        scored, _ = parser.score_entity_paths(
            original_question="test?",
            restored_question="test?",
            entity_start_nodes=[entity],
            entity_origin_paths=paths,
        )
        top = select_top_paths_by_entity(
            scored_paths=scored,
            entity_start_nodes=[entity],
            entity_origin_paths=paths,
            top_k=2,
        )

        self.assertEqual([item.path_id for item in top["e1"]], ["e1_p3", "e1_p2"])

    def test_two_entity_top2_cartesian_path_sets(self) -> None:
        top = {
            "e1": [
                ScoredEntityPath(entity_id="e1", path_id="e1_p1", score=95),
                ScoredEntityPath(entity_id="e1", path_id="e1_p2", score=80),
            ],
            "e2": [
                ScoredEntityPath(entity_id="e2", path_id="e2_p1", score=90),
                ScoredEntityPath(entity_id="e2", path_id="e2_p2", score=70),
            ],
        }

        path_sets = build_path_set_candidates(top_paths_by_entity=top)

        self.assertEqual(
            [(item.path_set_id, item.path_ids_by_entity) for item in path_sets],
            [
                ("ps1", {"e1": "e1_p1", "e2": "e2_p1"}),
                ("ps2", {"e1": "e1_p1", "e2": "e2_p2"}),
                ("ps3", {"e1": "e1_p2", "e2": "e2_p1"}),
                ("ps4", {"e1": "e1_p2", "e2": "e2_p2"}),
            ],
        )

    def test_prune_terminal_glue_paths_only_checks_terminal(self) -> None:
        paths = [
            _entity_path("e1_p1", "e1", ["Changed It", "performer", "of"]),
            _entity_path("e1_p2", "e1", ["Changed It", "of", "performer"]),
            _entity_path("e2_p1", "e2", ["Lothair II", "did"]),
            _entity_path("e2_p2", "e2", ["Lothair II", "mother"]),
            _entity_path("e3_p1", "e3", ["MovieA", "When"]),
            _entity_path("e3_p2", "e3", ["MovieA", "what"]),
        ]

        pruned, stats = prune_terminal_glue_paths(paths, keep_wh_terminals=True)

        self.assertEqual(
            [path.path_id for path in pruned],
            ["e1_p2", "e2_p2", "e3_p1", "e3_p2"],
        )
        self.assertEqual(stats["total_raw_paths"], 6)
        self.assertEqual(stats["total_kept_paths"], 4)
        self.assertEqual(stats["total_pruned_paths"], 2)
        self.assertFalse(stats["by_entity"]["e1"]["fallback_used"])
        self.assertEqual(stats["by_entity"]["e1"]["pruned_examples"][0]["terminal"], "of")
        self.assertEqual(stats["by_entity"]["e2"]["pruned_examples"][0]["terminal"], "did")

    def test_prune_terminal_glue_paths_fallback_keeps_entity_nonempty(self) -> None:
        paths = [
            _entity_path("e1_p1", "e1", ["MovieA", "of"]),
            _entity_path("e1_p2", "e1", ["MovieA", "?"]),
            _entity_path("e1_p3", "e1", ["MovieA", "did"]),
            _entity_path("e1_p4", "e1", ["MovieA", "the"]),
        ]

        pruned, stats = prune_terminal_glue_paths(paths, min_keep_per_entity=2)

        self.assertEqual(len(pruned), 2)
        self.assertEqual({path.entity_id for path in pruned}, {"e1"})
        self.assertTrue(stats["by_entity"]["e1"]["fallback_used"])
        self.assertEqual(stats["by_entity"]["e1"]["kept"], 2)
        self.assertEqual(stats["by_entity"]["e1"]["pruned"], 2)

    def test_grounded_atomic_dag_generation_uses_path_support(self) -> None:
        paths = [
            _entity_path("e1_p1", "e1", ["El Tonto", "director", "born"]),
            _entity_path("e2_p1", "e2", ["The Heart Of Doreon", "director", "born"]),
        ]
        path_sets = [
            PathSetCandidate(
                path_set_id="ps1",
                path_ids_by_entity={"e1": "e1_p1", "e2": "e2_p1"},
                mean_path_score=95,
            )
        ]
        selected_evidence = build_selected_dependency_path_evidence(
            path_set_candidates=path_sets,
            entity_origin_paths=paths,
        )
        llm = GroundedAtomicLLM(
            {
                "nodes": [
                    {
                        "node_id": "q1",
                        "question": "Who is the director of El Tonto?",
                        "dependencies": [],
                        "support": [
                            {
                                "path_set_id": "ps1",
                                "path_id": "e1_p1",
                                "node_texts": ["El Tonto", "director"],
                                "node_ids": ["1", "2"],
                            }
                        ],
                    },
                    {
                        "node_id": "q2",
                        "question": "When was q1's answer born?",
                        "dependencies": ["q1"],
                        "support": [
                            {
                                "path_set_id": "ps1",
                                "path_id": "e1_p1",
                                "node_texts": ["director", "born"],
                                "node_ids": ["2", "3"],
                            }
                        ],
                    },
                ],
                "selected_path_set_ids": ["ps1"],
                "reason": "test grounded DAG",
            }
        )
        parser = EntityPathSemanticParser(llm)

        dag, payload = parser.build_grounded_atomic_dag(
            original_question="Which film whose director was born first, El Tonto or The Heart Of Doreon?",
            selected_dependency_path_evidence=selected_evidence,
        )

        self.assertEqual([node.id for node in dag.nodes], ["q1", "q2"])
        self.assertEqual(dag.nodes[1].depends_on, ["q1"])
        self.assertEqual(dag.nodes[0].metadata["support_path_ids"], ["e1_p1"])
        self.assertEqual(dag.nodes[0].metadata["support"][0]["node_texts"], ["El Tonto", "director"])
        self.assertEqual(payload["selected_path_set_ids"], ["ps1"])

    def test_grounded_dag_requires_dependency_answer_variables(self) -> None:
        evidence = _selected_dependency_path_evidence_for_alphago()
        llm = SequenceGroundedAtomicLLM(
            [
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which company developed AlphaGo?",
                            "dependencies": [],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "e1_p1",
                                    "node_texts": ["AlphaGo", "developed", "company"],
                                }
                            ],
                        },
                        {
                            "node_id": "q2",
                            "question": "Who is the CEO of the company that developed AlphaGo?",
                            "dependencies": ["q1"],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "e1_p1",
                                    "node_texts": ["company", "CEO"],
                                }
                            ],
                        },
                    ]
                },
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which company developed AlphaGo?",
                            "dependencies": [],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "e1_p1",
                                    "node_texts": ["AlphaGo", "developed", "company"],
                                }
                            ],
                        },
                        {
                            "node_id": "q2",
                            "question": "Who is the CEO of q1's answer?",
                            "dependencies": ["q1"],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "e1_p1",
                                    "node_texts": ["company", "CEO"],
                                }
                            ],
                        },
                    ]
                },
            ]
        )
        parser = EntityPathSemanticParser(llm)

        dag, _ = parser.build_grounded_atomic_dag(
            original_question="Which university did the CEO of the company that developed AlphaGo graduate from?",
            selected_dependency_path_evidence=evidence,
        )

        self.assertEqual(llm.call_count, 2)
        self.assertEqual(dag.nodes[1].question, "Who is the CEO of q1's answer?")

    def test_step9_prompt_only_contains_question_and_selected_path_evidence(self) -> None:
        evidence = [
            {
                "path_set_id": "ps1",
                "paths": [
                    {
                        "entity_id": "e1",
                        "entity_text": "AlphaGo",
                        "path_id": "e1_p1",
                        "path_text": "AlphaGo -> developed -> company",
                        "node_texts": ["AlphaGo", "developed", "company"],
                    }
                ],
            }
        ]

        prompt = build_grounded_atomic_dag_prompt(
            original_question="Which company developed AlphaGo?",
            selected_dependency_path_evidence=evidence,
        )

        self.assertIn("Which company developed AlphaGo?", prompt)
        self.assertIn("Selected dependency path evidence", prompt)
        self.assertIn("AlphaGo -> developed -> company", prompt)
        self.assertNotIn("Full undirected graph edges", prompt)
        self.assertNotIn("Path-level scores", prompt)
        self.assertNotIn("Entity start nodes", prompt)
        self.assertNotIn("Restored/normalized question", prompt)
        self.assertNotIn("Question intent metadata", prompt)
        self.assertNotIn("direct semantic decomposition draft", prompt)

    def test_grounded_dag_requires_support(self) -> None:
        evidence = _selected_dependency_path_evidence_for_alphago()
        llm = SequenceGroundedAtomicLLM(
            [
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which company developed AlphaGo?",
                            "dependencies": [],
                        }
                    ]
                },
                _grounded_alphago_payload(),
            ]
        )
        parser = EntityPathSemanticParser(llm)

        dag, _ = parser.build_grounded_atomic_dag(
            original_question="Which company developed AlphaGo?",
            selected_dependency_path_evidence=evidence,
        )

        self.assertEqual(llm.call_count, 2)
        self.assertEqual([node.question for node in dag.nodes], ["Which company developed AlphaGo?"])
        self.assertIn("Previous output failed grounding validation", llm.prompts[1])

    def test_invalid_support_path_id_repaired_by_node_text_overlap(self) -> None:
        evidence = _selected_dependency_path_evidence_for_alphago()
        llm = SequenceGroundedAtomicLLM(
            [
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which company developed AlphaGo?",
                            "dependencies": [],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "fake_path",
                                    "node_texts": ["AlphaGo", "developed", "company"],
                                }
                            ],
                        }
                    ]
                },
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which company developed AlphaGo?",
                            "dependencies": [],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "fake_path",
                                    "node_texts": ["AlphaGo", "developed", "company"],
                                }
                            ],
                        }
                    ]
                },
            ]
        )
        parser = EntityPathSemanticParser(llm)

        dag, payload = parser.build_grounded_atomic_dag(
            original_question="Which company developed AlphaGo?",
            selected_dependency_path_evidence=evidence,
        )

        self.assertEqual(llm.call_count, 2)
        self.assertEqual(dag.nodes[0].metadata["support"][0]["path_id"], "e1_p1")
        warnings = "\n".join(payload.get("normalization_warnings") or [])
        self.assertIn("fake_path", warnings)
        self.assertIn("Repaired invalid support", warnings)

    def test_invalid_support_path_id_rejected_without_node_text_overlap(self) -> None:
        evidence = _selected_dependency_path_evidence_for_alphago()
        llm = SequenceGroundedAtomicLLM(
            [
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which company developed AlphaGo?",
                            "dependencies": [],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "fake_path",
                                    "node_texts": ["unrelated", "nodes"],
                                }
                            ],
                        }
                    ]
                },
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which company developed AlphaGo?",
                            "dependencies": [],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "fake_path",
                                    "node_texts": ["unrelated", "nodes"],
                                }
                            ],
                        }
                    ]
                },
            ]
        )
        parser = EntityPathSemanticParser(llm)

        with self.assertRaisesRegex(ValueError, "fake_path"):
            parser.build_grounded_atomic_dag(
                original_question="Which company developed AlphaGo?",
                selected_dependency_path_evidence=evidence,
            )
        self.assertEqual(llm.call_count, 2)

    def test_support_node_texts_mismatch_retries_then_warns(self) -> None:
        evidence = _selected_dependency_path_evidence_for_alphago()
        llm = SequenceGroundedAtomicLLM(
            [
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which university developed AlphaGo?",
                            "dependencies": [],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "e1_p1",
                                    "node_texts": ["AlphaGo", "university"],
                                }
                            ],
                        }
                    ]
                },
                {
                    "nodes": [
                        {
                            "node_id": "q1",
                            "question": "Which university developed AlphaGo?",
                            "dependencies": [],
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "e1_p1",
                                    "node_texts": ["AlphaGo", "university"],
                                }
                            ],
                        }
                    ]
                },
            ]
        )
        parser = EntityPathSemanticParser(llm)

        dag, payload = parser.build_grounded_atomic_dag(
            original_question="Which university developed AlphaGo?",
            selected_dependency_path_evidence=evidence,
        )

        self.assertEqual(llm.call_count, 2)
        self.assertEqual(dag.nodes[0].metadata["support"][0]["node_texts"], ["AlphaGo", "university"])
        self.assertIn("node_text_warnings", dag.nodes[0].metadata["support"][0])
        warnings = "\n".join(payload.get("normalization_warnings") or [])
        self.assertIn("university", warnings)
        self.assertIn("Previous output failed grounding validation", llm.prompts[1])

    def test_atomic_node_one_hop_fields_preserved(self) -> None:
        evidence = _selected_dependency_path_evidence_for_alphago()
        payload = _grounded_alphago_payload()
        payload["nodes"][0]["operation"] = "lookup"
        payload["nodes"][0]["input"] = {"type": "entity", "text": "AlphaGo"}
        payload["nodes"][0]["one_hop_relation"] = "developer company"
        payload["nodes"][0]["answer_type"] = "Organization"
        parser = EntityPathSemanticParser(GroundedAtomicLLM(payload))

        dag, _ = parser.build_grounded_atomic_dag(
            original_question="Which company developed AlphaGo?",
            selected_dependency_path_evidence=evidence,
        )

        metadata = dag.nodes[0].metadata
        self.assertEqual(metadata["operation"], "lookup")
        self.assertEqual(metadata["input"], {"type": "entity", "text": "AlphaGo"})
        self.assertEqual(metadata["one_hop_relation"], "developer company")
        self.assertEqual(metadata["answer_type"], "Organization")

    def test_no_final_operator_question_in_prompt(self) -> None:
        prompt = build_grounded_atomic_dag_prompt(
            original_question="Are AlphaGo and Lee Sedol from the same country?",
            selected_dependency_path_evidence=_selected_dependency_path_evidence_for_alphago(),
        )

        self.assertIn("Do not generate a final comparison question", prompt)
        self.assertIn("Do not generate a final yes/no question", prompt)
        self.assertIn("Do not generate a final ranking question", prompt)
        self.assertIn("Do not generate a final count or aggregation question", prompt)

    def test_no_candidate_node_llm_calls(self) -> None:
        question = "Which university did the CEO of the company that developed AlphaGo graduate from?"
        dependency_parse = _dependency_parse(
            ["GameA", "developed", "company", "CEO", "graduated", "university"],
            [(1, 2, "dep"), (2, 3, "obj"), (3, 4, "nmod:of"), (4, 5, "dep"), (5, 6, "obl:from")],
            pos_by_word={"GameA": "NNP"},
        )
        llm = NoCandidatePromptLLM()
        result = run_pipeline(
            record=QuestionRecord(question=question),
            index=1,
            mask_span_extractor=StaticMaskSpanExtractor(
                [
                    MaskSpan(
                        text="AlphaGo",
                        start_char=question.index("AlphaGo"),
                        end_char=question.index("AlphaGo") + len("AlphaGo"),
                        kind_hint="entity",
                        semantic_type_hint="Game",
                    )
                ]
            ),
            parser=StaticParser(dependency_parse),
            graph_builder=GraphBuilder(),
            anchor_selector=None,
            semantic_ast_optimizer=None,
            subquestion_generator=StaticSubquestionGenerator(llm),
            question_normalizer=IdentityNormalizer(),
            path_semantic_parser=EntityPathSemanticParser(llm),
            debug=False,
        )

        self.assertEqual([entity.text for entity in result["entity_start_nodes"]], ["AlphaGo"])
        self.assertTrue(result["scored_entity_paths"])
        self.assertTrue(result["path_set_candidates"])
        self.assertIn("grounded_atomic_dag_payload", result)
        self.assertNotIn("candidate_asts", result)
        self.assertNotIn("semantic_ast", result)
        self.assertEqual([node.question for node in result["subquestion_dag"].nodes], ["Which company developed AlphaGo?"])
        self.assertNotIn("problem_frame", result)
        self.assertNotIn("candidate_nodes", result)

    def test_run_pipeline_uses_grounded_atomic_dag(self) -> None:
        question = "Which university did the CEO of the company that developed AlphaGo graduate from?"
        dependency_parse = _dependency_parse(
            ["GameA", "developed", "company", "CEO", "graduated", "university"],
            [(1, 2, "dep"), (2, 3, "obj"), (3, 4, "nmod:of"), (4, 5, "dep"), (5, 6, "obl:from")],
            pos_by_word={"GameA": "NNP"},
        )
        llm = NoCandidatePromptLLM()

        result = run_pipeline(
            record=QuestionRecord(question=question),
            index=1,
            mask_span_extractor=StaticMaskSpanExtractor(
                [
                    MaskSpan(
                        text="AlphaGo",
                        start_char=question.index("AlphaGo"),
                        end_char=question.index("AlphaGo") + len("AlphaGo"),
                        kind_hint="entity",
                        semantic_type_hint="Game",
                    )
                ]
            ),
            parser=StaticParser(dependency_parse),
            graph_builder=GraphBuilder(),
            anchor_selector=None,
            semantic_ast_optimizer=None,
            subquestion_generator=StaticSubquestionGenerator(llm),
            question_normalizer=IdentityNormalizer(),
            path_semantic_parser=EntityPathSemanticParser(llm),
            debug=False,
        )

        self.assertNotIn("candidate_asts", result)
        self.assertNotIn("semantic_ast", result)
        self.assertIn("grounded_atomic_dag_payload", result)
        self.assertIsNotNone(result["subquestion_dag"])
        self.assertEqual([node.question for node in result["subquestion_dag"].nodes], ["Which company developed AlphaGo?"])
        self.assertEqual(result["subquestion_dag"].nodes[0].metadata["support_path_ids"], ["e1_p1"])

    def test_ordered_comparison_infers_age_from_younger_cue(self) -> None:
        selected_paths = [
            EntityOriginPath(
                path_id="e1_p1",
                entity_id="e1",
                entity_text="Term Of Trial",
                nodes=["Term Of Trial", "director", "younger"],
                node_ids=["1", "2", "3"],
                length=3,
            ),
            EntityOriginPath(
                path_id="e2_p1",
                entity_id="e2",
                entity_text="Would You Marry Me?",
                nodes=["Would You Marry Me?", "director", "younger"],
                node_ids=["4", "5", "6"],
                length=3,
            ),
        ]
        payload = {
            "nodes": [
                {"id": "term_of_trial", "label": "Term Of Trial", "kind": "entity", "source_path_ids": ["e1_p1"], "source_node_ids": ["1"]},
                {"id": "director_r1", "label": "director", "kind": "type_variable", "source_path_ids": ["e1_p1"], "source_node_ids": ["2"]},
                {"id": "age_r1", "label": "age", "kind": "value", "semantic_type": "Age", "source_path_ids": ["e1_p1"]},
                {"id": "would_you_marry_me", "label": "Would You Marry Me?", "kind": "entity", "source_path_ids": ["e2_p1"], "source_node_ids": ["4"]},
                {"id": "director_r2", "label": "director", "kind": "type_variable", "source_path_ids": ["e2_p1"], "source_node_ids": ["5"]},
                {"id": "age_r2", "label": "age", "kind": "value", "semantic_type": "Age", "source_path_ids": ["e2_p1"]},
            ],
            "edges": [
                {"source": "term_of_trial", "target": "director_r1", "relation": "director of Term Of Trial", "support_path_id": "e1_p1", "support_node_ids": ["1", "2"]},
                {"source": "director_r1", "target": "age_r1", "relation": "age of the director", "support_path_id": "e1_p1", "support_node_ids": ["2", "3"]},
                {"source": "would_you_marry_me", "target": "director_r2", "relation": "director of Would You Marry Me?", "support_path_id": "e2_p1", "support_node_ids": ["4", "5"]},
                {"source": "director_r2", "target": "age_r2", "relation": "age of the director", "support_path_id": "e2_p1", "support_node_ids": ["5", "6"]},
            ],
            "branch_terminals": {"e1": "age_r1", "e2": "age_r2"},
        }
        semantic_ast = parse_path_pruned_ast_payload(
            payload,
            selected_paths=selected_paths,
        )
        by_id = semantic_ast.node_by_id()
        self.assertEqual(by_id["age_r1"].source_graph_nodes, ["3"])
        self.assertEqual(by_id["age_r2"].source_graph_nodes, ["6"])

    def test_selected_path_semantic_transduction_accepts_llm_ast_without_validator(self) -> None:
        selected_paths = [
            EntityOriginPath(
                path_id="e1_p1",
                entity_id="e1",
                entity_text="Lothair II's",
                nodes=["Lothair II's", "mother", "die", "When"],
                node_ids=["1", "2", "3", "4"],
                length=4,
            )
        ]
        selected = [SelectedEntityPath(entity_id="e1", path_id="e1_p1")]
        llm = FakeEntityPathLLM(
            desired_paths={"e1": ["Lothair II's", "mother", "die", "When"]},
            ast_payload={
                "nodes": [
                    {"id": "lothair_ii_mother", "label": "Lothair II's mother", "kind": "entity", "source_path_ids": ["e1_p1"], "source_node_ids": ["1", "2"]},
                    {"id": "die", "label": "die", "kind": "type_variable", "source_path_ids": ["e1_p1"], "source_node_ids": ["3"]},
                    {"id": "when", "label": "When", "kind": "type_variable", "source_path_ids": ["e1_p1"], "source_node_ids": ["4"]},
                ],
                "edges": [
                    {"source": "lothair_ii_mother", "target": "die", "relation": "subject of die", "support_path_id": "e1_p1", "support_node_ids": ["1", "2", "3"]},
                    {"source": "when", "target": "die", "relation": "time of die", "support_path_id": "e1_p1", "support_node_ids": ["4", "3"]},
                ],
                "branch_terminals": {"e1": "when"},
            },
        )
        parser = EntityPathSemanticParser(llm)

        semantic_ast, _ = parser.build_selected_path_semantic_ast(
            original_question="When did Lothair II's mother die?",
            restored_question="When did Lothair II's mother die?",
            selected_entity_paths=selected,
            entity_origin_paths=selected_paths,
            undirected_graph_edges=[],
        )

        self.assertEqual(
            [(edge.source, edge.target, edge.relation_hint) for edge in semantic_ast.edges],
            [
                ("lothair_ii_mother", "die", "subject of die"),
                ("when", "die", "time of die"),
            ],
        )

    def test_merged_parallel_ast_is_localized_per_selected_path(self) -> None:
        selected_paths = [
            EntityOriginPath(
                path_id="e1_p1",
                entity_id="e1",
                entity_text="Edward Carfagno",
                nodes=["Edward Carfagno", "Miklos Rozsa", "worked", "screenplay", "What"],
                node_ids=["8", "10", "4", "2", "1"],
                length=5,
            ),
            EntityOriginPath(
                path_id="e2_p1",
                entity_id="e2",
                entity_text="Miklos Rozsa",
                nodes=["Miklos Rozsa", "Edward Carfagno", "worked", "screenplay", "What"],
                node_ids=["10", "8", "4", "2", "1"],
                length=5,
            ),
        ]
        ast_payload = {
            "nodes": [
                {"id": "edward_carfagno", "label": "Edward Carfagno", "kind": "entity", "source_path_ids": ["e1_p1"], "source_node_ids": ["8"]},
                {"id": "miklos_rozsa", "label": "Miklos Rozsa", "kind": "entity", "source_path_ids": ["e2_p1"], "source_node_ids": ["10"]},
                {"id": "screenplay", "label": "screenplay", "kind": "type_variable", "source_path_ids": ["e1_p1", "e2_p1"], "source_node_ids": ["2"]},
            ],
            "edges": [
                {"source": "edward_carfagno", "target": "screenplay", "relation": "screenplay worked on by Edward Carfagno", "support_path_id": "e1_p1", "support_node_ids": ["8", "4", "2"]},
                {"source": "miklos_rozsa", "target": "screenplay", "relation": "screenplay worked on by Miklos Rozsa", "support_path_id": "e2_p1", "support_node_ids": ["10", "4", "2"]},
            ],
            "branch_terminals": {"e1": "screenplay", "e2": "screenplay"},
        }
        llm = FakeEntityPathLLM(
            desired_paths={
                "e1": ["Edward Carfagno", "Miklos Rozsa", "worked", "screenplay", "What"],
                "e2": ["Miklos Rozsa", "Edward Carfagno", "worked", "screenplay", "What"],
            },
            ast_payload=ast_payload,
        )
        parser = EntityPathSemanticParser(llm)
        selected = [
            SelectedEntityPath(entity_id="e1", path_id="e1_p1"),
            SelectedEntityPath(entity_id="e2", path_id="e2_p1"),
        ]
        semantic_ast, _ = parser.build_path_pruned_ast(
            original_question="What screenplay was worked on by both Edward Carfagno and Miklos Rozsa?",
            restored_question="What screenplay was worked on by both Edward Carfagno and Miklos Rozsa?",
            selected_entity_paths=selected,
            entity_origin_paths=selected_paths,
            undirected_graph_edges=[],
        )
        self.assertEqual(
            [(edge.source, edge.target) for edge in semantic_ast.edges],
            [
                ("edward_carfagno", "screenplay_e1"),
                ("miklos_rozsa", "screenplay_e2"),
            ],
        )


class FakeEntityPathLLM:
    def __init__(self, desired_paths: dict[str, list[str]], ast_payload: dict[str, Any]) -> None:
        self.desired_paths = desired_paths
        self.ast_payload = ast_payload

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        if system_prompt == CANDIDATE_NODES_SYSTEM or system_prompt == PROBLEM_FRAME_SYSTEM:
            raise AssertionError("legacy candidate-node/problem-frame prompt was called")
        if "entity-origin dependency-path pipeline" in system_prompt:
            paths_by_entity = _json_after_marker(prompt, "Entity-origin dependency paths:")
            selected = []
            for entity_id, desired_nodes in self.desired_paths.items():
                path_id = next(
                    path["path_id"]
                    for path in paths_by_entity[entity_id]
                    if path["nodes"] == desired_nodes
                )
                selected.append({"entity_id": entity_id, "path_id": path_id, "reason": "test path"})
            return {"selected_paths": selected}
        if "Selected Path Semantic Transduction" in system_prompt or "entity-origin path-to-AST" in system_prompt:
            ast_payload = json.loads(json.dumps(self.ast_payload))
            selected_paths = _json_after_marker(prompt, "Selected entity-origin dependency paths:")
            by_entity = {path["entity_id"]: path["path_id"] for path in selected_paths}
            for node in ast_payload.get("nodes", []):
                node["source_path_ids"] = [by_entity.get(path_id.split("_", 1)[0], path_id) for path_id in node.get("source_path_ids", [])]
            for edge in ast_payload.get("edges", []):
                support_path_id = edge.get("support_path_id", "")
                edge["support_path_id"] = by_entity.get(support_path_id.split("_", 1)[0], support_path_id)
            return ast_payload
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


class CandidateFlowLLM:
    def __init__(
        self,
        *,
        path_scores: list[dict[str, Any]],
        ast_payloads_by_path_set: dict[str, dict[str, Any]] | None = None,
        best_candidate_id: str = "",
    ) -> None:
        self.path_scores = path_scores
        self.ast_payloads_by_path_set = ast_payloads_by_path_set or {}
        self.best_candidate_id = best_candidate_id

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        if system_prompt == CANDIDATE_NODES_SYSTEM or system_prompt == PROBLEM_FRAME_SYSTEM:
            raise AssertionError("legacy candidate-node/problem-frame prompt was called")
        if system_prompt == ENTITY_PATH_SCORING_SYSTEM or "dependency-path judge" in system_prompt:
            return {"path_scores": json.loads(json.dumps(self.path_scores))}
        if "Selected Path Semantic Transduction" in system_prompt or "Selected Path Semantic Transduction" in prompt:
            selected_paths = _json_after_marker(prompt, "Selected entity-origin dependency paths:")
            path_set_id = selected_paths[0].get("path_set_id", "ps1") if selected_paths else "ps1"
            return json.loads(json.dumps(self.ast_payloads_by_path_set[path_set_id]))
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


class GroundedAtomicLLM:
    def __init__(self, dag_payload: dict[str, Any]) -> None:
        self.dag_payload = dag_payload
        self.prompt = ""

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        if system_prompt == CANDIDATE_NODES_SYSTEM or system_prompt == PROBLEM_FRAME_SYSTEM:
            raise AssertionError("legacy candidate-node/problem-frame prompt was called")
        if "Selected Path Semantic Transduction" in system_prompt or "candidate Semantic ASTs" in system_prompt:
            raise AssertionError("AST pipeline prompt should not be called in grounded atomic mode")
        if system_prompt == GROUNDED_ATOMIC_DAG_SYSTEM:
            self.prompt = prompt
            return json.loads(json.dumps(self.dag_payload))
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


class SequenceGroundedAtomicLLM:
    def __init__(self, dag_payloads: list[dict[str, Any]]) -> None:
        self.dag_payloads = dag_payloads
        self.prompts: list[str] = []
        self.call_count = 0

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        if system_prompt != GROUNDED_ATOMIC_DAG_SYSTEM:
            raise AssertionError(f"Unexpected prompt: {system_prompt}")
        self.prompts.append(prompt)
        payload_index = min(self.call_count, len(self.dag_payloads) - 1)
        self.call_count += 1
        return json.loads(json.dumps(self.dag_payloads[payload_index]))


class NoCandidatePromptLLM:
    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        if system_prompt == CANDIDATE_NODES_SYSTEM or system_prompt == PROBLEM_FRAME_SYSTEM:
            raise AssertionError("legacy candidate-node/problem-frame prompt was called")
        if system_prompt == ENTITY_PATH_SCORING_SYSTEM or "dependency-path judge" in system_prompt:
            paths_by_entity = _json_after_marker(prompt, "Entity-origin dependency paths grouped by entity:")
            scores = []
            for entity_id, paths in paths_by_entity.items():
                for path in paths:
                    scores.append(
                        {
                            "entity_id": entity_id,
                            "path_id": path["path_id"],
                            "score": min(100, len(path["node_ids"]) * 20),
                            "valid": True,
                            "reason": "longer path score",
                        }
                    )
            return {"path_scores": scores}
        if "entity-origin dependency-path pipeline" in system_prompt:
            paths_by_entity = _json_after_marker(prompt, "Entity-origin dependency paths:")
            selected = []
            for entity_id, paths in paths_by_entity.items():
                path = max(paths, key=lambda item: len(item["node_ids"]))
                selected.append({"entity_id": entity_id, "path_id": path["path_id"], "reason": "longest useful path"})
            return {"selected_paths": selected}
        if "Selected Path Semantic Transduction" in system_prompt or "entity-origin path-to-AST" in system_prompt:
            selected_paths = _json_after_marker(prompt, "Selected entity-origin dependency paths:")
            path_id = selected_paths[0]["path_id"]
            return {
                "nodes": [
                    {"id": "alphago", "label": "AlphaGo", "kind": "entity", "source_path_ids": [path_id], "source_node_ids": ["1"]},
                    {"id": "company", "label": "company", "kind": "type_variable", "source_path_ids": [path_id], "source_node_ids": ["3"]},
                    {"id": "ceo", "label": "CEO", "kind": "type_variable", "source_path_ids": [path_id], "source_node_ids": ["4"]},
                    {"id": "university", "label": "university", "kind": "type_variable", "source_path_ids": [path_id], "source_node_ids": ["6"]},
                ],
                "edges": [
                    {"source": "alphago", "target": "company", "relation": "company that developed AlphaGo", "support_path_id": path_id, "support_node_ids": ["1", "3"]},
                    {"source": "company", "target": "ceo", "relation": "CEO of the company", "support_path_id": path_id, "support_node_ids": ["3", "4"]},
                    {"source": "ceo", "target": "university", "relation": "university the CEO graduated from", "support_path_id": path_id, "support_node_ids": ["4", "6"]},
                ],
                "branch_terminals": {"e1": "university"},
            }
        if system_prompt == GROUNDED_ATOMIC_DAG_SYSTEM:
            return {
                "nodes": [
                    {
                        "node_id": "q1",
                        "question": "Which company developed AlphaGo?",
                        "dependencies": [],
                        "support": [
                            {
                                "path_set_id": "ps1",
                                "path_id": "e1_p1",
                                "node_texts": ["AlphaGo", "developed", "company"],
                                "node_ids": ["1", "2", "3"],
                            }
                        ],
                    }
                ],
                "selected_path_set_ids": ["ps1"],
                "reason": "test grounded mode",
            }
        return {"question": "test question?"}


class StaticMaskSpanExtractor:
    def __init__(self, mask_spans: list[MaskSpan] | None = None) -> None:
        self.mask_spans = mask_spans or []

    def extract(self, question: str) -> MaskSpanResult:
        del question
        return MaskSpanResult(mask_spans=list(self.mask_spans))


class StaticParser:
    def __init__(self, dependency_parse: DependencyParse) -> None:
        self.dependency_parse = dependency_parse

    def parse(self, question: str) -> DependencyParse:
        del question
        return self.dependency_parse


class IdentityNormalizer:
    def normalize(self, question: str) -> SemanticNormalizationResult:
        return SemanticNormalizationResult(original_question=question, normalized_question=question, changed=False)


class StaticSubquestionGenerator:
    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def generate_dag(self, original_question: str, semantic_ast: Any) -> AtomicQuestionDAG:
        del original_question, semantic_ast
        return AtomicQuestionDAG()


def _entity_path(path_id: str, entity_id: str, nodes: list[str]) -> EntityOriginPath:
    return EntityOriginPath(
        path_id=path_id,
        entity_id=entity_id,
        entity_text=nodes[0],
        nodes=nodes,
        node_ids=[str(index) for index in range(1, len(nodes) + 1)],
        length=len(nodes),
    )


def _selected_dependency_path_evidence_for_alphago() -> list[dict[str, Any]]:
    return [
        {
            "path_set_id": "ps1",
            "paths": [
                {
                    "entity_id": "e1",
                    "entity_text": "AlphaGo",
                    "path_id": "e1_p1",
                    "path_text": "AlphaGo -> developed -> company",
                    "node_texts": ["AlphaGo", "developed", "company"],
                    "node_ids": ["1", "2", "3"],
                }
            ],
        }
    ]


def _grounded_alphago_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {
                "node_id": "q1",
                "question": "Which company developed AlphaGo?",
                "operation": "lookup",
                "input": {"type": "entity", "text": "AlphaGo"},
                "one_hop_relation": "developer company",
                "answer_type": "Organization",
                "dependencies": [],
                "support": [
                    {
                        "path_set_id": "ps1",
                        "path_id": "e1_p1",
                        "node_texts": ["AlphaGo", "developed", "company"],
                        "reason": "This path segment supports asking for the company that developed AlphaGo.",
                    }
                ],
            }
        ],
        "selected_path_set_ids": ["ps1"],
        "reason": "test grounded DAG",
    }


def _death_ast_payload(slot_id: str, semantic_type: str, relation: str, path_id: str) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "john_middleton_murry", "label": "John Middleton Murry", "kind": "entity", "semantic_type": "Person", "source_path_ids": [path_id], "source_node_ids": ["1"]},
            {"id": "wife", "label": "wife", "kind": "type_variable", "semantic_type": "Person", "source_path_ids": [path_id], "source_node_ids": ["2"]},
            {"id": slot_id, "label": slot_id, "kind": "value_slot", "semantic_type": semantic_type, "source_path_ids": [path_id], "source_node_ids": ["3", "4"]},
        ],
        "edges": [
            {"source": "john_middleton_murry", "target": "wife", "relation": "wife of John Middleton Murry", "support_path_id": path_id, "support_node_ids": ["1", "2"]},
            {"source": "wife", "target": slot_id, "relation": relation, "support_path_id": path_id, "support_node_ids": ["2", "3", "4"]},
        ],
        "branch_terminals": {"e1": slot_id},
    }


def _dependency_parse(
    words: list[str],
    edges: list[tuple[int, int, str]],
    pos_by_word: dict[str, str] | None = None,
) -> DependencyParse:
    pos_by_word = pos_by_word or {}
    return DependencyParse(
        tokens=[
            CoreNLPToken(index=index, word=word, pos=pos_by_word.get(word))
            for index, word in enumerate(words, start=1)
        ],
        edges=[
            DependencyEdge(
                source=words[source_index - 1],
                relation=relation,
                target=words[target_index - 1],
                source_index=source_index,
                target_index=target_index,
            )
            for source_index, target_index, relation in edges
        ],
    )


def _restored_candidate(node_id: str, placeholder: str, text: str, semantic_type: str) -> RestoredGraphNodeCandidate:
    return RestoredGraphNodeCandidate(
        node_id=node_id,
        token_index=int(node_id),
        graph_text=placeholder,
        placeholder=placeholder,
        restored_text=text,
        display_text=text,
        is_mask_placeholder=True,
        kind_hint="entity_candidate",
        semantic_type_hint=semantic_type,
        source_token_indices=[int(node_id)],
        text=text,
    )


def _mask_replacement(placeholder: str, text: str, semantic_type: str) -> MaskReplacement:
    return MaskReplacement(
        question=placeholder,
        mapping={placeholder: text},
        original_question=text,
        mask_mapping={
            placeholder: {
                "text": text,
                "kind": "entity",
                "semantic_type": semantic_type,
                "span": {"start": 0, "end": len(text)},
                "masked_span": {"start": 0, "end": len(placeholder)},
            }
        },
        mask_mappings=[
            MaskMapping(
                placeholder=placeholder,
                original_text=text,
                kind_hint="entity",
                semantic_type_hint=semantic_type,
                original_char_span=[0, len(text)],
                masked_char_span=[0, len(placeholder)],
            )
        ],
    )


def _parallel_replacement() -> MaskReplacement:
    replacement = _mask_replacement("FilmA", "Ten9Eight: Shoot For The Moon", "Film")
    replacement.mask_mapping["FilmB"] = {
        "text": "Sabotage (1936 Film)",
        "kind": "entity",
        "semantic_type": "Film",
        "span": {"start": 0, "end": 20},
        "masked_span": {"start": 0, "end": 5},
    }
    replacement.mask_mappings.append(
        MaskMapping(
            placeholder="FilmB",
            original_text="Sabotage (1936 Film)",
            kind_hint="entity",
            semantic_type_hint="Film",
            original_char_span=[0, 20],
            masked_char_span=[0, 5],
        )
    )
    return replacement


def _two_person_replacement() -> MaskReplacement:
    replacement = _mask_replacement("PersonA", "Edward Carfagno", "Person")
    replacement.mask_mapping["PersonB"] = {
        "text": "Miklos Rozsa",
        "kind": "entity",
        "semantic_type": "Person",
        "span": {"start": 0, "end": 12},
        "masked_span": {"start": 0, "end": 7},
    }
    replacement.mask_mappings.append(
        MaskMapping(
            placeholder="PersonB",
            original_text="Miklos Rozsa",
            kind_hint="entity",
            semantic_type_hint="Person",
            original_char_span=[0, 12],
            masked_char_span=[0, 7],
        )
    )
    return replacement


def _json_after_marker(prompt: str, marker: str) -> Any:
    start = prompt.index(marker) + len(marker)
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(prompt[start:].lstrip())
    return value


if __name__ == "__main__":
    unittest.main()
