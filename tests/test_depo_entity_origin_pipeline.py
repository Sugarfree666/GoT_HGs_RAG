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

from entity_path_pipeline import EntityPathSemanticParser  # noqa: E402
from entity_path_projector import (  # noqa: E402
    enumerate_entity_origin_paths,
    extract_entity_start_nodes,
    prune_terminal_glue_paths,
)
from models import (  # noqa: E402
    CoreNLPToken,
    DependencyEdge,
    DependencyParse,
    EntityOriginPath,
    EntityStartNode,
    ExplicitEntity,
    ExplicitEntityResult,
    MaskMapping,
    MaskReplacement,
    MaskSpanResult,
    AtomicEvidence,
    QuestionRecord,
    RestoredGraphNodeCandidate,
    SemanticNormalizationResult,
)
from path_projector import build_undirected_dependency_graph  # noqa: E402
from prompts import (  # noqa: E402
    ATOMIC_DAG_FROM_SEMANTIC_REASONING_PATH_SYSTEM,
    SEMANTIC_REASONING_PATH_SYSTEM,
    build_atomic_dag_from_semantic_reasoning_path_prompt,
    build_semantic_reasoning_path_prompt,
)


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

        useful_paths = [path for path in paths if path.nodes == ["Young Man Luther", "author", "spouse", "Who", "?"]]
        self.assertEqual(len(useful_paths), 1)

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

        self.assertIn(["Ten9Eight: Shoot For The Moon", "director", "nationality"], [path.nodes for path in paths])
        self.assertIn(["Sabotage (1936 Film)", "director", "nationality"], [path.nodes for path in paths])

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

    def test_step9_prompt_uses_only_atomic_evidences(self) -> None:
        atoms = [atom.to_dict() for atom in _walt_disney_atomic_evidence_pool()]
        raw_path_text = "Which Walt Disney film -- produced first -- The Apple Dumpling Gang -- Something Wicked This Way Comes"

        prompt = build_semantic_reasoning_path_prompt(
            original_question=(
                "Which Walt Disney film was produced first, "
                "The Apple Dumpling Gang or Something Wicked This Way Comes?"
            ),
            atomic_evidences=atoms,
        )

        self.assertIn("Atomic evidences:", prompt)
        self.assertIn("Which Walt Disney film produced first", prompt)
        self.assertNotIn("Selected dependency path evidence", prompt)
        self.assertNotIn(raw_path_text, prompt)
        self.assertNotIn(" -- ", prompt)

    def test_step10_prompt_only_contains_question_and_semantic_reasoning_paths(self) -> None:
        semantic_payload = _semantic_two_edge_payload()

        prompt = build_atomic_dag_from_semantic_reasoning_path_prompt(
            original_question="Which university did the CEO of the company that developed AlphaGo graduate from?",
            semantic_reasoning_paths=semantic_payload,
            validation_feedback="missing edge b1_e2",
        )

        self.assertIn("Which university did the CEO of the company that developed AlphaGo graduate from?", prompt)
        self.assertIn("Semantic Reasoning Paths", prompt)
        self.assertIn("missing edge b1_e2", prompt)
        self.assertIn("developer company", prompt)
        self.assertNotIn("AlphaGo -> developed -> company", prompt)

    def test_semantic_edge_coverage_rejects_missing_and_unknown_edges(self) -> None:
        semantic_payload = _semantic_two_edge_payload()

        cases = [
            (
                [_atomic_node_for_edge("q1", "b1_e1")],
                "skipped semantic reasoning edges",
            ),
            (
                [_atomic_node_for_edge("q1", "missing_edge")],
                "unknown source_semantic_edge_id",
            ),
        ]

        for nodes, error_pattern in cases:
            with self.subTest(error_pattern=error_pattern):
                parser = EntityPathSemanticParser(SemanticAtomicLLM({"nodes": nodes}))
                with self.assertRaisesRegex(ValueError, error_pattern):
                    parser.build_grounded_atomic_dag(
                        original_question="Which university did the CEO of the company that developed AlphaGo graduate from?",
                        semantic_reasoning_paths=semantic_payload,
                    )

    def test_step10_allows_multiple_atomic_nodes_for_one_semantic_edge(self) -> None:
        semantic_payload = _semantic_two_edge_payload()
        parser = EntityPathSemanticParser(
            SemanticAtomicLLM(
                {
                    "nodes": [
                        _atomic_node_for_edge("q1", "b1_e1"),
                        _atomic_node_for_edge("q2", "b1_e1", dependency="q1"),
                        _atomic_node_for_edge("q3", "b1_e2", dependency="q2"),
                    ]
                }
            )
        )

        dag, _ = parser.build_grounded_atomic_dag(
            original_question="Which university did the CEO of the company that developed AlphaGo graduate from?",
            semantic_reasoning_paths=semantic_payload,
        )

        self.assertEqual([node.metadata["source_semantic_edge_id"] for node in dag.nodes], ["b1_e1", "b1_e1", "b1_e2"])

    def test_step10_atomic_nodes_can_use_semantic_edge_support_only(self) -> None:
        semantic_payload = _semantic_two_edge_payload()
        atomic_payload = {
            "nodes": [
                {
                    key: value
                    for key, value in _atomic_node_for_edge("q1", "b1_e1").items()
                    if key != "support"
                },
                {
                    key: value
                    for key, value in _atomic_node_for_edge("q2", "b1_e2", dependency="q1").items()
                    if key != "support"
                },
            ]
        }
        parser = EntityPathSemanticParser(SemanticAtomicLLM(atomic_payload))

        dag, _ = parser.build_grounded_atomic_dag(
            original_question="Which university did the CEO of the company that developed AlphaGo graduate from?",
            semantic_reasoning_paths=semantic_payload,
        )

        self.assertEqual([node.metadata["source_semantic_edge_id"] for node in dag.nodes], ["b1_e1", "b1_e2"])
        self.assertEqual(
            dag.nodes[0].metadata["support"],
            [{"semantic_path_id": "b1", "semantic_edge_id": "b1_e1"}],
        )

    def test_lothair_step10_generates_self_contained_atomic_questions(self) -> None:
        evidence = [
            {
                "path_set_id": "ps1",
                "paths": [
                    {
                        "entity_id": "e1",
                        "entity_text": "Lothair II",
                        "path_id": "e1_p1",
                        "path_text": "Lothair II -> mother -> die -> When",
                        "node_texts": ["Lothair II", "mother", "die", "When"],
                    }
                ],
            }
        ]
        semantic_payload = {
            "semantic_reasoning_paths": [
                {
                    "branch_id": "b1",
                    "entity_id": "e1",
                    "source_path_id": "e1_p1",
                    "nodes": [
                        {"node_id": "b1_n1", "label": "Lothair II", "kind": "entity", "semantic_type": "Person"},
                        {"node_id": "b1_n2", "label": "mother", "kind": "semantic_object", "semantic_type": "Person"},
                        {"node_id": "b1_n3", "label": "death_date", "kind": "value_slot", "semantic_type": "Date"},
                    ],
                    "edges": [
                        {
                            "edge_id": "b1_e1",
                            "source": "b1_n1",
                            "target": "b1_n2",
                            "relation": "mother of person",
                            "answer_type": "Person",
                            "is_one_hop": True,
                            "support": [{"path_set_id": "ps1", "path_id": "e1_p1", "node_texts": ["Lothair II", "mother"]}],
                        },
                        {
                            "edge_id": "b1_e2",
                            "source": "b1_n2",
                            "target": "b1_n3",
                            "relation": "date of death of person",
                            "answer_type": "Date",
                            "is_one_hop": True,
                            "support": [{"path_set_id": "ps1", "path_id": "e1_p1", "node_texts": ["mother", "die", "When"]}],
                        },
                    ],
                }
            ]
        }
        atomic_payload = {
            "nodes": [
                {
                    "node_id": "q1",
                    "question": "Who was the mother of Lothair II?",
                    "operation": "lookup",
                    "one_hop_relation": "mother of person",
                    "answer_type": "Person",
                    "dependencies": [],
                    "source_semantic_path_id": "b1",
                    "source_semantic_edge_id": "b1_e1",
                },
                {
                    "node_id": "q2",
                    "question": "When did Lothair II's mother die?",
                    "operation": "lookup",
                    "one_hop_relation": "date of death of person",
                    "answer_type": "Date",
                    "dependencies": ["q1"],
                    "source_semantic_path_id": "b1",
                    "source_semantic_edge_id": "b1_e2",
                },
            ]
        }
        parser = EntityPathSemanticParser(SemanticAtomicLLM(atomic_payload))

        dag, _ = parser.build_grounded_atomic_dag(
            original_question="When did Lothair II's mother die?",
            semantic_reasoning_paths=semantic_payload,
        )

        self.assertEqual(
            [node.question for node in dag.nodes],
            ["Who was the mother of Lothair II?", "When did Lothair II's mother die?"],
        )
        forbidden_fragments = ("q1's answer", "previous answer", "answer to q1")
        self.assertFalse(any(fragment in node.question for node in dag.nodes for fragment in forbidden_fragments))

    def test_step10_rejects_variable_placeholder_questions(self) -> None:
        semantic_payload = _semantic_two_edge_payload()
        atomic_payload = {
            "nodes": [
                _atomic_node_for_edge("q1", "b1_e1"),
                {
                    **_atomic_node_for_edge("q2", "b1_e2", dependency="q1"),
                    "question": "Who is the CEO of q1's answer?",
                },
            ]
        }
        parser = EntityPathSemanticParser(SemanticAtomicLLM(atomic_payload))

        with self.assertRaisesRegex(ValueError, "forbidden dependency placeholder"):
            parser.build_grounded_atomic_dag(
                original_question="Which university did the CEO of the company that developed AlphaGo graduate from?",
                semantic_reasoning_paths=semantic_payload,
            )

    def test_semantic_reasoning_support_allows_induced_node_texts_not_in_dependency_path(self) -> None:
        evidence = [
            _atom("atom_1", "When The Stars Go Blue -> performer", node_texts=["When The Stars Go Blue", "performer"]),
            _atom("atom_2", "performer -> nationality", node_texts=["performer", "nationality"]),
        ]
        semantic_payload = {
            "semantic_reasoning_paths": [
                {
                    "branch_id": "b1",
                    "entity_id": "e1",
                    "source_path_id": "e1_p4",
                    "nodes": [
                        {"node_id": "b1_n1", "label": "When The Stars Go Blue", "kind": "entity", "semantic_type": "Song"},
                        {"node_id": "b1_n2", "label": "performer", "kind": "semantic_object", "semantic_type": "Person"},
                        {"node_id": "b1_n3", "label": "nationality", "kind": "value_slot", "semantic_type": "Nationality"},
                    ],
                    "edges": [
                        {
                            "edge_id": "b1_e1",
                            "source": "b1_n1",
                            "target": "b1_n2",
                            "relation": "performer of song",
                            "answer_type": "Person",
                            "is_one_hop": True,
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "e1_p4",
                                    "node_texts": ["When The Stars Go Blue", "performer"],
                                }
                            ],
                        },
                        {
                            "edge_id": "b1_e2",
                            "source": "b1_n2",
                            "target": "b1_n3",
                            "relation": "nationality of person",
                            "answer_type": "Nationality",
                            "is_one_hop": True,
                            "support": [
                                {
                                    "path_set_id": "ps1",
                                    "path_id": "e1_p4",
                                    "node_texts": ["performer", "nationality"],
                                }
                            ],
                        },
                    ],
                    "terminal_node_id": "b1_n3",
                }
            ],
            "operator_intent": {"type": "NONE", "handled_downstream": True},
        }
        parser = EntityPathSemanticParser(SemanticReasoningFlowLLM(
            semantic_payload=semantic_payload,
            atomic_payload={
                "nodes": [
                    _atomic_node_for_edge("q1", "b1_e1"),
                    _atomic_node_for_edge("q2", "b1_e2", dependency="q1"),
                ]
            },
        ))

        result, _ = parser.build_semantic_reasoning_paths(
            original_question="What nationality is the performer of song When The Stars Go Blue?",
            atomic_evidence_pool=evidence,
        )

        self.assertEqual([edge.relation for edge in result.paths[0].edges], ["performer of song", "nationality of person"])

    def test_semantic_reasoning_path_for_produced_first(self) -> None:
        evidence = _walt_disney_atomic_evidence_pool()

        llm = EvidenceGroundedSemanticLLM(
            {
                "semantic_reasoning_paths": [
                    {
                        "branch_id": "b1",
                        "branch_root": "The Apple Dumpling Gang",
                        "nodes": [
                            {"node_id": "b1_n1", "label": "The Apple Dumpling Gang", "kind": "entity", "semantic_type": "Film"},
                            {"node_id": "b1_n2", "label": "date_1", "kind": "value_slot", "semantic_type": "Date"},
                        ],
                        "edges": [
                            {
                                "edge_id": "b1_e1",
                                "source": "b1_n1",
                                "target": "b1_n2",
                                "relation": "production/release date",
                                "answer_type": "Date",
                                "supported_by": ["atom_2", "atom_1"],
                                "atomic_question_template": "When was The Apple Dumpling Gang released or produced?",
                            }
                        ],
                    },
                    {
                        "branch_id": "b2",
                        "branch_root": "Something Wicked This Way Comes",
                        "nodes": [
                            {"node_id": "b2_n1", "label": "Something Wicked This Way Comes", "kind": "entity", "semantic_type": "Film"},
                            {"node_id": "b2_n2", "label": "date_2", "kind": "value_slot", "semantic_type": "Date"},
                        ],
                        "edges": [
                            {
                                "edge_id": "b2_e1",
                                "source": "b2_n1",
                                "target": "b2_n2",
                                "relation": "production/release date",
                                "answer_type": "Date",
                                "supported_by": ["atom_5", "atom_1"],
                                "atomic_question_template": "When was Something Wicked This Way Comes released or produced?",
                            }
                        ],
                    },
                ],
                "operator_intent": {
                    "type": "ARGMIN",
                    "compare_attribute": "production/release date",
                    "surface_cues": ["first"],
                    "candidates": ["The Apple Dumpling Gang", "Something Wicked This Way Comes"],
                    "description": "Compare dates and choose the earlier one.",
                },
            }
        )
        parser = EntityPathSemanticParser(llm)

        result, payload = parser.build_semantic_reasoning_paths(
            original_question=(
                "Which Walt Disney film was produced first, "
                "The Apple Dumpling Gang or Something Wicked This Way Comes?"
            ),
            atomic_evidence_pool=evidence,
        )

        self.assertIn("Atomic evidences:", llm.prompts[0])
        self.assertIn("Do NOT directly convert dependency paths", llm.prompts[0])
        self.assertNotIn("Which Walt Disney film -- produced first -- The Apple Dumpling Gang -- Something Wicked This Way Comes", llm.prompts[0])
        self.assertEqual(payload["atomic_evidences"][0]["text"], "Which Walt Disney film produced first")
        self.assertEqual(
            [edge.relation for path in result.paths for edge in path.edges],
            ["production/release date", "production/release date"],
        )
        self.assertEqual(result.paths[0].edges[0].support[0]["atom_ids"], ["atom_2", "atom_1"])
        self.assertEqual(result.paths[1].edges[0].support[0]["supported_by"], ["atom_5", "atom_1"])
        self.assertEqual(result.operator_intent["type"], "ARGMIN")

    def test_step9_rejects_dependency_cue_nodes_and_relations(self) -> None:
        evidence = [
            _atom("atom_1", "The Apple Dumpling Gang produced first", node_texts=["The Apple Dumpling Gang", "produced first"]),
            _atom("atom_2", "produced first Something Wicked This Way Comes", node_texts=["produced first", "Something Wicked This Way Comes"]),
        ]
        llm = EvidenceGroundedSemanticLLM(
            {
                "semantic_reasoning_paths": [
                    {
                        "branch_id": "b1",
                        "nodes": [
                            {"node_id": "b1_n1", "label": "The Apple Dumpling Gang", "kind": "entity"},
                            {"node_id": "b1_n2", "label": "produced first", "kind": "semantic_object"},
                            {"node_id": "b1_n3", "label": "Something Wicked This Way Comes", "kind": "entity"},
                        ],
                        "edges": [
                            {
                                "edge_id": "b1_e1",
                                "source": "b1_n1",
                                "target": "b1_n2",
                                "relation": "produced first",
                                "supported_by": ["atom_1"],
                            },
                            {
                                "edge_id": "b1_e2",
                                "source": "b1_n2",
                                "target": "b1_n3",
                                "relation": "produced first",
                                "supported_by": ["atom_2"],
                            },
                        ],
                    }
                ],
                "operator_intent": {"type": "ARGMIN", "surface_cues": ["first"]},
            }
        )
        parser = EntityPathSemanticParser(llm)

        with self.assertRaisesRegex(ValueError, "forbidden"):
            parser.build_semantic_reasoning_paths(
                original_question=(
                    "Which Walt Disney film was produced first, "
                    "The Apple Dumpling Gang or Something Wicked This Way Comes?"
                ),
                atomic_evidence_pool=evidence,
            )

        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("Previous output failed Semantic Reasoning Path validation", llm.prompts[1])

    def test_parallel_nationality_semantic_paths_compile_to_branch_lookup_dag(self) -> None:
        question = (
            "Do director of film Ten9Eight: Shoot For The Moon and director of film "
            "Sabotage (1936 Film) share the same nationality?"
        )
        evidence = _parallel_nationality_evidence()
        llm = SemanticReasoningFlowLLM(
            semantic_payload=_semantic_parallel_nationality_payload(),
            atomic_payload=_atomic_parallel_nationality_payload(),
        )
        parser = EntityPathSemanticParser(llm)

        semantic_paths, _ = parser.build_semantic_reasoning_paths(
            original_question=question,
            atomic_evidence_pool=evidence,
        )
        dag, _ = parser.build_grounded_atomic_dag(
            original_question=question,
            semantic_reasoning_paths=semantic_paths,
        )

        self.assertEqual(
            [node.question for node in dag.nodes],
            [
                "Who directed Ten9Eight: Shoot For The Moon?",
                "What is the nationality of the director of Ten9Eight: Shoot For The Moon?",
                "Who directed Sabotage (1936 Film)?",
                "What is the nationality of the director of Sabotage (1936 Film)?",
            ],
        )
        self.assertEqual([node.depends_on for node in dag.nodes], [[], ["q1"], [], ["q3"]])
        self.assertEqual(
            [node.metadata["source_semantic_edge_id"] for node in dag.nodes],
            ["b1_e1", "b1_e2", "b2_e1", "b2_e2"],
        )
        self.assertEqual(semantic_paths.operator_intent.get("type"), "COMPARE_SAME")
        self.assertTrue(semantic_paths.operator_intent.get("handled_downstream"))
        self.assertFalse(any("share the same nationality" in node.question.lower() for node in dag.nodes))

class SemanticAtomicLLM:
    def __init__(self, dag_payload: dict[str, Any]) -> None:
        self.dag_payload = dag_payload

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        del prompt
        if system_prompt != ATOMIC_DAG_FROM_SEMANTIC_REASONING_PATH_SYSTEM:
            raise AssertionError(f"Unexpected prompt: {system_prompt}")
        return json.loads(json.dumps(self.dag_payload))


class SemanticReasoningFlowLLM:
    def __init__(self, *, semantic_payload: dict[str, Any], atomic_payload: dict[str, Any]) -> None:
        self.semantic_payload = semantic_payload
        self.atomic_payload = atomic_payload
        self.system_prompts: list[str] = []

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        del prompt
        self.system_prompts.append(system_prompt)
        if system_prompt == SEMANTIC_REASONING_PATH_SYSTEM:
            return json.loads(json.dumps(self.semantic_payload))
        if system_prompt == ATOMIC_DAG_FROM_SEMANTIC_REASONING_PATH_SYSTEM:
            return json.loads(json.dumps(self.atomic_payload))
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


class EvidenceGroundedSemanticLLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        if system_prompt != SEMANTIC_REASONING_PATH_SYSTEM:
            raise AssertionError(f"Unexpected prompt: {system_prompt}")
        return json.loads(json.dumps(self.payload))


def _entity_path(path_id: str, entity_id: str, nodes: list[str]) -> EntityOriginPath:
    return EntityOriginPath(
        path_id=path_id,
        entity_id=entity_id,
        entity_text=nodes[0],
        nodes=nodes,
        node_ids=[str(index) for index in range(1, len(nodes) + 1)],
        length=len(nodes),
    )


def _atom(evidence_id: str, text: str, *, node_texts: list[str] | None = None) -> AtomicEvidence:
    return AtomicEvidence(
        id=evidence_id,
        type="dependency_path",
        source="corenlp",
        text=text,
        metadata={"node_texts": node_texts or [text]},
    )


def _walt_disney_atomic_evidence_pool() -> list[AtomicEvidence]:
    return [
        _atom("atom_1", "Which Walt Disney film produced first", node_texts=["Which Walt Disney film", "produced first"]),
        _atom("atom_2", "produced first The Apple Dumpling Gang", node_texts=["produced first", "The Apple Dumpling Gang"]),
        _atom(
            "atom_3",
            "The Apple Dumpling Gang Something Wicked This Way Comes",
            node_texts=["The Apple Dumpling Gang", "Something Wicked This Way Comes"],
        ),
        _atom(
            "atom_4",
            "Something Wicked This Way Comes The Apple Dumpling Gang",
            node_texts=["Something Wicked This Way Comes", "The Apple Dumpling Gang"],
        ),
        _atom(
            "atom_5",
            "Something Wicked This Way Comes produced first",
            node_texts=["Something Wicked This Way Comes", "produced first"],
        ),
    ]


def _semantic_two_edge_payload() -> dict[str, Any]:
    return {
        "paths": [
            {
                "branch_id": "b1",
                "entity_id": "e1",
                "source_path_id": "e1_p1",
                "nodes": [
                    {"node_id": "b1_n1", "label": "AlphaGo", "kind": "entity", "semantic_type": "Game"},
                    {"node_id": "b1_n2", "label": "company", "kind": "semantic_object", "semantic_type": "Organization"},
                    {"node_id": "b1_n3", "label": "CEO", "kind": "semantic_object", "semantic_type": "Person"},
                ],
                "edges": [
                    {
                        "edge_id": "b1_e1",
                        "source": "b1_n1",
                        "target": "b1_n2",
                        "relation": "developer company",
                        "answer_type": "Organization",
                        "is_one_hop": True,
                        "support": [
                            {"supported_by": ["atom_1"], "atom_ids": ["atom_1"]}
                        ],
                    },
                    {
                        "edge_id": "b1_e2",
                        "source": "b1_n2",
                        "target": "b1_n3",
                        "relation": "CEO of company",
                        "answer_type": "Person",
                        "is_one_hop": True,
                        "support": [
                            {"supported_by": ["atom_2"], "atom_ids": ["atom_2"]}
                        ],
                    },
                ],
            }
        ],
    }


def _atomic_node_for_edge(node_id: str, edge_id: str, dependency: str | None = None) -> dict[str, Any]:
    dependencies = [dependency] if dependency else []
    if dependency:
        question = "What is the next fact in the AlphaGo semantic chain?"
        raw_input = {"type": "semantic_context", "text": "AlphaGo semantic chain"}
    else:
        question = "Which company developed AlphaGo?"
        raw_input = {"type": "entity", "text": "AlphaGo"}
    return {
        "node_id": node_id,
        "question": question,
        "operation": "lookup",
        "input": raw_input,
        "one_hop_relation": "test relation",
        "answer_type": "Entity",
        "dependencies": dependencies,
        "support": [
            {"supported_by": ["atom_1"], "atom_ids": ["atom_1"]}
        ],
        "source_semantic_path_id": "b1",
        "source_semantic_edge_id": edge_id,
    }


def _parallel_nationality_evidence() -> list[AtomicEvidence]:
    return [
        _atom("atom_1", "Ten9Eight: Shoot For The Moon director", node_texts=["Ten9Eight: Shoot For The Moon", "director"]),
        _atom("atom_2", "director nationality", node_texts=["director", "nationality"]),
        _atom("atom_3", "Sabotage (1936 Film) director", node_texts=["Sabotage (1936 Film)", "director"]),
        _atom("atom_4", "director nationality", node_texts=["director", "nationality"]),
    ]


def _semantic_parallel_nationality_payload() -> dict[str, Any]:
    return {
        "semantic_reasoning_paths": [
            {
                "branch_id": "b1",
                "entity_id": "e1",
                "source_path_id": "e1_p1",
                "nodes": [
                    {"node_id": "b1_n1", "label": "Ten9Eight: Shoot For The Moon", "kind": "entity", "semantic_type": "Film"},
                    {"node_id": "b1_n2", "label": "director_1", "kind": "semantic_object", "semantic_type": "Person"},
                    {"node_id": "b1_n3", "label": "nationality_1", "kind": "value_slot", "semantic_type": "Nationality"},
                ],
                "edges": [
                    {
                        "edge_id": "b1_e1",
                        "source": "b1_n1",
                        "target": "b1_n2",
                        "relation": "director of film",
                        "answer_type": "Person",
                        "is_one_hop": True,
                        "support": [
                            {"supported_by": ["atom_1"], "atom_ids": ["atom_1"]}
                        ],
                    },
                    {
                        "edge_id": "b1_e2",
                        "source": "b1_n2",
                        "target": "b1_n3",
                        "relation": "nationality of person",
                        "answer_type": "Nationality",
                        "is_one_hop": True,
                        "support": [
                            {"supported_by": ["atom_2"], "atom_ids": ["atom_2"]}
                        ],
                    },
                ],
                "terminal_node_id": "b1_n3",
                "score": 96,
            },
            {
                "branch_id": "b2",
                "entity_id": "e2",
                "source_path_id": "e2_p1",
                "nodes": [
                    {"node_id": "b2_n1", "label": "Sabotage (1936 Film)", "kind": "entity", "semantic_type": "Film"},
                    {"node_id": "b2_n2", "label": "director_2", "kind": "semantic_object", "semantic_type": "Person"},
                    {"node_id": "b2_n3", "label": "nationality_2", "kind": "value_slot", "semantic_type": "Nationality"},
                ],
                "edges": [
                    {
                        "edge_id": "b2_e1",
                        "source": "b2_n1",
                        "target": "b2_n2",
                        "relation": "director of film",
                        "answer_type": "Person",
                        "is_one_hop": True,
                        "support": [
                            {"supported_by": ["atom_3"], "atom_ids": ["atom_3"]}
                        ],
                    },
                    {
                        "edge_id": "b2_e2",
                        "source": "b2_n2",
                        "target": "b2_n3",
                        "relation": "nationality of person",
                        "answer_type": "Nationality",
                        "is_one_hop": True,
                        "support": [
                            {"supported_by": ["atom_4"], "atom_ids": ["atom_4"]}
                        ],
                    },
                ],
                "terminal_node_id": "b2_n3",
                "score": 96,
            },
        ],
        "operator_intent": {"type": "COMPARE_SAME", "handled_downstream": True, "surface_cues": ["share", "same"]},
        "score": 96,
    }


def _atomic_parallel_nationality_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {
                "node_id": "q1",
                "question": "Who directed Ten9Eight: Shoot For The Moon?",
                "operation": "lookup",
                "input": {"type": "entity", "text": "Ten9Eight: Shoot For The Moon"},
                "one_hop_relation": "director of film",
                "answer_type": "Person",
                "dependencies": [],
                "support": [{"supported_by": ["atom_1"], "atom_ids": ["atom_1"]}],
                "source_semantic_path_id": "b1",
                "source_semantic_edge_id": "b1_e1",
            },
            {
                "node_id": "q2",
                "question": "What is the nationality of the director of Ten9Eight: Shoot For The Moon?",
                "operation": "lookup",
                "input": {"type": "semantic_context", "text": "director of Ten9Eight: Shoot For The Moon"},
                "one_hop_relation": "nationality of person",
                "answer_type": "Nationality",
                "dependencies": ["q1"],
                "support": [{"supported_by": ["atom_2"], "atom_ids": ["atom_2"]}],
                "source_semantic_path_id": "b1",
                "source_semantic_edge_id": "b1_e2",
            },
            {
                "node_id": "q3",
                "question": "Who directed Sabotage (1936 Film)?",
                "operation": "lookup",
                "input": {"type": "entity", "text": "Sabotage (1936 Film)"},
                "one_hop_relation": "director of film",
                "answer_type": "Person",
                "dependencies": [],
                "support": [{"supported_by": ["atom_3"], "atom_ids": ["atom_3"]}],
                "source_semantic_path_id": "b2",
                "source_semantic_edge_id": "b2_e1",
            },
            {
                "node_id": "q4",
                "question": "What is the nationality of the director of Sabotage (1936 Film)?",
                "operation": "lookup",
                "input": {"type": "semantic_context", "text": "director of Sabotage (1936 Film)"},
                "one_hop_relation": "nationality of person",
                "answer_type": "Nationality",
                "dependencies": ["q3"],
                "support": [{"supported_by": ["atom_4"], "atom_ids": ["atom_4"]}],
                "source_semantic_path_id": "b2",
                "source_semantic_edge_id": "b2_e2",
            },
        ],
        "reason": "compiled per-branch nationality lookups only",
    }


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
        "reason": "test grounded DAG",
    }


def _semantic_alphago_payload() -> dict[str, Any]:
    return {
        "semantic_reasoning_paths": [
            {
                "branch_id": "b1",
                "entity_id": "e1",
                "source_path_id": "e1_p1",
                "nodes": [
                    {
                        "node_id": "b1_n1",
                        "label": "AlphaGo",
                        "kind": "entity",
                        "semantic_type": "Game",
                        "source_path_id": "e1_p1",
                        "source_node_texts": ["AlphaGo"],
                        "source_node_ids": ["1"],
                    },
                    {
                        "node_id": "b1_n2",
                        "label": "company",
                        "kind": "semantic_object",
                        "semantic_type": "Organization",
                        "source_path_id": "e1_p1",
                        "source_node_texts": ["developed", "company"],
                        "source_node_ids": ["2", "3"],
                    },
                    {
                        "node_id": "b1_n3",
                        "label": "CEO",
                        "kind": "semantic_object",
                        "semantic_type": "Person",
                        "source_path_id": "e1_p1",
                        "source_node_texts": ["company", "CEO"],
                        "source_node_ids": ["3", "4"],
                    },
                    {
                        "node_id": "b1_n4",
                        "label": "university",
                        "kind": "value_slot",
                        "semantic_type": "University",
                        "source_path_id": "e1_p1",
                        "source_node_texts": ["CEO", "graduated", "university"],
                        "source_node_ids": ["4", "5", "6"],
                    },
                ],
                "edges": [
                    {
                        "edge_id": "b1_e1",
                        "source": "b1_n1",
                        "target": "b1_n2",
                        "relation": "developer company",
                        "answer_type": "Organization",
                        "is_one_hop": True,
                        "support": [
                            {
                                "path_set_id": "ps1",
                                "path_id": "e1_p1",
                                "node_texts": ["AlphaGo", "developed", "company"],
                                "node_ids": ["1", "2", "3"],
                                "reason": "supports developer company lookup",
                            }
                        ],
                        "atomic_question_template": "Which company developed AlphaGo?",
                    },
                    {
                        "edge_id": "b1_e2",
                        "source": "b1_n2",
                        "target": "b1_n3",
                        "relation": "CEO of company",
                        "answer_type": "Person",
                        "is_one_hop": True,
                        "support": [
                            {
                                "path_set_id": "ps1",
                                "path_id": "e1_p1",
                                "node_texts": ["company", "CEO"],
                                "node_ids": ["3", "4"],
                                "reason": "supports CEO lookup",
                            }
                        ],
                        "atomic_question_template": "Who is the CEO of b1_n2's answer?",
                    },
                    {
                        "edge_id": "b1_e3",
                        "source": "b1_n3",
                        "target": "b1_n4",
                        "relation": "university graduated from",
                        "answer_type": "University",
                        "is_one_hop": True,
                        "support": [
                            {
                                "path_set_id": "ps1",
                                "path_id": "e1_p1",
                                "node_texts": ["CEO", "graduated", "university"],
                                "node_ids": ["4", "5", "6"],
                                "reason": "supports university lookup",
                            }
                        ],
                        "atomic_question_template": "Which university did b1_n3's answer graduate from?",
                    },
                ],
                "terminal_node_id": "b1_n4",
                "score": 95,
                "warnings": [],
            }
        ],
        "operator_intent": {"type": "NONE", "handled_downstream": True, "surface_cues": []},
        "score": 95,
        "score_breakdown": {},
        "warnings": [],
        "reason": "test semantic reasoning path",
    }


def _atomic_alphago_from_semantic_payload() -> dict[str, Any]:
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
                        "reason": "Copied from semantic reasoning edge b1_e1.",
                    }
                ],
                "source_semantic_path_id": "b1",
                "source_semantic_edge_id": "b1_e1",
            },
            {
                "node_id": "q2",
                "question": "Who is the CEO of the company that developed AlphaGo?",
                "operation": "lookup",
                "input": {"type": "semantic_context", "text": "company that developed AlphaGo"},
                "one_hop_relation": "CEO of company",
                "answer_type": "Person",
                "dependencies": ["q1"],
                "support": [
                    {
                        "path_set_id": "ps1",
                        "path_id": "e1_p1",
                        "node_texts": ["company", "CEO"],
                        "reason": "Copied from semantic reasoning edge b1_e2.",
                    }
                ],
                "source_semantic_path_id": "b1",
                "source_semantic_edge_id": "b1_e2",
            },
            {
                "node_id": "q3",
                "question": "Which university did the CEO of the company that developed AlphaGo graduate from?",
                "operation": "lookup",
                "input": {"type": "semantic_context", "text": "CEO of the company that developed AlphaGo"},
                "one_hop_relation": "university graduated from",
                "answer_type": "University",
                "dependencies": ["q2"],
                "support": [
                    {
                        "path_set_id": "ps1",
                        "path_id": "e1_p1",
                        "node_texts": ["CEO", "graduated", "university"],
                        "reason": "Copied from semantic reasoning edge b1_e3.",
                    }
                ],
                "source_semantic_path_id": "b1",
                "source_semantic_edge_id": "b1_e3",
            },
        ],
        "reason": "compiled from semantic reasoning path",
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
