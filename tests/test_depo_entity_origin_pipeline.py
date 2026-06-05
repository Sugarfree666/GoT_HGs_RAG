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
    parse_path_pruned_ast_payload,
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
    MaskMapping,
    MaskReplacement,
    MaskSpanResult,
    QuestionRecord,
    RestoredGraphNodeCandidate,
    SelectedEntityPath,
    SemanticNormalizationResult,
)
from path_projector import build_undirected_dependency_graph  # noqa: E402
from prompts import CANDIDATE_NODES_SYSTEM, PROBLEM_FRAME_SYSTEM  # noqa: E402
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

    def test_no_candidate_node_llm_calls(self) -> None:
        question = "Which university did the CEO of the company that developed AlphaGo graduate from?"
        dependency_parse = _dependency_parse(
            ["AlphaGo", "developed", "company", "CEO", "graduated", "university"],
            [(1, 2, "dep"), (2, 3, "obj"), (3, 4, "nmod:of"), (4, 5, "dep"), (5, 6, "obl:from")],
            pos_by_word={"AlphaGo": "NNP"},
        )
        llm = NoCandidatePromptLLM()
        result = run_pipeline(
            record=QuestionRecord(question=question),
            index=1,
            mask_span_extractor=StaticMaskSpanExtractor(),
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
        self.assertTrue(result["selected_entity_paths"])
        self.assertIsNone(result["problem_frame"])
        self.assertEqual(result["candidate_nodes"], [])

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

    def test_selected_path_semantic_transduction_retries_lothair_bad_ast(self) -> None:
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
        llm = RetryTransductionLLM(
            payloads=[
                {
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
                {
                    "nodes": [
                        {"id": "lothair_ii", "label": "Lothair II", "kind": "entity", "semantic_type": "Person", "source_path_ids": ["e1_p1"], "source_node_ids": ["1"]},
                        {"id": "mother", "label": "mother", "kind": "type_variable", "semantic_type": "Person", "source_path_ids": ["e1_p1"], "source_node_ids": ["2"]},
                        {"id": "death_date", "label": "death_date", "kind": "type_variable", "semantic_type": "Date", "source_path_ids": ["e1_p1"]},
                    ],
                    "edges": [
                        {"source": "lothair_ii", "target": "mother", "relation": "mother of Lothair II", "support_path_id": "e1_p1", "support_node_ids": ["1", "2"]},
                        {"source": "mother", "target": "death_date", "relation": "date of death of the mother", "support_path_id": "e1_p1", "support_node_ids": ["2", "3", "4"]},
                    ],
                    "branch_terminals": {"e1": "death_date"},
                },
            ]
        )
        parser = EntityPathSemanticParser(llm)

        semantic_ast, _ = parser.build_selected_path_semantic_ast(
            original_question="When did Lothair II's mother die?",
            restored_question="When did Lothair II's mother die?",
            selected_entity_paths=selected,
            entity_origin_paths=selected_paths,
            undirected_graph_edges=[],
        )

        self.assertEqual(llm.transduction_calls, 2)
        self.assertEqual(
            [(edge.source, edge.target, edge.relation_hint) for edge in semantic_ast.edges],
            [
                ("lothair_ii", "mother", "mother of Lothair II"),
                ("mother", "death_date", "date of death of the mother"),
            ],
        )
        self.assertEqual(semantic_ast.node_by_id()["death_date"].source_graph_nodes, ["3", "4"])

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


class RetryTransductionLLM:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.transduction_calls = 0

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        if system_prompt == CANDIDATE_NODES_SYSTEM or system_prompt == PROBLEM_FRAME_SYSTEM:
            raise AssertionError("legacy candidate-node/problem-frame prompt was called")
        if "Selected Path Semantic Transduction" not in system_prompt:
            raise AssertionError(f"Unexpected prompt: {system_prompt}")
        if self.transduction_calls > 0:
            self.assert_feedback_present(prompt)
        payload = self.payloads[min(self.transduction_calls, len(self.payloads) - 1)]
        self.transduction_calls += 1
        return json.loads(json.dumps(payload))

    @staticmethod
    def assert_feedback_present(prompt: str) -> None:
        if "Previous AST failed validation:" not in prompt:
            raise AssertionError("retry prompt did not include validation feedback")


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


class NoCandidatePromptLLM:
    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        if system_prompt == CANDIDATE_NODES_SYSTEM or system_prompt == PROBLEM_FRAME_SYSTEM:
            raise AssertionError("legacy candidate-node/problem-frame prompt was called")
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
        return {"question": "test question?"}


class StaticMaskSpanExtractor:
    def extract(self, question: str) -> MaskSpanResult:
        del question
        return MaskSpanResult(mask_spans=[])


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
