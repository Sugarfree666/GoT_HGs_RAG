from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from atomic_question_dag import restore_global_best_paths  # noqa: E402
from hanlp_sdp_parser import HanLPSDPParser  # noqa: E402
from entity_masking_preprocessor import EntityMaskingPreprocessor  # noqa: E402
from main import print_hanlp_sdp_result, run_hanlp_sdp_pipeline  # noqa: E402
from models import HanLPSDPEdge, HanLPSDPResult, QuestionRecord  # noqa: E402
from tri_sdp_reasoning_compiler import (  # noqa: E402
    _edge_cost,
    _path_cost_from_edge_map,
    _semantic_path_search_graph,
    _semantic_path_score,
    _shortest_semantic_boundary_path,
    build_evidence_graph,
    compile_token_reasoning_structure,
)


class HanLPSDPMainlineTest(unittest.TestCase):
    def test_hanlp_sdp_pipeline_runs_step5_from_entity_branch_paths_by_default(
        self,
    ) -> None:
        record = QuestionRecord(
            question="Who is older, Ryan Tubridy or Mauro Massironi?"
        )
        parser = FakeHanLPSDPParser()
        llm = FakePreprocessLLM()

        result = run_hanlp_sdp_pipeline(
            record=record,
            index=1,
            preprocessor=EntityMaskingPreprocessor(llm),
            parser=parser,
        )

        preprocess_result = result["preprocess_result"]
        self.assertEqual(
            preprocess_result.masked_question, "Who is older, ENTITYA or ENTITYB?"
        )
        self.assertEqual(
            preprocess_result.sdp_input_sentence, "Who is older, ENTITYA or ENTITYB?"
        )
        self.assertEqual(
            [mapping.placeholder for mapping in preprocess_result.mask_mappings],
            ["ENTITYA", "ENTITYB"],
        )
        self.assertEqual(parser.placeholders, ["ENTITYA", "ENTITYB"])
        self.assertEqual(
            result["hanlp_input_sentence"], "Who is older, ENTITYA or ENTITYB?"
        )
        self.assertEqual(parser.text, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(llm.calls, 2)

        step5_payload = json.loads(llm.step5_user_prompt)
        self.assertEqual(
            set(step5_payload), {"original_question", "topic_entities", "step4_paths"}
        )
        self.assertEqual(
            step5_payload["topic_entities"], ["Ryan Tubridy", "Mauro Massironi"]
        )
        self.assertEqual(
            step5_payload["step4_paths"],
            [["Ryan Tubridy", "older", "Who"], ["Mauro Massironi", "older", "Who"]],
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            print_hanlp_sdp_result(1, record, result)
        output = stream.getvalue()

        self.assertIn("[Original Question]", output)
        self.assertIn("[1. Explicit Entities]", output)
        self.assertIn(" - Ryan Tubridy", output)
        self.assertIn("[2. Entity Masking]", output)
        self.assertIn(" - ENTITYA -> Ryan Tubridy", output)
        self.assertIn("[3. HanLP SDP Parsing]", output)
        self.assertIn("[Raw SDP Edges]", output)
        self.assertIn("[4. Token Reasoning Structure]", output)
        self.assertIn("[Repaired Evidence Graph]", output)
        self.assertIn("Who[1] -- older[3]", output)
        self.assertIn("cost=1", output)
        self.assertNotIn("[Anchor Paths]", output)
        self.assertIn("[Entity Branch Best Paths]", output)
        self.assertIn("P1: ENTITYA ---- older ---- Who", output)
        self.assertIn("P2: ENTITYB ---- older ---- Who", output)
        self.assertIn("Branch SP:", output)
        self.assertNotIn("typed_wh_slot", output)
        self.assertIn("[5. Atomic Question DAG]", output)
        self.assertIn("q1: When was Ryan Tubridy born?", output)
        self.assertIn("q2: When was Mauro Massironi born?", output)

    def test_pipeline_normalizes_wh_order_before_masking_and_parsing(self) -> None:
        question = "Which country the composer of film Thunder On The Hill is from?"
        normalized = "Which country is the composer of film Thunder On The Hill from?"
        masked = "Which country is the composer of film ENTITYA from?"
        llm = StaticPreprocessLLM(
            {
                "explicit_entities": [
                    {
                        "surface": "Thunder On The Hill",
                        "type": "Work",
                    }
                ],
                "normalized_question": normalized,
                "normalization_changed": True,
                "normalization_note": "Inserted auxiliary is for wh-question order.",
                "warnings": [],
            }
        )
        parser = RecordingHanLPSDPParser()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_hanlp_sdp_pipeline(
                record=QuestionRecord(question=question),
                index=1,
                preprocessor=EntityMaskingPreprocessor(llm),
                parser=parser,
                debug=True,
                debug_dir=tmpdir,
                skip_step5=True,
            )
            debug_file = Path(result["token_reasoning_structure"].debug_file or "")
            self.assertTrue(debug_file.exists())
            debug_payload = json.loads(debug_file.read_text(encoding="utf-8"))

        preprocess_result = result["preprocess_result"]
        self.assertEqual(llm.calls, 1)
        self.assertEqual(preprocess_result.masked_question, masked)
        self.assertEqual(preprocess_result.sdp_input_sentence, masked)
        self.assertEqual(parser.text, masked)
        self.assertTrue(preprocess_result.normalization_changed)
        self.assertEqual(debug_payload["original_question"], question)
        self.assertEqual(debug_payload["normalized_question"], normalized)
        self.assertEqual(debug_payload["masked_question"], masked)
        self.assertEqual(
            debug_payload["step4_path_extraction"], "entity_branch_best_paths"
        )

    def test_hanlp_sdp_pipeline_can_skip_step5(self) -> None:
        record = QuestionRecord(
            question="Who is older, Ryan Tubridy or Mauro Massironi?"
        )
        parser = FakeHanLPSDPParser()
        llm = FakePreprocessLLM()

        result = run_hanlp_sdp_pipeline(
            record=record,
            index=1,
            preprocessor=EntityMaskingPreprocessor(llm),
            parser=parser,
            skip_step5=True,
        )

        self.assertEqual(llm.calls, 1)
        self.assertIsNone(result["atomic_question_dag"])
        stream = io.StringIO()
        with redirect_stdout(stream):
            print_hanlp_sdp_result(1, record, result)
        self.assertIn("(skipped: Step5 disabled)", stream.getvalue())

    def test_pipeline_runs_no_path_step5_when_step4_has_no_paths(self) -> None:
        question = "What is the capital of France?"
        llm = NoPathPipelineLLM(question)

        result = run_hanlp_sdp_pipeline(
            record=QuestionRecord(question=question),
            index=1,
            preprocessor=EntityMaskingPreprocessor(llm),
            parser=NoPathHanLPSDPParser(),
        )

        compiled = result["token_reasoning_structure"]
        self.assertEqual(compiled.path_type, "no_entity_branch_path")
        self.assertEqual(compiled.paths, [])
        self.assertEqual(llm.calls, 2)
        self.assertEqual(
            json.loads(llm.no_path_user_prompt),
            {"original_question": question},
        )
        dag = result["atomic_question_dag"]
        self.assertTrue(dag.valid, dag.validation_errors)
        self.assertEqual([node.question for node in dag.nodes], [question])

    def test_pipeline_sends_restored_entity_branch_paths_to_step5(self) -> None:
        question = "Which film has the director who was born later, Illusions (1982 Film) or It'S A Wonderful Afterlife?"
        llm = IllusionsPipelineLLM(question)
        result = run_hanlp_sdp_pipeline(
            record=QuestionRecord(question=question),
            index=1,
            preprocessor=EntityMaskingPreprocessor(llm),
            parser=FakeIllusionsBornLaterParser(),
        )

        compiled = result["token_reasoning_structure"]
        self.assertEqual(compiled.path_type, "entity_branch_best_paths")
        self.assertEqual(
            [list(path.nodes) for path in compiled.paths],
            [
                ["ENTITYA", "film", "has", "director", "born", "later"],
                ["ENTITYB", "film", "has", "director", "born", "later"],
            ],
        )
        payload = json.loads(llm.step5_user_prompt)
        self.assertEqual(
            payload["step4_paths"],
            [
                ["Illusions (1982 Film)", "film", "has", "director", "born", "later"],
                [
                    "It'S A Wonderful Afterlife",
                    "film",
                    "has",
                    "director",
                    "born",
                    "later",
                ],
            ],
        )
        self.assertNotIn("ENTITYA", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("ENTITYB", json.dumps(payload, ensure_ascii=False))
        dag = result["atomic_question_dag"]
        self.assertTrue(dag.valid, dag.validation_errors)
        self.assertEqual(dag.nodes[-1].depends_on, ("q2", "q4"))


class TriSDPEntityBranchCompilerTest(unittest.TestCase):
    def test_hanlp_parser_exposes_only_pas_edges_and_graphs(self) -> None:
        parser = HanLPSDPParser()
        tokens = ["Who", "older", "ENTITYA"]
        parser._pipeline = lambda text: {  # type: ignore[assignment]
            "tok": tokens,
            "sdp/dm": [[(2, "ARG1")], [(0, "root")], [(2, "ARG2")]],
            "sdp/pas": [[(2, "ARG1")], [(0, "root")], [(2, "ARG1")]],
            "sdp/psd": [[(2, "ACT-arg")], [(0, "root")], [(2, "PAT-arg")]],
            "udep": [[(2, "nsubj")], [(0, "root")], [(2, "obj")]],
        }
        parser.model_label = "fake"

        result = parser.parse("Who older ENTITYA")

        self.assertEqual(set(result.sdp_graphs), {"sdp/pas"})
        self.assertTrue(result.edges)
        self.assertEqual({edge.formalism for edge in result.edges}, {"sdp/pas"})
        self.assertEqual(result.syntax_heads, {"1": 2, "2": 0, "3": 2})
        self.assertEqual(result.syntax_head_source, "udep")
        self.assertIn("sdp/dm", result.raw)
        self.assertIn("sdp/psd", result.raw)

    def test_hanlp_parser_offsets_multi_sentence_syntax_heads(self) -> None:
        parser = HanLPSDPParser()
        parser._pipeline = lambda text: {  # type: ignore[assignment]
            "tok": [["ENTITYA", "runs"], ["ENTITYB", "walks"]],
            "sdp/pas": [[], []],
            "dep": [[(2, "nsubj"), (0, "root")], [(2, "nsubj"), (0, "root")]],
        }
        parser.model_label = "fake"

        result = parser.parse("ENTITYA runs. ENTITYB walks.")

        self.assertEqual(result.tokens, ["ENTITYA", "runs", "ENTITYB", "walks"])
        self.assertEqual(result.syntax_heads, {"1": 2, "2": 0, "3": 4, "4": 0})
        self.assertEqual(result.syntax_head_source, "dep")

    def test_pas_missing_does_not_fall_back_to_dm_or_psd(self) -> None:
        parser = HanLPSDPParser()
        parser._pipeline = lambda text: {  # type: ignore[assignment]
            "tok": ["ENTITYA", "target"],
            "sdp/dm": [[], [(1, "ARG1")]],
            "sdp/psd": [[], [(1, "PAT-arg")]],
        }
        parser.model_label = "fake"

        parsed = parser.parse("ENTITYA target")
        self.assertEqual(parsed.sdp_graphs, {})
        self.assertEqual(parsed.edges, [])
        self.assertTrue(any("sdp/pas" in warning for warning in parsed.warnings))

        compiled = compile_token_reasoning_structure(
            _hanlp_result(
                "ENTITYA target",
                ["ENTITYA", "target"],
                [_dm("ENTITYA", "ARG1", "target", 1, 2)],
            ),
            ["ENTITYA"],
        )
        self.assertEqual(compiled.paths, [])
        self.assertTrue(
            any("no sdp/pas edges" in warning for warning in compiled.warnings)
        )

    def test_build_evidence_graph_filters_mixed_input_to_pas_only(self) -> None:
        result = _hanlp_result(
            "ENTITYA target.",
            ["ENTITYA", "target", "."],
            [
                _dm("ENTITYA", "ARG1", "target", 1, 2),
                _pas("target", "ARG1", "ENTITYA", 2, 1),
                _psd("target", "PAT-arg", "ENTITYA", 2, 1),
            ],
        )

        state = build_evidence_graph(result)

        self.assertEqual(
            {item["formalism"] for item in state.normalized_edges}, {"sdp/pas"}
        )
        self.assertEqual(len(state.raw_edges), 1)
        self.assertTrue(
            any("ignored non-PAS edge" in warning for warning in state.warnings)
        )

    def test_pas_core_content_relation_costs_are_one(self) -> None:
        for relation in [
            "verb_ARG1",
            "VERB-ARG2",
            "verb/ARG3",
            "noun_ARG1",
            "noun.ARG2",
            "adj_ARG1",
        ]:
            with self.subTest(relation=relation):
                state = build_evidence_graph(
                    _hanlp_result(
                        "head dep",
                        ["head", "dep"],
                        [_pas("head", relation, "dep", 1, 2)],
                    )
                )
                self.assertEqual(
                    _edge_cost(_edge_between_texts(state, "head", "dep")), 1
                )

    def test_pas_structural_relation_costs_are_two(self) -> None:
        for relation in [
            "conj_ARG1",
            "conj_ARG2",
            "relative_ARG1",
            "relative_ARG2",
            "comp_ARG1",
            "comp_MOD",
            "verb_MOD",
        ]:
            with self.subTest(relation=relation):
                state = build_evidence_graph(
                    _hanlp_result(
                        "head dep",
                        ["head", "dep"],
                        [_pas("head", relation, "dep", 1, 2)],
                    )
                )
                self.assertEqual(
                    _edge_cost(_edge_between_texts(state, "head", "dep")), 2
                )

    def test_aggregated_pas_edge_uses_lowest_finite_relation_cost(self) -> None:
        state = build_evidence_graph(
            _hanlp_result(
                "head dep",
                ["head", "dep"],
                [
                    _pas("head", "comp_ARG1", "dep", 1, 2),
                    _pas("head", "verb_ARG1", "dep", 1, 2),
                ],
            )
        )

        self.assertEqual(_edge_cost(_edge_between_texts(state, "head", "dep")), 1)

    def test_non_noise_pas_labels_cost_three_and_remain_searchable(
        self,
    ) -> None:
        cost_three_relations = [
            "prep_ARG1",
            "prep_ARG2",
            "poss_ARG1",
            "poss_ARG2",
            "coord_ARG1",
            "coord_ARG2",
            "det_ARG1",
            "aux_ARG1",
            "unlisted_relation",
        ]
        for relation in cost_three_relations:
            with self.subTest(relation=relation):
                state = build_evidence_graph(
                    _hanlp_result(
                        "ENTITYA target",
                        ["ENTITYA", "target"],
                        [_pas("ENTITYA", relation, "target", 1, 2)],
                    )
                )
                edge = _edge_between_texts(state, "ENTITYA", "target")
                graph = _semantic_path_search_graph(state.nodes, state.edges)

                self.assertEqual(_edge_cost(edge), 3)
                self.assertTrue(_has_search_edge(graph, "1", "2"))

    def test_pure_noise_pas_labels_remain_excluded_from_search_graph(self) -> None:
        blocked_relations = [
            "punct_ARG1",
            "quote_ARG1",
            "lparen_ARG1",
            "rbracket_ARG1",
            "root",
        ]
        for relation in blocked_relations:
            with self.subTest(relation=relation):
                state = build_evidence_graph(
                    _hanlp_result(
                        "ENTITYA target",
                        ["ENTITYA", "target"],
                        [_pas("ENTITYA", relation, "target", 1, 2)],
                    )
                )
                edge = _edge_between_texts(state, "ENTITYA", "target")
                graph = _semantic_path_search_graph(state.nodes, state.edges)

                self.assertIsNone(_edge_cost(edge))
                self.assertFalse(_has_search_edge(graph, "1", "2"))

    def test_unrecognized_pas_labels_use_cost_three_fallback(
        self,
    ) -> None:
        compiled = compile_token_reasoning_structure(
            _hanlp_result(
                "ENTITYA target",
                ["ENTITYA", "target"],
                [_pas("ENTITYA", "unknown_relation", "target", 1, 2)],
            ),
            ["ENTITYA"],
        )

        self.assertEqual([path.nodes for path in compiled.paths], [["ENTITYA", "target"]])
        self.assertEqual(
            compiled.global_selection["entity_branch_results"][0]["selected"][
                "path_cost"
            ],
            3,
        )
        self.assertEqual(compiled.debug_payload["unsearchable_pas_edges"], [])

    def test_answer_anchor_debug_fields_are_removed_from_step4_mainline(self) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), ["ENTITYA", "ENTITYB"]
        )

        self.assertIsNone(compiled.answer_anchor)
        self.assertIsNone(compiled.answer_anchor_id)
        self.assertEqual(compiled.anchor_path_results, [])
        self.assertIn("repaired_evidence_edges", compiled.debug_payload)
        for legacy_key in [
            "answer_anchor_candidates",
            "global_candidates",
            "selected_global_candidate",
            "selected_global_candidates",
            "selected_global_selection",
            "query_focus",
            "candidate_paths",
            "parallel_entity_sets",
        ]:
            self.assertNotIn(legacy_key, compiled.debug_payload)
        serialized_debug = json.dumps(compiled.debug_payload, ensure_ascii=False)
        for source in [
            "typed_wh_slot",
            "wh_anchor",
            "root_projection",
            "modifier_projection",
            "comparative_focus",
            "bare_wh_predicate_root",
            "clause_predicate",
            "explicit_entity",
        ]:
            self.assertNotIn(source, serialized_debug)

    def test_each_explicit_entity_is_an_independent_branch_without_global_competition(
        self,
    ) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), ["ENTITYA", "ENTITYB"]
        )

        self.assertEqual(compiled.path_type, "entity_branch_best_paths")
        self.assertEqual(len(compiled.paths), 2)
        self.assertEqual(
            [path.nodes[0] for path in compiled.paths], ["ENTITYA", "ENTITYB"]
        )
        self.assertEqual(
            [path.nodes[-1] for path in compiled.paths], ["later", "later"]
        )
        self.assertEqual(
            compiled.global_selection["selection_type"], "entity_branch_best_paths"
        )
        self.assertEqual(len(compiled.global_selection["entity_branch_results"]), 2)

    def test_semantic_boundaries_are_degree_one_non_entity_nodes_in_filtered_graph(
        self,
    ) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), ["ENTITYA", "ENTITYB"]
        )
        boundaries = compiled.global_selection["semantic_boundary_nodes"]
        boundary_by_text = {item["text"]: item for item in boundaries}

        self.assertEqual(boundary_by_text["later"]["degree"], 1)
        self.assertNotIn("Which", boundary_by_text)
        self.assertNotIn("film", boundary_by_text)
        self.assertIn("director", boundary_by_text)
        self.assertNotIn("ENTITYA", boundary_by_text)
        self.assertNotIn("ENTITYB", boundary_by_text)

    def test_wh_function_word_can_be_semantic_boundary(self) -> None:
        result = _hanlp_result(
            "Which film ENTITYA?",
            ["Which", "film", "ENTITYA", "?"],
            [
                _pas("film", "noun_ARG1", "Which", 2, 1),
                _pas("film", "noun_ARG2", "ENTITYA", 2, 3),
            ],
        )
        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])
        boundaries = {
            item["text"]: item
            for item in compiled.global_selection["semantic_boundary_nodes"]
        }

        self.assertIn("Which", boundaries)
        self.assertEqual(boundaries["Which"]["kind"], "function")

    def test_root_punctuation_function_and_coord_words_are_not_boundaries(self) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), ["ENTITYA", "ENTITYB"]
        )
        boundary_texts = {
            item["text"]
            for item in compiled.global_selection["semantic_boundary_nodes"]
        }

        for excluded in {"ROOT", ",", "?", "has", "the", "or"}:
            self.assertNotIn(excluded, boundary_texts)

    def test_function_words_remain_searchable_internal_bridges(self) -> None:
        state = build_evidence_graph(
            _hanlp_result(
                "ENTITYA from country same?",
                ["ENTITYA", "from", "country", "same", "?"],
                [
                    _pas("ENTITYA", "verb_ARG1", "from", 1, 2),
                    _pas("from", "prep_ARG2", "country", 2, 3),
                    _pas("country", "adj_ARG1", "same", 3, 4),
                ],
            )
        )
        graph = _semantic_path_search_graph(state.nodes, state.edges)

        self.assertEqual(_edge_cost(_edge_between_texts(state, "from", "country")), 3)
        self.assertTrue(_has_search_edge(graph, "1", "2"))
        self.assertTrue(_has_search_edge(graph, "2", "3"))
        path, cost = _shortest_semantic_boundary_path(
            graph, state.nodes, "1", "4", blocked_internal_ids=set()
        )
        self.assertEqual(
            [state.nodes[node_id].text for node_id in path],
            ["ENTITYA", "from", "country", "same"],
        )
        self.assertEqual(cost, 5)

    def test_coordination_and_scope_words_remain_searchable_internal_bridges(
        self,
    ) -> None:
        for bridge in ["and", "or", "both", "between", "than"]:
            with self.subTest(bridge=bridge):
                state = build_evidence_graph(
                    _hanlp_result(
                        f"ENTITYA {bridge} country",
                        ["ENTITYA", bridge, "country"],
                        [
                            _pas("ENTITYA", "coord_ARG1", bridge, 1, 2),
                            _pas(bridge, "coord_ARG2", "country", 2, 3),
                        ],
                    )
                )
                graph = _semantic_path_search_graph(state.nodes, state.edges)

                self.assertEqual(
                    _edge_cost(_edge_between_texts(state, "ENTITYA", bridge)), 3
                )
                self.assertTrue(_has_search_edge(graph, "1", "2"))
                self.assertTrue(_has_search_edge(graph, "2", "3"))
                path, cost = _shortest_semantic_boundary_path(
                    graph, state.nodes, "1", "3", blocked_internal_ids=set()
                )
                self.assertEqual(
                    [state.nodes[node_id].text for node_id in path],
                    ["ENTITYA", bridge, "country"],
                )
                self.assertEqual(cost, 6)

    def test_root_and_punctuation_nodes_cannot_enter_search_paths(self) -> None:
        state = build_evidence_graph(
            _hanlp_result(
                "ENTITYA ? country",
                ["ENTITYA", "?", "country"],
                [
                    _pas("ENTITYA", "verb_ARG1", "?", 1, 2),
                    _pas("?", "verb_ARG2", "country", 2, 3),
                ],
            )
        )
        graph = _semantic_path_search_graph(state.nodes, state.edges)

        self.assertFalse(_has_search_edge(graph, "1", "2"))
        self.assertFalse(_has_search_edge(graph, "2", "3"))

    def test_dijkstra_uses_existing_edge_cost_for_lowest_cost_path(self) -> None:
        result = _hanlp_result(
            "ENTITYA can reach goal.",
            ["ENTITYA", "can", "mid", "goal", "."],
            [
                _pas("ENTITYA", "unknown_relation", "goal", 1, 4),
                _pas("mid", "verb_ARG1", "ENTITYA", 3, 1),
                _pas("mid", "verb_ARG2", "goal", 3, 4),
            ],
        )
        state = build_evidence_graph(result)
        graph = _semantic_path_search_graph(state.nodes, state.edges)

        path, cost = _shortest_semantic_boundary_path(
            graph, state.nodes, "1", "4", blocked_internal_ids=set()
        )

        self.assertEqual(
            [state.nodes[node_id].text for node_id in path], ["ENTITYA", "mid", "goal"]
        )
        self.assertEqual(cost, 2)

    def test_path_cost_sums_edge_costs_without_infinite_penalty(self) -> None:
        result = _hanlp_result(
            "ENTITYA mid goal.",
            ["ENTITYA", "mid", "goal", "."],
            [
                _pas("mid", "verb_ARG1", "ENTITYA", 2, 1),
                _pas("mid", "verb_ARG2", "goal", 2, 3),
            ],
        )
        state = build_evidence_graph(result)
        graph = _semantic_path_search_graph(state.nodes, state.edges)

        path, cost = _shortest_semantic_boundary_path(
            graph, state.nodes, "1", "3", blocked_internal_ids=set()
        )

        self.assertEqual(cost, 2)
        self.assertEqual(_path_cost_from_edge_map(state.edges, path), 2)

    def test_single_structural_edge_ties_two_direct_semantic_edges_by_total_cost(
        self,
    ) -> None:
        result = _hanlp_result(
            "ENTITYA mid target.",
            ["ENTITYA", "mid", "target", "."],
            [
                _pas("mid", "verb_ARG1", "ENTITYA", 2, 1),
                _pas("mid", "verb_ARG2", "target", 2, 3),
                _pas("target", "comp_ARG1", "ENTITYA", 3, 1),
            ],
        )
        state = build_evidence_graph(result)
        graph = _semantic_path_search_graph(state.nodes, state.edges)

        path, cost = _shortest_semantic_boundary_path(
            graph, state.nodes, "1", "3", blocked_internal_ids=set()
        )

        self.assertEqual(cost, 2)
        self.assertEqual(
            [state.nodes[node_id].text for node_id in path], ["ENTITYA", "target"]
        )

    def test_entity_branch_sp_tie_keeps_first_candidate_without_cost_tiebreak(
        self,
    ) -> None:
        result = _hanlp_result(
            "ENTITYA cheap expensive?",
            ["ENTITYA", "cheap", "expensive", "?"],
            [
                _pas("cheap", "comp_ARG1", "ENTITYA", 2, 1),
                _pas("expensive", "adj_ARG1", "ENTITYA", 3, 1),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])
        branch = compiled.global_selection["entity_branch_results"][0]
        candidates = {
            candidate["boundary"]: candidate for candidate in branch["candidates"]
        }

        self.assertEqual(candidates["cheap"]["path_cost"], 2)
        self.assertEqual(candidates["expensive"]["path_cost"], 1)
        self.assertEqual(
            candidates["cheap"]["sp_score"], candidates["expensive"]["sp_score"]
        )
        self.assertEqual(branch["selected"]["boundary"], "cheap")
        self.assertNotIn("rank", candidates["cheap"])
        self.assertNotIn("rank_components", candidates["cheap"])

    def test_semantic_path_score_prefers_complete_compact_paths(self) -> None:
        state = build_evidence_graph(
            _hanlp_result(
                "ENTITYA film can director born later?",
                ["ENTITYA", "film", "can", "director", "born", "later", "?"],
                [],
            )
        )
        branch_semantic_ids = {"2", "4", "5", "6"}

        compact_score, compact_components = _semantic_path_score(
            state=state,
            entity_id="1",
            path_ids=["1", "2", "4", "5", "6"],
            branch_semantic_ids=branch_semantic_ids,
        )
        noisy_score, noisy_components = _semantic_path_score(
            state=state,
            entity_id="1",
            path_ids=["1", "2", "3", "4", "5", "6"],
            branch_semantic_ids=branch_semantic_ids,
        )
        short_score, short_components = _semantic_path_score(
            state=state,
            entity_id="1",
            path_ids=["1", "2"],
            branch_semantic_ids=branch_semantic_ids,
        )

        self.assertEqual(compact_score, 1.0)
        self.assertAlmostEqual(noisy_score, 8 / 9)
        self.assertEqual(short_score, 0.4)
        self.assertGreater(compact_score, noisy_score)
        self.assertGreater(noisy_score, short_score)
        self.assertEqual(compact_components["covered_semantic_nodes_count"], 4)
        self.assertEqual(noisy_components["path_nodes_without_entity_count"], 5)
        self.assertEqual(short_components["covered_semantic_node_ids"], ["2"])

    def test_entity_branch_sp_uses_per_entity_candidate_semantics(self) -> None:
        result = _hanlp_result(
            "ENTITYA alpha beta ENTITYB gamma delta?",
            ["ENTITYA", "alpha", "beta", "ENTITYB", "gamma", "delta", "?"],
            [
                _pas("alpha", "verb_ARG1", "ENTITYA", 2, 1),
                _pas("alpha", "verb_ARG2", "beta", 2, 3),
                _pas("gamma", "verb_ARG1", "ENTITYB", 5, 4),
                _pas("gamma", "verb_ARG2", "delta", 5, 6),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])
        branch_by_entity = {
            branch["entity"]: branch
            for branch in compiled.global_selection["entity_branch_results"]
        }

        entity_a_semantics = set(
            branch_by_entity["ENTITYA"]["selected"]["sp_components"][
                "branch_semantic_nodes"
            ]
        )
        entity_b_semantics = set(
            branch_by_entity["ENTITYB"]["selected"]["sp_components"][
                "branch_semantic_nodes"
            ]
        )

        self.assertEqual(entity_a_semantics, {"alpha", "beta"})
        self.assertEqual(entity_b_semantics, {"gamma", "delta"})
        self.assertTrue(entity_a_semantics.isdisjoint(entity_b_semantics))

    def test_other_explicit_entities_cannot_be_internal_nodes(self) -> None:
        result = _hanlp_result(
            "ENTITYA ENTITYB target?",
            ["ENTITYA", "ENTITYB", "target", "?"],
            [
                _pas("ENTITYB", "verb_ARG1", "ENTITYA", 2, 1),
                _pas("ENTITYB", "verb_ARG2", "target", 2, 3),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(
            [list(path.nodes) for path in compiled.paths], [["ENTITYB", "target"]]
        )
        self.assertTrue(
            any(
                "ENTITYA" in warning and "no reachable semantic boundary" in warning
                for warning in compiled.warnings
            )
        )

    def test_pas_preposition_contraction_requires_paired_arguments(self) -> None:
        paired = _hanlp_result(
            "director of ENTITYA born?",
            ["director", "of", "ENTITYA", "born", "?"],
            [
                _pas("of", "prep_ARG1", "director", 2, 1),
                _pas("of", "prep_ARG2", "ENTITYA", 2, 3),
                _pas("born", "verb_ARG1", "director", 4, 1),
            ],
        )

        compiled = compile_token_reasoning_structure(paired, ["ENTITYA"])
        preposition_edges = _virtual_edges_by_rule(
            compiled, "pas_preposition_contraction"
        )

        self.assertEqual(len(preposition_edges), 1)
        self.assertEqual(
            {preposition_edges[0]["source_text"], preposition_edges[0]["target_text"]},
            {"director", "ENTITYA"},
        )
        self.assertEqual(preposition_edges[0]["edge_cost"], 1)
        provenance = preposition_edges[0]["provenance"][0]
        self.assertEqual(provenance["preposition"], "of")
        self.assertEqual(provenance["formalism"], "sdp/pas")
        self.assertEqual(len(provenance["source_edges"]), 2)

        incomplete = _hanlp_result(
            "director of ENTITYA?",
            ["director", "of", "ENTITYA", "?"],
            [_pas("of", "prep_ARG1", "director", 2, 1)],
        )
        incomplete_compiled = compile_token_reasoning_structure(incomplete, ["ENTITYA"])
        self.assertEqual(
            _virtual_edges_by_rule(incomplete_compiled, "pas_preposition_contraction"),
            [],
        )

    def test_pas_possessive_contraction_merges_duplicate_markers(self) -> None:
        result = _hanlp_result(
            "ENTITYA ' s mother died?",
            ["ENTITYA", "'", "s", "mother", "died", "?"],
            [
                _pas("'", "poss_ARG2", "ENTITYA", 2, 1),
                _pas("'", "poss_ARG1", "mother", 2, 4),
                _pas("s", "poss_ARG2", "ENTITYA", 3, 1),
                _pas("s", "adj_ARG1", "mother", 3, 4),
                _pas("died", "verb_ARG1", "mother", 5, 4),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])
        possessive_edges = _virtual_edges_by_rule(
            compiled, "pas_possessive_contraction"
        )

        self.assertEqual(len(possessive_edges), 1)
        self.assertEqual(
            {possessive_edges[0]["source_text"], possessive_edges[0]["target_text"]},
            {"ENTITYA", "mother"},
        )
        self.assertEqual(possessive_edges[0]["edge_cost"], 1)
        self.assertEqual(len(possessive_edges[0]["provenance"]), 2)
        self.assertEqual(
            {item["marker"] for item in possessive_edges[0]["provenance"]}, {"'", "s"}
        )
        self.assertFalse(
            any(
                {"'", "s"}.intersection(
                    {edge["source_text"], edge["target_text"]}
                )
                for edge in compiled.debug_payload["repaired_evidence_edges"]
            )
        )

    def test_pas_possessive_contraction_keeps_single_marker_behavior(self) -> None:
        result = _hanlp_result(
            "ENTITYA 's father died?",
            ["ENTITYA", "'s", "father", "died", "?"],
            [
                _pas("'s", "poss_ARG2", "ENTITYA", 2, 1),
                _pas("'s", "noun_ARG1", "father", 2, 3),
                _pas("died", "verb_ARG1", "father", 4, 3),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])
        possessive_edges = _virtual_edges_by_rule(
            compiled, "pas_possessive_contraction"
        )

        self.assertEqual(len(possessive_edges), 1)
        self.assertEqual(
            {possessive_edges[0]["source_text"], possessive_edges[0]["target_text"]},
            {"ENTITYA", "father"},
        )
        self.assertEqual(possessive_edges[0]["edge_cost"], 1)
        self.assertEqual(possessive_edges[0]["provenance"][0]["marker"], "'s")
        self.assertFalse(
            any(
                "'s" in {edge["source_text"], edge["target_text"]}
                for edge in compiled.debug_payload["repaired_evidence_edges"]
            )
        )

    def test_pas_possessive_contraction_merges_split_quote_and_s_markers(
        self,
    ) -> None:
        result = _hanlp_result(
            "ENTITYA ' s father died?",
            ["ENTITYA", "'", "s", "father", "died", "?"],
            [
                _pas("'", "poss_ARG2", "ENTITYA", 2, 1),
                _pas("s", "poss_ARG1", "father", 3, 4),
                _pas("died", "verb_ARG1", "father", 5, 4),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])
        possessive_edges = _virtual_edges_by_rule(
            compiled, "pas_possessive_contraction"
        )

        self.assertEqual(len(possessive_edges), 1)
        self.assertEqual(
            {possessive_edges[0]["source_text"], possessive_edges[0]["target_text"]},
            {"ENTITYA", "father"},
        )
        self.assertEqual(possessive_edges[0]["edge_cost"], 1)
        provenance = possessive_edges[0]["provenance"][0]
        self.assertEqual(provenance["marker"], "'s")
        self.assertEqual(provenance["marker_ids"], ["2", "3"])
        self.assertEqual(len(provenance["source_edges"]), 2)
        self.assertFalse(
            any(
                {"'", "s"}.intersection(
                    {edge["source_text"], edge["target_text"]}
                )
                for edge in compiled.debug_payload["repaired_evidence_edges"]
            )
        )

    def test_pas_preposition_and_possessive_contractions_do_not_receive_derived_penalty(
        self,
    ) -> None:
        preposition = compile_token_reasoning_structure(
            _hanlp_result(
                "director of ENTITYA born?",
                ["director", "of", "ENTITYA", "born", "?"],
                [
                    _pas("of", "prep_ARG1", "director", 2, 1),
                    _pas("of", "prep_ARG2", "ENTITYA", 2, 3),
                    _pas("born", "verb_ARG1", "director", 4, 1),
                ],
            ),
            ["ENTITYA"],
        )
        possessive = compile_token_reasoning_structure(
            _hanlp_result(
                "ENTITYA ' mother died?",
                ["ENTITYA", "'", "mother", "died", "?"],
                [
                    _pas("'", "poss_ARG2", "ENTITYA", 2, 1),
                    _pas("'", "poss_ARG1", "mother", 2, 3),
                    _pas("died", "verb_ARG1", "mother", 4, 3),
                ],
            ),
            ["ENTITYA"],
        )

        for compiled in [preposition, possessive]:
            with self.subTest(path=compiled.paths[0].nodes):
                selected = compiled.global_selection["entity_branch_results"][0][
                    "selected"
                ]
                self.assertEqual(selected["path_cost"], 2)
                self.assertEqual(
                    [item["edge_cost"] for item in selected["edge_costs"]], [1, 1]
                )
                self.assertNotIn("rank", selected)
                self.assertNotIn("rank_components", selected)
                self.assertIn("sp_score", selected)
                self.assertIn("sp_components", selected)

    def test_plain_content_s_is_not_possessive_contracted(self) -> None:
        result = _hanlp_result(
            "ENTITYA s signal?",
            ["ENTITYA", "s", "signal", "?"],
            [
                _pas("s", "ARG1", "ENTITYA", 2, 1),
                _pas("s", "ARG2", "signal", 2, 3),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])

        self.assertEqual(
            _virtual_edges_by_rule(compiled, "pas_possessive_contraction"), []
        )

    def test_pas_coordination_contraction_through_and(self) -> None:
        result = _hanlp_result(
            "Do both films ENTITYA and ENTITYB have directors from the same country?",
            [
                "Do",
                "both",
                "films",
                "ENTITYA",
                "and",
                "ENTITYB",
                "have",
                "directors",
                "from",
                "the",
                "same",
                "country",
                "?",
            ],
            [
                _pas("have", "verb_ARG2", "and", 7, 5),
                _pas("and", "coord_ARG1", "ENTITYA", 5, 4),
                _pas("and", "coord_ARG2", "ENTITYB", 5, 6),
                _pas("have", "verb_ARG1", "directors", 7, 8),
                _pas("directors", "noun_ARG1", "country", 8, 12),
                _pas("country", "adj_ARG1", "same", 12, 11),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])
        contraction_edges = _virtual_edges_by_rule(
            compiled, "pas_coordination_contraction"
        )

        self.assertEqual(len(contraction_edges), 2)
        self.assertEqual(
            {
                frozenset((edge["source_text"], edge["target_text"]))
                for edge in contraction_edges
            },
            {frozenset(("have", "ENTITYA")), frozenset(("have", "ENTITYB"))},
        )
        for edge in contraction_edges:
            self.assertTrue(edge["derived"])
            self.assertEqual(edge["edge_cost"], 2)
            self.assertEqual(
                edge["provenance"][0]["semantic_relation"], "verb_arg2"
            )
        self.assertFalse(
            any(
                "and" in {edge["source_text"], edge["target_text"]}
                for edge in compiled.debug_payload["repaired_evidence_edges"]
            )
        )

    def test_pas_coordination_contraction_through_or(self) -> None:
        result = _hanlp_result(
            "Does ENTITYA or ENTITYB have a director?",
            ["Does", "ENTITYA", "or", "ENTITYB", "have", "director", "?"],
            [
                _pas("have", "verb_ARG2", "or", 5, 3),
                _pas("or", "coord_ARG1", "ENTITYA", 3, 2),
                _pas("or", "coord_ARG2", "ENTITYB", 3, 4),
                _pas("have", "verb_ARG1", "director", 5, 6),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])
        contraction_edges = _virtual_edges_by_rule(
            compiled, "pas_coordination_contraction"
        )

        self.assertEqual(
            {
                frozenset((edge["source_text"], edge["target_text"]))
                for edge in contraction_edges
            },
            {frozenset(("have", "ENTITYA")), frozenset(("have", "ENTITYB"))},
        )
        self.assertTrue(all(edge["edge_cost"] == 2 for edge in contraction_edges))
        self.assertFalse(
            any(
                "or" in {edge["source_text"], edge["target_text"]}
                for edge in compiled.debug_payload["repaired_evidence_edges"]
            )
        )

    def test_pas_coordination_contraction_skips_predicate_coordination(self) -> None:
        result = _hanlp_result(
            "Do ENTITYA write and ENTITYB direct?",
            ["Do", "ENTITYA", "write", "and", "ENTITYB", "direct", "?"],
            [
                _pas("Do", "verb_ARG2", "and", 1, 4),
                _pas("and", "coord_ARG1", "write", 4, 3),
                _pas("and", "coord_ARG2", "direct", 4, 6),
                _pas("write", "verb_ARG1", "ENTITYA", 3, 2),
                _pas("direct", "verb_ARG1", "ENTITYB", 6, 5),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(
            _virtual_edges_by_rule(compiled, "pas_coordination_contraction"), []
        )

    def test_pas_coordination_candidates_attach_by_syntactic_head(self) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), ["ENTITYA", "ENTITYB"]
        )
        attachment_edges = _virtual_edges_by_rule(
            compiled, "pas_coordination_candidate_attachment"
        )

        self.assertEqual(len(attachment_edges), 2)
        self.assertEqual(
            {
                frozenset((edge["source_text"], edge["target_text"]))
                for edge in attachment_edges
            },
            {frozenset(("ENTITYA", "film")), frozenset(("ENTITYB", "film"))},
        )
        for edge in attachment_edges:
            self.assertEqual(edge["edge_cost"], 2)
            provenance = edge["provenance"][0]
            self.assertEqual(provenance["syntactic_attachment"], "film")
            self.assertEqual(provenance["basis"], "syntactic_coordination_head")
            self.assertEqual(provenance["formalism"], "sdp/pas")
            self.assertEqual(provenance["syntax_head_source"], "test/udep")
            self.assertIn(provenance["member_id"], provenance["syntax_head_chain"])
            self.assertEqual(len(provenance["coordination_edges"]), 2)

    def test_pas_coordination_attachment_skips_without_syntax_heads(self) -> None:
        result = _hanlp_result(
            "ENTITYA or ENTITYB?",
            ["ENTITYA", "or", "ENTITYB", "?"],
            [
                _pas("or", "coord_ARG1", "ENTITYA", 2, 1),
                _pas("or", "coord_ARG2", "ENTITYB", 2, 3),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(
            _virtual_edges_by_rule(compiled, "pas_coordination_candidate_attachment"),
            [],
        )
        self.assertTrue(
            any(
                "syntactic dependency heads are missing" in warning
                for warning in compiled.warnings
            )
        )

    def test_pas_coordination_attachment_skips_root_syntax_attachment(self) -> None:
        result = _hanlp_result(
            "ENTITYA or ENTITYB?",
            ["ENTITYA", "or", "ENTITYB", "?"],
            [
                _pas("or", "coord_ARG1", "ENTITYA", 2, 1),
                _pas("or", "coord_ARG2", "ENTITYB", 2, 3),
            ],
            syntax_heads={"1": 0, "2": 3, "3": 1},
            syntax_head_source="test/udep",
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(
            _virtual_edges_by_rule(compiled, "pas_coordination_candidate_attachment"),
            [],
        )
        self.assertTrue(
            any(
                "syntactic attachment ROOT[0] is invalid" in warning
                for warning in compiled.warnings
            )
        )

    def test_pas_coordination_attachment_skips_different_syntactic_attachments(
        self,
    ) -> None:
        result = _hanlp_result(
            "film book ENTITYA or ENTITYB?",
            ["film", "book", "ENTITYA", "or", "ENTITYB", "?"],
            [
                _pas("or", "coord_ARG1", "ENTITYA", 4, 3),
                _pas("or", "coord_ARG2", "ENTITYB", 4, 5),
            ],
            syntax_heads={"3": 1, "4": 5, "5": 2},
            syntax_head_source="test/udep",
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(
            _virtual_edges_by_rule(compiled, "pas_coordination_candidate_attachment"),
            [],
        )
        self.assertTrue(
            any(
                "different syntactic attachments" in warning
                for warning in compiled.warnings
            )
        )

    def test_pas_coordination_attachment_connects_three_entity_parent_chain(
        self,
    ) -> None:
        result = _hanlp_result(
            "Which film ENTITYA or ENTITYB or ENTITYC?",
            ["Which", "film", "ENTITYA", "or", "ENTITYB", "ENTITYC", "?"],
            [
                _pas("film", "noun_ARG1", "Which", 2, 1),
                _pas("or", "coord_ARG1", "ENTITYA", 4, 3),
                _pas("or", "coord_ARG2", "ENTITYB", 4, 5),
                _pas("or", "coord_ARG2", "ENTITYC", 4, 6),
            ],
            syntax_heads={"3": 2, "4": 6, "5": 3, "6": 5},
            syntax_head_source="test/udep",
        )

        compiled = compile_token_reasoning_structure(
            result, ["ENTITYA", "ENTITYB", "ENTITYC"]
        )
        attachment_edges = _virtual_edges_by_rule(
            compiled, "pas_coordination_candidate_attachment"
        )

        self.assertEqual(len(attachment_edges), 3)
        self.assertEqual(
            {
                frozenset((edge["source_text"], edge["target_text"]))
                for edge in attachment_edges
            },
            {
                frozenset(("ENTITYA", "film")),
                frozenset(("ENTITYB", "film")),
                frozenset(("ENTITYC", "film")),
            },
        )
        self.assertTrue(
            all(
                edge["provenance"][0]["basis"] == "syntactic_coordination_head"
                for edge in attachment_edges
            )
        )

    def test_console_prints_repaired_graph_with_pas_repair_edges(self) -> None:
        hanlp_result = _hanlp_result(
            "Which film, ENTITYA or ENTITYB, ENTITYC's mother of director?",
            [
                "Which",
                "film",
                "ENTITYA",
                "or",
                "ENTITYB",
                "ENTITYC",
                "'",
                "mother",
                "of",
                "director",
                "?",
            ],
            [
                _pas("film", "noun_ARG1", "Which", 2, 1),
                _pas("or", "coord_ARG1", "ENTITYA", 4, 3),
                _pas("or", "coord_ARG2", "ENTITYB", 4, 5),
                _pas("'", "poss_ARG2", "ENTITYC", 7, 6),
                _pas("'", "poss_ARG1", "mother", 7, 8),
                _pas("of", "prep_ARG1", "mother", 9, 8),
                _pas("of", "prep_ARG2", "director", 9, 10),
            ],
            syntax_heads={"3": 2, "4": 5, "5": 3},
            syntax_head_source="test/udep",
        )
        compiled = compile_token_reasoning_structure(
            hanlp_result, ["ENTITYA", "ENTITYB", "ENTITYC"]
        )
        result = _printable_pipeline_result(
            hanlp_result, compiled, ["ENTITYA", "ENTITYB", "ENTITYC"]
        )
        stream = io.StringIO()

        with redirect_stdout(stream):
            print_hanlp_sdp_result(
                1, QuestionRecord(question=hanlp_result.text), result
            )
        output = stream.getvalue()

        self.assertIn("[Repaired Evidence Graph]", output)
        self.assertIn("film[2] -- ENTITYA[3]", output)
        self.assertIn("film[2] -- ENTITYB[5]", output)
        self.assertIn("ENTITYC[6] -- mother[8]", output)
        self.assertIn("mother[8] -- director[10]", output)
        self.assertIn("film[2] -- ENTITYA[3] (cost=2)", output)
        self.assertIn("ENTITYC[6] -- mother[8] (cost=1)", output)
        self.assertIn("mother[8] -- director[10] (cost=1)", output)
        self.assertIn("ENTITYA[3] -- or[4] (cost=3)", output)
        self.assertNotIn("; rule=", output)
        self.assertNotIn("; rel=", output)
        self.assertNotIn("; derived", output)

    def test_each_entity_keeps_only_one_top1_candidate(self) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), ["ENTITYA", "ENTITYB"]
        )
        branch_results = compiled.global_selection["entity_branch_results"]

        self.assertEqual(len(compiled.paths), 2)
        for branch in branch_results:
            self.assertGreater(branch["candidate_count"], 1)
            self.assertEqual(branch["selected"]["boundary"], "later")
        self.assertEqual(
            [list(path.nodes) for path in compiled.paths],
            [
                ["ENTITYA", "film", "has", "director", "born", "later"],
                ["ENTITYB", "film", "has", "director", "born", "later"],
            ],
        )

    def test_entity_branch_sp_prefers_complete_semantic_path(
        self,
    ) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), ["ENTITYA", "ENTITYB"]
        )
        branch = compiled.global_selection["entity_branch_results"][0]
        candidate_by_boundary = {
            candidate["boundary"]: candidate for candidate in branch["candidates"]
        }

        self.assertEqual(branch["selected"]["boundary"], "later")
        self.assertGreater(
            candidate_by_boundary["later"]["sp_score"],
            candidate_by_boundary["director"]["sp_score"],
        )
        self.assertAlmostEqual(candidate_by_boundary["later"]["sp_score"], 8 / 9)
        self.assertEqual(
            candidate_by_boundary["later"]["sp_components"]["branch_semantic_nodes"],
            ["film", "director", "born", "later"],
        )
        self.assertEqual(
            candidate_by_boundary["later"]["sp_components"]["covered_semantic_nodes"],
            ["film", "director", "born", "later"],
        )
        self.assertEqual(branch["selected"]["path_cost"], 6)
        self.assertEqual(
            [item["edge_cost"] for item in branch["selected"]["edge_costs"]],
            [2, 1, 1, 1, 1],
        )
        self.assertNotIn("rank", branch["selected"])
        self.assertNotIn("rank_components", branch["selected"])

    def test_step5_restore_accepts_multiple_entity_branch_paths(self) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), ["ENTITYA", "ENTITYB"]
        )
        restored = restore_global_best_paths(
            compiled.paths,
            [
                SimpleNamespace(
                    placeholder="ENTITYA", original_text="Illusions (1982 Film)"
                ),
                SimpleNamespace(
                    placeholder="ENTITYB", original_text="It'S A Wonderful Afterlife"
                ),
            ],
        )

        self.assertEqual(
            restored,
            [
                ["Illusions (1982 Film)", "film", "has", "director", "born", "later"],
                [
                    "It'S A Wonderful Afterlife",
                    "film",
                    "has",
                    "director",
                    "born",
                    "later",
                ],
            ],
        )

    def test_no_explicit_entities_returns_empty_paths_and_warning(self) -> None:
        compiled = compile_token_reasoning_structure(
            _which_film_born_later_result(), []
        )

        self.assertEqual(compiled.path_type, "no_entity_branch_path")
        self.assertEqual(compiled.paths, [])
        self.assertTrue(
            any("no explicit entity starts" in warning for warning in compiled.warnings)
        )

    def test_no_semantic_boundary_returns_empty_paths_and_warning(self) -> None:
        result = _hanlp_result(
            "ENTITYA or ENTITYB?",
            ["ENTITYA", "or", "ENTITYB", "?"],
            [
                _pas("or", "coord_ARG1", "ENTITYA", 2, 1),
                _pas("or", "coord_ARG2", "ENTITYB", 2, 3),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.path_type, "no_entity_branch_path")
        self.assertEqual(compiled.paths, [])
        self.assertTrue(
            any(
                "no semantic boundary nodes" in warning for warning in compiled.warnings
            )
        )

    def test_disconnected_entity_warns_but_other_branch_survives(self) -> None:
        result = _hanlp_result(
            "ENTITYA ENTITYB target?",
            ["ENTITYA", "ENTITYB", "target", "?"],
            [_pas("target", "verb_ARG1", "ENTITYB", 3, 2)],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(
            [list(path.nodes) for path in compiled.paths], [["ENTITYB", "target"]]
        )
        self.assertTrue(
            any(
                "ENTITYA" in warning and "no reachable semantic boundary" in warning
                for warning in compiled.warnings
            )
        )


def _atomic_question(
    question_id: str,
    question: str,
    depends_on: list[str],
    output_type: str,
    operation: str = "lookup",
) -> dict[str, object]:
    return {
        "id": question_id,
        "question": question,
        "depends_on": depends_on,
        "operation": operation,
        "output_type": output_type,
    }


def _older_step5_payload() -> dict[str, object]:
    return {
        "atomic_questions": [
            _atomic_question("q1", "When was Ryan Tubridy born?", [], "date"),
            _atomic_question("q2", "When was Mauro Massironi born?", [], "date"),
            _atomic_question(
                "q3",
                "Based on q1's answer and q2's answer, who is older: Ryan Tubridy or Mauro Massironi?",
                ["q1", "q2"],
                "person",
                "select",
            ),
        ],
    }


def _illusions_step5_payload() -> dict[str, object]:
    return {
        "atomic_questions": [
            _atomic_question(
                "q1", "Who is the director of Illusions (1982 Film)?", [], "person"
            ),
            _atomic_question("q2", "When was q1's answer born?", ["q1"], "date"),
            _atomic_question(
                "q3", "Who is the director of It'S A Wonderful Afterlife?", [], "person"
            ),
            _atomic_question("q4", "When was q3's answer born?", ["q3"], "date"),
            _atomic_question(
                "q5",
                "Which film has the director born later, Illusions (1982 Film) or It'S A Wonderful Afterlife, based on q2's answer and q4's answer?",
                ["q2", "q4"],
                "work",
                "select",
            ),
        ],
    }


class FakePreprocessLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.step5_user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        if "DEPO Step 5" in system_prompt or "Atomic Question DAG" in system_prompt:
            self.step5_user_prompt = user_prompt
            payload = json.loads(user_prompt)
            assert set(payload) == {
                "original_question",
                "topic_entities",
                "step4_paths",
            }
            assert payload["topic_entities"] == ["Ryan Tubridy", "Mauro Massironi"]
            assert payload["step4_paths"] == [
                ["Ryan Tubridy", "older", "Who"],
                ["Mauro Massironi", "older", "Who"],
            ]
            serialized = json.dumps(payload, ensure_ascii=False)
            assert "ENTITYA" not in serialized
            assert "ENTITYB" not in serialized
            assert "masked_question" not in serialized
            assert "answer_anchor" not in serialized
            return _older_step5_payload()
        assert "DEPO Step 2: topic entity extraction" in system_prompt
        assert "Deterministic entity candidates" in user_prompt
        assert "Who is older, Ryan Tubridy or Mauro Massironi?" in user_prompt
        return {
            "entities": [
                {
                    "text": "Ryan Tubridy",
                    "semantic_type_hint": "Person",
                    "start_char": 14,
                    "end_char": 26,
                    "confidence": 1.0,
                    "reason": "explicit person name",
                },
                {
                    "text": "Mauro Massironi",
                    "semantic_type_hint": "Person",
                    "start_char": 30,
                    "end_char": 45,
                    "confidence": 1.0,
                    "reason": "explicit person name",
                },
            ],
            "warnings": [],
        }


class NoPathPipelineLLM:
    def __init__(self, question: str) -> None:
        self.question = question
        self.calls = 0
        self.no_path_user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        if "action trace generation" in system_prompt:
            self.no_path_user_prompt = user_prompt
            return {
                "actions": [
                    {
                        "id": "q1",
                        "consume": [],
                        "produce": "q1_answer",
                        "question": self.question,
                    }
                ]
            }
        return {"verified_entities": [], "warnings": []}


class IllusionsPipelineLLM:
    def __init__(self, question: str) -> None:
        self.question = question
        self.calls = 0
        self.step5_user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        if "DEPO Step 5" in system_prompt or "Atomic Question DAG" in system_prompt:
            self.step5_user_prompt = user_prompt
            payload = json.loads(user_prompt)
            assert set(payload) == {
                "original_question",
                "topic_entities",
                "step4_paths",
            }
            assert payload["topic_entities"] == [
                "Illusions (1982 Film)",
                "It'S A Wonderful Afterlife",
            ]
            assert payload["step4_paths"] == [
                ["Illusions (1982 Film)", "film", "has", "director", "born", "later"],
                [
                    "It'S A Wonderful Afterlife",
                    "film",
                    "has",
                    "director",
                    "born",
                    "later",
                ],
            ]
            return _illusions_step5_payload()
        assert "DEPO Step 2: topic entity extraction" in system_prompt
        return {
            "entities": [
                _entity(self.question, "Illusions (1982 Film)", "Work"),
                _entity(self.question, "It'S A Wonderful Afterlife", "Work"),
            ],
            "warnings": [],
        }


class StaticPreprocessLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0
        self.system_prompt = ""
        self.user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.payload


class RecordingHanLPSDPParser:
    def __init__(self) -> None:
        self.text = ""
        self.placeholders: list[str] = []

    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        self.text = text
        self.placeholders = list(placeholders or [])
        tokens = [
            "Which",
            "country",
            "is",
            "the",
            "composer",
            "of",
            "film",
            "ENTITYA",
            "from",
            "?",
        ]
        return HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=["tok", "sdp/pas"],
            sdp_graphs={"sdp/pas": []},
            edges=[
                HanLPSDPEdge("sdp/pas", 0, "ROOT", "root", 2, "country"),
                HanLPSDPEdge("sdp/pas", 5, "composer", "noun_ARG1", 8, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 2, "country", "noun_ARG1", 5, "composer"),
            ],
            raw={"tok": tokens, "sdp/pas": []},
            model="fake",
        )


class NoPathHanLPSDPParser:
    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        del placeholders
        tokens = ["What", "is", "the", "capital", "of", "France", "?"]
        return HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=["tok", "sdp/pas"],
            sdp_graphs={"sdp/pas": []},
            edges=[HanLPSDPEdge("sdp/pas", 0, "ROOT", "root", 2, "is")],
            raw={"tok": tokens, "sdp/pas": []},
            warnings=[],
            model="fake.no-path.model",
        )


class FakeHanLPSDPParser:
    def __init__(self) -> None:
        self.placeholders: list[str] = []
        self.text = ""

    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        self.placeholders = list(placeholders or [])
        self.text = text
        tokens = ["Who", "is", "older", ",", "ENTITYA", "or", "ENTITYB", "?"]
        return HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=["tok", "sdp/pas"],
            sdp_graphs={
                "sdp/pas": [
                    [(3, "adj_ARG1")],
                    [],
                    [(0, "root")],
                    [],
                    [(3, "adj_ARG1")],
                    [],
                    [(3, "adj_ARG1")],
                    [],
                ],
            },
            edges=[
                HanLPSDPEdge("sdp/pas", 3, "older", "adj_ARG1", 1, "Who"),
                HanLPSDPEdge("sdp/pas", 3, "older", "adj_ARG1", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 3, "older", "adj_ARG1", 7, "ENTITYB"),
                HanLPSDPEdge("sdp/pas", 6, "or", "coord_ARG1", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 6, "or", "coord_ARG2", 7, "ENTITYB"),
            ],
            raw={"tok": tokens, "sdp/pas": []},
            warnings=[],
            model="fake.hanlp.model",
            mask_token_checks={placeholder: "OK" for placeholder in self.placeholders},
        )


class FakeIllusionsBornLaterParser:
    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        del text, placeholders
        return _which_film_born_later_result()


def _which_film_born_later_result() -> HanLPSDPResult:
    return _hanlp_result(
        "Which film has the director who was born later, ENTITYA or ENTITYB?",
        [
            "Which",
            "film",
            "has",
            "the",
            "director",
            "who",
            "was",
            "born",
            "later",
            ",",
            "ENTITYA",
            "or",
            "ENTITYB",
            "?",
        ],
        [
            _pas("has", "verb_ARG1", "film", 3, 2),
            _pas("has", "verb_ARG2", "director", 3, 5),
            _pas("the", "det_ARG1", "director", 4, 5),
            _pas("born", "verb_ARG2", "director", 8, 5),
            _pas("later", "adj_ARG1", "born", 9, 8),
            _pas("or", "coord_ARG1", "ENTITYA", 12, 11),
            _pas("or", "coord_ARG2", "ENTITYB", 12, 13),
        ],
        syntax_heads={"13": 11, "11": 2, "12": 13},
        syntax_head_source="test/udep",
    )


def _entity(
    question: str, text: str, semantic_type: str = "Entity"
) -> dict[str, object]:
    start = question.index(text)
    return {
        "text": text,
        "semantic_type_hint": semantic_type,
        "start_char": start,
        "end_char": start + len(text),
        "confidence": 1.0,
        "reason": "test entity",
    }


def _dm(
    head: str, relation: str, dep: str, head_idx: int, dep_idx: int
) -> HanLPSDPEdge:
    return HanLPSDPEdge("sdp/dm", head_idx, head, relation, dep_idx, dep)


def _pas(
    head: str, relation: str, dep: str, head_idx: int, dep_idx: int
) -> HanLPSDPEdge:
    return HanLPSDPEdge("sdp/pas", head_idx, head, relation, dep_idx, dep)


def _psd(
    head: str, relation: str, dep: str, head_idx: int, dep_idx: int
) -> HanLPSDPEdge:
    return HanLPSDPEdge("sdp/psd", head_idx, head, relation, dep_idx, dep)


def _virtual_edges_by_rule(compiled: object, rule: str) -> list[dict[str, object]]:
    return [
        edge
        for edge in getattr(compiled, "debug_payload", {}).get("virtual_edges", [])
        if edge.get("rule") == rule
        or any(item.get("rule") == rule for item in edge.get("provenance", []))
    ]


def _edge_between_texts(state: object, left_text: str, right_text: str) -> object:
    for edge in getattr(state, "edges").values():
        if {edge.source_text, edge.target_text} == {left_text, right_text}:
            return edge
    raise AssertionError(f"missing edge between {left_text!r} and {right_text!r}")


def _has_search_edge(
    graph: dict[str, list[tuple[str, int, tuple[str, str]]]],
    left_id: str,
    right_id: str,
) -> bool:
    return any(
        neighbor_id == right_id for neighbor_id, _cost, _key in graph.get(left_id, [])
    )


def _printable_pipeline_result(
    hanlp_result: HanLPSDPResult, compiled: object, explicit_entities: list[str]
) -> dict[str, object]:
    mappings = [
        SimpleNamespace(placeholder=placeholder, original_text=placeholder)
        for placeholder in explicit_entities
    ]
    preprocess_result = SimpleNamespace(
        explicit_entities=SimpleNamespace(
            entities=[
                SimpleNamespace(text=placeholder) for placeholder in explicit_entities
            ]
        ),
        mask_mappings=mappings,
        normalized_question=hanlp_result.text,
        warnings=[],
        masked_question=hanlp_result.text,
    )
    return {
        "preprocess_result": preprocess_result,
        "hanlp_sdp_result": hanlp_result,
        "token_reasoning_structure": compiled,
        "hanlp_input_sentence": hanlp_result.text,
        "atomic_question_dag": None,
    }


def _hanlp_result(
    text: str,
    tokens: list[str],
    edges: list[HanLPSDPEdge],
    *,
    syntax_heads: dict[str, int] | None = None,
    syntax_head_source: str = "",
) -> HanLPSDPResult:
    formalisms = sorted({edge.formalism for edge in edges})
    return HanLPSDPResult(
        text=text,
        tokens=tokens,
        available_keys=["tok", *formalisms],
        sdp_graphs={formalism: [] for formalism in formalisms},
        edges=edges,
        raw={"tok": tokens},
        syntax_heads=dict(syntax_heads or {}),
        syntax_head_source=syntax_head_source,
    )


if __name__ == "__main__":
    unittest.main()
