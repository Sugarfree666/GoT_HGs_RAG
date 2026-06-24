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

from entity_masking_preprocessor import EntityMaskingPreprocessor  # noqa: E402
from main import (  # noqa: E402
    print_hanlp_sdp_result,
    run_hanlp_sdp_pipeline,
)
from models import HanLPSDPEdge, HanLPSDPResult, QuestionRecord  # noqa: E402
from tri_sdp_reasoning_compiler import (  # noqa: E402
    TokenReasoningEdge,
    TokenReasoningNode,
    TokenReasoningPath,
    _AnswerAnchorCandidate,
    _WorkingState,
    _rank_global_path_candidate,
    _select_global_best_path,
    build_evidence_graph,
    compile_token_reasoning_structure,
)


def _global_candidate_stub(
    rank: tuple[int, int, int, int],
    anchor_index: int,
    path_index: int,
    node_ids: list[str],
) -> SimpleNamespace:
    return SimpleNamespace(
        global_rank=rank,
        anchor_index=anchor_index,
        path_index=path_index,
        path=SimpleNamespace(node_ids=node_ids),
    )


class HanLPSDPMainlineTest(unittest.TestCase):
    def test_hanlp_sdp_pipeline_runs_step5_from_global_best_path_by_default(self) -> None:
        record = QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?")
        parser = FakeHanLPSDPParser()
        llm = FakePreprocessLLM()
        preprocessor = EntityMaskingPreprocessor(llm)

        result = run_hanlp_sdp_pipeline(
            record=record,
            index=1,
            preprocessor=preprocessor,
            parser=parser,
        )

        preprocess_result = result["preprocess_result"]
        self.assertEqual(preprocess_result.masked_question, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(preprocess_result.sdp_input_sentence, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual([mapping.placeholder for mapping in preprocess_result.mask_mappings], ["ENTITYA", "ENTITYB"])
        self.assertEqual(parser.placeholders, ["ENTITYA", "ENTITYB"])
        self.assertEqual(result["hanlp_input_sentence"], "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(parser.text, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(llm.calls, 2)
        step5_payload = json.loads(llm.step5_user_prompt)
        self.assertEqual(set(step5_payload), {"original_question", "paths"})
        self.assertEqual(len(step5_payload["paths"]), 1)
        self.assertEqual(step5_payload["paths"][0]["path_id"], result["token_reasoning_structure"].paths[0].path_id)

        stream = io.StringIO()
        with redirect_stdout(stream):
            print_hanlp_sdp_result(1, record, result)
        output = stream.getvalue()

        self.assertIn("[Original Question]", output)
        self.assertIn("[1. Explicit Entities]", output)
        self.assertIn(" - Ryan Tubridy", output)
        self.assertNotIn(" - Ryan Tubridy [Person]", output)
        self.assertIn("[2. Entity Masking]", output)
        self.assertIn(" - ENTITYA -> Ryan Tubridy", output)
        self.assertIn("Normalized question: unchanged", output)
        self.assertIn("Masked question: Who is older, ENTITYA or ENTITYB?", output)
        self.assertNotIn("SDP input sentence:", output)
        self.assertIn("[3. HanLP SDP Parsing]", output)
        self.assertIn("HanLP input sentence: Who is older, ENTITYA or ENTITYB?", output)
        self.assertNotIn("[HanLP Tokens]", output)
        self.assertNotIn("[Available HanLP Keys]", output)
        self.assertIn("[Raw SDP Edges]", output)
        self.assertIn("[SDP: sdp/dm]", output)
        self.assertIn("[SDP: sdp/pas]", output)
        self.assertIn("[SDP: sdp/psd]", output)
        self.assertIn("older[3] --ARG1--> Who[1]", output)
        self.assertIn("older[3] --ARG1--> ENTITYA[5]", output)
        self.assertIn("older[3] --ACT-arg--> ENTITYB[7]", output)
        self.assertIn("[4. Token Reasoning Structure]", output)
        self.assertNotIn("[Graph]", output)
        self.assertIn("[Anchor Paths]", output)
        self.assertIn("Anchor A1: older[3] sources=comparative_focus,clause_predicate", output)
        self.assertIn("  P1: ENTITYA ---- older", output)
        self.assertIn("Anchor A2: ENTITYA[5] sources=explicit_entity", output)
        self.assertIn("Anchor A3: ENTITYB[7] sources=explicit_entity", output)
        self.assertNotIn("answer_anchor:", output)
        self.assertNotIn("entity_anchors:", output)
        self.assertIn("[5. Atomic Question DAG]", output)
        self.assertNotIn("(skipped: Step5 disabled)", output)
        self.assertIn("q1: When was Ryan Tubridy born?", output)
        self.assertIn("q2: When was Mauro Massironi born?", output)
        self.assertNotIn("[4. Content Reasoning Chains]", output)
        self.assertNotIn("[4. Simplified SDP/DM Graph]", output)
        self.assertNotIn("[Kept / Derived Edges]", output)
        self.assertNotIn("older --ARG2--> ENTITYA", output)
        self.assertNotIn("older --ARG1--> ANSWER\n", output)

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
        self.assertEqual(preprocess_result.normalized_question, normalized)
        self.assertTrue(preprocess_result.normalization_changed)
        self.assertEqual(preprocess_result.normalization_note, "Inserted auxiliary is for wh-question order.")
        self.assertEqual([entity.text for entity in preprocess_result.explicit_entities.entities], ["Thunder On The Hill"])
        self.assertEqual(
            [
                (
                    entity.start_char,
                    entity.end_char,
                    question[entity.start_char : entity.end_char],
                )
                for entity in preprocess_result.explicit_entities.entities
            ],
            [(35, 54, "Thunder On The Hill")],
        )
        self.assertEqual(preprocess_result.masked_question, masked)
        self.assertEqual(preprocess_result.sdp_input_sentence, masked)
        self.assertEqual(result["hanlp_input_sentence"], masked)
        self.assertEqual(parser.text, masked)
        self.assertEqual(parser.placeholders, ["ENTITYA"])

        self.assertEqual(debug_payload["original_question"], question)
        self.assertEqual(debug_payload["normalized_question"], normalized)
        self.assertTrue(debug_payload["normalization_changed"])
        self.assertEqual(debug_payload["normalization_note"], "Inserted auxiliary is for wh-question order.")
        self.assertEqual(debug_payload["masked_question"], masked)

    def test_hanlp_sdp_pipeline_can_skip_step5(self) -> None:
        record = QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?")
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

    def test_preprocessor_smoke_examples(self) -> None:
        cases = [
            (
                "Who is the spouse of Young Man Luther's author?",
                {
                    "entities": [
                        _entity("Who is the spouse of Young Man Luther's author?", "Young Man Luther"),
                    ],
                    "warnings": [],
                },
                "Who is the spouse of ENTITYA's author?",
                "Who is the spouse of ENTITYA's author?",
            ),
            (
                "What is the date of death of the director of film FilmA?",
                {
                    "entities": [
                        _entity("What is the date of death of the director of film FilmA?", "FilmA"),
                    ],
                    "warnings": [],
                },
                "What is the date of death of the director of film ENTITYA?",
                "What is the date of death of the director of film ENTITYA?",
            ),
            (
                "Where was the person who wrote about the rioting being a dividing factor in Birmingham educated?",
                {
                    "entities": [
                        _entity(
                            "Where was the person who wrote about the rioting being a dividing factor in Birmingham educated?",
                            "Birmingham",
                        ),
                    ],
                    "warnings": [],
                },
                "Where was the person who wrote about the rioting being a dividing factor in ENTITYA educated?",
                "Where was the person who wrote about the rioting being a dividing factor in ENTITYA educated?",
            ),
            (
                "Who is older, Ryan Tubridy or Mauro Massironi?",
                {
                    "entities": [
                        _entity("Who is older, Ryan Tubridy or Mauro Massironi?", "Ryan Tubridy", "Person"),
                        _entity("Who is older, Ryan Tubridy or Mauro Massironi?", "Mauro Massironi", "Person"),
                    ],
                    "warnings": [],
                },
                "Who is older, ENTITYA or ENTITYB?",
                "Who is older, ENTITYA or ENTITYB?",
            ),
        ]
        for question, payload, expected_masked, expected_sdp in cases:
            with self.subTest(question=question):
                llm = StaticPreprocessLLM(payload)
                result = EntityMaskingPreprocessor(llm).preprocess(question)

                self.assertEqual(llm.calls, 1)
                self.assertEqual(result.masked_question, expected_masked)
                self.assertEqual(result.sdp_input_sentence, expected_sdp)
                for mapping in result.mask_mappings:
                    self.assertTrue(mapping.placeholder.startswith("ENTITY"))
                    self.assertIn(mapping.placeholder, result.masked_question)
                    self.assertIn(mapping.placeholder, result.sdp_input_sentence)

    def test_preprocessor_uses_generic_entities_only_schema(self) -> None:
        question = "What music school did the singer of The Search for Everything: Wave One attend?"
        llm = StaticPreprocessLLM(
            {
                "entities": [_entity(question, "The Search for Everything: Wave One", "Work")],
                "warnings": [],
            }
        )

        result = EntityMaskingPreprocessor(llm).preprocess(question)

        self.assertEqual([entity.text for entity in result.explicit_entities.entities], ["The Search for Everything: Wave One"])
        self.assertEqual(result.masked_question, "What music school did the singer of ENTITYA attend?")
        self.assertEqual(result.sdp_input_sentence, result.masked_question)
        self.assertEqual([(mapping.placeholder, mapping.original_text) for mapping in result.mask_mappings], [("ENTITYA", "The Search for Everything: Wave One")])
        self.assertNotIn("attend", result.explicit_entities.entities[0].text)
        self.assertIn("DEPO Step 2: topic entity extraction", llm.system_prompt)
        self.assertNotIn("semantic_type_hint", llm.user_prompt)

    def test_preprocessor_recovers_numeric_title_in_typed_comparison_list(self) -> None:
        question = "Which film came out first, 3 Dots or Dying God?"
        llm = StaticPreprocessLLM(
            {
                "entities": [_entity(question, "Dying God", "Film")],
                "warnings": [],
            }
        )

        result = EntityMaskingPreprocessor(llm).preprocess(question)

        self.assertEqual([entity.text for entity in result.explicit_entities.entities], ["3 Dots", "Dying God"])
        self.assertEqual(result.masked_question, "Which film came out first, ENTITYA or ENTITYB?")
        self.assertEqual(
            [(mapping.placeholder, mapping.original_text) for mapping in result.mask_mappings],
            [("ENTITYA", "3 Dots"), ("ENTITYB", "Dying God")],
        )
        self.assertNotIn("Dots or Dying God", [entity.text for entity in result.explicit_entities.entities])
        self.assertIn("typed coordinate title candidate", result.explicit_entities.entities[0].reason)

    def test_preprocessor_ignores_legacy_mask_fields_but_accepts_legacy_entities(self) -> None:
        question = "Who is the spouse of Young Man Luther's author?"
        llm = StaticPreprocessLLM(
            {
                "explicit_entities": [_entity(question, "Young Man Luther", "Work")],
                "mask_mappings": [{"placeholder": "WRONG", "original_text": "Young Man Luther"}],
                "masked_question": "SHOULD NOT BE USED",
                "warnings": [],
            }
        )

        result = EntityMaskingPreprocessor(llm).preprocess(question)

        self.assertEqual(result.masked_question, "Who is the spouse of ENTITYA's author?")
        self.assertEqual([(mapping.placeholder, mapping.original_text) for mapping in result.mask_mappings], [("ENTITYA", "Young Man Luther")])

class TriSDPReasoningCompilerTest(unittest.TestCase):
    def test_dell_constraint_entity_does_not_enter_main_path(self) -> None:
        result = self._dell_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "feature")
        self.assertEqual(anchor["path_type"], "fallback_main_path")
        path = anchor["paths"][0]  # type: ignore[index]
        self.assertEqual(path["nodes"][0], "ENTITYB")
        self.assert_ordered_subsequence(path["nodes"], ["ENTITYB", "drives", "iterations", "interface", "replacing", "letting", "feature", "call", "What"])
        self.assertEqual(len(path["node_ids"]), len(set(path["node_ids"])))
        selected = [record for record in anchor["candidate_paths"] if record.get("selected")][0]  # type: ignore[index]
        self.assertIn("missing_semantic_nodes_count", selected["rank_components"])
        self.assert_path_union_graph(compiled)

    def test_johnny_majors_main_path_excludes_constraint_entity(self) -> None:
        result = self._johnny_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "year")
        self.assertEqual(anchor["path_type"], "fallback_main_path")
        path = anchor["paths"][0]  # type: ignore[index]
        self.assertEqual(path["nodes"], ["ENTITYA", "defeated", "player", "born", "year"])
        self.assertNotIn("ENTITYB", path["nodes"])
        self.assertNotIn("1956", path["nodes"])
        self.assertEqual(len(path["node_ids"]), len(set(path["node_ids"])))
        self.assert_path_union_graph(compiled)

    def test_role_coordinated_nationality_parallel_cover(self) -> None:
        result = self._nationality_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "nationality")
        self.assertEqual(anchor["anchor_id"], "14")
        self.assertEqual(anchor["path_type"], "candidate_path_cover")
        self.assertEqual(self.anchor_path_ids(compiled, "nationality"), [["5", "2", "11", "13", "14"], ["10", "7", "11", "13", "14"]])
        self.assertEqual(self.anchor_path_nodes(compiled, "nationality"), [["ENTITYA", "director", "share", "same", "nationality"], ["ENTITYB", "director", "share", "same", "nationality"]])
        self.assertTrue(all(len(path["node_ids"]) == len(set(path["node_ids"])) for path in anchor["paths"]))  # type: ignore[index]
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assert_path_union_graph(compiled)

    def test_lifted_coordination_parallel_cover_is_structural_not_lexical(self) -> None:
        result = _hanlp_result(
            "Do alpha near ENTITYA and beta near ENTITYB align the common omega?",
            ["Do", "alpha", "near", "ENTITYA", "and", "beta", "near", "ENTITYB", "align", "the", "common", "omega", "?"],
            [
                _root("sdp/dm", "align", 9),
                _psd("and", "CONJ.member", "alpha", 5, 2),
                _psd("and", "CONJ.member", "beta", 5, 6),
                _psd("alpha", "PAT-arg", "ENTITYA", 2, 4),
                _psd("beta", "PAT-arg", "ENTITYB", 6, 8),
                _psd("align", "ACT-arg", "alpha", 9, 2),
                _psd("align", "ACT-arg", "beta", 9, 6),
                _psd("align", "PAT-arg", "omega", 9, 12),
                _dm("common", "RSTR", "omega", 11, 12),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "omega")
        self.assertEqual(anchor["path_type"], "candidate_path_cover")
        self.assertEqual(self.anchor_path_ids(compiled, "omega"), [["4", "2", "9", "11", "12"], ["8", "6", "9", "11", "12"]])
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assert_path_union_graph(compiled)

    def test_lifted_coordination_same_bound_entity_is_not_parallel(self) -> None:
        result = _hanlp_result(
            "Do left and right merge target near ENTITYA beside ENTITYB?",
            ["Do", "left", "and", "right", "merge", "target", "near", "ENTITYA", "beside", "ENTITYB", "?"],
            [
                _root("sdp/dm", "merge", 5),
                _psd("and", "CONJ.member", "left", 3, 2),
                _psd("and", "CONJ.member", "right", 3, 4),
                _psd("left", "PAT-arg", "ENTITYA", 2, 8),
                _psd("right", "PAT-arg", "ENTITYA", 4, 8),
                _psd("merge", "ACT-arg", "left", 5, 2),
                _psd("merge", "ACT-arg", "right", 5, 4),
                _psd("merge", "PAT-arg", "target", 5, 6),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "target")
        self.assertEqual(anchor["path_type"], "fallback_main_path")
        self.assertEqual(len(anchor["paths"]), 1)
        self.assertNotIn("ENTITYB", anchor["paths"][0]["nodes"])  # type: ignore[index]
        self.assert_path_union_graph(compiled)

    def test_candidate_comparison_uses_typed_slot_substitution(self) -> None:
        result = self._film_director_died_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "film")
        self.assertEqual(anchor["path_type"], "candidate_path_cover")
        self.assertEqual(self.anchor_path_nodes(compiled, "film"), [["ENTITYA", "director", "died"], ["ENTITYB", "director", "died"]])
        self.assertTrue(any(item["text"] == "first" for item in compiled.constraints))
        rendered = "\n".join(" ---- ".join(path) for path in self.anchor_path_nodes(compiled, "film"))
        self.assertNotIn(" or ", rendered)
        derived_rules = {edge.rule for edge in compiled.edges if edge.derived}
        self.assertTrue(any("candidate_slot_substitution" in rule for rule in derived_rules))
        substitution_edges = [edge for edge in compiled.edges if "candidate_slot_substitution" in edge.rule]
        self.assertTrue(all(edge.provenance for edge in substitution_edges))
        self.assert_path_union_graph(compiled)

    def test_candidate_comparison_recovers_surface_typed_wh_slot(self) -> None:
        result = self._which_film_director_younger_result()
        self.assertFalse(any(edge.head == "Which" and edge.dep == "film" for edge in result.edges))

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "film")
        self.assertEqual(anchor["path_type"], "candidate_path_cover")
        self.assertEqual(self.anchor_path_ids(compiled, "film"), [["8", "4", "6"], ["10", "4", "6"]])
        self.assertEqual(self.anchor_path_nodes(compiled, "film"), [["ENTITYA", "director", "younger"], ["ENTITYB", "director", "younger"]])
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assertTrue(all(len(path["node_ids"]) == len(set(path["node_ids"])) for path in anchor["paths"]))  # type: ignore[index]
        self.assert_path_union_graph(compiled)

        substitution_edges = [
            edge
            for edge in self.anchor_result(compiled, "film").get("virtual_edges", [])  # type: ignore[union-attr]
            if "candidate_slot_substitution" in str(edge.get("rule", ""))
        ]
        self.assertEqual(len(substitution_edges), 2)
        for edge in substitution_edges:
            provenance = edge["provenance"][0]
            self.assertEqual(provenance["typed_wh_slot_id"], "2")
            self.assertEqual(provenance["schema_path_ids"], ["2", "4", "6"])
            self.assertEqual(provenance["typed_wh_evidence"]["surface_adjacency"]["slot"], "film")
            self.assertTrue(provenance["candidate_set_evidence"])

    def test_surface_typed_wh_adjacency_positive_and_negative_cases(self) -> None:
        what_year = _hanlp_result(
            "What year was ENTITYA born?",
            ["What", "year", "was", "ENTITYA", "born", "?"],
            [
                _root("sdp/dm", "born", 5),
                _dm("born", "ARG1", "ENTITYA", 5, 4),
                _dm("born", "TWHEN", "year", 5, 2),
            ],
        )
        self.assertTrue(self.anchor_result(compile_token_reasoning_structure(what_year, ["ENTITYA"]), "year"))

        what_does = _hanlp_result(
            "What does ENTITYA call target?",
            ["What", "does", "ENTITYA", "call", "target", "?"],
            [
                _root("sdp/dm", "call", 4),
                _dm("call", "ARG1", "What", 4, 1),
                _dm("call", "ARG2", "target", 4, 5),
                _dm("call", "ARG1", "ENTITYA", 4, 3),
            ],
        )
        self.assertFalse(
            [
                result
                for result in compile_token_reasoning_structure(what_does, ["ENTITYA"]).anchor_path_results
                if result.get("anchor_text") == "does"
            ]
        )

        which_of = _hanlp_result(
            "Which of ENTITYA is older?",
            ["Which", "of", "ENTITYA", "is", "older", "?"],
            [
                _root("sdp/dm", "older", 5),
                _dm("older", "ARG1", "ENTITYA", 5, 3),
            ],
        )
        compiled_which_of = compile_token_reasoning_structure(which_of, ["ENTITYA"])
        self.assertFalse([result for result in compiled_which_of.anchor_path_results if result.get("anchor_text") == "of"])

    def test_bridge_contraction_edges_are_strong_and_semantic(self) -> None:
        result = _hanlp_result(
            "Who is the spouse of the director of film ENTITYA?",
            ["Who", "is", "the", "spouse", "of", "the", "director", "of", "film", "ENTITYA", "?"],
            [
                _root("sdp/dm", "is", 2),
                _dm("is", "ARG2", "Who", 2, 1),
                _dm("is", "ARG2", "spouse", 2, 4),
                _dm("of", "ARG1", "spouse", 5, 4),
                _dm("of", "ARG2", "director", 5, 7),
                _pas("of", "prep_ARG1", "director", 8, 7),
                _pas("of", "prep_ARG2", "ENTITYA", 8, 10),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])

        self.assertEqual(self.anchor_path_nodes(compiled, "spouse")[0], ["ENTITYA", "director", "spouse"])
        selected = [record for record in self.anchor_result(compiled, "spouse")["candidate_paths"] if record.get("selected")][0]  # type: ignore[index]
        components = selected["rank_components"]
        self.assertEqual(components["semantic_nodes"], ["spouse", "director", "ENTITYA"])
        self.assertEqual(components["missing_semantic_nodes_count"], 0)
        self.assertEqual(components["weak_edge_count"], 0)
        edges = {
            frozenset((edge.source_text, edge.target_text)): edge
            for edge in compiled.edges
        }
        self.assertEqual(edges[frozenset(("ENTITYA", "director"))].edge_quality, "STRONG")
        self.assertEqual(edges[frozenset(("director", "spouse"))].edge_quality, "STRONG")
        self.assertTrue(edges[frozenset(("ENTITYA", "director"))].derived)
        self.assertTrue(edges[frozenset(("director", "spouse"))].derived)

    def test_local_rank_prefers_semantic_coverage_over_short_entity_path(self) -> None:
        result = _hanlp_result(
            "When was ENTITYA constructed in the city where ENTITYB from the region where ENTITYC is located died?",
            [
                "When",
                "was",
                "ENTITYA",
                "constructed",
                "in",
                "the",
                "city",
                "where",
                "ENTITYB",
                "from",
                "the",
                "region",
                "where",
                "ENTITYC",
                "is",
                "located",
                "died",
                "?",
            ],
            [
                _root("sdp/dm", "constructed", 4),
                _dm("constructed", "ARG1", "ENTITYA", 4, 3),
                _dm("constructed", "loc", "city", 4, 7),
                _dm("died", "loc", "city", 17, 7),
                _dm("died", "ARG1", "ENTITYB", 17, 9),
                _dm("from", "ARG1", "ENTITYB", 10, 9),
                _dm("from", "ARG2", "region", 10, 12),
                _dm("located", "ARG1", "ENTITYC", 16, 14),
                _dm("located", "loc", "region", 16, 12),
                _pas("from", "prep_ARG1", "ENTITYB", 10, 9),
                _pas("from", "prep_ARG2", "region", 10, 12),
                _psd("located", "ACT-arg", "ENTITYC", 16, 14),
                _psd("located", "LOC", "region", 16, 12),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB", "ENTITYC"])

        self.assertEqual(
            self.anchor_path_nodes(compiled, "ENTITYC")[0],
            ["ENTITYA", "constructed", "city", "died", "ENTITYB", "region", "located", "ENTITYC"],
        )
        anchor = self.anchor_result(compiled, "ENTITYC")
        selected = [record for record in anchor["candidate_paths"] if record.get("selected")][0]  # type: ignore[index]
        self.assertEqual(selected["rank_components"]["missing_semantic_nodes_count"], 0)
        short_entity = [
            record
            for record in anchor["candidate_paths"]  # type: ignore[index]
            if record["nodes"] == ["ENTITYC"]
        ][0]
        partial = [
            record
            for record in anchor["candidate_paths"]  # type: ignore[index]
            if record["nodes"] == ["ENTITYB", "region", "located", "ENTITYC"]
        ][0]
        self.assertGreater(short_entity["rank_components"]["missing_semantic_nodes_count"], selected["rank_components"]["missing_semantic_nodes_count"])
        self.assertGreater(partial["rank_components"]["missing_semantic_nodes_count"], selected["rank_components"]["missing_semantic_nodes_count"])

    def test_bare_wh_candidate_substitution_born_later(self) -> None:
        result = self._born_later_result()
        self.assertFalse(any(edge.head_idx == 0 for edge in result.edges))

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "born")
        self.assertEqual(anchor["path_type"], "candidate_path_cover")
        self.assertEqual(self.anchor_path_ids(compiled, "born"), [["6", "3", "4"], ["8", "3", "4"]])
        self.assertEqual(self.anchor_path_nodes(compiled, "born"), [["ENTITYA", "born", "later"], ["ENTITYB", "born", "later"]])
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assertTrue(all(len(path["node_ids"]) > 1 for path in anchor["paths"]))  # type: ignore[index]
        self.assertTrue(all(len(path["node_ids"]) == len(set(path["node_ids"])) for path in anchor["paths"]))  # type: ignore[index]
        self.assert_path_union_graph(compiled)

        substitution_edges = [
            edge
            for edge in self.anchor_result(compiled, "born").get("virtual_edges", [])  # type: ignore[union-attr]
            if "candidate_bare_wh_substitution" in str(edge.get("rule", ""))
        ]
        self.assertEqual(len(substitution_edges), 2)
        self.assertTrue(all(edge.get("derived") and edge.get("provenance") for edge in substitution_edges))
        for edge in substitution_edges:
            provenance = edge["provenance"][0]
            self.assertEqual(provenance["rule"], "candidate_bare_wh_substitution")
            self.assertEqual(provenance["bare_wh_slot_id"], "1")
            self.assertEqual(provenance["query_predicate_id"], "3")
            self.assertEqual(provenance["schema_path_ids"], ["1", "3", "4"])
            self.assertTrue(provenance["candidate_set_evidence"])

    def test_bare_wh_candidate_substitution_is_structural_not_lexical(self) -> None:
        result = _hanlp_result(
            "Whom did pivot afterward, ENTITYA or ENTITYB?",
            ["Whom", "did", "pivot", "afterward", ",", "ENTITYA", "or", "ENTITYB", "?"],
            [
                _dm("pivot", "ARG2", "Whom", 3, 1),
                _pas("pivot", "verb_ARG2", "Whom", 3, 1),
                _psd("pivot", "PAT-arg", "Whom", 3, 1),
                _dm("pivot", "TWHEN", "afterward", 3, 4),
                _psd("afterward", "adj_ARG1", "pivot", 4, 3),
                _pas("or", "coord_ARG1", "ENTITYA", 7, 6),
                _pas("or", "coord_ARG2", "ENTITYB", 7, 8),
                _psd("or", "DISJ.member", "ENTITYA", 7, 6),
                _psd("or", "DISJ.member", "ENTITYB", 7, 8),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        anchor = self.anchor_result(compiled, "pivot")
        self.assertEqual(anchor["path_type"], "candidate_path_cover")
        self.assertEqual(self.anchor_path_ids(compiled, "pivot"), [["6", "3", "4"], ["8", "3", "4"]])
        self.assertTrue(any("candidate_bare_wh_substitution" in edge.rule for edge in compiled.edges))
        self.assert_path_union_graph(compiled)

    def test_bare_wh_candidate_substitution_requires_direct_wh_core_argument(self) -> None:
        result = _hanlp_result(
            "Who did marker shift afterward, ENTITYA or ENTITYB?",
            ["Who", "did", "marker", "shift", "afterward", ",", "ENTITYA", "or", "ENTITYB", "?"],
            [
                _dm("marker", "ARG1", "Who", 3, 1),
                _dm("shift", "ARG2", "marker", 4, 3),
                _dm("shift", "TWHEN", "afterward", 4, 5),
                _pas("or", "coord_ARG1", "ENTITYA", 8, 7),
                _pas("or", "coord_ARG2", "ENTITYB", 8, 9),
                _psd("or", "DISJ.member", "ENTITYA", 8, 7),
                _psd("or", "DISJ.member", "ENTITYB", 8, 9),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.debug_payload["selection_mode"], "global_best_path")
        self.assertFalse(
            any(
                result.get("selection_mode") == "candidate_bare_wh_substitution"
                for result in compiled.debug_payload["per_anchor_results"]
            )
        )
        self.assertFalse(any("candidate_bare_wh_substitution" in edge.rule for edge in compiled.edges))
        self.assertFalse(any("candidate_bare_wh_substitution" in str(edge) for edge in compiled.debug_payload["virtual_edges"]))

    def test_simple_main_chain_and_answer_anchor_regressions(self) -> None:
        director_born = self._director_born_result()

        compiled = compile_token_reasoning_structure(director_born, ["ENTITYA"])

        self.assert_multi_anchor_result(compiled)
        born_anchor = self.anchor_result(compiled, "born")
        self.assertEqual(born_anchor["path_type"], "single_main_path")
        self.assertEqual(self.anchor_path_nodes(compiled, "born")[0], ["ENTITYA", "director", "born"])
        self.assert_path_union_graph(compiled)

        for result, entities, anchor_text in [
            (self._older_result(), ["ENTITYA", "ENTITYB"], "older"),
            (director_born, ["ENTITYA"], "born"),
            (self._typed_year_result(), ["ENTITYA"], "year"),
            (self._nationality_result(), ["ENTITYA", "ENTITYB"], "nationality"),
            (self._born_later_result(), ["ENTITYA", "ENTITYB"], "born"),
            (self._which_film_director_younger_result(), ["ENTITYA", "ENTITYB"], "film"),
        ]:
            self.assertTrue(self.anchor_result(compile_token_reasoning_structure(result, entities), anchor_text))

    def test_possessive_clitic_marker_is_contracted_from_entity_mother_path(self) -> None:
        result = _hanlp_result(
            "When did ENTITYA's mother die?",
            ["When", "did", "ENTITYA", "'", "s", "mother", "die", "?"],
            [
                _root("sdp/dm", "die", 7),
                _pas("s", "poss_ARG2", "ENTITYA", 5, 3),
                _pas("s", "poss_ARG1", "mother", 5, 6),
                _dm("die", "ARG1", "mother", 7, 6),
                _pas("die", "verb_ARG1", "mother", 7, 6),
                _dm("When", "TWHEN", "die", 1, 7),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])

        self.assertEqual(self.anchor_path_nodes(compiled, "die")[0], ["ENTITYA", "mother", "die"])
        self.assertNotIn("s", self.anchor_path_nodes(compiled, "die")[0])
        self.assertNotIn("s", {node.text for node in compiled.nodes})
        self.assertIn(frozenset(("ENTITYA", "mother")), _edge_text_pairs(compiled))
        self.assertIn(frozenset(("mother", "die")), _edge_text_pairs(compiled))
        possessive_edges = [
            edge for edge in compiled.edges
            if {edge.source_text, edge.target_text} == {"ENTITYA", "mother"}
        ]
        self.assertTrue(any("possessive_marker_contraction" in edge.rule for edge in possessive_edges))

    def test_possessive_clitic_with_adj_possessed_edge_is_contracted(self) -> None:
        result = _hanlp_result(
            "What is the ENTITYA dom ensamma performer's birth date?",
            ["What", "is", "the", "ENTITYA", "dom", "ensamma", "performer", "'", "s", "birth", "date", "?"],
            [
                _root("sdp/dm", "date", 11),
                _dm("What", "BV", "date", 1, 11),
                _pas("performer", "noun_ARG1", "ENTITYA", 7, 4),
                _pas("s", "poss_ARG2", "performer", 9, 7),
                _pas("s", "adj_ARG1", "date", 9, 11),
                _dm("birth", "compound", "date", 10, 11),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])

        self.assertEqual(self.anchor_path_nodes(compiled, "date")[0], ["ENTITYA", "performer", "date"])
        self.assertNotIn("s", self.anchor_path_nodes(compiled, "date")[0])
        self.assertNotIn("s", {node.text for node in compiled.nodes})
        self.assertIn(frozenset(("ENTITYA", "performer")), _edge_text_pairs(compiled))
        self.assertIn(frozenset(("performer", "date")), _edge_text_pairs(compiled))

    def test_non_possessive_s_is_not_globally_treated_as_function(self) -> None:
        result = _hanlp_result(
            "ENTITYA s target?",
            ["ENTITYA", "s", "target", "?"],
            [
                _root("sdp/dm", "s", 2),
                _dm("s", "ARG1", "ENTITYA", 2, 1),
                _dm("s", "ARG2", "target", 2, 3),
            ],
        )

        state = build_evidence_graph(result)

        self.assertEqual(state.nodes["2"].text, "s")
        self.assertEqual(state.nodes["2"].kind, "content")

    def test_broadcaster_headquarters_remains_single_main_path(self) -> None:
        result = self._broadcaster_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])

        self.assertEqual(self.anchor_path_nodes(compiled, "headquarters")[0], ["ENTITYA", "show", "broadcaster", "headquarters"])
        self.assert_path_union_graph(compiled)

    def test_role_descriptor_lifting_is_debug_evidence_not_forced_terminal_cover(self) -> None:
        result = self._role_series_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB", "ENTITYC"])

        json.dumps(compiled.to_dict())
        self.assert_multi_anchor_result(compiled)
        series_paths = self.anchor_result(compiled, "series")["paths"]  # type: ignore[index]
        self.assertEqual(len(series_paths), 1)
        self.assertEqual(len(series_paths[0]["node_ids"]), len(set(series_paths[0]["node_ids"])))  # type: ignore[index]
        self.assert_path_union_graph(compiled)
        self.assertTrue(
            any(
                "descriptor_lifting" in str(edge.get("rule", ""))
                for result in compiled.debug_payload["per_anchor_results"]
                for edge in result.get("virtual_edges", [])
            )
        )

    def test_works_collection_museum_houses_main_path_and_debug(self) -> None:
        result = self._works_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            compiled = compile_token_reasoning_structure(
                result,
                ["ENTITYA"],
                masked_question=result.text,
                question_id="works_case",
                debug=True,
                debug_dir=tmpdir,
            )
            self.assertTrue(Path(compiled.debug_file or "").exists())
            json.dumps(compiled.to_dict())

        self.assertEqual(self.anchor_path_nodes(compiled, "houses")[0], ["ENTITYA", "Works", "part", "collection", "museum", "houses"])
        self.assertTrue(any(item["text"] == "65,000" for item in compiled.constraints))
        final_texts = {node.text for node in compiled.nodes}
        self.assertNotIn("ROOT", final_texts)
        self.assertNotIn("a", final_texts)
        self.assertNotIn("?", final_texts)
        self.assert_path_union_graph(compiled)

    def test_structure_ranking_not_lexical_shortcut(self) -> None:
        result = _hanlp_result(
            "Alpha ENTITYA links target while ENTITYB label also links target?",
            ["Alpha", "ENTITYA", "links", "target", "while", "ENTITYB", "label", "also", "?"],
            [
                _root("sdp/dm", "links", 3),
                _dm("links", "ARG1", "ENTITYA", 3, 2),
                _dm("links", "ARG2", "target", 3, 4),
                _dm("label", "compound", "ENTITYB", 7, 6),
                _dm("links", "ARG1", "label", 3, 7),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assert_multi_anchor_result(compiled)
        self.assertEqual(self.anchor_path_nodes(compiled, "target")[0], ["ENTITYB", "label", "links", "target"])
        selected = [record for record in self.anchor_result(compiled, "target")["candidate_paths"] if record.get("selected")][0]  # type: ignore[index]
        self.assertEqual(selected["rank_components"]["missing_semantic_nodes_count"], 1)
        self.assertIn("ENTITYB", selected["rank_components"]["covered_semantic_nodes"])
        self.assertIn("label", selected["rank_components"]["covered_semantic_nodes"])
        self.assert_path_union_graph(compiled)

    def test_global_path_union_invariants_and_determinism(self) -> None:
        cases = [
            (self._dell_result(), ["ENTITYA", "ENTITYB"]),
            (self._johnny_result(), ["ENTITYA", "ENTITYB"]),
            (self._nationality_result(), ["ENTITYA", "ENTITYB"]),
            (self._film_director_died_result(), ["ENTITYA", "ENTITYB"]),
            (self._which_film_director_younger_result(), ["ENTITYA", "ENTITYB"]),
            (self._born_later_result(), ["ENTITYA", "ENTITYB"]),
            (self._director_born_result(), ["ENTITYA"]),
        ]
        for result, entities in cases:
            with self.subTest(question=result.text):
                first = compile_token_reasoning_structure(result, entities)
                second = compile_token_reasoning_structure(result, entities)
                self.assertEqual(first.to_dict(), second.to_dict())
                self.assert_path_union_graph(first)
                edge_pairs = [(edge.source, edge.target) for edge in first.edges]
                self.assertEqual(len(edge_pairs), len(set(frozenset(pair) for pair in edge_pairs)))
                for path in first.paths:
                    self.assertEqual(len(path.node_ids), len(set(path.node_ids)))
                self.assertEqual(first.path_type, "global_best_path")
                self.assertEqual(len(first.paths), 1)
                self.assertTrue(first.global_selection)

    def test_global_rank_tiebreak_prefers_typed_anchor_fit(self) -> None:
        selected = _select_global_best_path(
            [
                _global_candidate_stub((0, 0, 2, 6), 2, 1, ["8", "5", "2"]),
                _global_candidate_stub((0, 0, 2, 0), 1, 1, ["8", "5", "2"]),
            ]
        )

        self.assertEqual(selected.anchor_index, 1)  # type: ignore[union-attr]

    def test_global_rank_priorities_are_lexicographic(self) -> None:
        semantic_winner = _select_global_best_path(
            [
                _global_candidate_stub((1, 0, 1, 0), 1, 1, ["1"]),
                _global_candidate_stub((0, 9, 9, 6), 2, 1, ["1", "2", "3"]),
            ]
        )
        self.assertEqual(semantic_winner.anchor_index, 2)  # type: ignore[union-attr]

        dirty_winner = _select_global_best_path(
            [
                _global_candidate_stub((0, 1, 2, 0), 1, 1, ["1", "2"]),
                _global_candidate_stub((0, 0, 9, 6), 2, 1, ["1", "2", "3"]),
            ]
        )
        self.assertEqual(dirty_winner.anchor_index, 2)  # type: ignore[union-attr]

        length_winner = _select_global_best_path(
            [
                _global_candidate_stub((0, 0, 3, 0), 1, 1, ["1", "2", "3", "4"]),
                _global_candidate_stub((0, 0, 2, 6), 2, 1, ["1", "2", "3"]),
            ]
        )
        self.assertEqual(length_winner.anchor_index, 2)  # type: ignore[union-attr]

    def test_global_rank_stable_when_tuples_match(self) -> None:
        selected = _select_global_best_path(
            [
                _global_candidate_stub((0, 0, 2, 0), 2, 1, ["1", "3"]),
                _global_candidate_stub((0, 0, 2, 0), 1, 2, ["1", "2"]),
                _global_candidate_stub((0, 0, 2, 0), 1, 1, ["1", "4"]),
            ]
        )

        self.assertEqual((selected.anchor_index, selected.path_index), (1, 1))  # type: ignore[union-attr]

    def test_global_rank_components_count_dirty_path_and_missing_edges(self) -> None:
        state = _WorkingState(
            nodes={
                "1": TokenReasoningNode("1", "ENTITYA", 1, "entity"),
                "2": TokenReasoningNode("2", "of", 2, "function"),
                "3": TokenReasoningNode("3", "target", 3, "content"),
            },
            raw_edges={},
            edges={
                ("1", "2"): TokenReasoningEdge(
                    "1",
                    "2",
                    "ENTITYA",
                    "of",
                    support=0.67,
                    edge_quality="MEDIUM",
                    derived=True,
                    rule="test_edge",
                )
            },
            normalized_edges=[],
            virtual_edges=[],
            warnings=[],
        )
        warnings: list[str] = []

        rank, components = _rank_global_path_candidate(
            TokenReasoningPath("A1.P1", ["ENTITYA", "of", "target"], ["1", "2", "3"]),
            _AnswerAnchorCandidate("3", "target", ["clause_predicate"], 0),
            state,
            {"1", "3"},
            warnings=warnings,
        )

        self.assertEqual(rank, (0, 4, 2, 6))
        self.assertEqual(components["medium_edge_count"], 1)
        self.assertEqual(components["weak_edge_count"], 1)
        self.assertEqual(components["function_node_count"], 1)
        self.assertEqual(components["derived_edge_count"], 1)
        self.assertEqual(components["dirty_path_count"], 4)
        self.assertTrue(warnings)

    def test_no_global_candidate_fallback(self) -> None:
        result = HanLPSDPResult(
            text="?",
            tokens=["?"],
            available_keys=[],
            sdp_graphs={},
            edges=[],
            raw={},
        )

        compiled = compile_token_reasoning_structure(result, [])

        self.assertEqual(compiled.path_type, "no_global_path")
        self.assertEqual(compiled.paths, [])
        self.assertEqual(compiled.global_selection, {})
        self.assertIsNone(compiled.answer_anchor)
        self.assertIsNone(compiled.answer_anchor_id)
        self.assertIn("global path selection found no candidates", compiled.warnings)

    def assert_ordered_subsequence(self, values: list[str], expected: list[str]) -> None:
        cursor = 0
        for value in values:
            if cursor < len(expected) and value == expected[cursor]:
                cursor += 1
        self.assertEqual(cursor, len(expected), values)

    def assert_path_union_graph(self, compiled: object) -> None:
        path_node_ids: set[str] = set()
        path_pairs: set[frozenset[str]] = set()
        for path in compiled.paths:  # type: ignore[attr-defined]
            path_node_ids.update(path.node_ids)
            for left, right in zip(path.node_ids, path.node_ids[1:]):
                path_pairs.add(frozenset((left, right)))
        final_node_ids = {node.id for node in compiled.nodes}  # type: ignore[attr-defined]
        final_pairs = {frozenset((edge.source, edge.target)) for edge in compiled.edges}  # type: ignore[attr-defined]
        self.assertEqual(final_node_ids, path_node_ids)
        self.assertEqual(final_pairs, path_pairs)

    def assert_multi_anchor_result(self, compiled: object) -> None:
        self.assertEqual(compiled.path_type, "global_best_path")  # type: ignore[attr-defined]
        self.assertIsNotNone(compiled.answer_anchor)  # type: ignore[attr-defined]
        self.assertIsNotNone(compiled.answer_anchor_id)  # type: ignore[attr-defined]
        self.assertTrue(compiled.anchor_path_results)  # type: ignore[attr-defined]
        self.assertTrue(compiled.global_selection)  # type: ignore[attr-defined]
        self.assertEqual(len(compiled.paths), 1)  # type: ignore[attr-defined]

    def anchor_result(self, compiled: object, anchor_text: str) -> dict[str, object]:
        matches = [
            result
            for result in compiled.anchor_path_results  # type: ignore[attr-defined]
            if result.get("anchor_text") == anchor_text
        ]
        self.assertTrue(matches, f"missing anchor {anchor_text!r}")
        return matches[0]

    def anchor_path_nodes(self, compiled: object, anchor_text: str) -> list[list[str]]:
        return [list(path["nodes"]) for path in self.anchor_result(compiled, anchor_text).get("paths", [])]  # type: ignore[index,union-attr]

    def anchor_path_ids(self, compiled: object, anchor_text: str) -> list[list[str]]:
        return [list(path["node_ids"]) for path in self.anchor_result(compiled, anchor_text).get("paths", [])]  # type: ignore[index,union-attr]

    def _dell_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "What does dell call the feature letting the interface replacing ENTITYA in later iterations of the ENTITYB drives to remain powered when the computer is off?",
            [
                "What",
                "does",
                "dell",
                "call",
                "the",
                "feature",
                "letting",
                "the",
                "interface",
                "replacing",
                "ENTITYA",
                "in",
                "later",
                "iterations",
                "of",
                "the",
                "ENTITYB",
                "drives",
                "to",
                "remain",
                "powered",
                "when",
                "the",
                "computer",
                "is",
                "off",
                "?",
            ],
            [
                _root("sdp/dm", "call", 4),
                _dm("call", "ARG1", "What", 4, 1),
                _dm("call", "ARG2", "feature", 4, 6),
                _pas("call", "verb_ARG2", "feature", 4, 6),
                _dm("feature", "RSTR", "letting", 6, 7),
                _dm("letting", "ARG1", "interface", 7, 9),
                _dm("interface", "RSTR", "replacing", 9, 10),
                _dm("replacing", "ARG2", "ENTITYA", 10, 11),
                _pas("in", "prep_ARG1", "interface", 12, 9),
                _pas("in", "prep_ARG2", "iterations", 12, 14),
                _dm("iterations", "RSTR", "drives", 14, 18),
                _pas("of", "prep_ARG1", "iterations", 15, 14),
                _pas("of", "prep_ARG2", "drives", 15, 18),
                _dm("drives", "ARG1", "ENTITYB", 18, 17),
            ],
        )

    def _johnny_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "The player who defeated ENTITYA for the ENTITYB in 1956 was born in what year?",
            ["The", "player", "who", "defeated", "ENTITYA", "for", "the", "ENTITYB", "in", "1956", "was", "born", "in", "what", "year", "?"],
            [
                _root("sdp/dm", "born", 12),
                _dm("what", "BV", "year", 14, 15),
                _dm("born", "ARG1", "player", 12, 2),
                _dm("born", "TWHEN", "year", 12, 15),
                _dm("defeated", "ARG1", "player", 4, 2),
                _dm("defeated", "ARG2", "ENTITYA", 4, 5),
                _pas("for", "prep_ARG1", "defeated", 6, 4),
                _pas("for", "prep_ARG2", "ENTITYB", 6, 8),
                _dm("1956", "ARG1", "defeated", 10, 4),
            ],
        )

    def _nationality_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "Do director of film ENTITYA and director of film ENTITYB share the same nationality?",
            ["Do", "director", "of", "film", "ENTITYA", "and", "director", "of", "film", "ENTITYB", "share", "the", "same", "nationality", "?"],
            [
                _root("sdp/dm", "share", 11),
                _psd("and", "CONJ.member", "director", 6, 2),
                _psd("and", "CONJ.member", "director", 6, 7),
                _psd("director", "PAT-arg", "ENTITYA", 2, 5),
                _psd("director", "PAT-arg", "ENTITYB", 7, 10),
                _dm("share", "ARG2", "nationality", 11, 14),
                _dm("same", "RSTR", "nationality", 13, 14),
                _pas("share", "verb_ARG1", "director", 11, 2),
                _pas("share", "verb_ARG1", "director", 11, 7),
                _pas("share", "verb_ARG2", "nationality", 11, 14),
                _psd("share", "ACT-arg", "director", 11, 2),
                _psd("share", "ACT-arg", "director", 11, 7),
                _psd("share", "PAT-arg", "nationality", 11, 14),
            ],
        )

    def _film_director_died_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "Which film has the director who died first, ENTITYA or ENTITYB?",
            ["Which", "film", "has", "the", "director", "who", "died", "first", ",", "ENTITYA", "or", "ENTITYB", "?"],
            [
                _dm("Which", "BV", "film", 1, 2),
                _dm("has", "ARG1", "film", 3, 2),
                _dm("has", "ARG2", "director", 3, 5),
                _dm("died", "ARG1", "director", 7, 5),
                _dm("first", "ARG1", "died", 8, 7),
                _pas("or", "coord_ARG1", "ENTITYA", 11, 10),
                _pas("or", "coord_ARG2", "ENTITYB", 11, 12),
                _psd("or", "DISJ.member", "ENTITYA", 11, 10),
                _psd("or", "DISJ.member", "ENTITYB", 11, 12),
                _psd("director", "RSTR", "died", 5, 7),
            ],
        )

    def _which_film_director_younger_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "Which film whose director is younger, ENTITYA or ENTITYB?",
            ["Which", "film", "whose", "director", "is", "younger", ",", "ENTITYA", "or", "ENTITYB", "?"],
            [
                _dm("film", "poss", "director", 2, 4),
                _pas("director", "ARG1", "film", 4, 2),
                _dm("younger", "ARG1", "director", 6, 4),
                _psd("younger", "adj_ARG1", "director", 6, 4),
                _pas("is", "verb_ARG2", "younger", 5, 6),
                _psd("is", "PAT-arg", "younger", 5, 6),
                _pas("or", "coord_ARG1", "ENTITYA", 9, 8),
                _pas("or", "coord_ARG2", "ENTITYB", 9, 10),
                _psd("or", "DISJ.member", "ENTITYA", 9, 8),
                _psd("or", "DISJ.member", "ENTITYB", 9, 10),
            ],
        )

    def _director_born_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "Where was the director of film ENTITYA born?",
            ["Where", "was", "the", "director", "of", "film", "ENTITYA", "born", "?"],
            [
                _dm("director", "ARG1", "ENTITYA", 4, 7),
                _dm("born", "ARG2", "director", 8, 4),
                _dm("Where", "loc", "born", 1, 8),
                _pas("of", "prep_ARG1", "director", 5, 4),
                _pas("of", "prep_ARG2", "ENTITYA", 5, 7),
                _psd("born", "ACT-arg", "director", 8, 4),
            ],
        )

    def _older_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "Who is older, ENTITYA or ENTITYB?",
            ["Who", "is", "older", ",", "ENTITYA", "or", "ENTITYB", "?"],
            [
                _root("sdp/dm", "older", 3),
                _dm("older", "ARG1", "Who", 3, 1),
                _dm("older", "ARG2", "ENTITYA", 3, 5),
                _psd("older", "ACT-arg", "ENTITYB", 3, 7),
            ],
        )

    def _born_later_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "Who was born later, ENTITYA or ENTITYB?",
            ["Who", "was", "born", "later", ",", "ENTITYA", "or", "ENTITYB", "?"],
            [
                _dm("born", "ARG2", "Who", 3, 1),
                _pas("born", "verb_ARG2", "Who", 3, 1),
                _psd("born", "PAT-arg", "Who", 3, 1),
                _dm("later", "loc", "born", 4, 3),
                _psd("later", "adj_ARG1", "born", 4, 3),
                _dm("born", "TWHEN", "later", 3, 4),
                _pas("or", "coord_ARG1", "ENTITYA", 7, 6),
                _pas("or", "coord_ARG2", "ENTITYB", 7, 8),
                _psd("or", "DISJ.member", "ENTITYA", 7, 6),
                _psd("or", "DISJ.member", "ENTITYB", 7, 8),
            ],
        )

    def _typed_year_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "What year was ENTITYA born?",
            ["What", "year", "was", "ENTITYA", "born", "?"],
            [
                _root("sdp/dm", "born", 5),
                _dm("What", "BV", "year", 1, 2),
                _dm("born", "ARG1", "ENTITYA", 5, 4),
                _dm("born", "TWHEN", "year", 5, 2),
            ],
        )

    def _broadcaster_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "Where is the headquarters for the service broadcaster the show ENTITYA is on?",
            ["Where", "is", "the", "headquarters", "for", "the", "service", "broadcaster", "the", "show", "ENTITYA", "is", "on", "?"],
            [
                _dm("for", "ARG1", "headquarters", 5, 4),
                _dm("for", "ARG2", "broadcaster", 5, 8),
                _dm("on", "ARG1", "show", 13, 10),
                _pas("show", "noun_ARG1", "ENTITYA", 10, 11),
                _pas("on", "prep_ARG1", "ENTITYA", 13, 11),
                _psd("show", "ID", "ENTITYA", 10, 11),
                _psd("broadcaster", "RSTR", "is", 8, 12),
                _psd("show", "RSTR", "is", 10, 12),
                _psd("is", "ACT-arg", "show", 12, 10),
            ],
        )

    def _role_series_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "ENTITYA is an American actress and voice actress, known for her role as ENTITYB on what American animated television series produced for ENTITYC?",
            ["ENTITYA", "is", "an", "American", "actress", "and", "voice", "actress", "known", "for", "her", "role", "as", "ENTITYB", "on", "what", "American", "animated", "television", "series", "produced", "for", "ENTITYC", "?"],
            [
                _pas("is", "ARG1", "ENTITYA", 2, 1),
                _pas("is", "ARG2", "actress", 2, 5),
                _dm("known", "ARG1", "actress", 9, 5),
                _pas("for", "prep_ARG1", "known", 10, 9),
                _pas("for", "prep_ARG2", "role", 10, 12),
                _pas("as", "prep_ARG1", "role", 13, 12),
                _pas("as", "prep_ARG2", "ENTITYB", 13, 14),
                _pas("on", "prep_ARG1", "role", 15, 12),
                _pas("on", "prep_ARG2", "series", 15, 20),
                _dm("produced", "ARG1", "series", 21, 20),
                _pas("for", "prep_ARG1", "produced", 22, 21),
                _pas("for", "prep_ARG2", "ENTITYC", 22, 23),
                _dm("what", "BV", "series", 16, 20),
            ],
        )

    def _works_result(self) -> HanLPSDPResult:
        return _hanlp_result(
            "Works by ENTITYA are part of a collection in a museum that houses approximately 65,000 what?",
            ["Works", "by", "ENTITYA", "are", "part", "of", "a", "collection", "in", "a", "museum", "that", "houses", "approximately", "65,000", "what", "?"],
            [
                _pas("by", "prep_ARG1", "Works", 2, 1),
                _pas("by", "prep_ARG2", "ENTITYA", 2, 3),
                _pas("are", "ARG1", "Works", 4, 1),
                _pas("are", "ARG2", "part", 4, 5),
                _dm("part", "ARG2", "collection", 5, 8),
                _pas("in", "prep_ARG1", "collection", 9, 8),
                _pas("in", "prep_ARG2", "museum", 9, 11),
                _dm("houses", "ARG1", "museum", 13, 11),
                _dm("houses", "ARG2", "what", 13, 16),
                _dm("what", "ARG1", "65,000", 16, 15),
                _dm("approximately", "ARG1", "65,000", 14, 15),
            ],
        )


class FakePreprocessLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.step5_user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        if "DEPO Step 5" in system_prompt:
            self.step5_user_prompt = user_prompt
            payload = json.loads(user_prompt)
            assert set(payload) == {"original_question", "paths"}
            assert len(payload["paths"]) == 1
            path_id = payload["paths"][0]["path_id"]
            serialized = json.dumps(payload, ensure_ascii=False)
            assert "ENTITYA" not in serialized
            assert "ENTITYB" not in serialized
            assert "masked_question" not in serialized
            assert "answer_anchor" not in serialized
            return {
                "nodes": [
                    {
                        "id": "q1",
                        "question": "When was Ryan Tubridy born?",
                        "depends_on": [],
                        "support": {"path_id": path_id, "start_index": 0, "end_index": 1},
                    },
                    {
                        "id": "q2",
                        "question": "When was Mauro Massironi born?",
                        "depends_on": [],
                        "support": {"path_id": path_id, "start_index": 0, "end_index": 0},
                    },
                ]
            }
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


class StaticPreprocessLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0
        self.system_prompt = ""
        self.user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.calls += 1
        return self.payload


class RecordingHanLPSDPParser:
    def __init__(self) -> None:
        self.placeholders: list[str] = []
        self.text = ""

    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        self.placeholders = list(placeholders or [])
        self.text = text
        tokens = ["Which", "country", "is", "the", "composer", "of", "film", "ENTITYA", "from", "?"]
        return HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=["tok", "sdp/dm"],
            sdp_graphs={"sdp/dm": []},
            edges=[
                HanLPSDPEdge("sdp/dm", 0, "ROOT", "root", 2, "country"),
                HanLPSDPEdge("sdp/dm", 5, "composer", "ARG1", 8, "ENTITYA"),
                HanLPSDPEdge("sdp/dm", 2, "country", "ARG1", 5, "composer"),
            ],
            raw={"tok": tokens, "sdp/dm": []},
            model="fake",
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
            available_keys=["tok", "sdp/dm", "sdp/pas"],
            sdp_graphs={
                "sdp/dm": [[(3, "ARG1")], [], [(0, "root")], [], [(3, "ARG2")], [], [(3, "ARG2")], []],
                "sdp/pas": [[(3, "ARG1")]],
                "sdp/psd": [[(3, "ACT-arg")]],
            },
            edges=[
                HanLPSDPEdge("sdp/dm", 3, "older", "ARG1", 1, "Who"),
                HanLPSDPEdge("sdp/dm", 0, "ROOT", "root", 3, "older"),
                HanLPSDPEdge("sdp/dm", 3, "older", "ARG2", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/dm", 3, "older", "ARG2", 7, "ENTITYB"),
                HanLPSDPEdge("sdp/pas", 3, "older", "ARG1", 1, "Who"),
                HanLPSDPEdge("sdp/pas", 3, "older", "ARG1", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 6, "or", "coord_ARG1", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 6, "or", "coord_ARG2", 7, "ENTITYB"),
                HanLPSDPEdge("sdp/psd", 3, "older", "ACT-arg", 7, "ENTITYB"),
            ],
            raw={"tok": tokens, "sdp/dm": [], "sdp/pas": [], "sdp/psd": []},
            warnings=[],
            model="fake.hanlp.model",
            mask_token_checks={placeholder: "OK" for placeholder in self.placeholders},
        )


def _entity(question: str, text: str, semantic_type: str = "Entity") -> dict[str, object]:
    start = question.index(text)
    return {
        "text": text,
        "semantic_type_hint": semantic_type,
        "start_char": start,
        "end_char": start + len(text),
        "confidence": 1.0,
        "reason": "test entity",
    }


def _dm(head: str, relation: str, dep: str, head_idx: int, dep_idx: int) -> HanLPSDPEdge:
    return HanLPSDPEdge("sdp/dm", head_idx, head, relation, dep_idx, dep)


def _pas(head: str, relation: str, dep: str, head_idx: int, dep_idx: int) -> HanLPSDPEdge:
    return HanLPSDPEdge("sdp/pas", head_idx, head, relation, dep_idx, dep)


def _psd(head: str, relation: str, dep: str, head_idx: int, dep_idx: int) -> HanLPSDPEdge:
    return HanLPSDPEdge("sdp/psd", head_idx, head, relation, dep_idx, dep)


def _root(formalism: str, dep: str, dep_idx: int) -> HanLPSDPEdge:
    return HanLPSDPEdge(formalism, 0, "ROOT", "root", dep_idx, dep)


def _hanlp_result(text: str, tokens: list[str], edges: list[HanLPSDPEdge]) -> HanLPSDPResult:
    formalisms = sorted({edge.formalism for edge in edges})
    return HanLPSDPResult(
        text=text,
        tokens=tokens,
        available_keys=["tok", *formalisms],
        sdp_graphs={formalism: [] for formalism in formalisms},
        edges=edges,
        raw={"tok": tokens},
    )


def _edge_text_pairs(compiled: object) -> set[frozenset[str]]:
    return {frozenset((edge.source_text, edge.target_text)) for edge in compiled.edges}  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
