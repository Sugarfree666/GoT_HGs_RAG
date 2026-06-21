from __future__ import annotations

import io
import inspect
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from entity_masking_preprocessor import EntityMaskingPreprocessor  # noqa: E402
from main import (  # noqa: E402
    _pipeline_debug_record,
    _write_run_debug_json,
    print_hanlp_sdp_result,
    run_hanlp_sdp_pipeline,
)
from models import HanLPSDPEdge, HanLPSDPResult, QuestionRecord  # noqa: E402
from tri_sdp_reasoning_compiler import compile_token_reasoning_structure  # noqa: E402


class HanLPSDPMainlineTest(unittest.TestCase):
    def test_hanlp_sdp_pipeline_preprocesses_once_and_parses_masked_question(self) -> None:
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
        self.assertEqual(llm.step2_calls, 1)
        self.assertEqual(llm.step5_calls, 1)

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
        self.assertIn("[Graph]", output)
        self.assertIn("ENTITYA ---- older", output)
        self.assertIn("ENTITYB ---- older", output)
        self.assertIn("[Paths]", output)
        self.assertIn("P1: ENTITYA ---- older", output)
        self.assertIn("P2: ENTITYB ---- older", output)
        self.assertIn("answer_anchor: older", output)
        self.assertIn("entity_anchors: ENTITYA, ENTITYB", output)
        self.assertIn("[5. Atomic Question DAG]", output)
        self.assertIn("q1: When was Ryan Tubridy born?", output)
        self.assertIn("q2: When was Mauro Massironi born?", output)
        self.assertIn("support: P1[0:1]", output)
        self.assertIn("support: P2[0:1]", output)
        self.assertNotIn("[4. Content Reasoning Chains]", output)
        self.assertNotIn("[4. Simplified SDP/DM Graph]", output)
        self.assertNotIn("[Kept / Derived Edges]", output)
        self.assertNotIn("older --ARG2--> ENTITYA", output)
        self.assertNotIn("older --ARG1--> ANSWER\n", output)

    def test_pipeline_llm_call_budget_and_skip_step5(self) -> None:
        record = QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?")

        llm = FakePreprocessLLM()
        result = run_hanlp_sdp_pipeline(
            record=record,
            index=1,
            preprocessor=EntityMaskingPreprocessor(llm),
            parser=FakeHanLPSDPParser(),
        )
        self.assertIsNotNone(result["atomic_question_dag"])
        self.assertEqual(llm.step2_calls, 1)
        self.assertEqual(llm.step5_calls, 1)
        self.assertEqual(llm.calls, 2)

        skip_llm = FakePreprocessLLM()
        skipped = run_hanlp_sdp_pipeline(
            record=record,
            index=1,
            preprocessor=EntityMaskingPreprocessor(skip_llm),
            parser=FakeHanLPSDPParser(),
            skip_step5=True,
        )
        self.assertIsNone(skipped["atomic_question_dag"])
        self.assertEqual(skip_llm.step2_calls, 1)
        self.assertEqual(skip_llm.step5_calls, 0)
        self.assertEqual(skip_llm.calls, 1)

        signature = inspect.signature(compile_token_reasoning_structure)
        self.assertNotIn("llm_client", signature.parameters)
        self.assertNotIn("llm", signature.parameters)

    def test_cli_debug_file_overwrites_and_contains_scored_candidate_paths(self) -> None:
        record = QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?")
        result = run_hanlp_sdp_pipeline(
            record=record,
            index=1,
            preprocessor=EntityMaskingPreprocessor(FakePreprocessLLM()),
            parser=FakeHanLPSDPParser(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            debug_path = Path(tmpdir) / "depo_debug.json"
            debug_path.write_text("old debug content", encoding="utf-8")

            written = _write_run_debug_json([_pipeline_debug_record(1, record, result)], debug_dir=tmpdir)
            payload = json.loads(Path(written).read_text(encoding="utf-8"))

        self.assertEqual(Path(written).name, "depo_debug.json")
        self.assertEqual(payload["records"][0]["original_question"], record.question)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("old debug content", serialized)
        step4_debug = payload["records"][0]["step4"]["debug_payload"]
        self.assertIn("reasoning_candidates", step4_debug)
        self.assertTrue(any(candidate.get("candidate_paths") for candidate in step4_debug["reasoning_candidates"]))
        self.assertTrue(
            any(
                "rank_components" in path_record
                for candidate in step4_debug["reasoning_candidates"]
                for path_record in candidate.get("candidate_paths", [])
            )
        )

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

        self.assertEqual(compiled.path_type, "single_main_path")
        self.assertEqual(len(compiled.paths), 1)
        path = compiled.paths[0]
        self.assertEqual(path.nodes[0], "ENTITYA")
        self.assert_ordered_subsequence(path.nodes, ["ENTITYA", "replacing", "interface", "letting", "feature", "call", "What"])
        self.assertNotIn("ENTITYB", path.nodes)
        self.assertEqual(len(path.node_ids), len(set(path.node_ids)))
        self.assertEqual(compiled.entity_anchors, ["ENTITYA"])
        self.assertNotIn("ENTITYB", {node.text for node in compiled.nodes})
        self.assert_path_union_graph(compiled)

    def test_johnny_majors_main_path_excludes_constraint_entity(self) -> None:
        result = self._johnny_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.path_type, "single_main_path")
        self.assertEqual(len(compiled.paths), 1)
        self.assertEqual(compiled.paths[0].nodes, ["ENTITYA", "defeated", "player", "born", "year"])
        self.assertNotIn("ENTITYB", compiled.paths[0].nodes)
        self.assertNotIn("1956", compiled.paths[0].nodes)
        self.assertNotIn("ENTITYB", {node.text for node in compiled.nodes})
        self.assertNotIn("1956", {node.text for node in compiled.nodes})
        self.assertEqual(compiled.entity_anchors, ["ENTITYA"])
        self.assertEqual(len(compiled.paths[0].node_ids), len(set(compiled.paths[0].node_ids)))
        self.assert_path_union_graph(compiled)

    def test_role_coordinated_nationality_parallel_cover(self) -> None:
        result = self._nationality_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.answer_anchor, "nationality")
        self.assertEqual(compiled.answer_anchor_id, "14")
        self.assertEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual([path.node_ids for path in compiled.paths], [["5", "2", "11", "14"], ["10", "7", "11", "14"]])
        self.assertEqual([path.nodes for path in compiled.paths], [["ENTITYA", "director", "share", "nationality"], ["ENTITYB", "director", "share", "nationality"]])
        self.assertTrue(all(len(path.node_ids) == len(set(path.node_ids)) for path in compiled.paths))
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assertEqual(compiled.entity_anchors, ["ENTITYA", "ENTITYB"])
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

        self.assertEqual(compiled.answer_anchor, "omega")
        self.assertEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual([path.node_ids for path in compiled.paths], [["4", "2", "9", "12"], ["8", "6", "9", "12"]])
        self.assertEqual(compiled.entity_anchors, ["ENTITYA", "ENTITYB"])
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assert_path_union_graph(compiled)

    def test_global_anchor_selection_handles_both_directors_same_nationality(self) -> None:
        result = _hanlp_result(
            "Do both directors of films ENTITYA and ENTITYB have the same nationality?",
            ["Do", "both", "directors", "of", "films", "ENTITYA", "and", "ENTITYB", "have", "the", "same", "nationality", "?"],
            [
                _dm("both", "BV", "directors", 2, 3),
                _dm("have", "ARG1", "directors", 9, 3),
                _dm("ENTITYA", "_and_c", "ENTITYB", 6, 8),
                _dm("have", "ARG2", "nationality", 9, 12),
                _dm("the", "BV", "nationality", 10, 12),
                _dm("same", "ARG1", "nationality", 11, 12),
                _pas("Do", "aux_ARG1", "directors", 1, 3),
                _pas("both", "det_ARG1", "directors", 2, 3),
                _pas("of", "prep_ARG1", "directors", 4, 3),
                _pas("have", "verb_ARG1", "directors", 9, 3),
                _pas("films", "noun_ARG1", "ENTITYA", 5, 6),
                _pas("and", "coord_ARG1", "ENTITYA", 7, 6),
                _pas("of", "prep_ARG2", "and", 4, 7),
                _pas("films", "noun_ARG1", "and", 5, 7),
                _pas("and", "coord_ARG2", "ENTITYB", 7, 8),
                _root("sdp/pas", "have", 9),
                _pas("Do", "aux_ARG2", "have", 1, 9),
                _pas("have", "verb_ARG2", "nationality", 9, 12),
                _pas("the", "det_ARG1", "nationality", 10, 12),
                _pas("same", "adj_ARG1", "nationality", 11, 12),
                _psd("directors", "RSTR", "both", 3, 2),
                _psd("have", "ACT-arg", "directors", 9, 3),
                _psd("and", "CONJ.member", "ENTITYA", 7, 6),
                _psd("and", "CONJ.member", "ENTITYB", 7, 8),
                _root("sdp/psd", "have", 9),
                _psd("nationality", "RSTR", "same", 12, 11),
                _psd("have", "PAT-arg", "nationality", 9, 12),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.answer_anchor, "nationality")
        self.assertNotEqual(compiled.answer_anchor, "both")
        self.assertEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual(compiled.entity_anchors, ["ENTITYA", "ENTITYB"])
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assertEqual(len(compiled.paths), 2)
        for path, entity in zip(compiled.paths, ["ENTITYA", "ENTITYB"]):
            self.assertEqual(path.nodes[0], entity)
            self.assertIn("directors", path.nodes)
            self.assertEqual(path.nodes[-1], "nationality")
            self.assertGreater(len(path.node_ids), 1)
            self.assertEqual(len(path.node_ids), len(set(path.node_ids)))
        self.assert_path_union_graph(compiled)

    def test_global_anchor_selection_is_structural_not_lexical_for_quantified_attribute(self) -> None:
        result = _hanlp_result(
            "Do quantifier_token role_token of container_token ENTITYA and ENTITYB predicate_token modifier_token attribute_token?",
            [
                "Do",
                "quantifier_token",
                "role_token",
                "of",
                "container_token",
                "ENTITYA",
                "and",
                "ENTITYB",
                "predicate_token",
                "modifier_token",
                "attribute_token",
                "?",
            ],
            [
                _dm("quantifier_token", "BV", "role_token", 2, 3),
                _dm("predicate_token", "ARG1", "role_token", 9, 3),
                _dm("ENTITYA", "_and_c", "ENTITYB", 6, 8),
                _dm("predicate_token", "ARG2", "attribute_token", 9, 11),
                _dm("modifier_token", "ARG1", "attribute_token", 10, 11),
                _pas("of", "prep_ARG1", "role_token", 4, 3),
                _pas("predicate_token", "verb_ARG1", "role_token", 9, 3),
                _pas("container_token", "noun_ARG1", "ENTITYA", 5, 6),
                _pas("and", "coord_ARG1", "ENTITYA", 7, 6),
                _pas("of", "prep_ARG2", "and", 4, 7),
                _pas("container_token", "noun_ARG1", "and", 5, 7),
                _pas("and", "coord_ARG2", "ENTITYB", 7, 8),
                _root("sdp/pas", "predicate_token", 9),
                _pas("predicate_token", "verb_ARG2", "attribute_token", 9, 11),
                _pas("modifier_token", "adj_ARG1", "attribute_token", 10, 11),
                _psd("role_token", "RSTR", "quantifier_token", 3, 2),
                _psd("predicate_token", "ACT-arg", "role_token", 9, 3),
                _psd("and", "CONJ.member", "ENTITYA", 7, 6),
                _psd("and", "CONJ.member", "ENTITYB", 7, 8),
                _root("sdp/psd", "predicate_token", 9),
                _psd("attribute_token", "RSTR", "modifier_token", 11, 10),
                _psd("predicate_token", "PAT-arg", "attribute_token", 9, 11),
            ],
        )

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.answer_anchor, "attribute_token")
        self.assertNotEqual(compiled.answer_anchor, "quantifier_token")
        self.assertEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual(compiled.entity_anchors, ["ENTITYA", "ENTITYB"])
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assertEqual(len(compiled.paths), 2)
        self.assertTrue(all(path.nodes[-1] == "attribute_token" for path in compiled.paths))
        self.assertTrue(all("role_token" in path.nodes for path in compiled.paths))
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

        self.assertNotEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual(compiled.path_type, "single_main_path")
        self.assertEqual(compiled.entity_anchors, ["ENTITYA"])
        self.assertEqual(len(compiled.paths), 1)
        self.assertNotIn("ENTITYB", {node.text for node in compiled.nodes})
        self.assert_path_union_graph(compiled)

    def test_candidate_comparison_uses_typed_slot_substitution(self) -> None:
        result = self._film_director_died_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual([path.nodes for path in compiled.paths], [["ENTITYA", "director", "died"], ["ENTITYB", "director", "died"]])
        self.assertTrue(any(item["text"] == "first" for item in compiled.constraints))
        rendered = "\n".join(" ---- ".join(path.nodes) for path in compiled.paths)
        self.assertNotIn(" or ", rendered)
        self.assertNotIn("film", {node.text for node in compiled.nodes})
        derived_rules = {edge.rule for edge in compiled.edges if edge.derived}
        self.assertTrue(any("candidate_slot_substitution" in rule for rule in derived_rules))
        substitution_edges = [edge for edge in compiled.edges if "candidate_slot_substitution" in edge.rule]
        self.assertTrue(all(edge.provenance for edge in substitution_edges))
        self.assert_path_union_graph(compiled)

    def test_candidate_comparison_recovers_surface_typed_wh_slot(self) -> None:
        result = self._which_film_director_younger_result()
        self.assertFalse(any(edge.head == "Which" and edge.dep == "film" for edge in result.edges))

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.answer_anchor, "film")
        self.assertEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual([path.node_ids for path in compiled.paths], [["8", "4", "6"], ["10", "4", "6"]])
        self.assertEqual([path.nodes for path in compiled.paths], [["ENTITYA", "director", "younger"], ["ENTITYB", "director", "younger"]])
        self.assertEqual(compiled.entity_anchors, ["ENTITYA", "ENTITYB"])
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assertTrue(all(len(path.node_ids) == len(set(path.node_ids)) for path in compiled.paths))
        self.assert_path_union_graph(compiled)

        substitution_edges = [edge for edge in compiled.edges if "candidate_slot_substitution" in edge.rule]
        self.assertEqual(len(substitution_edges), 2)
        for edge in substitution_edges:
            provenance = edge.provenance[0]
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
        self.assertEqual(compile_token_reasoning_structure(what_year, ["ENTITYA"]).answer_anchor, "year")

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
        self.assertNotEqual(compile_token_reasoning_structure(what_does, ["ENTITYA"]).answer_anchor, "does")

        which_of = _hanlp_result(
            "Which of ENTITYA is older?",
            ["Which", "of", "ENTITYA", "is", "older", "?"],
            [
                _root("sdp/dm", "older", 5),
                _dm("older", "ARG1", "ENTITYA", 5, 3),
            ],
        )
        compiled_which_of = compile_token_reasoning_structure(which_of, ["ENTITYA"])
        self.assertNotEqual(compiled_which_of.answer_anchor, "of")
        self.assertNotEqual(compiled_which_of.answer_anchor, "ENTITYA")

    def test_bare_wh_candidate_substitution_born_later(self) -> None:
        result = self._born_later_result()
        self.assertFalse(any(edge.head_idx == 0 for edge in result.edges))

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB"])

        self.assertEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual(compiled.answer_anchor, "born")
        self.assertEqual([path.node_ids for path in compiled.paths], [["6", "3", "4"], ["8", "3", "4"]])
        self.assertEqual([path.nodes for path in compiled.paths], [["ENTITYA", "born", "later"], ["ENTITYB", "born", "later"]])
        self.assertEqual(compiled.entity_anchors, ["ENTITYA", "ENTITYB"])
        self.assertIn(["ENTITYA", "ENTITYB"], compiled.candidate_sets)
        self.assertTrue(all(len(path.node_ids) > 1 for path in compiled.paths))
        self.assertTrue(all(len(path.node_ids) == len(set(path.node_ids)) for path in compiled.paths))
        self.assert_path_union_graph(compiled)

        substitution_edges = [edge for edge in compiled.edges if "candidate_bare_wh_substitution" in edge.rule]
        self.assertEqual(len(substitution_edges), 2)
        self.assertTrue(all(edge.derived and edge.provenance for edge in substitution_edges))
        for edge in substitution_edges:
            provenance = edge.provenance[0]
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

        self.assertEqual(compiled.path_type, "candidate_path_cover")
        self.assertEqual(compiled.answer_anchor, "pivot")
        self.assertEqual([path.node_ids for path in compiled.paths], [["6", "3", "4"], ["8", "3", "4"]])
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

        self.assertNotEqual(compiled.debug_payload["selection_mode"], "candidate_bare_wh_substitution")
        self.assertFalse(any("candidate_bare_wh_substitution" in edge.rule for edge in compiled.edges))
        self.assertFalse(any("candidate_bare_wh_substitution" in str(edge) for edge in compiled.debug_payload["virtual_edges"]))

    def test_simple_main_chain_and_answer_anchor_regressions(self) -> None:
        director_born = self._director_born_result()

        compiled = compile_token_reasoning_structure(director_born, ["ENTITYA"])

        self.assertEqual(compiled.path_type, "single_main_path")
        self.assertEqual(compiled.paths[0].nodes, ["ENTITYA", "director", "born"])
        self.assertEqual(_edge_text_pairs(compiled), {frozenset(("ENTITYA", "director")), frozenset(("director", "born"))})
        self.assert_path_union_graph(compiled)

        self.assertEqual(compile_token_reasoning_structure(self._older_result(), ["ENTITYA", "ENTITYB"]).answer_anchor, "older")
        self.assertEqual(compile_token_reasoning_structure(director_born, ["ENTITYA"]).answer_anchor, "born")
        self.assertEqual(compile_token_reasoning_structure(self._typed_year_result(), ["ENTITYA"]).answer_anchor, "year")
        self.assertEqual(compile_token_reasoning_structure(self._nationality_result(), ["ENTITYA", "ENTITYB"]).answer_anchor, "nationality")
        self.assertEqual(compile_token_reasoning_structure(self._born_later_result(), ["ENTITYA", "ENTITYB"]).answer_anchor, "born")
        self.assertEqual(compile_token_reasoning_structure(self._which_film_director_younger_result(), ["ENTITYA", "ENTITYB"]).answer_anchor, "film")

    def test_broadcaster_headquarters_remains_single_main_path(self) -> None:
        result = self._broadcaster_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA"])

        self.assertEqual(compiled.paths[0].nodes, ["ENTITYA", "show", "broadcaster", "headquarters"])
        self.assert_path_union_graph(compiled)

    def test_role_descriptor_lifting_is_debug_evidence_not_forced_terminal_cover(self) -> None:
        result = self._role_series_result()

        compiled = compile_token_reasoning_structure(result, ["ENTITYA", "ENTITYB", "ENTITYC"])

        json.dumps(compiled.to_dict())
        self.assertEqual(len(compiled.paths), 1)
        self.assertEqual(len(compiled.paths[0].node_ids), len(set(compiled.paths[0].node_ids)))
        self.assert_path_union_graph(compiled)
        self.assertTrue(
            any("descriptor_lifting" in str(edge.get("rule", "")) for edge in compiled.debug_payload["virtual_edges"])
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

        self.assertEqual(compiled.paths[0].nodes, ["ENTITYA", "Works", "part", "collection", "museum", "houses", "what"])
        self.assertTrue(any(item["text"] == "65,000" for item in compiled.constraints))
        self.assertNotIn("65,000", {node.text for node in compiled.nodes})
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

        self.assertEqual(compiled.answer_anchor, "target")
        self.assertEqual(compiled.paths[0].nodes, ["ENTITYA", "links", "target"])
        self.assertEqual(compiled.entity_anchors, ["ENTITYA"])
        self.assertNotIn("ENTITYB", {node.text for node in compiled.nodes})
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
                if first.path_type == "single_main_path":
                    unselected = set(entities) - set(first.entity_anchors)
                    self.assertFalse(unselected & {node.text for node in first.nodes})

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
        self.step2_calls = 0
        self.step5_calls = 0
        self.step5_user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        if "DEPO Step 5" in system_prompt:
            self.step5_calls += 1
            self.step5_user_prompt = user_prompt
            payload = json.loads(user_prompt)
            assert set(payload) == {"original_question", "paths"}
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
                        "support": {"path_id": "P1", "start_index": 0, "end_index": 1},
                    },
                    {
                        "id": "q2",
                        "question": "When was Mauro Massironi born?",
                        "depends_on": [],
                        "support": {"path_id": "P2", "start_index": 0, "end_index": 1},
                    },
                ]
            }
        assert "DEPO Step 2: topic entity extraction" in system_prompt
        self.step2_calls += 1
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
