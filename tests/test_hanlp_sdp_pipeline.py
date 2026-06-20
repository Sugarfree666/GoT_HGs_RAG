from __future__ import annotations

import io
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
    print_corenlp_dependency_result,
    print_hanlp_sdp_result,
    run_corenlp_dependency_pipeline,
    run_hanlp_sdp_pipeline,
)
from models import CoreNLPToken, DependencyEdge, DependencyParse, HanLPSDPEdge, HanLPSDPResult, QuestionRecord  # noqa: E402
from sdp_dm_simplifier import SDPDMSimplifier  # noqa: E402
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
        self.assertEqual(llm.calls, 1)

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
        self.assertNotIn("[4. Content Reasoning Chains]", output)
        self.assertNotIn("[4. Simplified SDP/DM Graph]", output)
        self.assertNotIn("[Kept / Derived Edges]", output)
        self.assertNotIn("older --ARG2--> ENTITYA", output)
        self.assertNotIn("older --ARG1--> ANSWER\n", output)
        self.assertNotIn("Relation-Carrier Declarative Views", output)
        self.assertNotIn("CoreNLP + OpenIE View Annotations", output)
        self.assertNotIn("Semantic Reasoning Path Induction", output)
        self.assertNotIn("Semantic-Path-Guided Atomic DAG", output)

    def test_corenlp_mainline_reuses_hanlp_preprocess_and_stops_after_dependency_parse(self) -> None:
        record = QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?")
        parser = FakeCoreNLPParser()
        llm = FakePreprocessLLM()
        preprocessor = EntityMaskingPreprocessor(llm)

        result = run_corenlp_dependency_pipeline(
            record=record,
            index=1,
            preprocessor=preprocessor,
            parser=parser,
        )

        preprocess_result = result["preprocess_result"]
        self.assertEqual(preprocess_result.masked_question, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(preprocess_result.sdp_input_sentence, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(result["corenlp_input_sentence"], "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(parser.text, "Who is older, ENTITYA or ENTITYB?")
        self.assertEqual(llm.calls, 1)
        self.assertNotIn("corenlp_view_annotations", result)
        self.assertNotIn("dependency_graph", result)
        self.assertNotIn("atomic_evidences", result)
        self.assertNotIn("semantic_reasoning_paths", result)
        self.assertNotIn("subquestion_dag", result)

        stream = io.StringIO()
        with redirect_stdout(stream):
            print_corenlp_dependency_result(1, record, result)
        output = stream.getvalue()

        self.assertIn("[Original Question]", output)
        self.assertIn("[1. Explicit Entities]", output)
        self.assertIn(" - Ryan Tubridy", output)
        self.assertNotIn(" - Ryan Tubridy [Person]", output)
        self.assertIn("[2. Entity Masking]", output)
        self.assertIn("Masked question: Who is older, ENTITYA or ENTITYB?", output)
        self.assertNotIn("SDP input sentence:", output)
        self.assertIn("CoreNLP input sentence: Who is older, ENTITYA or ENTITYB?", output)
        self.assertIn("[3. CoreNLP Dependency Parsing]", output)
        self.assertIn("[CoreNLP Dependency Edges]", output)
        self.assertIn("older[3] --nsubj--> Who[1]", output)
        self.assertNotIn("[4.", output)
        self.assertNotIn("CoreNLP + OpenIE View Annotations", output)
        self.assertNotIn("Semantic Reasoning Path Induction", output)
        self.assertNotIn("Semantic-Path-Guided Atomic DAG", output)

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

    def test_sdp_dm_simplifier_removes_glue_and_keeps_core_edges(self) -> None:
        result = HanLPSDPResult(
            text="ANSWER is whether the directors of film ENTITYA and ENTITYB share the same nationality.",
            tokens=[],
            available_keys=["tok", "sdp/dm"],
            sdp_graphs={"sdp/dm": []},
            edges=[
                _dm("is", "ARG1", "ANSWER", 2, 1),
                _dm("the", "BV", "directors", 4, 5),
                _dm("share", "ARG1", "directors", 11, 5),
                _dm("directors", "ARG1", "ENTITYA", 5, 8),
                _dm("film", "compound", "ENTITYA", 7, 8),
                _dm("ENTITYA", "_and_c", "ENTITYB", 8, 10),
                _dm("is", "ARG2", "share", 2, 11),
                _dm("share", "ARG2", "nationality", 11, 14),
                _dm("the", "BV", "nationality", 12, 14),
                _dm("same", "ARG1", "nationality", 13, 14),
                HanLPSDPEdge("sdp/pas", 11, "share", "ARG0", 5, "directors"),
            ],
            raw={},
        )

        graph = SDPDMSimplifier().simplify(result)

        self.assertEqual(
            [edge.display() for edge in graph.edges],
            [
                "share --ARG1--> directors",
                "directors --ARG1--> ENTITYA",
                "ENTITYA --_and_c--> ENTITYB",
                "share --ARG2--> nationality",
                "same --ARG1--> nationality",
            ],
        )
        self.assertIn("is --ARG1--> ANSWER", graph.removed_edges)
        self.assertIn("is --ARG2--> share", graph.removed_edges)
        self.assertIn("the --BV--> directors", graph.removed_edges)
        self.assertIn("film --compound--> ENTITYA", graph.removed_edges)
        self.assertNotIn("share --ARG0--> directors", [edge.display() for edge in graph.edges])

    def test_sdp_dm_simplifier_collapses_preposition_arg_pairs(self) -> None:
        result = HanLPSDPResult(
            text="The sister of ANSWER played Susie in ENTITYA.",
            tokens=[],
            available_keys=["tok", "sdp/dm"],
            sdp_graphs={"sdp/dm": []},
            edges=[
                _dm("The", "BV", "sister", 1, 2),
                _dm("of", "ARG1", "sister", 3, 2),
                _dm("played", "ARG1", "sister", 5, 2),
                _dm("of", "ARG2", "ANSWER", 3, 4),
                _dm("in", "ARG1", "played", 7, 5),
                _dm("played", "ARG2", "Susie", 5, 6),
                _dm("in", "ARG2", "ENTITYA", 7, 8),
            ],
            raw={},
        )

        graph = SDPDMSimplifier().simplify(result)

        self.assertEqual(
            [edge.display() for edge in graph.edges],
            [
                "played --in--> ENTITYA",
                "played --ARG1--> sister",
                "played --ARG2--> Susie",
            ],
        )
        derived = [edge for edge in graph.edges if edge.derived]
        self.assertEqual([edge.rule for edge in derived], ["collapse_preposition_arg_pair"])
        self.assertIn("in --ARG1--> played", derived[0].provenance)
        self.assertIn("in --ARG2--> ENTITYA", derived[0].provenance)
        self.assertIn("of --ARG1--> sister", graph.removed_edges)
        self.assertIn("of --ARG2--> ANSWER", graph.removed_edges)
        self.assertIn("The --BV--> sister", graph.removed_edges)


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

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
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


class FakeCoreNLPParser:
    def __init__(self) -> None:
        self.text = ""

    def parse(self, text: str) -> DependencyParse:
        self.text = text
        tokens = ["Who", "is", "older", ",", "ENTITYA", "or", "ENTITYB", "?"]
        return DependencyParse(
            tokens=[CoreNLPToken(index=index, word=word) for index, word in enumerate(tokens, start=1)],
            edges=[
                DependencyEdge("older", "nsubj", "Who", 3, 1),
                DependencyEdge("older", "cop", "is", 3, 2),
                DependencyEdge("older", "obj", "ENTITYA", 3, 5),
                DependencyEdge("ENTITYA", "conj:or", "ENTITYB", 5, 7),
            ],
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
