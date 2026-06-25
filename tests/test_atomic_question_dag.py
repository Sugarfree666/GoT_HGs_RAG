from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from atomic_question_dag import (  # noqa: E402
    ATOMIC_QUESTION_DAG_SYSTEM,
    PathAlignedAtomicDAGGenerator,
    restore_entity_paths,
    restore_global_best_path,
    restore_global_best_paths,
)
from entity_masking_preprocessor import EntityMaskingPreprocessor  # noqa: E402
from main import run_hanlp_sdp_pipeline  # noqa: E402
from models import HanLPSDPEdge, HanLPSDPResult, MaskMapping, QuestionRecord  # noqa: E402


class AtomicQuestionDAGTest(unittest.TestCase):
    def test_llm_input_contract_contains_only_original_entities_and_global_best_paths(self) -> None:
        llm = RecordingStep5LLM(_johnny_payload())

        PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Do director of film Ten9Eight: Shoot For The Moon share the same nationality?",
            explicit_entities=["Ten9Eight: Shoot For The Moon"],
            global_best_paths=[["Ten9Eight: Shoot For The Moon", "director", "share", "nationality"]],
        )

        payload = json.loads(llm.user_prompts[0])
        self.assertEqual(set(payload), {"original_question", "explicit_entities", "global_best_paths"})
        self.assertEqual(payload["explicit_entities"], ["Ten9Eight: Shoot For The Moon"])
        self.assertEqual(
            payload["global_best_paths"],
            [["Ten9Eight: Shoot For The Moon", "director", "share", "nationality"]],
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        forbidden = [
            "answer_anchor",
            "entity_anchors",
            "constraints",
            "candidate_sets",
            "path_type",
            "masked_question",
            "entity_map",
            "path_id",
            "start_index",
            "end_index",
            "ENTITYA",
            "ENTITYB",
        ]
        for item in forbidden:
            self.assertNotIn(item, serialized)

    def test_restore_entity_paths_replaces_complete_placeholder_tokens_only(self) -> None:
        paths = [
            SimpleNamespace(path_id="P1", nodes=["ENTITYA", "director", "ENTITYBish"]),
            SimpleNamespace(path_id="P2", nodes=["ENTITYB", "born"]),
        ]
        mappings = [
            MaskMapping("ENTITYA", "Ten9Eight: Shoot For The Moon", "entity"),
            MaskMapping("ENTITYB", "Sabotage (1936 Film)", "entity"),
        ]

        restored = restore_entity_paths(paths, mappings)

        self.assertEqual(restored[0].nodes, ("Ten9Eight: Shoot For The Moon", "director", "ENTITYBish"))
        self.assertEqual(restored[1].nodes, ("Sabotage (1936 Film)", "born"))

    def test_restore_global_best_path_replaces_placeholders(self) -> None:
        restored = restore_global_best_path(
            {"nodes": ["ENTITYA", "signed", "person"]},
            [MaskMapping("ENTITYA", "Barcelona", "entity")],
        )

        self.assertEqual(restored, ["Barcelona", "signed", "person"])

    def test_restore_global_best_paths_replaces_each_selected_path(self) -> None:
        paths = [
            SimpleNamespace(nodes=["ENTITYA", "director", "born"]),
            SimpleNamespace(nodes=["ENTITYB", "director", "born"]),
        ]
        restored = restore_global_best_paths(
            paths,
            [
                MaskMapping("ENTITYA", "Illusions (1982 Film)", "entity"),
                MaskMapping("ENTITYB", "It'S A Wonderful Afterlife", "entity"),
            ],
        )

        self.assertEqual(
            restored,
            [
                ["Illusions (1982 Film)", "director", "born"],
                ["It'S A Wonderful Afterlife", "director", "born"],
            ],
        )

    def test_restore_global_best_path_rejects_unmapped_placeholder(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unresolved entity placeholder"):
            restore_global_best_path({"nodes": ["ENTITYA", "born"]}, [])

    def test_single_path_predicate_chain_generates_dependent_atomic_questions(self) -> None:
        result = _generate(_johnny_payload())

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual([node.id for node in result.nodes], ["q1", "q2"])
        self.assertEqual(result.nodes[1].depends_on, ("q1",))
        self.assertEqual(result.edges[0].to_dict(), {"source": "q1", "target": "q2"})
        self.assertEqual(result.leaf_node_ids, ["q2"])
        self.assertTrue(all(node.support is None for node in result.nodes))

    def test_single_node_global_best_path_is_sent_to_llm_without_indices(self) -> None:
        llm = RecordingStep5LLM(_single_action_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="What is the nationality of Some Entity?",
            explicit_entities=["Some Entity"],
            global_best_paths=[["Some Entity"]],
        )

        self.assertTrue(result.valid, result.validation_errors)
        payload = json.loads(llm.user_prompts[0])
        self.assertEqual(payload["global_best_paths"], [["Some Entity"]])
        self.assertNotIn("index", json.dumps(payload, ensure_ascii=False))
        self.assertIsNone(result.nodes[0].support)

    def test_empty_global_best_path_fails_before_llm_call(self) -> None:
        llm = RecordingStep5LLM(_single_action_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Question?",
            explicit_entities=[],
            global_best_paths=[],
        )

        self.assertFalse(result.valid)
        self.assertEqual(llm.user_prompts, [])
        self.assertIn("Step5 requires at least one non-empty global_best_paths entry.", result.validation_errors)

    def test_multi_path_cover_is_sent_to_llm(self) -> None:
        llm = RecordingStep5LLM(_parallel_born_later_payload())
        result = PathAlignedAtomicDAGGenerator(llm).generate(
            original_question="Which film has the director who was born later, Illusions (1982 Film) or It'S A Wonderful Afterlife?",
            explicit_entities=["Illusions (1982 Film)", "It'S A Wonderful Afterlife"],
            global_best_paths=[
                ["Illusions (1982 Film)", "director", "born"],
                ["It'S A Wonderful Afterlife", "director", "born"],
            ],
        )

        self.assertTrue(result.valid, result.validation_errors)
        payload = json.loads(llm.user_prompts[0])
        self.assertEqual(
            payload["global_best_paths"],
            [
                ["Illusions (1982 Film)", "director", "born"],
                ["It'S A Wonderful Afterlife", "director", "born"],
            ],
        )
        self.assertEqual(result.nodes[-1].depends_on, ("q2", "q4"))

    def test_parallel_nationality_actions_can_feed_final_comparison(self) -> None:
        result = _generate(
            _parallel_nationality_payload(),
            question=(
                "Do director of film Ten9Eight: Shoot For The Moon and director of film "
                "Sabotage (1936 Film) share the same nationality?"
            ),
            entities=["Ten9Eight: Shoot For The Moon", "Sabotage (1936 Film)"],
            path=["Ten9Eight: Shoot For The Moon", "director", "share", "nationality"],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(
            [edge.to_dict() for edge in result.edges],
            [
                {"source": "q1", "target": "q2"},
                {"source": "q3", "target": "q4"},
                {"source": "q2", "target": "q5"},
                {"source": "q4", "target": "q5"},
            ],
        )
        self.assertEqual(result.leaf_node_ids, ["q5"])
        self.assertIn("same nationality", result.nodes[-1].question)

    def test_born_later_actions_generate_complete_selection_dag(self) -> None:
        result = _generate(
            _born_later_payload(),
            question="Who was born later, Gideon Johnson Pillow or Holm Jølsen?",
            entities=["Gideon Johnson Pillow", "Holm Jølsen"],
            path=["Gideon Johnson Pillow", "born", "later"],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(
            [node.question for node in result.nodes],
            [
                "When was Gideon Johnson Pillow born?",
                "When was Holm Jølsen born?",
                "Who was born later, Gideon Johnson Pillow or Holm Jølsen, based on q1's answer and q2's answer?",
            ],
        )
        self.assertEqual(
            [edge.to_dict() for edge in result.edges],
            [{"source": "q1", "target": "q3"}, {"source": "q2", "target": "q3"}],
        )

    def test_comparison_action_trace_allows_final_multi_parent_node(self) -> None:
        result = _generate(
            _younger_director_payload(),
            question="Which film whose director is younger, Dangerously They Live or Salad By The Roots?",
            entities=["Dangerously They Live", "Salad By The Roots"],
            path=["Dangerously They Live", "director", "younger"],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(
            [edge.to_dict() for edge in result.edges],
            [
                {"source": "q1", "target": "q2"},
                {"source": "q3", "target": "q4"},
                {"source": "q2", "target": "q5"},
                {"source": "q4", "target": "q5"},
            ],
        )
        self.assertEqual(result.leaf_node_ids, ["q5"])
        self.assertEqual(result.nodes[-1].depends_on, ("q2", "q4"))
        self.assertTrue(all(node.support is None for node in result.nodes))

    def test_dell_long_path_generates_dag_without_support_cover(self) -> None:
        result = _generate(
            _dell_payload(),
            question=(
                "What does dell call the feature letting the interface replacing FireWire "
                "to remain powered when the computer is off?"
            ),
            entities=["FireWire"],
            path=["FireWire", "replacing", "interface", "letting", "feature", "call", "What"],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertTrue(all(node.support is None for node in result.nodes))
        self.assertEqual(
            [edge.to_dict() for edge in result.edges],
            [{"source": "q1", "target": "q2"}, {"source": "q2", "target": "q3"}],
        )

    def test_messi_barcelona_action_trace_regression(self) -> None:
        result = _generate(
            _messi_barcelona_payload(),
            question="When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?",
            entities=["Messi", "Copa del Rey", "Barcelona"],
            path=["Barcelona", "signed", "get", "person", "compared", "goals", "Messi"],
        )

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(
            [node.question for node in result.nodes],
            [
                "Who is the person that Messi's goals in Copa del Rey were compared to?",
                "When did q1's answer get signed by Barcelona?",
            ],
        )
        self.assertEqual(result.nodes[1].depends_on, ("q1",))
        self.assertEqual([edge.to_dict() for edge in result.edges], [{"source": "q1", "target": "q2"}])
        self.assertNotIn("born", result.nodes[1].question.lower())

    def test_actions_payload_is_required(self) -> None:
        result = _generate({"nodes": [{"id": "q1", "question": "What is A?"}]})

        self.assertFalse(result.valid)
        self.assertIn("actions must be a non-empty list.", result.validation_errors)

    def test_future_dependency_is_rejected(self) -> None:
        payload = {
            "actions": [
                {"id": "q1", "consume": ["q2_answer"], "produce": "q1_answer", "question": "What follows q2's answer?"},
                {"id": "q2", "consume": ["B"], "produce": "q2_answer", "question": "What is B?"},
            ]
        }
        result = _generate(payload)

        self.assertFalse(result.valid)
        self.assertIn("qN reference points to non-previous node 'q2'", "; ".join(result.validation_errors))

    def test_unresolved_entity_placeholder_in_question_is_rejected(self) -> None:
        payload = {
            "actions": [
                {"id": "q1", "consume": ["A", "director"], "produce": "q1_answer", "question": "Who directed ENTITYA?"},
            ]
        }
        result = _generate(payload)

        self.assertFalse(result.valid)
        self.assertIn("question contains unresolved ENTITY placeholder", "; ".join(result.validation_errors))

    def test_unresolved_entity_placeholder_in_consume_is_rejected(self) -> None:
        payload = {
            "actions": [
                {"id": "q1", "consume": ["ENTITYA", "director"], "produce": "q1_answer", "question": "Who directed A?"},
            ]
        }
        result = _generate(payload)

        self.assertFalse(result.valid)
        self.assertIn("consume contains unresolved ENTITY placeholder", "; ".join(result.validation_errors))

    def test_question_and_consume_references_are_auto_dependencies(self) -> None:
        payload = {
            "actions": [
                {"id": "q1", "consume": ["A", "director"], "produce": "q1_answer", "question": "Who directed A?"},
                {"id": "q2", "consume": ["nationality", "q1_answer"], "produce": "q2_answer", "question": "What is the nationality?"},
                {
                    "id": "q3",
                    "consume": ["q1_answer", "q2_answer"],
                    "produce": "q3_answer",
                    "question": "What nationality does q1's answer have based on q2_answer?",
                },
            ]
        }
        result = _generate(payload)

        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(result.nodes[1].depends_on, ("q1",))
        self.assertEqual(result.nodes[2].depends_on, ("q1", "q2"))

    def test_braced_question_reference_is_rejected(self) -> None:
        payload = {
            "actions": [
                {"id": "q1", "consume": ["A", "director"], "produce": "q1_answer", "question": "Who directed A?"},
                {
                    "id": "q2",
                    "consume": ["nationality", "q1_answer"],
                    "produce": "q2_answer",
                    "question": "What is the nationality of {{q1}}?",
                },
            ]
        }
        result = _generate(payload)

        self.assertFalse(result.valid)
        self.assertIn("must use qN's answer references", "; ".join(result.validation_errors))

    def test_bad_action_id_and_produce_are_rejected(self) -> None:
        payload = {
            "actions": [
                {"id": "q2", "consume": ["A"], "produce": "q2_answer", "question": "What is A?"},
            ]
        }
        result = _generate(payload)

        self.assertFalse(result.valid)
        errors = "; ".join(result.validation_errors)
        self.assertIn("action id must be q1", errors)
        self.assertIn("produce must be 'q1_answer'", errors)

    def test_non_question_is_rejected(self) -> None:
        payload = {
            "actions": [
                {"id": "q1", "consume": ["A"], "produce": "q1_answer", "question": "Tell me A."},
            ]
        }
        result = _generate(payload)

        self.assertFalse(result.valid)
        self.assertIn("question must be a non-empty single question", "; ".join(result.validation_errors))

    def test_deterministic_output_for_same_payload(self) -> None:
        first = _generate(_johnny_payload())
        second = _generate(_johnny_payload())

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_pipeline_integration_returns_atomic_question_dag(self) -> None:
        llm = FullPipelineLLM()
        result = run_hanlp_sdp_pipeline(
            record=QuestionRecord(question="Who is older, Ryan Tubridy or Mauro Massironi?"),
            index=1,
            preprocessor=EntityMaskingPreprocessor(llm),
            parser=FakeOlderParser(),
        )

        dag = result["atomic_question_dag"]
        self.assertTrue(dag.valid, dag.validation_errors)
        self.assertEqual([node.question for node in dag.nodes], ["When was Ryan Tubridy born?", "When was Mauro Massironi born?"])
        payload = json.loads(llm.step5_user_prompt)
        self.assertEqual(set(payload), {"original_question", "explicit_entities", "global_best_paths"})
        self.assertEqual(payload["explicit_entities"], ["Ryan Tubridy", "Mauro Massironi"])
        self.assertNotIn("ENTITYA", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("ENTITYB", json.dumps(payload, ensure_ascii=False))


class RecordingStep5LLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        self.assert_step5_prompt(system_prompt, user_prompt)
        return self.payload

    @staticmethod
    def assert_step5_prompt(system_prompt: str, user_prompt: str) -> None:
        if system_prompt != ATOMIC_QUESTION_DAG_SYSTEM:
            raise AssertionError("Unexpected system prompt")
        payload = json.loads(user_prompt)
        if set(payload) != {"original_question", "explicit_entities", "global_best_paths"}:
            raise AssertionError(f"Unexpected Step5 prompt keys: {set(payload)}")


class FullPipelineLLM:
    def __init__(self) -> None:
        self.step5_user_prompt = ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "DEPO Step 2" in system_prompt:
            question = "Who is older, Ryan Tubridy or Mauro Massironi?"
            return {
                "entities": [
                    _entity(question, "Ryan Tubridy"),
                    _entity(question, "Mauro Massironi"),
                ],
                "warnings": [],
            }
        if system_prompt == ATOMIC_QUESTION_DAG_SYSTEM:
            self.step5_user_prompt = user_prompt
            payload = json.loads(user_prompt)
            if set(payload) != {"original_question", "explicit_entities", "global_best_paths"}:
                raise AssertionError(f"Unexpected Step5 prompt keys: {set(payload)}")
            return {
                "actions": [
                    {
                        "id": "q1",
                        "consume": ["Ryan Tubridy", "older"],
                        "produce": "q1_answer",
                        "question": "When was Ryan Tubridy born?",
                    },
                    {
                        "id": "q2",
                        "consume": ["Mauro Massironi", "older"],
                        "produce": "q2_answer",
                        "question": "When was Mauro Massironi born?",
                    },
                ]
            }
        raise AssertionError(f"Unexpected system prompt: {system_prompt}")


class FakeOlderParser:
    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        del placeholders
        tokens = ["Who", "is", "older", ",", "ENTITYA", "or", "ENTITYB", "?"]
        return HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=["tok", "sdp/dm", "sdp/pas", "sdp/psd"],
            sdp_graphs={"sdp/dm": [], "sdp/pas": [], "sdp/psd": []},
            edges=[
                HanLPSDPEdge("sdp/dm", 0, "ROOT", "root", 3, "older"),
                HanLPSDPEdge("sdp/dm", 3, "older", "ARG1", 1, "Who"),
                HanLPSDPEdge("sdp/dm", 3, "older", "ARG2", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 6, "or", "coord_ARG1", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/pas", 6, "or", "coord_ARG2", 7, "ENTITYB"),
                HanLPSDPEdge("sdp/psd", 6, "or", "DISJ.member", 5, "ENTITYA"),
                HanLPSDPEdge("sdp/psd", 6, "or", "DISJ.member", 7, "ENTITYB"),
                HanLPSDPEdge("sdp/psd", 3, "older", "ACT-arg", 7, "ENTITYB"),
            ],
            raw={"tok": tokens},
            model="fake",
        )


def _generate(
    payload: dict[str, Any],
    *,
    question: str = "Question?",
    entities: list[str] | None = None,
    path: list[str] | None = None,
) -> Any:
    return PathAlignedAtomicDAGGenerator(RecordingStep5LLM(payload)).generate(
        original_question=question,
        explicit_entities=entities or ["A"],
        global_best_paths=[path or ["A", "B"]],
    )


def _single_action_payload(question: str = "What is the supported fact?") -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "q1",
                "consume": ["A"],
                "produce": "q1_answer",
                "question": question,
            }
        ]
    }


def _johnny_payload() -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "q1",
                "consume": ["Johnny Majors", "defeated", "player"],
                "produce": "q1_answer",
                "question": "Who defeated Johnny Majors for the Heisman Trophy in 1956?",
            },
            {
                "id": "q2",
                "consume": ["q1_answer", "born", "year"],
                "produce": "q2_answer",
                "question": "What year was q1's answer born?",
            },
        ]
    }


def _parallel_nationality_payload() -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "q1",
                "consume": ["Ten9Eight: Shoot For The Moon", "director"],
                "produce": "q1_answer",
                "question": "Who directed Ten9Eight: Shoot For The Moon?",
            },
            {
                "id": "q2",
                "consume": ["q1_answer", "nationality"],
                "produce": "q2_answer",
                "question": "What is the nationality of q1's answer?",
            },
            {
                "id": "q3",
                "consume": ["Sabotage (1936 Film)", "director"],
                "produce": "q3_answer",
                "question": "Who directed Sabotage (1936 Film)?",
            },
            {
                "id": "q4",
                "consume": ["q3_answer", "nationality"],
                "produce": "q4_answer",
                "question": "What is the nationality of q3's answer?",
            },
            {
                "id": "q5",
                "consume": ["q2_answer", "q4_answer", "share", "nationality"],
                "produce": "q5_answer",
                "question": "Do the directors have the same nationality based on q2's answer and q4's answer?",
            },
        ]
    }


def _born_later_payload() -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "q1",
                "consume": ["Gideon Johnson Pillow", "born"],
                "produce": "q1_answer",
                "question": "When was Gideon Johnson Pillow born?",
            },
            {
                "id": "q2",
                "consume": ["Holm Jølsen", "born"],
                "produce": "q2_answer",
                "question": "When was Holm Jølsen born?",
            },
            {
                "id": "q3",
                "consume": ["q1_answer", "q2_answer", "later"],
                "produce": "q3_answer",
                "question": "Who was born later, Gideon Johnson Pillow or Holm Jølsen, based on q1's answer and q2's answer?",
            },
        ]
    }


def _younger_director_payload() -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "q1",
                "consume": ["Dangerously They Live", "director"],
                "produce": "q1_answer",
                "question": "Who directed Dangerously They Live?",
            },
            {
                "id": "q2",
                "consume": ["q1_answer", "younger"],
                "produce": "q2_answer",
                "question": "When was q1's answer born?",
            },
            {
                "id": "q3",
                "consume": ["Salad By The Roots", "director"],
                "produce": "q3_answer",
                "question": "Who directed Salad By The Roots?",
            },
            {
                "id": "q4",
                "consume": ["q3_answer", "younger"],
                "produce": "q4_answer",
                "question": "When was q3's answer born?",
            },
            {
                "id": "q5",
                "consume": ["q2_answer", "q4_answer", "younger"],
                "produce": "q5_answer",
                "question": "Which film has the younger director, Dangerously They Live or Salad By The Roots, based on q2's answer and q4's answer?",
            },
        ]
    }


def _dell_payload() -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "q1",
                "consume": ["FireWire", "replacing", "interface"],
                "produce": "q1_answer",
                "question": "What interface replaced FireWire?",
            },
            {
                "id": "q2",
                "consume": ["q1_answer", "letting", "feature"],
                "produce": "q2_answer",
                "question": "What feature lets q1's answer remain powered?",
            },
            {
                "id": "q3",
                "consume": ["q2_answer", "call"],
                "produce": "q3_answer",
                "question": "What does Dell call q2's answer?",
            },
        ]
    }


def _messi_barcelona_payload() -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "q1",
                "consume": ["person", "compared", "goals", "Messi"],
                "produce": "q1_answer",
                "question": "Who is the person that Messi's goals in Copa del Rey were compared to?",
            },
            {
                "id": "q2",
                "consume": ["Barcelona", "signed", "get", "q1_answer"],
                "produce": "q2_answer",
                "question": "When did q1's answer get signed by Barcelona?",
            },
        ]
    }


def _parallel_born_later_payload() -> dict[str, Any]:
    return {
        "actions": [
            {
                "id": "q1",
                "consume": ["Illusions (1982 Film)", "director"],
                "produce": "q1_answer",
                "question": "Who is the director of Illusions (1982 Film)?",
            },
            {
                "id": "q2",
                "consume": ["q1_answer", "born"],
                "produce": "q2_answer",
                "question": "When was q1's answer born?",
            },
            {
                "id": "q3",
                "consume": ["It'S A Wonderful Afterlife", "director"],
                "produce": "q3_answer",
                "question": "Who is the director of It'S A Wonderful Afterlife?",
            },
            {
                "id": "q4",
                "consume": ["q3_answer", "born"],
                "produce": "q4_answer",
                "question": "When was q3's answer born?",
            },
            {
                "id": "q5",
                "consume": ["q2_answer", "q4_answer"],
                "produce": "q5_answer",
                "question": "Which film has the director born later, Illusions (1982 Film) or It'S A Wonderful Afterlife, based on q2's answer and q4's answer?",
            },
        ]
    }


def _entity(question: str, text: str) -> dict[str, Any]:
    start = question.index(text)
    return {
        "text": text,
        "start_char": start,
        "end_char": start + len(text),
        "confidence": 1.0,
        "reason": "test entity",
    }


if __name__ == "__main__":
    unittest.main()
