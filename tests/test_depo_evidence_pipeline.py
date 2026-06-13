from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from atomic_evidence_extractor import AtomicEvidenceExtractor  # noqa: E402
from entity_path_pipeline import (  # noqa: E402
    ATOMIC_DAG_FROM_SEMANTIC_REASONING_PATH_SYSTEM,
    SEMANTIC_REASONING_PATH_SYSTEM,
    EntityPathSemanticParser,
)
from graph_builder import GraphBuilder  # noqa: E402
from main import run_pipeline  # noqa: E402
from models import (  # noqa: E402
    CoreNLPToken,
    CoreNLPViewAnnotation,
    DeclarativeView,
    DependencyEdge,
    ExplicitEntityResult,
    MaskMapping,
    MaskSpan,
    MaskSpanResult,
    OpenIETriple,
    QuestionRecord,
    RelationCarrierViewResult,
)
from question_normalizer import RelationCarrierDeclarativeGenerator  # noqa: E402


class EvidencePipelineTest(unittest.TestCase):
    def test_declarative_views_generated_after_masking_preserve_placeholders(self) -> None:
        generator = RelationCarrierDeclarativeGenerator(None)

        result = generator.generate_relation_carrier_views(
            original_question="What is the nationality of the performer of song Changed It?",
            masked_question="What is the nationality of the performer of song SongA?",
            placeholders=["SongA"],
        )

        self.assertEqual(result.masked_question, "What is the nationality of the performer of song SongA?")
        self.assertEqual(
            [view.sentence for view in result.declarative_views],
            ["The song SongA has a performer.", "The performer has a nationality."],
        )
        self.assertIn("SongA", result.declarative_views[0].sentence)

    def test_extractor_builds_corenlp_structural_evidence(self) -> None:
        extractor = AtomicEvidenceExtractor()
        evidence = extractor.extract(
            masked_question="Are LocationA and LocationB both located in the same country?",
            mask_mappings=[
                _mapping("LocationA", "Marufabad", "Location"),
                _mapping("LocationB", "Nasamkhrali", "Location"),
            ],
            declarative_views=[DeclarativeView(id="view_1", sentence="LocationA and LocationB are located in the same country.")],
            corenlp_annotations=[
                CoreNLPViewAnnotation(
                    view_id="view_1",
                    text="LocationA and LocationB are located in the same country.",
                    tokens=[
                        _token(1, "LocationA", "NNP"),
                        _token(2, "and", "CC"),
                        _token(3, "LocationB", "NNP"),
                        _token(5, "located", "VBN"),
                        _token(9, "country", "NN"),
                    ],
                    edges=[
                        DependencyEdge("located", "nsubj", "LocationA", 5, 1),
                        DependencyEdge("LocationA", "conj:and", "LocationB", 1, 3),
                        DependencyEdge("located", "obl:in", "country", 5, 9),
                    ],
                )
            ],
            operator_intent={"type": "boolean", "cues": ["both", "same"]},
        )

        by_type = _by_type(evidence)
        self.assertIn("dependency_edge", by_type)
        self.assertIn("role_or_attribute", by_type)
        self.assertIn("operator_cue", by_type)
        self.assertTrue(any(item.dependency_relation == "obl:in" for item in by_type["dependency_edge"]))
        self.assertTrue(any(item.text == "country" for item in by_type["role_or_attribute"]))

    def test_extractor_builds_openie_relational_evidence(self) -> None:
        extractor = AtomicEvidenceExtractor()
        evidence = extractor.extract(
            masked_question="What is the nationality of the performer of song SongA?",
            mask_mappings=[_mapping("SongA", "Changed It", "Song")],
            declarative_views=[DeclarativeView(id="view_1", sentence="The song SongA has a performer.")],
            corenlp_annotations=[
                CoreNLPViewAnnotation(
                    view_id="view_1",
                    text="The song SongA has a performer.",
                    openie_triples=[OpenIETriple(subject="song SongA", relation="has", object="performer", confidence=0.91)],
                )
            ],
            operator_intent={"type": "lookup", "target_hint": "nationality"},
        )

        by_type = _by_type(evidence)
        self.assertIn("openie_triple", by_type)
        self.assertIn("relation_phrase", by_type)
        self.assertIn("relation_direction", by_type)
        self.assertEqual(by_type["openie_triple"][0].subject, "song SongA")
        self.assertEqual(by_type["relation_phrase"][0].relation, "has")
        self.assertTrue(by_type["openie_triple"][0].metadata["surface_relation_hint"])

    def test_run_pipeline_uses_evidence_pool_and_skips_path_scoring(self) -> None:
        question = "What is the nationality of the performer of song SongA?"
        llm = EvidencePipelineLLM()

        result = run_pipeline(
            record=QuestionRecord(question=question),
            index=1,
            mask_span_extractor=StaticMaskSpanExtractor(
                [
                    MaskSpan(
                        text="SongA",
                        start_char=question.index("SongA"),
                        end_char=question.index("SongA") + len("SongA"),
                        kind_hint="entity",
                        semantic_type_hint="Song",
                    )
                ]
            ),
            parser=StaticOpenIEParser(),
            graph_builder=GraphBuilder(),
            question_normalizer=StaticRelationCarrierGenerator(),
            path_semantic_parser=EntityPathSemanticParser(llm),
        )

        self.assertIn(SEMANTIC_REASONING_PATH_SYSTEM, llm.system_prompts)
        self.assertIn(ATOMIC_DAG_FROM_SEMANTIC_REASONING_PATH_SYSTEM, llm.system_prompts)
        self.assertTrue(all(edge.support for path in result["semantic_reasoning_paths"].paths for edge in path.edges))
        self.assertTrue(any(atom["id"] == "openie_triple_1" for atom in result["atomic_evidences"]))
        self.assertTrue(any(atom["id"] == "corenlp_path_1" for atom in result["atomic_evidences"]))
        self.assertEqual(
            [node.question for node in result["subquestion_dag"].nodes],
            [
                "Who is the performer of SongA?",
                "What is the nationality of the performer of SongA?",
            ],
        )
        self.assertIn("supported_by", result["subquestion_dag"].nodes[0].metadata)


class StaticRelationCarrierGenerator:
    def generate_relation_carrier_views(
        self,
        *,
        original_question: str,
        masked_question: str,
        placeholders: list[str],
    ) -> RelationCarrierViewResult:
        del original_question, placeholders
        return RelationCarrierViewResult(
            masked_question=masked_question,
            declarative_views=[
                DeclarativeView(id="view_1", sentence="The song SongA has a performer."),
                DeclarativeView(id="view_2", sentence="The performer has a nationality."),
            ],
            operator_intent={"type": "lookup", "target_hint": "nationality", "answer_type_hint": "Nationality", "cues": ["what nationality"]},
        )


class StaticMaskSpanExtractor:
    def __init__(self, mask_spans: list[MaskSpan]) -> None:
        self.mask_spans = mask_spans

    def extract(self, question: str) -> MaskSpanResult:
        del question
        return MaskSpanResult(mask_spans=list(self.mask_spans))


class StaticOpenIEParser:
    def annotate_views(self, views: list[dict[str, Any]], *, enable_openie: bool = True) -> list[CoreNLPViewAnnotation]:
        del views, enable_openie
        return [
            CoreNLPViewAnnotation(
                view_id="view_1",
                text="The song SongA has a performer.",
                tokens=[
                    _token(1, "The", "DT"),
                    _token(2, "song", "NN"),
                    _token(3, "SongA", "NNP"),
                    _token(4, "has", "VBZ"),
                    _token(6, "performer", "NN"),
                ],
                edges=[
                    DependencyEdge("SongA", "compound", "song", 3, 2),
                    DependencyEdge("has", "nsubj", "SongA", 4, 3),
                    DependencyEdge("has", "obj", "performer", 4, 6),
                    DependencyEdge("SongA", "dep", "performer", 3, 6),
                ],
                openie_triples=[OpenIETriple(subject="song SongA", relation="has", object="performer")],
            ),
            CoreNLPViewAnnotation(
                view_id="view_2",
                text="The performer has a nationality.",
                tokens=[
                    _token(1, "The", "DT"),
                    _token(2, "performer", "NN"),
                    _token(3, "has", "VBZ"),
                    _token(5, "nationality", "NN"),
                ],
                edges=[DependencyEdge("has", "nsubj", "performer", 3, 2), DependencyEdge("has", "obj", "nationality", 3, 5)],
                openie_triples=[OpenIETriple(subject="performer", relation="has", object="nationality")],
            ),
        ]

    def parse(self, question: str) -> Any:
        del question
        return self.annotate_views([])[0].to_dependency_parse()


class EvidencePipelineLLM:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        self.system_prompts.append(system_prompt)
        if system_prompt == SEMANTIC_REASONING_PATH_SYSTEM:
            self.assert_prompt_contains_evidence(prompt)
            return {
                "semantic_reasoning_paths": [
                    {
                        "branch_id": "b1",
                        "nodes": [
                            {"node_id": "b1_n1", "label": "SongA", "kind": "entity", "semantic_type": "Song"},
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
                                "supported_by": ["openie_triple_1", "corenlp_path_1"],
                            },
                            {
                                "edge_id": "b1_e2",
                                "source": "b1_n2",
                                "target": "b1_n3",
                                "relation": "nationality of performer",
                                "answer_type": "Nationality",
                                "is_one_hop": True,
                                "supported_by": ["openie_triple_2", "role_4"],
                            },
                        ],
                    }
                ],
                "operator_intent": {"type": "lookup", "target_hint": "nationality"},
            }
        if system_prompt == ATOMIC_DAG_FROM_SEMANTIC_REASONING_PATH_SYSTEM:
            return {
                "nodes": [
                    {
                        "node_id": "q1",
                        "question": "Who is the performer of SongA?",
                        "operation": "lookup",
                        "one_hop_relation": "performer of song",
                        "answer_type": "Person",
                        "dependencies": [],
                        "source_semantic_path_id": "b1",
                        "source_semantic_edge_id": "b1_e1",
                    },
                    {
                        "node_id": "q2",
                        "question": "What is the nationality of the performer of SongA?",
                        "operation": "lookup",
                        "one_hop_relation": "nationality of performer",
                        "answer_type": "Nationality",
                        "dependencies": ["q1"],
                        "source_semantic_path_id": "b1",
                        "source_semantic_edge_id": "b1_e2",
                    },
                ]
            }
        raise AssertionError(f"Unexpected prompt: {system_prompt}")

    def assert_prompt_contains_evidence(self, prompt: str) -> None:
        if "Atomic evidence pool:" not in prompt:
            raise AssertionError("Step 9 prompt did not include atomic evidence pool.")
        if "Selected dependency path evidence" in prompt:
            raise AssertionError("Step 9 prompt leaked selected dependency path evidence.")


def _mapping(placeholder: str, original: str, semantic_type: str) -> MaskMapping:
    return MaskMapping(
        placeholder=placeholder,
        original_text=original,
        kind_hint="entity",
        semantic_type_hint=semantic_type,
    )


def _token(index: int, word: str, pos: str) -> CoreNLPToken:
    return CoreNLPToken(index=index, word=word, pos=pos)


def _by_type(evidence: list[Any]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for item in evidence:
        result.setdefault(item.type, []).append(item)
    return result


if __name__ == "__main__":
    unittest.main()
