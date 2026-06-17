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
from models import (  # noqa: E402
    CoreNLPToken,
    CoreNLPViewAnnotation,
    DeclarativeView,
    DependencyEdge,
    MaskMapping,
    OpenIETriple,
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
