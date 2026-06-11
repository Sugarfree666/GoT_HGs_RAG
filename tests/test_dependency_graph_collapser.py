from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from dependency_graph_collapser import COLLAPSIBLE_RELS, collapse_dependency_graph  # noqa: E402
from entity_path_projector import build_entity_start_nodes_from_explicit_entities  # noqa: E402
from models import CoreNLPToken, DependencyEdge, DependencyParse, MaskMapping, MaskReplacement  # noqa: E402
from path_projector import build_undirected_dependency_graph  # noqa: E402


def _dependency_parse(
    tokens: list[tuple[int, str]],
    edges: list[tuple[int, int, str]],
) -> DependencyParse:
    token_by_index = {index: word for index, word in tokens}
    return DependencyParse(
        tokens=[
            CoreNLPToken(index=index, word=word, lemma=word.lower(), pos="NN")
            for index, word in tokens
        ],
        edges=[
            DependencyEdge(
                source=token_by_index[head],
                relation=relation,
                target=token_by_index[child],
                source_index=head,
                target_index=child,
            )
            for head, child, relation in edges
        ],
        raw={},
    )


def _sample_graph():
    parse = _dependency_parse(
        [
            (1, "Which"),
            (2, "OrganizationA"),
            (3, "film"),
            (4, "was"),
            (5, "produced"),
            (6, "first"),
            (8, "FilmA"),
            (10, "FilmB"),
        ],
        [
            (3, 1, "det"),
            (3, 2, "compound"),
            (5, 3, "nsubj:pass"),
            (5, 4, "aux:pass"),
            (5, 6, "advmod"),
            (5, 8, "obj"),
            (5, 10, "obj"),
        ],
    )
    return build_undirected_dependency_graph(parse)


class DependencyGraphCollapserTest(unittest.TestCase):
    def test_collapses_only_requested_relations_and_sorts_text_by_token_index(self) -> None:
        collapsed = collapse_dependency_graph(_sample_graph())

        self.assertEqual(COLLAPSIBLE_RELS, {"compound", "case", "det", "advmod"})
        self.assertNotIn("1", collapsed)
        self.assertNotIn("2", collapsed)
        self.assertNotIn("6", collapsed)
        self.assertEqual(collapsed.nodes["3"]["text"], "Which OrganizationA film")
        self.assertEqual(collapsed.nodes["5"]["text"], "produced first")

        self.assertTrue(collapsed.has_edge("5", "3"))
        self.assertTrue(collapsed.has_edge("5", "4"))
        self.assertTrue(collapsed.has_edge("5", "8"))
        self.assertTrue(collapsed.has_edge("5", "10"))
        self.assertEqual(collapsed.edges["5", "3"]["relations"], ["nsubj:pass"])
        self.assertEqual(collapsed.edges["5", "4"]["relations"], ["aux:pass"])
        self.assertEqual(collapsed.edges["5", "8"]["relations"], ["obj"])
        self.assertEqual(collapsed.edges["5", "10"]["relations"], ["obj"])

        remaining_relations = {
            relation
            for _source, _target, attrs in collapsed.edges(data=True)
            for relation in attrs.get("relations", [])
        }
        self.assertFalse(remaining_relations & COLLAPSIBLE_RELS)

    def test_collapsed_placeholder_can_still_be_entity_start(self) -> None:
        collapsed = collapse_dependency_graph(_sample_graph())
        replacement = MaskReplacement(
            masked_question="Which OrganizationA film was produced first, FilmA or FilmB?",
            mask_mappings=[
                MaskMapping(
                    placeholder="OrganizationA",
                    original_text="Walt Disney",
                    kind_hint="entity",
                    semantic_type_hint="organization",
                    token_indices=[2],
                )
            ],
        )

        starts = build_entity_start_nodes_from_explicit_entities(
            dependency_graph=collapsed,
            restored_graph_node_candidates=[],
            replacement=replacement,
        )

        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].text, "Walt Disney")
        self.assertEqual(starts[0].graph_node_ids, ["3"])
        self.assertIn(2, starts[0].token_ids)


if __name__ == "__main__":
    unittest.main()
