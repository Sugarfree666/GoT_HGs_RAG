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

from models import (  # noqa: E402
    CandidateNode,
    CandidatePath,
    CoreNLPToken,
    DependencyEdge,
    DependencyParse,
    ProblemFrame,
    Requirement,
    SelectedPath,
)
from path_ast_builder import (  # noqa: E402
    labeled_ast_from_skeleton,
    prefer_endpoint_complete_selected_paths,
    selected_paths_to_ast_skeleton,
    validate_selected_paths,
)
from path_pipeline import PathBasedSemanticParser  # noqa: E402
from path_projector import (  # noqa: E402
    build_candidate_projected_graph,
    build_undirected_dependency_graph,
    enumerate_candidate_paths,
    filter_candidate_paths,
)


class FakePathLLM:
    def __init__(self, candidate_frame_payload: dict[str, Any], desired_paths: dict[str, list[str]]) -> None:
        self.candidate_frame_payload = candidate_frame_payload
        self.desired_paths = desired_paths

    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        if "candidate node pool" in system_prompt:
            return {"candidate_nodes": self.candidate_frame_payload.get("candidate_nodes", [])}
        if "lightweight Problem Frame" in system_prompt:
            return self.candidate_frame_payload.get("problem_frame", self.candidate_frame_payload)
        if "choose exactly one" in system_prompt:
            paths = _json_after_marker(prompt, "Filtered candidate paths:")
            selected = []
            for requirement_id, desired_nodes in self.desired_paths.items():
                path_id = next(
                    path["path_id"]
                    for path in paths
                    if path["nodes"] == desired_nodes
                )
                selected.append({"requirement_id": requirement_id, "path_id": path_id})
            return {"selected_paths": selected}
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


class PathProjectionPipelineTest(unittest.TestCase):
    def test_single_chain_multihop_path_builds_ast_edges(self) -> None:
        question = "Which university did the CEO of the company that developed AlphaGo graduate from?"
        dependency_parse = _dependency_parse(
            ["AlphaGo", "company", "CEO", "university"],
            [(1, 2, "acl"), (2, 3, "nmod:of"), (3, 4, "nmod:from")],
        )
        fake_llm = FakePathLLM(
            candidate_frame_payload={
                "candidate_nodes": [
                    {"id": "AlphaGo", "text": "AlphaGo", "kind": "entity", "graph_node_ids": ["1"]},
                    {"id": "company", "text": "company", "kind": "role", "graph_node_ids": ["2"]},
                    {"id": "CEO", "text": "CEO", "kind": "role", "graph_node_ids": ["3"]},
                    {"id": "university", "text": "university", "kind": "slot", "graph_node_ids": ["4"]},
                ],
                "problem_frame": {
                    "operator": "NONE",
                    "requirements": [
                        {"id": "r1", "root": "AlphaGo", "target": "university"},
                    ],
                },
            },
            desired_paths={"r1": ["AlphaGo", "company", "CEO", "university"]},
        )

        path_parser = PathBasedSemanticParser(fake_llm)
        candidate_nodes, problem_frame, _ = path_parser.build_candidate_nodes_and_frame(question, question, [])
        dependency_graph = build_undirected_dependency_graph(dependency_parse)
        projected_graph = build_candidate_projected_graph(dependency_graph, candidate_nodes)
        enumerated_paths = enumerate_candidate_paths(projected_graph)
        filtered_paths = filter_candidate_paths(enumerated_paths, problem_frame.requirements)
        selected_paths, _ = path_parser.select_paths(question, problem_frame, filtered_paths)

        validate_selected_paths(selected_paths, problem_frame.requirements, filtered_paths)
        skeleton = selected_paths_to_ast_skeleton(
            problem_frame=problem_frame,
            selected_paths=selected_paths,
            filtered_paths=filtered_paths,
            candidate_nodes=candidate_nodes,
        )

        self.assertEqual(
            [(edge.source, edge.target) for edge in skeleton.edges],
            [("AlphaGo", "company"), ("company", "CEO"), ("CEO", "university")],
        )
        self.assertEqual(skeleton.operator.operator, "NONE")
        self.assertEqual(skeleton.operator.inputs, ["university"])

    def test_parallel_compare_paths_clone_shared_surface_nodes(self) -> None:
        candidate_nodes = [
            CandidateNode(id="MovieA", text="MovieA", kind="entity"),
            CandidateNode(id="MovieB", text="MovieB", kind="entity"),
            CandidateNode(id="film", text="film", kind="type_qualifier"),
            CandidateNode(id="director", text="director", kind="role"),
            CandidateNode(id="nationality", text="nationality", kind="slot"),
        ]
        problem_frame = ProblemFrame(
            operator="COMPARE_SAME",
            answer_mode="boolean",
            requirements=[
                Requirement(id="r1", root="MovieA", target="nationality"),
                Requirement(id="r2", root="MovieB", target="nationality"),
            ],
        )
        paths = [
            CandidatePath(
                path_id="p1",
                nodes=["MovieA", "director", "nationality"],
                node_ids=["MovieA", "director", "nationality"],
                candidate_for=["r1"],
            ),
            CandidatePath(
                path_id="p2",
                nodes=["MovieB", "director", "nationality"],
                node_ids=["MovieB", "director", "nationality"],
                candidate_for=["r2"],
            ),
        ]
        selected_paths = [
            SelectedPath(requirement_id="r1", path_id="p1"),
            SelectedPath(requirement_id="r2", path_id="p2"),
        ]

        skeleton = selected_paths_to_ast_skeleton(problem_frame, selected_paths, paths, candidate_nodes)

        self.assertEqual(
            [(edge.source, edge.target) for edge in skeleton.edges],
            [
                ("MovieA", "director_r1"),
                ("director_r1", "nationality_r1"),
                ("MovieB", "director_r2"),
                ("director_r2", "nationality_r2"),
            ],
        )
        self.assertEqual(skeleton.operator.operator, "COMPARE_SAME")
        self.assertEqual(skeleton.operator.inputs, ["nationality_r1", "nationality_r2"])

    def test_filter_removes_reverse_duplicates_and_requirement_irrelevant_paths(self) -> None:
        paths = [
            CandidatePath(
                path_id="p1",
                nodes=["MovieA", "director", "nationality"],
                node_ids=["MovieA", "director", "nationality"],
            ),
            CandidatePath(
                path_id="p2",
                nodes=["nationality", "director", "MovieA"],
                node_ids=["nationality", "director", "MovieA"],
            ),
            CandidatePath(
                path_id="p3",
                nodes=["director", "producer"],
                node_ids=["director", "producer"],
            ),
        ]
        requirements = [Requirement(id="r1", root="MovieA", target="nationality")]

        filtered = filter_candidate_paths(paths, requirements)

        self.assertEqual([path.path_id for path in filtered], ["p1"])
        self.assertEqual(filtered[0].candidate_for, ["r1"])

    def test_selection_repair_prefers_shortest_root_to_target_path(self) -> None:
        requirements = [Requirement(id="r1", root="The Good Shepherd", target="played")]
        paths = [
            CandidatePath(
                path_id="p3",
                nodes=["director", "played"],
                node_ids=["director", "played"],
                candidate_for=["r1"],
            ),
            CandidatePath(
                path_id="p9",
                nodes=["The Good Shepherd", "director", "played"],
                node_ids=["The Good Shepherd", "director", "played"],
                candidate_for=["r1"],
            ),
            CandidatePath(
                path_id="p14",
                nodes=["The Good Shepherd", "director", "played", "The Godfather"],
                node_ids=["The Good Shepherd", "director", "played", "The Godfather"],
                candidate_for=["r1"],
            ),
        ]

        selected, actions = prefer_endpoint_complete_selected_paths(
            [SelectedPath(requirement_id="r1", path_id="p3")],
            requirements,
            paths,
        )

        self.assertEqual(selected, [SelectedPath(requirement_id="r1", path_id="p9")])
        self.assertTrue(actions)

    def test_ast_skeleton_trims_selected_path_to_requirement_target(self) -> None:
        candidate_nodes = [
            CandidateNode(id="The Good Shepherd", text="The Good Shepherd", kind="entity"),
            CandidateNode(id="director", text="director", kind="role"),
            CandidateNode(id="played", text="played", kind="slot"),
            CandidateNode(id="The Godfather", text="The Godfather", kind="entity"),
        ]
        frame = ProblemFrame(
            operator="NONE",
            requirements=[Requirement(id="r1", root="The Good Shepherd", target="played")],
        )
        paths = [
            CandidatePath(
                path_id="p14",
                nodes=["The Good Shepherd", "director", "played", "The Godfather"],
                node_ids=["The Good Shepherd", "director", "played", "The Godfather"],
                candidate_for=["r1"],
            )
        ]

        skeleton = selected_paths_to_ast_skeleton(
            frame,
            [SelectedPath(requirement_id="r1", path_id="p14")],
            paths,
            candidate_nodes,
        )

        self.assertEqual(
            [(edge.source, edge.target) for edge in skeleton.edges],
            [("the_good_shepherd", "director"), ("director", "played")],
        )
        self.assertEqual(skeleton.branch_terminals["r1"], "played")

    def test_labeled_ast_preserves_terminal_requirement_context(self) -> None:
        candidate_nodes = [
            CandidateNode(id="The Good Shepherd", text="The Good Shepherd", kind="entity"),
            CandidateNode(id="director", text="director", kind="role"),
            CandidateNode(id="played", text="played", kind="slot"),
        ]
        frame = ProblemFrame(
            operator="NONE",
            requirements=[
                Requirement(
                    id="r1",
                    root="The Good Shepherd",
                    target="played",
                    description="character played by the director of The Good Shepherd in The Godfather",
                )
            ],
        )
        paths = [
            CandidatePath(
                path_id="p1",
                nodes=["The Good Shepherd", "director", "played"],
                node_ids=["The Good Shepherd", "director", "played"],
                candidate_for=["r1"],
            )
        ]
        skeleton = selected_paths_to_ast_skeleton(
            frame,
            [SelectedPath(requirement_id="r1", path_id="p1")],
            paths,
            candidate_nodes,
        )

        ast = labeled_ast_from_skeleton(
            skeleton,
            {
                "edges": [
                    {"source": "the_good_shepherd", "target": "director", "relation": "director of The Good Shepherd"},
                    {"source": "director", "target": "played", "relation": "character played by the director"},
                ],
                "operator": {"type": "NONE", "inputs": ["played"], "output": "answer"},
            },
            frame,
        )

        self.assertEqual(
            ast.edges[1].relation_hint,
            "character played by the director of The Good Shepherd in The Godfather",
        )


def _dependency_parse(words: list[str], edges: list[tuple[int, int, str]]) -> DependencyParse:
    return DependencyParse(
        tokens=[CoreNLPToken(index=index, word=word) for index, word in enumerate(words, start=1)],
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


def _json_after_marker(prompt: str, marker: str) -> Any:
    start = prompt.index(marker) + len(marker)
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(prompt[start:].lstrip())
    return value


if __name__ == "__main__":
    unittest.main()
