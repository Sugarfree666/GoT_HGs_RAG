from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from models import SemanticASTEdge, SemanticASTNode, SemanticASTPrimaryOperator, SemanticASTResult
from ast_validator import repair_missing_value_slots, validate_ast_completeness
from subquestion_generator import SubquestionGenerator
from surface_validator import contains_bare_variable, validate_atomic_dag_surface
from hyper_branch.atomic import AtomicDagExecutor


class FakeSurfaceLLM:
    def chat_json(self, system_prompt: str, prompt: str) -> dict[str, str]:
        del system_prompt
        if '"step_id": "q3"' in prompt and "ARGMIN" in prompt:
            return {"question": "Which of X1 and X2 was released first?"}
        if '"step_id": "q1"' in prompt and "release date" in prompt:
            return {"question": "What is the release date of Aas Ka Panchhi?"}
        if '"step_id": "q2"' in prompt and "release date" in prompt:
            return {"question": "What is the release date of Phoolwari?"}
        if '"step_id": "q1"' in prompt:
            return {"question": "Who is the director of Wrong Turn 5: Bloodlines?"}
        if '"step_id": "q2"' in prompt:
            return {"question": "What is the nationality of X1?"}
        return {"question": "What is the requested fact?"}


class DepoSurfaceRealizationTest(unittest.TestCase):
    def test_bridge_node_does_not_surface_intermediate_variable(self) -> None:
        ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(operator="NONE"),
            nodes=[
                SemanticASTNode(id="film", label="Wrong Turn 5: Bloodlines", kind="entity"),
                SemanticASTNode(id="director", label="director", kind="type_variable"),
                SemanticASTNode(id="nationality", label="nationality", kind="type_variable"),
            ],
            edges=[
                SemanticASTEdge(source="film", target="director", relation_hint="directed by"),
                SemanticASTEdge(source="director", target="nationality", relation_hint="nationality"),
            ],
        )

        dag = SubquestionGenerator(FakeSurfaceLLM()).generate_dag(
            "What is the nationality of the director of Wrong Turn 5: Bloodlines?",
            ast,
        )
        payload = dag.to_dict()

        self.assertEqual(validate_atomic_dag_surface(payload), [])
        questions = [node["question"] for node in payload["nodes"]]
        self.assertFalse(any(contains_bare_variable(question) for question in questions))
        self.assertNotIn("What is the nationality of X1?", questions)
        self.assertEqual(
            questions[1],
            "What is the nationality of the director of Wrong Turn 5: Bloodlines?",
        )
        self.assertEqual(payload["nodes"][1]["dependencies"], ["q1"])
        normalized = AtomicDagExecutor.normalize_dag_payload(payload)
        self.assertEqual([node.node_id for node in AtomicDagExecutor.topological_sort(normalized)], ["q1", "q2"])

    def test_selection_node_uses_candidate_metadata_not_variable_contract(self) -> None:
        ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(
                operator="ARGMIN",
                inputs=["date_a", "date_b"],
                output="FINAL",
                cue_text="released first",
            ),
            nodes=[
                SemanticASTNode(id="film_a", label="Aas Ka Panchhi", kind="entity"),
                SemanticASTNode(id="date_a", label="release date", kind="type_variable"),
                SemanticASTNode(id="film_b", label="Phoolwari", kind="entity"),
                SemanticASTNode(id="date_b", label="release date", kind="type_variable"),
            ],
            edges=[
                SemanticASTEdge(source="film_a", target="date_a", relation_hint="release date"),
                SemanticASTEdge(source="film_b", target="date_b", relation_hint="release date"),
            ],
        )

        dag = SubquestionGenerator(FakeSurfaceLLM()).generate_dag(
            "Which film was released first, Aas Ka Panchhi or Phoolwari?",
            ast,
        )
        payload = dag.to_dict()

        self.assertEqual(validate_atomic_dag_surface(payload), [])
        self.assertEqual([node["dependencies"] for node in payload["nodes"]], [[], [], ["q1", "q2"]])
        self.assertFalse(any(contains_bare_variable(node["question"]) for node in payload["nodes"]))
        self.assertNotIn("Which of X1 and X2 was released first?", [node["question"] for node in payload["nodes"]])

        final_node = payload["nodes"][2]
        self.assertEqual(final_node["question"], "Which film was released first, Aas Ka Panchhi or Phoolwari?")
        self.assertEqual(final_node["metadata"]["operator"], "ARGMIN")
        self.assertEqual(
            final_node["metadata"]["candidates"],
            [
                {"label": "Aas Ka Panchhi", "source_node_id": "q1"},
                {"label": "Phoolwari", "source_node_id": "q2"},
            ],
        )
        normalized = AtomicDagExecutor.normalize_dag_payload(payload)
        self.assertEqual([node.node_id for node in AtomicDagExecutor.topological_sort(normalized)], ["q1", "q2", "q3"])

    def test_compare_operator_with_single_chain_gets_none_feedback(self) -> None:
        ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(
                operator="COMPARE_DIFF",
                inputs=["death_date"],
                output="answer",
                cue_text="date of death",
            ),
            nodes=[
                SemanticASTNode(id="film", label="Madame La Presidente", kind="entity"),
                SemanticASTNode(id="director", label="director", kind="type_variable"),
                SemanticASTNode(id="death_date", label="date of death", kind="type_variable"),
                SemanticASTNode(id="COMPARE_DIFF", label="COMPARE_DIFF", kind="operator"),
            ],
            edges=[
                SemanticASTEdge(source="film", target="director", relation_hint="director of film"),
                SemanticASTEdge(source="director", target="death_date", relation_hint="date of death"),
                SemanticASTEdge(source="death_date", target="COMPARE_DIFF", edge_type="operator", relation_hint="COMPARE_DIFF"),
            ],
        )

        warnings = validate_ast_completeness(
            "What is the date of death of the director of film Madame La Presidente?",
            ast,
        )

        self.assertTrue(any("requires at least 2" in warning for warning in warnings))
        self.assertTrue(any("single-chain" in warning for warning in warnings))

    def test_single_chain_compare_operator_is_demoted_before_dag_generation(self) -> None:
        ast = SemanticASTResult(
            status="ok",
            primary_operator=SemanticASTPrimaryOperator(
                operator="COMPARE_DIFF",
                inputs=["death_date"],
                output="answer",
                cue_text="date of death",
            ),
            nodes=[
                SemanticASTNode(id="film", label="Madame La Presidente", kind="entity"),
                SemanticASTNode(id="director", label="director", kind="type_variable"),
                SemanticASTNode(id="death_date", label="date of death", kind="type_variable"),
                SemanticASTNode(id="COMPARE_DIFF", label="COMPARE_DIFF", kind="operator"),
            ],
            edges=[
                SemanticASTEdge(source="film", target="director", relation_hint="director of film"),
                SemanticASTEdge(source="director", target="death_date", relation_hint="date of death"),
                SemanticASTEdge(source="death_date", target="COMPARE_DIFF", edge_type="operator", relation_hint="COMPARE_DIFF"),
            ],
        )

        repaired = repair_missing_value_slots(
            "What is the date of death of the director of film Madame La Presidente?",
            ast,
        )
        self.assertEqual(repaired.primary_operator.operator, "NONE")

        dag = SubquestionGenerator(FakeSurfaceLLM()).generate_dag(
            "What is the date of death of the director of film Madame La Presidente?",
            ast,
        )
        payload = dag.to_dict()

        self.assertEqual(validate_atomic_dag_surface(payload), [])
        self.assertEqual(len(payload["nodes"]), 2)
        self.assertFalse(any(node.get("metadata", {}).get("operator") for node in payload["nodes"]))
        self.assertEqual(payload["nodes"][1]["dependencies"], ["q1"])


if __name__ == "__main__":
    unittest.main()
