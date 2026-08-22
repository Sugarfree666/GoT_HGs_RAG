from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hyper_branch.config import load_config
from hyper_branch.pipeline import HyperBranchPipeline
from tests.agriculture_fixture import ensure_agriculture_fixture


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_runs_end_to_end(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        ensure_agriculture_fixture(project_root)
        config = load_config(project_root / "configs" / "agriculture.yaml", project_root)

        question = "How can urban farms build community support while dealing with lead contamination in soil?"
        with (
            patch("hyper_branch.pipeline.OpenAICompatibleClient", PipelineTestClient),
            patch("hyper_branch.pipeline.OpenAIAtomicLLMService", PipelineTestLLMService),
        ):
            pipeline = HyperBranchPipeline(config, logging.getLogger("hyper_branch.test"))
            result = pipeline.run(
                question,
                {
                    "nodes": [
                        {
                            "id": "q1",
                            "question": question,
                            "depends_on": [],
                            "operation": "lookup",
                        }
                    ]
                },
            )

        self.assertEqual(
            set(result),
            {"question", "dag", "atomic_answers", "final_answer"},
        )
        self.assertTrue(result["final_answer"]["answer"])
        self.assertEqual(len(result["atomic_answers"]), 1)


class PipelineTestClient:
    def __init__(self, config: object) -> None:
        pass

    def embed_texts(self, texts: list[str], stage: str) -> list[np.ndarray]:
        vector = np.zeros(1536, dtype=np.float32)
        vector[0] = 1.0
        return [vector for _ in texts]


class PipelineTestLLMService:
    def __init__(self, client: object, prompts: object) -> None:
        pass

    def analyze_atomic_question(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, object]],
    ) -> dict[str, list[str]]:
        return {"entities": ["urban farms"]}

    def answer_atomic_question(
        self,
        atomic_question: str,
        answer_contract: dict[str, object],
        dependency_answers: list[dict[str, object]],
        evidence: dict[str, list[dict[str, object]]],
        original_question: str = "",
    ) -> dict[str, str]:
        return {"answer": "community support"}

if __name__ == "__main__":
    unittest.main()
