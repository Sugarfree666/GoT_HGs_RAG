from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hyper_branch.pipeline import HyperBranchPipeline
from tests.agriculture_fixture import ensure_agriculture_fixture


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_runs_end_to_end(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        ensure_agriculture_fixture(project_root)
        question = "How can urban farms build community support while dealing with lead contamination in soil?"
        with patch("hyper_branch.pipeline.OpenAIClient", PipelineTestClient):
            pipeline = HyperBranchPipeline(
                project_root / "datasets" / "agriculture",
                model="test-model",
                embedding_model="test-embedding-model",
                timeout_seconds=120,
                temperature=0.2,
                api_key="test-key",
            )
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
                ["urban farms"],
            )

        self.assertEqual(
            set(result),
            {"question", "dag", "topic_entity_ids", "atomic_answers", "final_answer"},
        )
        self.assertTrue(result["final_answer"]["answer"])
        self.assertEqual(len(result["atomic_answers"]), 1)


class PipelineTestClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(1536, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def chat_json(
        self,
        _system_prompt: str,
        _user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        if max_tokens is None:
            return {"entities": ["urban farms", "community support"]}
        assert max_tokens == 900
        return {"answer": "community support"}


if __name__ == "__main__":
    unittest.main()
