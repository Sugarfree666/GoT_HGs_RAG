from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from hyper_branch.pipeline import HyperBranchPipeline


class AtomicPipelineTest(unittest.TestCase):
    def test_pipeline_rewrites_dependencies_and_merges_all_ancestor_pools(self) -> None:
        database = RecordingDatabase(
            pools=[
                {"H-original": set()},
                {"H-q1": set()},
                {"H-q2": {"H-q1"}},
                {"H-q3": {"H-q2"}},
            ]
        )
        client = RecordingClient(["B", "C", "Done"])
        dag = {
            "nodes": [
                {"id": "q3", "question": "What follows q2's answer?", "depends_on": ["q2"]},
                {"id": "q1", "question": "Who is linked to A?", "depends_on": []},
                {"id": "q2", "question": "Where was q1's answer recorded?", "depends_on": ["q1"]},
            ]
        }

        with (
            patch("hyper_branch.pipeline.HypergraphDatabase", return_value=database),
            patch("hyper_branch.pipeline.OpenAIClient", return_value=client),
        ):
            pipeline = HyperBranchPipeline(
                Path("unused"),
                top_k=10,
                model="test-model",
                embedding_model="test-embedding-model",
                timeout_seconds=120,
                temperature=0.2,
                api_key="test-key",
            )
            result = pipeline.run("What eventually follows A?", dag, ["A"])

        self.assertEqual(
            [item["question"] for item in result["atomic_answers"]],
            ["Who is linked to A?", "Where was B recorded?", "What follows C?"],
        )
        self.assertEqual(result["final_answer"], {"answer": "Done"})
        self.assertEqual(database.anchor_calls, [["A"], ["A"], ["B"], ["C"]])
        self.assertEqual(
            set(database.rank_calls[-1]["candidates"]),
            {"H-original", "H-q1", "H-q2", "H-q3"},
        )
        self.assertEqual(client.chat_calls[1]["dependency_context"][0]["answer"], "B")

class RecordingDatabase:
    def __init__(self, pools: list[dict[str, set[str]]]) -> None:
        self.pools = pools
        self.anchor_calls: list[list[str]] = []
        self.rank_calls: list[dict[str, Any]] = []

    def candidate_pool(self, mentions: list[str], embedder: object) -> dict[str, set[str]]:
        self.anchor_calls.append(list(mentions))
        return self.pools.pop(0)

    def rank(
        self,
        question: str,
        candidates: dict[str, set[str]],
        embedder: object,
        top_k: int,
    ) -> list[dict[str, Any]]:
        self.rank_calls.append({"question": question, "candidates": candidates, "top_k": top_k})
        return [
            {
                "id": hyperedge_id,
                "text": hyperedge_id,
                "chunks": [(hyperedge_id, f"Evidence for {hyperedge_id}")],
                "first_hop_texts": sorted(candidates[hyperedge_id]),
            }
            for hyperedge_id in list(candidates)[:top_k]
        ]


class RecordingClient:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.chat_calls: list[dict[str, Any]] = []

    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(3, dtype=np.float32)

    def chat_json(self, _system_prompt: str, user_prompt: str, *, max_tokens: int) -> dict[str, str]:
        self.chat_calls.append(json.loads(user_prompt))
        self.assert_max_tokens(max_tokens)
        return {"answer": self.answers.pop(0)}

    @staticmethod
    def assert_max_tokens(max_tokens: int) -> None:
        assert max_tokens == 900
if __name__ == "__main__":
    unittest.main()
