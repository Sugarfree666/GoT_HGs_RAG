from __future__ import annotations

import json
import unittest
from pathlib import Path

from hyper_branch.config import load_config
from hyper_branch.logging_utils import TraceStore, configure_logging, create_run_dir
from hyper_branch.pipeline import HyperBranchPipeline
from tests.agriculture_fixture import ensure_agriculture_fixture


class PipelineSmokeTest(unittest.TestCase):
    def test_mock_pipeline_runs_end_to_end(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        ensure_agriculture_fixture(project_root)
        config = load_config(project_root / "configs" / "agriculture.yaml", project_root)
        config.llm.use_mock = True

        question = "How can urban farms build community support while dealing with lead contamination in soil?"
        run_dir = create_run_dir(config.runtime.base_run_dir, "test smoke run")
        logger = configure_logging(run_dir, config.runtime.log_level)
        trace_store = TraceStore(run_dir)

        pipeline = HyperBranchPipeline(config=config, run_dir=run_dir, logger=logger, trace_store=trace_store)
        result = pipeline.run(question)

        self.assertTrue(result["final_answer"]["answer"])
        self.assertTrue((run_dir / "artifacts" / "final_answer.json").exists())
        self.assertTrue((run_dir / "artifacts" / "dag_input.json").exists())
        self.assertTrue((run_dir / "artifacts" / "original_question_analysis.json").exists())
        self.assertTrue((run_dir / "artifacts" / "shared_candidate_pool.json").exists())
        self.assertFalse((run_dir / "artifacts" / "shared_candidate_pool_initial.json").exists())
        self.assertFalse((run_dir / "artifacts" / "shared_candidate_pool_final.json").exists())
        self.assertTrue((run_dir / "artifacts" / "atomic_question_analyses.json").exists())
        self.assertTrue((run_dir / "artifacts" / "atomic_retrieval.json").exists())
        self.assertTrue((run_dir / "artifacts" / "atomic_answers.json").exists())
        self.assertTrue((run_dir / "run.log").exists())

        retrieval = json.loads(
            (run_dir / "artifacts" / "atomic_retrieval.json").read_text(encoding="utf-8")
        )[0]
        self.assertIn("candidate_hyperedge_count", retrieval)
        self.assertIn("answerer_evidence", retrieval)
        self.assertNotIn("candidate_hyperedge_ids", retrieval)
        self.assertNotIn("candidate_sources", retrieval)
        self.assertNotIn("top_evidence", retrieval)
        self.assertNotIn("evidence", retrieval["atomic_answer"])

        answers = json.loads(
            (run_dir / "artifacts" / "atomic_answers.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("evidence", answers[0])

if __name__ == "__main__":
    unittest.main()
