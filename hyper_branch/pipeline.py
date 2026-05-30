from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .atomic import (
    AtomicDagExecutor,
    AtomicEvidenceFusion,
    AtomicHyperedgeRetriever,
    AtomicQuestionAnalyzer,
    FinalAnswerComposer,
)
from .config import Config
from .data.loaders import HypergraphDatasetLoader
from .llm import LocalHashEmbeddingClient, MockAtomicLLMService, OpenAIAtomicLLMService, OpenAICompatibleClient, PromptManager
from .logging_utils import TraceStore


class HyperBranchPipeline:
    def __init__(
        self,
        config: Config,
        run_dir: Path,
        logger: logging.Logger,
        trace_store: TraceStore,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.logger = logger
        self.trace_store = trace_store

        loader = HypergraphDatasetLoader(config.dataset, logger)
        self.dataset = loader.load()
        self.trace_store.save_artifact("dataset_summary.json", self.dataset.summary)

        prompts = PromptManager(config.prompts.directory)
        if config.llm.use_mock:
            self.embedder = LocalHashEmbeddingClient()
            self.llm_service = MockAtomicLLMService()
        else:
            client = OpenAICompatibleClient(config.llm, trace_store=trace_store)
            self.embedder = client
            self.llm_service = OpenAIAtomicLLMService(client=client, prompts=prompts)

        analyzer = AtomicQuestionAnalyzer(llm_service=self.llm_service)
        retriever = AtomicHyperedgeRetriever(
            dataset=self.dataset,
            embedder=self.embedder,
            config=config.retrieval,
            llm_service=self.llm_service,
            logger=logger,
        )
        fusion = AtomicEvidenceFusion(
            config=config.retrieval,
            embedder=self.embedder,
            hyperedge_store=self.dataset.hyperedge_store,
            chunk_store=self.dataset.chunk_store,
        )
        composer = FinalAnswerComposer(llm_service=self.llm_service)
        self.executor = AtomicDagExecutor(
            analyzer=analyzer,
            retriever=retriever,
            fusion=fusion,
            composer=composer,
            llm_service=self.llm_service,
            logger=logger,
        )

    def run(self, question: str, dag_payload: Any | None = None) -> dict[str, Any]:
        self.logger.info("Starting HyperBranch pipeline for question: %s", question)
        if dag_payload is None:
            self.logger.info(
                "No DAG payload supplied; treating the question as a single atomic node for compatibility."
            )
        result = self.executor.run(original_question=question, dag_payload=dag_payload)
        artifacts = result.artifacts
        self.trace_store.save_artifact("artifacts/dag_input.json", artifacts["dag_input"])
        self.trace_store.save_artifact("artifacts/atomic_question_analyses.json", artifacts["atomic_question_analyses"])
        self.trace_store.save_artifact("artifacts/atomic_retrieval.json", artifacts["atomic_retrieval"])
        self.trace_store.save_artifact("artifacts/atomic_answers.json", artifacts["atomic_answers"])
        self.trace_store.save_artifact("artifacts/final_answer.json", result.final_answer)

        payload = result.to_dict()
        payload["run_dir"] = str(self.run_dir)
        self.logger.info("Pipeline finished. Artifacts saved under %s", self.run_dir)
        return payload
