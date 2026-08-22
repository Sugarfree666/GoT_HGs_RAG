"""Execute one DEPO DAG with a shared HyperBranch dataset snapshot."""

from __future__ import annotations

import logging
from typing import Any

from .atomic import AtomicDagExecutor, AtomicHyperedgeRetriever, AtomicQuestionAnalyzer
from .config import Config
from .data.loaders import HypergraphDatasetLoader
from .llm import (
    OpenAIAtomicLLMService,
    OpenAICompatibleClient,
    PromptManager,
)


class HyperBranchPipeline:
    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.logger = logger
        self.dataset = HypergraphDatasetLoader(config.dataset, logger).load()
        client = OpenAICompatibleClient(config.llm)
        self.embedder = client
        self.llm_service = OpenAIAtomicLLMService(client, PromptManager(config.prompts.directory))
        retriever = AtomicHyperedgeRetriever(self.dataset, self.embedder, config.retrieval)
        self.executor = AtomicDagExecutor(AtomicQuestionAnalyzer(self.llm_service), retriever, self.llm_service)

    def run(
        self,
        question: str,
        dag: dict[str, Any],
        original_question_entities: list[str] | None = None,
    ) -> dict[str, Any]:
        result = self.executor.run(question, dag, original_question_entities)
        return {
            "question": question,
            "dag": dag,
            "atomic_answers": [item.to_dict() for item in result.atomic_results],
            "final_answer": result.final_answer,
        }
