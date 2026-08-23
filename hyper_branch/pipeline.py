"""Execute one DEPO DAG with a shared HyperBranch dataset snapshot."""

from __future__ import annotations

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
    def __init__(self, config: Config) -> None:
        #加载超图数据库
        self.dataset = HypergraphDatasetLoader(config.dataset).load()
        #创建LLM客户端
        client = OpenAICompatibleClient(config.llm)
        #保存嵌入服务
        self.embedder = client
        #用于回答原子问题
        self.llm_service = OpenAIAtomicLLMService(client, PromptManager(config.prompts.directory))
        #创建超边检索器
        retriever = AtomicHyperedgeRetriever(self.dataset, self.embedder, config.retrieval)
        #创建原子问题DAG执行器，AtomicQuestionAnalyzer解析问题，retriever检索证据，self.llm_service用于回答问题
        self.executor = AtomicDagExecutor(AtomicQuestionAnalyzer(self.llm_service), retriever, self.llm_service)

    def run(
        self,
        question: str,
        dag: dict[str, Any],
        original_question_entities: list[str],
    ) -> dict[str, Any]:
        #进入推理
        result = self.executor.run(question, dag, original_question_entities)
        #生成最终结果
        return {
            "question": question,
            "dag": dag,
            "atomic_answers": [item.to_dict() for item in result.atomic_results],
            "final_answer": result.final_answer,
        }
