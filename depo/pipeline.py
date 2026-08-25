"""DEPO 的唯一算法入口。"""

from __future__ import annotations

from typing import Any

from atomic_question_dag import generate_atomic_question_dag, restore_paths
from entity_masking_preprocessor import preprocess_question
from hanlp_sdp_parser import HanLPSDPParser
from llm_client import LLMClient
from tri_sdp_reasoning_compiler import compile_token_reasoning_structure


def run_depo(question: str, parser: HanLPSDPParser, llm_client: LLMClient) -> dict[str, Any]:
    """从自然语言问题生成原子问题 DAG。"""
    #问题处理，实体识别+掩码处理
    preprocessed = preprocess_question(question, llm_client)
    #用于调用HanLP用于语义依存解析
    pas_result = parser.parse(preprocessed.masked_question)
    #构建推理结构
    masked_paths = compile_token_reasoning_structure(
        pas_result,
        list(preprocessed.mask_mapping),
    )
    #生成原子问题DAG
    dag = generate_atomic_question_dag(
        llm_client,
        question,
        preprocessed.entities,
        restore_paths(masked_paths, preprocessed.mask_mapping),
    )
    #返回识别到的实体和原子问题DAG给后续检索过程
    return {"entities": preprocessed.entities, "atomic_question_dag": dag}
