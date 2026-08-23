from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import QuestionRecord


if TYPE_CHECKING:
    from entity_masking_preprocessor import EntityMaskingPreprocessor
    from hanlp_sdp_parser import HanLPSDPParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", help="Process one question.")
    parser.add_argument("--questions-file", default="questions.json")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [QuestionRecord(question=args.question)] if args.question else read_questions(args.questions_file)
    return _run_hanlp_sdp_cli(args, records)

#depo的初始化，
def _run_hanlp_sdp_cli(args: argparse.Namespace, records: list[QuestionRecord]) -> int:
    #读取API,优先环境变量，否则命令行
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    from entity_masking_preprocessor import EntityMaskingPreprocessor
    from hanlp_sdp_parser import HanLPSDPParser
    from llm_client import LLMClient
    #初始化LLM
    llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
    #初始化实体处理器
    preprocessor = EntityMaskingPreprocessor(llm_client)
    #初始化HanLP
    parser = HanLPSDPParser()
    #循环处理每一个问题
    for index, record in enumerate(records, start=1):
        print(f"[run {index}/{len(records)}] {record.question}")
        #真正算法入口
        run_hanlp_sdp_pipeline(
            record=record,
            preprocessor=preprocessor,
            parser=parser,
            llm_client=llm_client,
        )
        print(f"[ok]  #{index}")
    return 0


def run_hanlp_sdp_pipeline(
    record: QuestionRecord,
    #实体预处理器，用来将实体提取出来并且替换成占位符
    preprocessor: "EntityMaskingPreprocessor",
    #HanLP语义依存解析器
    parser: "HanLPSDPParser",
    llm_client: Any,
) -> dict[str, Any]:
    #导入两个函数：1.用于生成原子问题DAG 2.恢复实体名称
    from atomic_question_dag import QuestionStructureAtomicDAGGenerator, restore_global_best_paths
    from tri_sdp_reasoning_compiler import compile_token_reasoning_structure
    #实体预处理，
    preprocess_result = preprocessor.preprocess(record.question)
    #获取实体列表
    explicit_entities = [mapping.placeholder for mapping in preprocess_result.mask_mappings]
    #语义依存解析
    hanlp_sdp_result = parser.parse(preprocess_result.masked_question)
    #SDP图转换推理结构
    token_reasoning_structure = compile_token_reasoning_structure(
        hanlp_sdp_result,
        explicit_entities,
    )
    #恢复结构中的实体
    question_structure = restore_global_best_paths(
        token_reasoning_structure.paths,
        preprocess_result.mask_mappings,
    )
    #生成原子问题DAG
    atomic_question_dag = QuestionStructureAtomicDAGGenerator(llm_client).generate(
        original_question=record.question,
        question_entities=[entity.text for entity in preprocess_result.explicit_entities.entities],
        question_structure=question_structure,
    )
    return {
        "preprocess_result": preprocess_result,
        "atomic_question_dag": atomic_question_dag,
    }


if __name__ == "__main__":
    raise SystemExit(main())
