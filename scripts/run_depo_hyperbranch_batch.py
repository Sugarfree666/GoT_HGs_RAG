from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

#项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#加入paython的搜索路径
sys.path[:0] = [str(PROJECT_ROOT / "depo"), str(PROJECT_ROOT)]

from hyper_branch.pipeline import HyperBranchPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DEPO and HyperBranch for one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    api_key = args.api_key or os.environ["OPENAI_API_KEY"]
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    #读取指定数据集的问题文件，然后根据参数截取运行的问题数量
    questions = json.loads(
        (PROJECT_ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8")
    )[args.start - 1 : args.end]
    #需要运行的题目数量
    if args.limit is not None:
        questions = questions[: args.limit]

    from hanlp_sdp_parser import HanLPSDPParser
    from llm_client import LLMClient
    from pipeline import run_depo
    #创建llm客户端
    llm = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
    #读取yaml配置文件，转成字典格式
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8")
    )

    #创建一个检索器对象，并完成初始化
    hyperbranch = HyperBranchPipeline(
        #超图数据库路径
        PROJECT_ROOT / config["dataset_root"],
        top_k=config["top_k"],
        model=args.llm_model,
        #嵌入模型
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        api_key=api_key,
        base_url=base_url,
    )
    #创建解析器对象
    sdp_parser = HanLPSDPParser()
    #输出目录路径
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / run_id
    #读取问题列表
    for offset, item in enumerate(questions, start=1):
        #计算真实问题编号
        index = args.start + offset - 1
        #获取问题文本，
        question = item["question"].strip()
        #输出文件位置
        result_file = output_dir / f"{index:05d}" / "result.json"
        #判断是否运行过，断点续跑
        if args.resume and result_file.exists():
            continue
        #跑问题分解算法，返回实体列表和原子问题DAG
        decomposition = run_depo(question, sdp_parser, llm)
        #跑当前原子问题DAG的检索和回答
        result = hyperbranch.run(
            question,
            decomposition["atomic_question_dag"],
            decomposition["entities"],
        )
        #创建结果目录
        result_file.parent.mkdir(parents=True, exist_ok=True)
        #把结果写入result.json
        result_file.write_text(
            json.dumps(
                {
                    "question": question,
                    "gold_answer": item["answer"],
                    "dag": result["dag"],
                    "atomic_answers": result["atomic_answers"],
                    "final_answer": result["final_answer"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            #指定编码
            encoding="utf-8",
        )
        print(f"{args.dataset} #{index}: {result['final_answer']['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
