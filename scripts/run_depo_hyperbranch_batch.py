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
from hyper_branch.client import OpenAIClient


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
    from pipeline import run_depo
    #创建llm客户端
    #读取yaml配置文件，转成字典格式
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8")
    )
    llm = OpenAIClient(
        api_key=api_key,
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=base_url,
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
        client=llm,
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
        try:
            decomposition = run_depo(question, sdp_parser, llm)
            result = hyperbranch.run(
                question,
                decomposition["atomic_question_dag"],
                decomposition["entities"],
            )
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(
                    {
                        "topic_entities": decomposition["entities"],
                        "nodes": [
                            {
                                "id": node["node_id"],
                                "rewritten_question": node["question"],
                                "entities": node["entities"],
                                "evidence_blocks": node["evidence_blocks"],
                                "answer": node["answer"],
                            }
                            for node in result["atomic_answers"]
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{args.dataset} #{index}: {result['final_answer']['answer']}")
        except Exception as exc:
            print(f"{args.dataset} #{index} failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
