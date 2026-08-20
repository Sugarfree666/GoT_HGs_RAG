from __future__ import annotations

# main.py 是 DEPO 分解流程的命令行入口：
# 原问题 -> 实体识别/掩码 -> HanLP PAS 语义依存解析
# -> Step4 推理路径构建 -> Step5 原子问题 DAG 生成 -> 结果打印

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Any

# 项目内部工具：读取问题文件，以及各阶段使用的数据结构
from io_utils import read_questions
from models import (
    HanLPSDPPreprocessResult,
    HanLPSDPResult,
    ExplicitEntityResult,
    QuestionRecord,
)

# 仅用于类型检查，程序真正运行时不会在这里导入这些较重的模块
if TYPE_CHECKING:
    from hanlp_sdp_parser import HanLPSDPParser
    from entity_masking_preprocessor import EntityMaskingPreprocessor


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="DEPO HanLP-SDP parsing with explicit entity masking and token reasoning output."
    )
    # 直接处理单个问题；若未提供，则从 questions.json 读取
    parser.add_argument(
        "--question",
        help="Run one manually supplied question instead of questions.json.",
    )
    parser.add_argument(
        "--questions-file", default="questions.json", help="Path to questions.json."
    )
    # OpenAI 配置：优先读取环境变量，其次使用命令行参数
    parser.add_argument(
        "--api-key", help="OpenAI API key. Used only if OPENAI_API_KEY is not set."
    )
    parser.add_argument(
        "--base-url", help="OpenAI base URL. Used only if OPENAI_BASE_URL is not set."
    )
    # 可指定 HanLP 模型；不指定时由 HanLPSDPParser 使用默认配置
    parser.add_argument(
        "--hanlp-model",
        help="HanLP pretrained constant name from hanlp.pretrained.mtl/sdp, or a local model path.",
    )
    # debug 模式会打印/保存更多中间结构，便于排查 Step3/Step4
    parser.add_argument(
        "--debug", action="store_true", help="Print detailed intermediate structures."
    )
    parser.add_argument(
        "--debug-dir",
        default="debug/hanlp_sdp",
        help="Directory for HanLP PAS debug JSON files when --debug is enabled.",
    )
    # 调试 Step4 时可跳过 Step5 的 LLM 原子问题 DAG 生成
    parser.add_argument(
        "--skip-step5", action="store_true", help="Skip Step5 DAG generation."
    )
    parser.add_argument(
        "--run-step5",
        action="store_true",
        help="Compatibility flag; Step5 runs by default.",
    )
    return parser.parse_args()


def main() -> int:
    """程序入口：读取问题，然后启动完整 DEPO 流程。"""
    args = parse_args()

    # 将单个问题或问题文件统一整理为 QuestionRecord 列表
    records = (
        [QuestionRecord(question=args.question)]
        if args.question
        else read_questions(args.questions_file)
    )
    return _run_hanlp_sdp_cli(args, records)


def _run_hanlp_sdp_cli(args: argparse.Namespace, records: list[QuestionRecord]) -> int:
    """初始化模型/客户端，并逐题运行 DEPO 管线。"""

    # 环境变量优先，避免每次运行都在命令行明文传 API Key
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print(
            "This HanLP SDP branch requires LLM calls for explicit entity masking and Step5 DAG generation.",
            file=sys.stderr,
        )
        print(
            "Set OPENAI_API_KEY or pass --api-key. Use --skip-step5 only when debugging Step4.",
            file=sys.stderr,
        )
        return 2

    try:
        # 延迟导入：只有真正运行流程时才加载 HanLP 和 LLM 相关模块
        from hanlp_sdp_parser import HanLPSDPParser
        from entity_masking_preprocessor import EntityMaskingPreprocessor
        from llm_client import LLMClient

        # 同一个 LLM 客户端同时服务实体预处理和 Step5
        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")

        # Step1/2：显式实体识别、问题规范化和实体掩码
        preprocessor = EntityMaskingPreprocessor(llm_client)

        # Step3：HanLP SDP（当前主要使用 PAS）解析器
        parser = HanLPSDPParser(args.hanlp_model)

        print("If this is the first run, HanLP may download the model automatically.")
        print("You can set HANLP_HOME to control the cache directory.")
        print()

        # 逐个问题执行完整流程，并将中间结果和最终 DAG 打印出来
        for index, record in enumerate(records, start=1):
            result = run_hanlp_sdp_pipeline(
                record=record,
                index=index,
                preprocessor=preprocessor,
                parser=parser,
                debug=args.debug,
                debug_dir=args.debug_dir,
                llm_client=llm_client,
                skip_step5=args.skip_step5,
                run_step5=args.run_step5 or not args.skip_step5,
            )
            print_hanlp_sdp_result(index, record, result, debug=args.debug)
    #异常处理，如果缺少python包，就提示安装
    except ModuleNotFoundError as exc:
        if "hanlp" in str(exc).lower() or getattr(exc, "name", "") == "hanlp":
            print("Missing dependency: hanlp", file=sys.stderr)
            print("Run: pip install hanlp", file=sys.stderr)
            return 2
        print(
            f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def run_hanlp_sdp_pipeline(
    record: QuestionRecord,
    index: int,
    preprocessor: "EntityMaskingPreprocessor",
    parser: "HanLPSDPParser",
    debug: bool = False,
    debug_dir: str | None = None,
    llm_client: Any | None = None,
    skip_step5: bool = False,
    run_step5: bool = True,
) -> dict[str, Any]:
    """
    对一个问题运行完整 DEPO 流程。

    返回值保留各阶段中间结果，既方便终端打印，也方便其他模块复用。
    """
    from tri_sdp_reasoning_compiler import compile_token_reasoning_structure

    # ---------- Step1/2：实体识别、问题规范化与实体掩码 ----------
    preprocess_result = preprocessor.preprocess(record.question)
    #取得处理后的句子
    hanlp_input_sentence = preprocess_result.sdp_input_sentence
    # 取得实体占位符，ENTITYA/ENTITYB等等
    explicit_entities = [
        mapping.placeholder for mapping in preprocess_result.mask_mappings
    ]

    # ---------- Step3：HanLP SDP 语义依存解析 ----------
    hanlp_sdp_result = parser.parse(
        hanlp_input_sentence,
        placeholders=explicit_entities,
    )
    # ---------- Step4：修复语义图并选择问题结构/推理路径 ----------
    # 该函数会根据 PAS 图和实体位置构建 token reasoning structure，
    # 最终的最佳分支路径保存在 token_reasoning_structure.paths 中。
    token_reasoning_structure = compile_token_reasoning_structure(
        hanlp_sdp_result,
        explicit_entities=explicit_entities,
        masked_question=preprocess_result.masked_question,
        original_question=preprocess_result.original_question,
        normalized_question=preprocess_result.normalized_question
        or preprocess_result.original_question,
        normalization_changed=preprocess_result.normalization_changed,
        normalization_note=preprocess_result.normalization_note,
        question_id=record.qid or f"q{index}",
        debug=debug,
        debug_dir=debug_dir,
    )
    # ---------- Step5：根据原问题 + question_structure 生成原子问题 DAG ----------
    atomic_question_dag = None
    if run_step5 and not skip_step5:
        from atomic_question_dag import (
            QuestionStructureAtomicDAGGenerator,
            invalid_atomic_question_dag,
            restore_global_best_paths,
        )

        # 优先使用外部传入的 LLM；没有时尝试从预处理器内部复用
        step5_llm = llm_client or _llm_client_from_preprocessor(preprocessor)
        if step5_llm is None:
            atomic_question_dag = invalid_atomic_question_dag(
                ["Step5 requires an LLM client."]
            )
        else:
            try:
                # Step4 路径中仍是 ENTITYA 等占位符，
                # 这里恢复为真实实体文本，作为 Step5 的 question_structure。
                question_structure = restore_global_best_paths(
                    token_reasoning_structure.paths,
                    preprocess_result.mask_mappings,
                )
            except ValueError as exc:
                atomic_question_dag = invalid_atomic_question_dag([str(exc)])
            else:
                # 将原问题、显式实体和问题结构交给 LLM，生成原子问题 DAG
                atomic_question_dag = QuestionStructureAtomicDAGGenerator(
                    step5_llm
                ).generate(
                    original_question=record.question,
                    question_entities=[
                        entity.text
                        for entity in preprocess_result.explicit_entities.entities
                    ],
                    question_structure=question_structure,
                )
    # 统一返回所有关键中间结果，方便调试和下游模块直接读取
    return {
        "preprocess_result": preprocess_result,
        "explicit_entities": preprocess_result.explicit_entities,
        "explicit_entity_payload": preprocess_result.explicit_entities.raw_payload,
        "original_question": preprocess_result.original_question,
        "normalized_question": preprocess_result.normalized_question,
        "normalization_changed": preprocess_result.normalization_changed,
        "normalization_note": preprocess_result.normalization_note,
        "masked_question": preprocess_result.masked_question,
        "sdp_input_sentence": preprocess_result.sdp_input_sentence,
        "hanlp_input_sentence": hanlp_input_sentence,
        "entity_mask_mappings": preprocess_result.mask_mappings,
        "hanlp_sdp_result": hanlp_sdp_result,
        "token_reasoning_structure": token_reasoning_structure,
        "atomic_question_dag": atomic_question_dag,
    }


def run_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """兼容旧代码的别名，实际仍调用 run_hanlp_sdp_pipeline。"""
    return run_hanlp_sdp_pipeline(*args, **kwargs)


def _llm_client_from_preprocessor(
    preprocessor: "EntityMaskingPreprocessor",
) -> Any | None:
    """尝试从预处理器内部取出已经创建好的 LLMClient。复用GPT"""
    #尝试取出preprocessor.explicit_extractor，如果没有就返回None
    extractor = getattr(preprocessor, "explicit_extractor", None)
    return getattr(extractor, "llm_client", None)


def print_hanlp_sdp_result(
    index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False
) -> None:
    """按 Step1~Step5 的顺序，把一个问题的处理结果打印到终端。"""
    preprocess_result: HanLPSDPPreprocessResult = result["preprocess_result"]
    explicit_entities: ExplicitEntityResult = preprocess_result.explicit_entities
    hanlp_result: HanLPSDPResult = result["hanlp_sdp_result"]
    token_reasoning_structure = result["token_reasoning_structure"]

    separator = "=" * 60
    print(separator)
    title = f"Question {index}"
    if record.qid:
        title += f" ({record.qid})"
    print(title)
    print(separator)
    print()

    # 原始输入问题
    print("[Original Question]")
    print(record.question)
    print()

    # Step1：LLM 识别出的显式实体
    print("[1. Explicit Entities]")
    if explicit_entities.entities:
        for entity in explicit_entities.entities: 
            print(f" - {entity.text}")
    else:
        print(" (none)")
    print()

    # Step2：实体 -> ENTITYA/ENTITYB 映射，以及规范化/掩码后的问题
    print("[2. Entity Masking]")
    if preprocess_result.mask_mappings:
        for mapping in preprocess_result.mask_mappings:
            print(f" - {mapping.placeholder} -> {mapping.original_text}")
    else:
        print(" (none)")
    if (
        preprocess_result.normalized_question
        and preprocess_result.normalized_question != record.question
    ):
        print(f"Normalized question: {preprocess_result.normalized_question}")
    else:
        print("Normalized question: unchanged")
    print(f"Masked question: {preprocess_result.masked_question}")
    print()

    # Step3：显示 HanLP 使用的模型和实际送入 HanLP 的句子
    print("[3. HanLP SDP Parsing]")
    print(f" Model: {hanlp_result.model or '(unknown)'}")
    print(
        f" HanLP input sentence: {result.get('hanlp_input_sentence') or preprocess_result.masked_question}"
    )
    print()

    # 检查 ENTITYA 等占位符经过 HanLP 分词后是否仍保持完整
    print("[Mask Token Check]")
    if hanlp_result.mask_token_checks:
        for placeholder, status in hanlp_result.mask_token_checks.items():
            print(f"{placeholder}: {status}")
    else:
        print("(none)")
    print()

    # HanLP 原始语义依存边
    print("[Raw SDP Edges]")
    _print_all_hanlp_sdp_edges(hanlp_result)
    print()

    # Step4：修复后的图，以及每个实体分支最终选择的最佳路径
    print("[4. Token Reasoning Structure]")
    print("[Repaired Evidence Graph]")
    _print_repaired_evidence_graph(token_reasoning_structure)
    print()

    # paths 是 Step4 最终输出的 question_structure 基础
    selected_paths = getattr(token_reasoning_structure, "paths", []) or []
    global_selection = getattr(token_reasoning_structure, "global_selection", {}) or {}
    path_type = getattr(token_reasoning_structure, "path_type", "")
    if path_type == "entity_branch_best_paths":
        print(
            "[Entity Branch Best Paths]"
            if len(selected_paths) > 1
            else "[Entity Branch Best Path]"
        )
        selected_payloads = global_selection.get("paths") or []
        for path_index, path in enumerate(selected_paths, start=1):
            nodes = list(getattr(path, "nodes", []))
            selected_payload = (
                selected_payloads[path_index - 1]
                if path_index - 1 < len(selected_payloads)
                else {}
            )
            # SP 分数用于表示该分支路径的选择评分
            sp_score = selected_payload.get("sp_score")
            print(f"P{path_index}: {' ---- '.join(nodes)}")
            if sp_score is not None:
                print(f"  Branch SP: {sp_score:.6g}")
    else:
        print("[Entity Branch Best Paths]")
        print("(no path selected)")
    combined_warnings = [*preprocess_result.warnings, *hanlp_result.warnings]
    if debug and combined_warnings:
        print()
        print("[HanLP SDP Warnings]")
        for warning in combined_warnings:
            print(f" - {warning}")
    print()

    # Step5：打印 LLM 生成并通过校验的原子问题 DAG
    atomic_question_dag = result.get("atomic_question_dag")
    if atomic_question_dag is None:
        print("[5. Atomic Question DAG]")
        print("(skipped: Step5 disabled)")
        print()
        return

    print("[5. Atomic Question DAG]")
    # 无效时输出校验错误和 LLM 原始返回，便于定位提示词/格式问题
    if not atomic_question_dag.valid:
        print("(invalid)")
        for error in atomic_question_dag.validation_errors:
            print(f" - {error}")
        if atomic_question_dag.raw_payload is not None:
            print("raw_payload:")
            print(
                json.dumps(
                    atomic_question_dag.raw_payload, ensure_ascii=False, indent=2
                )
            )
        print()
        return
    # 有效 DAG：逐节点打印原子问题及其依赖关系
    for node in atomic_question_dag.nodes:
        print(f"{node.id}: {node.question}")
        print(
            f"  depends_on: {', '.join(node.depends_on) if node.depends_on else '(none)'}"
        )
        print()


def _print_repaired_evidence_graph(token_reasoning_structure: Any) -> None:
    """打印 Step4 修复后的证据图边。"""
    debug_payload = getattr(token_reasoning_structure, "debug_payload", {}) or {}
    edges = list(debug_payload.get("repaired_evidence_edges") or [])
    if not edges:
        edges = [
            edge.to_dict() if hasattr(edge, "to_dict") else edge
            for edge in getattr(token_reasoning_structure, "edges", []) or []
        ]
    if not edges:
        print("(none)")
        return

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("source") or "?")
        target_id = str(edge.get("target") or "?")
        source_text = str(edge.get("source_text") or source_id)
        target_text = str(edge.get("target_text") or target_id)
        metadata = _format_repaired_edge_metadata(edge)
        suffix = f" ({metadata})" if metadata else ""
        print(f"{source_text}[{source_id}] -- {target_text}[{target_id}]{suffix}")


def _format_repaired_edge_metadata(edge: dict[str, Any]) -> str:
    """将修复图边的代价格式化为便于阅读的文本。"""
    edge_cost = edge.get("edge_cost")
    if edge_cost is None:
        return "cost=blocked"
    return f"cost={edge_cost}"


def _print_hanlp_edges_for_formalism(
    hanlp_result: HanLPSDPResult, formalism: str
) -> None:
    """打印指定 SDP formalism（例如 sdp/pas）的依存边。"""
    if formalism not in hanlp_result.sdp_graphs:
        print("(none)")
        return
    print(f"[SDP: {formalism}]")
    edges = [edge for edge in hanlp_result.edges if edge.formalism == formalism]
    if not edges:
        print("(no readable edges)")
        return
    for edge in edges:
        print(edge.display())


def _print_all_hanlp_sdp_edges(hanlp_result: HanLPSDPResult) -> None:
    """按预设顺序打印 HanLP 返回的全部 SDP 图。"""
    if not hanlp_result.sdp_graphs:
        print("(none)")
        return
    for formalism in sorted(hanlp_result.sdp_graphs, key=_hanlp_sdp_formalism_sort_key):
        _print_hanlp_edges_for_formalism(hanlp_result, formalism)


def _hanlp_sdp_formalism_sort_key(formalism: str) -> tuple[int, str]:
    """控制 SDP 图的打印顺序：PAS 优先。"""
    order = {"sdp/pas": 0}
    return (order.get(formalism, 100), formalism)


# 直接执行 python main.py 时进入这里；
# SystemExit 会把 main() 返回的状态码交给操作系统。
if __name__ == "__main__":
    raise SystemExit(main())
