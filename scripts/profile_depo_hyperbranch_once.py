from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (DEPO_ROOT, PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import run_depo_decomposition_batch as depo_batch  # noqa: E402
from hyper_branch.config import load_config  # noqa: E402
from hyper_branch.logging_utils import TraceStore, configure_logging  # noqa: E402
from hyper_branch.pipeline import HyperBranchPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one online DEPO + HyperBranch run by stage.")
    parser.add_argument("--dataset", default="musique", help="Dataset folder under questions/.")
    parser.add_argument("--questions-file", help="Specific questions JSON file. Defaults to questions/<dataset>/questions.json.")
    parser.add_argument("--question-index", type=int, default=1, help="1-based index in the questions file.")
    parser.add_argument("--question", help="Manual question. Overrides --questions-file/--question-index.")
    parser.add_argument("--config", help="HyperBranch YAML config. Defaults to configs/<dataset>.yaml.")
    parser.add_argument("--output-dir", default="runs/profile_depo_hyperbranch", help="Profile output root.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="Model for DEPO and HyperBranch chat calls.")
    parser.add_argument("--embedding-model", help="Override HyperBranch embedding model.")
    parser.add_argument("--hanlp-model", help="HanLP pretrained constant name or local model path.")
    parser.add_argument("--hyperbranch-mock-llm", action="store_true", help="Profile DEPO online, then use HyperBranch mock LLM/local embeddings.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose HyperBranch console logs.")
    return parser.parse_args()


class Profile:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def timed(self, name: str) -> "_Timer":
        return _Timer(self, name)

    def add(self, name: str, seconds: float, **metadata: Any) -> None:
        payload = {"name": name, "seconds": seconds}
        payload.update(metadata)
        self.events.append(payload)

    def summary(self) -> list[dict[str, Any]]:
        return sorted(self.events, key=lambda item: float(item.get("seconds", 0.0)), reverse=True)


class _Timer:
    def __init__(self, profile: Profile, name: str) -> None:
        self.profile = profile
        self.name = name
        self.started = 0.0

    def __enter__(self) -> "_Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.profile.add(self.name, time.perf_counter() - self.started)


class TimedDepoLLMClient:
    def __init__(self, base_client: Any, profile: Profile) -> None:
        self._base_client = base_client
        self._profile = profile
        self.phase = "depo_llm"
        self.model = getattr(base_client, "model", "")

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return self._base_client.chat_json(system_prompt, user_prompt, max_retries=max_retries)
        finally:
            self._profile.add(
                "llm.depo.chat_json",
                time.perf_counter() - started,
                phase=self.phase,
                model=self.model,
                user_prompt_chars=len(user_prompt or ""),
            )


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2
    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    try:
        from atomic_question_dag import QuestionStructureAtomicDAGGenerator, restore_global_best_paths
        from entity_masking_preprocessor import EntityMaskingPreprocessor
        from hanlp_sdp_parser import HanLPSDPParser
        from llm_client import LLMClient
        from models import QuestionRecord
        from tri_sdp_reasoning_compiler import compile_token_reasoning_structure
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run in the project environment.", file=sys.stderr)
        return 2

    profile = Profile()
    _patch_hyperbranch_client_timing(profile)

    with profile.timed("load_question"):
        dataset = args.dataset
        questions_file = _repo_path(args.questions_file or Path("questions") / dataset / "questions.json")
        if args.question:
            item = {"index": args.question_index, "question": args.question, "answer": ""}
        else:
            items = depo_batch._read_question_items(questions_file)
            if args.question_index < 1 or args.question_index > len(items):
                raise IndexError(f"--question-index must be between 1 and {len(items)}.")
            item = items[args.question_index - 1]
        record = QuestionRecord(question=item["question"], qid=item.get("qid"))

    print(f"Profiling: dataset={dataset}, index={item['index']}, question={record.question}")

    with profile.timed("init.depo_llm_client"):
        base_llm = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
        llm_client = TimedDepoLLMClient(base_llm, profile)

    with profile.timed("init.depo_preprocessor"):
        preprocessor = EntityMaskingPreprocessor(llm_client)

    with profile.timed("init.hanlp_parser"):
        parser = HanLPSDPParser(args.hanlp_model)

    llm_client.phase = "depo.step2_entity_extraction"
    with profile.timed("depo.step2_preprocess"):
        preprocess_result = preprocessor.preprocess(record.question)

    explicit_placeholders = [mapping.placeholder for mapping in preprocess_result.mask_mappings]
    with profile.timed("depo.step3_hanlp_sdp_parse"):
        hanlp_sdp_result = parser.parse(
            preprocess_result.sdp_input_sentence,
            placeholders=explicit_placeholders,
        )

    with profile.timed("depo.step4_compile_token_reasoning_structure"):
        token_reasoning_structure = compile_token_reasoning_structure(
            hanlp_sdp_result,
            explicit_entities=explicit_placeholders,
            masked_question=preprocess_result.masked_question,
            original_question=preprocess_result.original_question,
            normalized_question=preprocess_result.normalized_question or preprocess_result.original_question,
            normalization_changed=preprocess_result.normalization_changed,
            normalization_note=preprocess_result.normalization_note,
            question_id=record.qid or f"q{item['index']}",
        )

    with profile.timed("depo.restore_question_structure"):
        question_structure = restore_global_best_paths(
            token_reasoning_structure.paths,
            preprocess_result.mask_mappings,
        )

    llm_client.phase = "depo.step5_action_trace"
    with profile.timed("depo.step5_atomic_dag"):
        atomic_question_dag = QuestionStructureAtomicDAGGenerator(llm_client).generate(
            original_question=record.question,
            question_entities=[
                entity.text for entity in preprocess_result.explicit_entities.entities
            ],
            question_structure=question_structure,
        )

    with profile.timed("depo.build_decomposition_payload"):
        result = {
            "preprocess_result": preprocess_result,
            "explicit_entities": preprocess_result.explicit_entities,
            "explicit_entity_payload": preprocess_result.explicit_entities.raw_payload,
            "original_question": preprocess_result.original_question,
            "normalized_question": preprocess_result.normalized_question,
            "normalization_changed": preprocess_result.normalization_changed,
            "normalization_note": preprocess_result.normalization_note,
            "masked_question": preprocess_result.masked_question,
            "sdp_input_sentence": preprocess_result.sdp_input_sentence,
            "hanlp_input_sentence": preprocess_result.sdp_input_sentence,
            "entity_mask_mappings": preprocess_result.mask_mappings,
            "hanlp_sdp_result": hanlp_sdp_result,
            "token_reasoning_structure": token_reasoning_structure,
            "atomic_question_dag": atomic_question_dag,
        }
        decomposition_payload = depo_batch.build_decomposition_payload(
            dataset=dataset,
            questions_file=questions_file,
            item=item,
            result=result,
            question_structure=question_structure,
        )
        dag_payload = _hyperbranch_dag_payload(decomposition_payload)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _repo_path(args.output_dir) / dataset / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = _repo_path(args.config or Path("configs") / f"{dataset}.yaml")

    with profile.timed("hyperbranch.load_config"):
        config = load_config(config_path, PROJECT_ROOT)
        if args.hyperbranch_mock_llm:
            config.llm.use_mock = True
        config.llm.model = args.llm_model
        if args.embedding_model:
            config.llm.embedding_model = args.embedding_model

    hyperbranch_run_dir = output_dir / "hyperbranch_run"
    hyperbranch_run_dir.mkdir(parents=True, exist_ok=True)
    with profile.timed("hyperbranch.configure_logging"):
        logger = configure_logging(hyperbranch_run_dir, config.runtime.log_level, verbose_console=args.verbose)
        trace_store = TraceStore(hyperbranch_run_dir)

    try:
        with profile.timed("hyperbranch.init_pipeline_dataset_and_clients"):
            pipeline = HyperBranchPipeline(
                config=config,
                run_dir=hyperbranch_run_dir,
                logger=logger,
                trace_store=trace_store,
            )
            _wrap_executor_methods(pipeline, profile)

        with profile.timed("hyperbranch.pipeline_run"):
            hyperbranch_result = pipeline.run(
                record.question,
                dag_payload=dag_payload,
                original_question_entities=_depo_explicit_entity_texts(decomposition_payload),
            )
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    payload = {
        "dataset": dataset,
        "index": item["index"],
        "question": record.question,
        "gold_answer": _gold_answer(item),
        "base_url": base_url,
        "llm_model": args.llm_model,
        "embedding_model": config.llm.embedding_model,
        "dag_node_count": len(dag_payload.get("nodes") or []),
        "dag_valid": bool(atomic_question_dag.valid),
        "final_answer": hyperbranch_result.get("final_answer", {}),
        "events": profile.events,
        "events_slowest_first": profile.summary(),
    }
    profile_path = output_dir / "profile.json"
    profile_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(payload, profile_path)
    return 0


def _patch_hyperbranch_client_timing(profile: Profile) -> None:
    from hyper_branch.llm.client import OpenAICompatibleClient

    if getattr(OpenAICompatibleClient, "_profile_patched", False):
        return

    original_chat_text = OpenAICompatibleClient.chat_text
    original_embed_texts = OpenAICompatibleClient.embed_texts

    def chat_text(self: Any, stage: str, system_prompt: str, user_payload: dict[str, Any], max_tokens: int = 1400, temperature: float | None = None) -> str:
        started = time.perf_counter()
        try:
            return original_chat_text(self, stage, system_prompt, user_payload, max_tokens=max_tokens, temperature=temperature)
        finally:
            profile.add(
                "llm.hyperbranch.chat_text",
                time.perf_counter() - started,
                stage=stage,
                model=getattr(self.config, "model", ""),
                max_tokens=max_tokens,
            )

    def embed_texts(self: Any, texts: list[str], stage: str) -> list[Any]:
        uncached = [text for text in texts if text not in getattr(self, "embedding_cache", {})]
        started = time.perf_counter()
        try:
            return original_embed_texts(self, texts, stage)
        finally:
            profile.add(
                "llm.hyperbranch.embed_texts",
                time.perf_counter() - started,
                stage=stage,
                model=getattr(self.config, "embedding_model", ""),
                count=len(texts),
                uncached_count=len(uncached),
            )

    OpenAICompatibleClient.chat_text = chat_text
    OpenAICompatibleClient.embed_texts = embed_texts
    OpenAICompatibleClient._profile_patched = True


def _wrap_executor_methods(pipeline: HyperBranchPipeline, profile: Profile) -> None:
    executor = pipeline.executor
    _wrap_method(executor.analyzer, "analyze", "hyperbranch.node.analyze_atomic_question", profile)
    _wrap_method(executor.retriever, "retrieve", "hyperbranch.node.retrieve", profile)
    _wrap_method(executor.fusion, "fuse", "hyperbranch.node.fuse", profile)
    _wrap_method(executor, "_answer_atomic_question", "hyperbranch.node.answer_atomic_question", profile)
    _wrap_method(executor.composer, "compose", "hyperbranch.compose_final_answer", profile)


def _wrap_method(obj: Any, method_name: str, event_name: str, profile: Profile) -> None:
    original = getattr(obj, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            profile.add(event_name, time.perf_counter() - started)

    setattr(obj, method_name, wrapped)


def _hyperbranch_dag_payload(decomposition_payload: dict[str, Any]) -> dict[str, Any]:
    dag = (((decomposition_payload.get("stages") or {}).get("6_atomic_question_dag")) or {})
    if not isinstance(dag, dict):
        raise ValueError("DEPO decomposition does not contain stages.6_atomic_question_dag.")
    if not dag.get("valid"):
        errors = dag.get("validation_errors") or []
        raise ValueError(f"DEPO atomic DAG is invalid: {errors}")
    nodes = dag.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("DEPO atomic DAG does not contain any nodes.")
    topic_entities = _depo_explicit_entity_texts(decomposition_payload)
    return {
        "question": decomposition_payload.get("question", ""),
        "topic_entities": topic_entities,
        "original_question_entities": topic_entities,
        "nodes": nodes,
        "edges": dag.get("edges") or [],
        "leaf_node_ids": dag.get("leaf_node_ids") or [],
        "source": "depo_stages.6_atomic_question_dag",
    }


def _depo_explicit_entity_texts(decomposition_payload: dict[str, Any]) -> list[str]:
    explicit = (((decomposition_payload.get("stages") or {}).get("1_explicit_entities")) or {})
    entities = explicit.get("entities") if isinstance(explicit, dict) else []
    entity_items = entities if isinstance(entities, list) else []
    texts: list[str] = []
    seen: set[str] = set()
    for item in entity_items:
        raw_text = item.get("text") if isinstance(item, dict) else item
        text = str(raw_text or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            texts.append(text)
    return texts


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def _gold_answer(item: dict[str, Any]) -> Any:
    for key in ("answer", "gold_answer", "answers", "gold_answers"):
        if key in item:
            return item[key]
    return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return value


def _print_report(payload: dict[str, Any], profile_path: Path) -> None:
    print()
    print("[Profile]")
    print(f"Output: {profile_path}")
    print(f"DAG nodes: {payload['dag_node_count']}, final answer: {payload.get('final_answer', {}).get('answer', '')!r}")
    print()
    print("Slowest events:")
    for event in payload["events_slowest_first"][:20]:
        seconds = float(event.get("seconds", 0.0))
        metadata = " ".join(
            f"{key}={value}"
            for key, value in event.items()
            if key not in {"name", "seconds"} and value not in (None, "", [])
        )
        print(f"{seconds:8.2f}s  {event['name']}{('  ' + metadata) if metadata else ''}")


if __name__ == "__main__":
    raise SystemExit(main())
