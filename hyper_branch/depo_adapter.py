from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import Config


def generate_atomic_dag_with_depo(
    question: str,
    config: Config,
    *,
    corenlp_url: str = "http://localhost:9000",
    corenlp_memory: str = "4G",
    corenlp_home: str | None = None,
    corenlp_timeout_ms: int = 60000,
) -> dict[str, Any]:
    """Run the existing DEPO pipeline and return its atomic DAG payload.

    This is intentionally a thin adapter. It constructs DEPO's existing
    components and calls depo.main.run_pipeline without changing DEPO's
    decomposition, DAG construction, or dependency semantics.
    """

    api_key = os.getenv(config.llm.api_key_env, "").strip()
    base_url = os.getenv(config.llm.base_url_env, "").strip() or None
    if not api_key:
        raise RuntimeError(
            f"DEPO requires an online LLM API key. Set {config.llm.api_key_env}, "
            "or pass --dag with a precomputed DEPO DAG, or use --mock-llm for single-node mock execution."
        )

    depo_dir = (config.project_root / "depo").resolve()
    if not depo_dir.exists():
        raise RuntimeError(f"DEPO directory not found: {depo_dir}")

    with _prepend_sys_path(depo_dir):
        try:
            main_module = _load_depo_main(depo_dir)
            corenlp_parser_module = importlib.import_module("corenlp_parser")
            entity_path_pipeline_module = importlib.import_module("entity_path_pipeline")
            graph_builder_module = importlib.import_module("graph_builder")
            llm_client_module = importlib.import_module("llm_client")
            mask_span_module = importlib.import_module("mask_span_extractor")
            models_module = importlib.import_module("models")
            normalizer_module = importlib.import_module("question_normalizer")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"DEPO dependency is missing: {exc.name}. Install DEPO dependencies before using --question without --dag."
            ) from exc

        llm_client = llm_client_module.LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=config.llm.model,
        )
        question_normalizer = normalizer_module.SemanticQuestionNormalizer(llm_client)
        mask_span_extractor = mask_span_module.MaskSpanExtractor(llm_client)
        graph_builder = graph_builder_module.GraphBuilder()
        path_semantic_parser = entity_path_pipeline_module.EntityPathSemanticParser(llm_client)
        record = models_module.QuestionRecord(question=question)

        try:
            with corenlp_parser_module.CoreNLPParser(
                corenlp_url,
                timeout_ms=corenlp_timeout_ms,
                memory=corenlp_memory,
                corenlp_home=corenlp_home,
            ) as parser:
                result = main_module.run_pipeline(
                    record=record,
                    index=1,
                    mask_span_extractor=mask_span_extractor,
                    parser=parser,
                    graph_builder=graph_builder,
                    question_normalizer=question_normalizer,
                    path_semantic_parser=path_semantic_parser,
                    debug=False,
                )
        except Exception as exc:
            raise RuntimeError(f"DEPO failed to generate an atomic DAG: {exc}") from exc

    return _extract_dag_payload(result)


def _extract_dag_payload(depo_result: dict[str, Any]) -> dict[str, Any]:
    subquestion_dag = depo_result.get("subquestion_dag")
    if subquestion_dag is not None and hasattr(subquestion_dag, "to_dict"):
        return subquestion_dag.to_dict()
    if isinstance(subquestion_dag, dict):
        return subquestion_dag

    subquestions = depo_result.get("subquestions", [])
    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(subquestions, start=1):
        payload = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        nodes.append(
            {
                "id": payload.get("id") or f"q{index}",
                "question": payload.get("question", ""),
                "dependencies": payload.get("dependencies", payload.get("depends_on", [])),
                "metadata": payload,
            }
        )
    if nodes:
        return {"nodes": nodes, "edges": []}
    raise RuntimeError("DEPO completed but did not return an atomic DAG or subquestions.")


def _load_depo_main(depo_dir: Path) -> Any:
    module_name = "_hyper_branch_depo_main"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, depo_dir / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DEPO main module from {depo_dir / 'main.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _prepend_sys_path(path: Path) -> Any:
    text = str(path)
    added = False
    if text not in sys.path:
        sys.path.insert(0, text)
        added = True
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(text)
            except ValueError:
                pass
