from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_DIR = PROJECT_ROOT / "depo"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DEPO_DIR) not in sys.path:
    sys.path.insert(0, str(DEPO_DIR))

from hyper_branch.config import Config, load_config
from hyper_branch.data.loaders import HypergraphDatasetLoader
from hyper_branch.llm import (
    LocalHashEmbeddingClient,
    MockReasoningService,
    OpenAICompatibleClient,
    OpenAIReasoningService,
    PromptManager,
)
from hyper_branch.logging_utils import TraceStore
from hyper_branch.reasoning.controller import ThoughtController
from hyper_branch.reasoning.operations import ThoughtOperationExecutor
from hyper_branch.reasoning.scoring import ThoughtScorer
from hyper_branch.reasoning.taskframe import TaskFrameBuilder, TaskFrameRegistry
from hyper_branch.retrieval.evidence import EvidenceRetriever
from hyper_branch.utils import extract_json_payload, pretty_json, short_text, slugify

from anchor_selector import AnchorSelector
from ast_builder import SemanticASTOptimizer
from corenlp_parser import CoreNLPConnectionError, CoreNLPParser
from graph_builder import GraphBuilder
from main import run_pipeline as run_depo_pipeline
from mask_span_extractor import MaskSpanExtractor
from models import AtomicQuestionDAG, QuestionRecord
from question_normalizer import SemanticQuestionNormalizer
from subquestion_generator import SubquestionGenerator


FINAL_SYNTHESIS_SYSTEM = """You are the final answer synthesizer for a DEPO + HyperBranch experiment.

Return JSON only:
{
  "answer": "...",
  "reasoning_summary": "...",
  "confidence": 0.0,
  "remaining_gaps": ["..."]
}

Rules:
- Answer the original question directly.
- The answer must be the shortest grounded answer span, not an explanatory sentence.
- For yes/no questions, answer exactly "yes" or "no" when the evidence supports it.
- Prefer 1 to 5 words in "answer"; never exceed 8 words unless the gold entity name itself is longer.
- Use the atomic subquestion DAG only as the reasoning plan.
- Use the subquestion answers and evidence as the factual basis.
- If evidence is insufficient or contradictory, keep the best grounded short answer and list gaps in "remaining_gaps".
- Put explanation only in "reasoning_summary".
- Keep "confidence" between 0 and 1.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DEPO decomposition + HyperBranch on 2WikiMultiHopQA.")
    parser.add_argument("--question-file", default="questions/2wikimultihopqa/questions.json")
    parser.add_argument("--config", default="configs/2wikimultihopqa.yaml")
    parser.add_argument("--output-dir", default="runs/depo_2wiki")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="", help="Override chat model for both DEPO and HyperBranch.")
    parser.add_argument("--embedding-model", default="", help="Override embedding model for HyperBranch.")
    parser.add_argument("--api-key", default="", help="OpenAI-compatible API key. Prefer env var for shared runs.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL, e.g. https://api.example.com/v1.")
    parser.add_argument("--mock-llm", action="store_true", help="Use HyperBranch mock LLM. DEPO still uses online LLM.")
    parser.add_argument("--resume", action="store_true", help="Skip samples with an existing final_answer.json.")
    parser.add_argument("--corenlp-url", default="http://localhost:9000")
    parser.add_argument("--corenlp-memory", default="4G")
    parser.add_argument("--corenlp-home", default="")
    parser.add_argument("--corenlp-timeout-ms", type=int, default=60000)
    parser.add_argument(
        "--corenlp-backend",
        choices=("auto", "stanza", "java"),
        default="auto",
        help="Use stanza CoreNLPClient, direct Java server management, or auto fallback.",
    )
    return parser.parse_args()


class HyperBranchExperimentRunner:
    def __init__(self, config: Config, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader_logger = _file_logger("depo_2wiki.loader", self.output_dir / "hyperbranch_loader.log")
        loader = HypergraphDatasetLoader(config.dataset, self.loader_logger)
        self.dataset = loader.load()

        if config.llm.use_mock:
            self.client: OpenAICompatibleClient | None = None
            self.embedder = LocalHashEmbeddingClient()
            self.llm_service = MockReasoningService()
        else:
            self.client = OpenAICompatibleClient(config.llm, trace_store=None)
            self.embedder = self.client
            prompts = PromptManager(config.prompts.directory)
            self.llm_service = OpenAIReasoningService(client=self.client, prompts=prompts)

    def run(self, question: str, run_dir: Path) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_store = TraceStore(run_dir)
        logger = _file_logger(f"depo_2wiki.hyperbranch.{run_dir.name}", run_dir / "run.log", self.config.runtime.log_level)
        if self.client is not None:
            self.client.trace_store = trace_store
        trace_store.save_artifact("artifacts/dataset_summary.json", self.dataset.summary)

        taskframe_builder = TaskFrameBuilder(self.llm_service, self.dataset, logger, trace_store)
        registry = TaskFrameRegistry(
            embedder=self.embedder,
            threshold=self.config.retrieval.taskframe_registration_threshold,
            logger=logger,
            trace_store=trace_store,
        )
        scorer = ThoughtScorer(embedder=self.embedder, config=self.config.reasoning, logger=logger)
        evidence_retriever = EvidenceRetriever(
            dataset=self.dataset,
            embedder=self.embedder,
            config=self.config.retrieval,
            logger=logger,
            reasoning_config=self.config.reasoning,
        )
        executor = ThoughtOperationExecutor(logger=logger, trace_store=trace_store)
        controller = ThoughtController(
            config=self.config,
            dataset=self.dataset,
            taskframe_builder=taskframe_builder,
            registry=registry,
            scorer=scorer,
            evidence_retriever=evidence_retriever,
            executor=executor,
            llm_service=self.llm_service,
            logger=logger,
            trace_store=trace_store,
        )
        result = controller.run(question)
        trace_store.save_artifact("artifacts/task_frame.json", result["task_frame"])
        trace_store.save_artifact("artifacts/thought_graph.json", result["thought_graph"])
        trace_store.save_artifact("artifacts/evidence_subgraph.json", result["evidence_subgraph"])
        if "llm_evidence_view" in result:
            trace_store.save_artifact("artifacts/llm_evidence_view.json", result["llm_evidence_view"])
        trace_store.save_artifact("artifacts/final_answer.json", result["final_answer"])
        result["run_dir"] = str(run_dir)
        _close_logger(logger)
        return result

    def synthesize_final_answer(
        self,
        original_question: str,
        dag: AtomicQuestionDAG,
        subquestion_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.client is None:
            return _fallback_final_answer(original_question, subquestion_results)

        payload = {
            "original_question": original_question,
            "atomic_subquestion_dag": dag.to_dict(),
            "subquestion_results": _subquestion_synthesis_view(subquestion_results),
            "evidence": _combined_evidence(subquestion_results, limit=24),
        }
        response_text = self.client.chat_text(
            stage="depo_final_answer",
            system_prompt=FINAL_SYNTHESIS_SYSTEM,
            user_payload=payload,
            max_tokens=900,
            temperature=0.0,
        )
        parsed = extract_json_payload(response_text)
        if not isinstance(parsed, dict):
            raise ValueError("Final synthesis did not return a JSON object.")
        return _normalize_final_answer_payload(parsed)


class HyperBranchClientAdapter:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(max(1, max_retries)):
            prompt = user_prompt
            if attempt:
                prompt += (
                    "\n\nYour previous response was not valid JSON or failed to parse. "
                    "Return only one valid JSON object."
                )
            try:
                text = self._chat_raw_prompt(system_prompt, prompt)
                parsed = extract_json_payload(text)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON object, got {type(parsed).__name__}.")
                return parsed
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"DEPO LLM call failed after {max_retries} attempts: {last_error}")

    def _chat_raw_prompt(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.client.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
        }
        response = self.client._post_json("/chat/completions", payload)
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if self.client.trace_store is not None:
            self.client.trace_store.log_llm_call("depo_decomposition", payload, {"content": content})
        return str(content)


class JavaCoreNLPParser:
    def __init__(
        self,
        url: str,
        timeout_ms: int,
        memory: str,
        corenlp_home: str | None,
        log_path: Path,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.memory = memory
        self.corenlp_home = corenlp_home
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self._payload_parser = CoreNLPParser(url=url, timeout_ms=timeout_ms, memory=memory, corenlp_home=corenlp_home)
        self.properties = {
            "annotators": "tokenize,ssplit,pos,lemma,depparse",
            "outputFormat": "json",
            "depparse.extradependencies": "MAXIMAL",
        }

    def __enter__(self) -> "JavaCoreNLPParser":
        if self._is_ready():
            return self
        home = self._resolve_corenlp_home()
        classpath = os.pathsep.join(str(path) for path in sorted(home.glob("*.jar")))
        if not classpath:
            raise CoreNLPConnectionError(f"CoreNLP home does not contain jar files: {home}")
        port = parse.urlparse(self.url).port or 9000
        command = [
            "java",
            f"-mx{self.memory}",
            "-cp",
            classpath,
            "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
            "-port",
            str(port),
            "-timeout",
            str(self.timeout_ms),
            "-quiet",
            "true",
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=self._log_handle,
            text=True,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise CoreNLPConnectionError(
                    f"CoreNLP Java server exited early with code {self.process.returncode}. See {self.log_path}."
                )
            if self._is_ready():
                return self
            time.sleep(1)
        raise CoreNLPConnectionError(f"Timed out waiting for CoreNLP Java server at {self.url}. See {self.log_path}.")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        finally:
            self.process = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def parse(self, text: str) -> Any:
        payload = self._annotate(text)
        return self._payload_parser._parse_payload(payload)

    def _is_ready(self) -> bool:
        try:
            self._annotate("CoreNLP readiness check.")
            return True
        except Exception:
            return False

    def _annotate(self, text: str) -> dict[str, Any]:
        properties = parse.quote(json.dumps(self.properties, separators=(",", ":")))
        req = request.Request(
            url=f"{self.url}/?properties={properties}",
            data=text.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=max(5, self.timeout_ms / 1000)) as response:
                body = response.read().decode("utf-8")
        except error.URLError as exc:
            raise CoreNLPConnectionError(f"CoreNLP Java server request failed at {self.url}: {exc}") from exc
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise CoreNLPConnectionError("CoreNLP Java server returned a non-object JSON payload.")
        return parsed

    def _resolve_corenlp_home(self) -> Path:
        candidates: list[Path] = []
        if self.corenlp_home:
            candidates.append(Path(self.corenlp_home).expanduser())
        if os.getenv("CORENLP_HOME"):
            candidates.append(Path(os.environ["CORENLP_HOME"]).expanduser())
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            candidates.extend(sorted((Path(local_appdata) / "StanfordNLP" / "stanza" / "Cache").glob("*/corenlp"), reverse=True))
        candidates.append(Path.home() / "stanza_corenlp")
        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("stanford-corenlp*.jar")):
                return candidate
        raise CoreNLPConnectionError("Could not find a Stanford CoreNLP directory with stanford-corenlp*.jar files.")


def main() -> int:
    args = parse_args()
    question_file = _resolve_path(args.question_file)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = _normalize_base_url(args.base_url)

    config = load_config(_resolve_path(args.config), PROJECT_ROOT)
    config.runtime.base_run_dir = output_dir
    if args.model:
        config.llm.model = args.model
    if args.embedding_model:
        config.llm.embedding_model = args.embedding_model
    if args.mock_llm:
        config.llm.use_mock = True

    questions = _load_questions(question_file, args.start_index, args.limit)
    hb_runner = HyperBranchExperimentRunner(config=config, output_dir=output_dir)
    if hb_runner.client is None:
        raise RuntimeError("DEPO decomposition requires an online LLM client; do not use --mock-llm for the full experiment.")
    depo_llm = HyperBranchClientAdapter(hb_runner.client)
    depo_components = _build_depo_components(depo_llm)

    parser_cm = _build_parser(args, output_dir)
    aggregate_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    fatal_failure: dict[str, Any] | None = None
    with parser_cm as parser_backend:
        progress = tqdm(total=len(questions), desc="DEPO+HyperBranch", unit="q", dynamic_ncols=True)
        for offset, question_entry in enumerate(questions):
            dataset_index = args.start_index + offset
            run_dir = output_dir / f"q{dataset_index + 1:04d}_{slugify(question_entry['question'], 48)}"
            final_path = run_dir / "artifacts" / "final_answer.json"
            error_path = run_dir / "artifacts" / "error.json"
            if args.resume and final_path.exists() and not error_path.exists():
                record = _load_existing_record(question_entry, run_dir)
                aggregate_records.append(record)
                progress.update(1)
                continue
            try:
                record = _run_one_question(
                    question_entry=question_entry,
                    dataset_index=dataset_index,
                    run_dir=run_dir,
                    parser_backend=parser_backend,
                    depo_components=depo_components,
                    hb_runner=hb_runner,
                )
                aggregate_records.append(record)
            except Exception as exc:
                failure = _write_failure(question_entry, dataset_index, run_dir, exc)
                failures.append(failure)
                aggregate_records.append(_failed_aggregate_record(question_entry, run_dir, failure))
                if _is_fatal_online_error(exc):
                    fatal_failure = failure
                    progress.update(1)
                    break
            finally:
                _write_aggregate_files(output_dir, aggregate_records, failures, args, question_file)
                if fatal_failure is None:
                    progress.update(1)
        progress.close()

    return 1 if fatal_failure is not None else 0


def _run_one_question(
    question_entry: dict[str, Any],
    dataset_index: int,
    run_dir: Path,
    parser_backend: Any,
    depo_components: dict[str, Any],
    hb_runner: HyperBranchExperimentRunner,
) -> dict[str, Any]:
    question = question_entry["question"]
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stale_error = artifacts_dir / "error.json"
    if stale_error.exists():
        stale_error.unlink()

    depo_result = run_depo_pipeline(
        record=QuestionRecord(question=question),
        index=dataset_index + 1,
        parser=parser_backend,
        debug=False,
        **depo_components,
    )
    dag = _ensure_dag(depo_result)
    _write_json(artifacts_dir / "depo_decomposition.json", _depo_result_to_artifact(depo_result))

    variable_bindings: dict[str, str] = {}
    subquestion_results: list[dict[str, Any]] = []
    for node in dag.nodes:
        execution_bindings = _execution_bindings_for_node(node.to_dict(), variable_bindings)
        executable_question = _substitute_variables(node.question, execution_bindings)
        sub_run_dir = run_dir / "subquestions" / node.id
        hb_result = hb_runner.run(executable_question, sub_run_dir)
        answer = str(hb_result.get("final_answer", {}).get("answer", "") or "").strip()
        if node.output and node.output != "FINAL" and answer:
            variable_bindings[node.output] = answer
        subquestion_results.append(
            {
                "node": node.to_dict(),
                "question": node.question,
                "executable_question": executable_question,
                "answer": answer,
                "output_variable": node.output,
                "hyperbranch_run_dir": str(sub_run_dir),
                "final_answer": hb_result.get("final_answer", {}),
                "evidence": _extract_evidence(hb_result),
                "status": "success",
            }
        )

    final_answer = hb_runner.synthesize_final_answer(question, dag, subquestion_results)
    combined_evidence = _combined_evidence(subquestion_results, limit=80)

    _write_json(artifacts_dir / "subquestion_results.json", subquestion_results)
    _write_json(artifacts_dir / "final_answer.json", final_answer)
    _write_json(artifacts_dir / "evidence_subgraph.json", _synthetic_evidence_subgraph(combined_evidence))
    _write_json(artifacts_dir / "thought_graph.json", _synthetic_thought_graph(question, final_answer, subquestion_results, combined_evidence))
    _write_json(artifacts_dir / "task_frame.json", _synthetic_task_frame(question, dag))

    return {
        "question": question,
        "golden_answers": _gold_answers(question_entry),
        "context": list(question_entry.get("context", [])),
        "nhops": question_entry.get("nhops"),
        "run_dir": str(run_dir),
        "run_status": "success",
        "answer": final_answer.get("answer", ""),
        "generation": f"<answer>{final_answer.get('answer', '')}</answer>" if final_answer.get("answer") else "",
        "generation_explanation": str(final_answer.get("reasoning_summary", "") or ""),
        "retrieved": combined_evidence,
        "retrieved_knowledge": "\n\n".join(item.get("content", "") for item in combined_evidence if item.get("content")),
        "subquestion_count": len(dag.nodes),
    }


def _build_depo_components(llm_client: HyperBranchClientAdapter) -> dict[str, Any]:
    return {
        "question_normalizer": SemanticQuestionNormalizer(llm_client),
        "mask_span_extractor": MaskSpanExtractor(llm_client),
        "graph_builder": GraphBuilder(),
        "anchor_selector": AnchorSelector(llm_client),
        "semantic_ast_optimizer": SemanticASTOptimizer(llm_client),
        "subquestion_generator": SubquestionGenerator(llm_client),
    }


def _build_parser(args: argparse.Namespace, output_dir: Path) -> Any:
    if args.corenlp_backend == "java":
        return JavaCoreNLPParser(
            url=args.corenlp_url,
            timeout_ms=args.corenlp_timeout_ms,
            memory=args.corenlp_memory,
            corenlp_home=args.corenlp_home or None,
            log_path=output_dir / "corenlp_server.log",
        )
    stanza_parser = CoreNLPParser(
        args.corenlp_url,
        timeout_ms=args.corenlp_timeout_ms,
        memory=args.corenlp_memory,
        corenlp_home=args.corenlp_home or None,
    )
    if args.corenlp_backend == "stanza":
        return stanza_parser
    try:
        import stanza.server  # noqa: F401

        return stanza_parser
    except ModuleNotFoundError:
        return JavaCoreNLPParser(
            url=args.corenlp_url,
            timeout_ms=args.corenlp_timeout_ms,
            memory=args.corenlp_memory,
            corenlp_home=args.corenlp_home or None,
            log_path=output_dir / "corenlp_server.log",
        )


def _load_questions(question_file: Path, start_index: int, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(question_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {question_file}.")
    selected = payload[start_index : start_index + limit]
    questions: list[dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("Question entries must be JSON objects.")
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError("Question entry has no non-empty 'question'.")
        questions.append(item)
    return questions


def _ensure_dag(depo_result: dict[str, Any]) -> AtomicQuestionDAG:
    dag = depo_result.get("subquestion_dag")
    if isinstance(dag, AtomicQuestionDAG):
        return dag
    subquestions = depo_result.get("subquestions")
    if not subquestions:
        raise ValueError("DEPO produced no atomic subquestion DAG.")
    raise ValueError("DEPO result did not include an AtomicQuestionDAG.")


def _depo_result_to_artifact(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "semantic_normalization",
        "mask_spans",
        "replacement",
        "restored_graph_node_candidates",
        "anchor_selection",
        "restored_anchor_connected_subgraph",
        "semantic_ast",
        "subquestion_dag",
        "subquestions",
    )
    return {key: _to_jsonable(result.get(key)) for key in keys}


def _substitute_variables(question: str, bindings: dict[str, str]) -> str:
    substituted = question
    for variable, value in sorted(bindings.items(), key=lambda item: len(item[0]), reverse=True):
        if not value:
            continue
        substituted = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])", value, substituted)
    return substituted


def _execution_bindings_for_node(node: dict[str, Any], variable_bindings: dict[str, str]) -> dict[str, str]:
    bindings = dict(variable_bindings)
    candidate_bindings = node.get("candidate_bindings", [])
    if not isinstance(candidate_bindings, list):
        return bindings
    for item in candidate_bindings:
        if not isinstance(item, dict):
            continue
        variable = str(item.get("value", "") or "").strip()
        candidate = str(item.get("candidate", "") or "").strip()
        if not variable or not candidate or variable in bindings:
            continue
        bindings[variable] = candidate
    return bindings


def _extract_evidence(hb_result: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_subgraph = hb_result.get("evidence_subgraph", {})
    evidence = evidence_subgraph.get("evidence", []) if isinstance(evidence_subgraph, dict) else []
    extracted: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            key = str(item.get("chunk_id", "") or content)
            if key in seen:
                continue
            seen.add(key)
            extracted.append(
                {
                    "chunk_id": str(item.get("chunk_id", "") or ""),
                    "content": content,
                    "score": item.get("score"),
                    "source_node_ids": item.get("source_node_ids", []),
                    "source_edge_ids": item.get("source_edge_ids", []),
                }
            )
    return extracted


def _combined_evidence(subquestion_results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in subquestion_results:
        node = result.get("node", {})
        node_id = str(node.get("id", "") if isinstance(node, dict) else "")
        for item in result.get("evidence", []):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            key = str(item.get("chunk_id", "") or content)
            if key in seen:
                continue
            seen.add(key)
            combined.append({**item, "subquestion_id": node_id})
            if len(combined) >= limit:
                return combined
    return combined


def _subquestion_synthesis_view(subquestion_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    for result in subquestion_results:
        node = result.get("node", {})
        evidence = [
            {
                "chunk_id": item.get("chunk_id", ""),
                "content": short_text(str(item.get("content", "") or ""), 500),
            }
            for item in result.get("evidence", [])[:5]
            if isinstance(item, dict)
        ]
        view.append(
            {
                "id": node.get("id") if isinstance(node, dict) else "",
                "type": node.get("type") if isinstance(node, dict) else "",
                "question": result.get("question", ""),
                "executable_question": result.get("executable_question", ""),
                "answer": result.get("answer", ""),
                "output_variable": result.get("output_variable", ""),
                "depends_on": node.get("depends_on", []) if isinstance(node, dict) else [],
                "candidate_bindings": node.get("candidate_bindings", []) if isinstance(node, dict) else [],
                "evidence": evidence,
            }
        )
    return view


def _normalize_final_answer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    answer = str(payload.get("answer", "") or "").strip()
    answer = re.sub(r"^(?:answer\s*:\s*|the answer is\s+|it is\s+|it's\s+)", "", answer, flags=re.IGNORECASE).strip()
    remaining_gaps = payload.get("remaining_gaps", [])
    if not isinstance(remaining_gaps, list):
        remaining_gaps = [str(remaining_gaps)]
    confidence = payload.get("confidence", 0.0)
    try:
        confidence_float = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_float = 0.0
    return {
        "answer": answer,
        "reasoning_summary": str(payload.get("reasoning_summary", "") or "").strip(),
        "confidence": confidence_float,
        "remaining_gaps": [str(item).strip() for item in remaining_gaps if str(item).strip()],
    }


def _fallback_final_answer(original_question: str, subquestion_results: list[dict[str, Any]]) -> dict[str, Any]:
    for result in reversed(subquestion_results):
        answer = str(result.get("answer", "") or "").strip()
        if answer:
            return {
                "answer": answer,
                "reasoning_summary": "Fallback used the last non-empty atomic subquestion answer.",
                "confidence": 0.2,
                "remaining_gaps": ["No online final synthesis was available.", f"Original question: {original_question}"],
            }
    return {
        "answer": "",
        "reasoning_summary": "No atomic subquestion produced an answer.",
        "confidence": 0.0,
        "remaining_gaps": ["No answer was produced."],
    }


def _synthetic_evidence_subgraph(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_ids = [item.get("chunk_id", "") for item in evidence if item.get("chunk_id")]
    return {
        "hyperedge_ids": [],
        "entity_ids": [],
        "chunk_ids": list(dict.fromkeys(chunk_ids)),
        "evidence": evidence,
        "summary_text": " | ".join(short_text(item.get("content", ""), 160) for item in evidence[:5]),
    }


def _synthetic_thought_graph(
    question: str,
    final_answer: dict[str, Any],
    subquestion_results: list[dict[str, Any]],
    combined_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    thoughts: dict[str, Any] = {
        "depo-root": {
            "thought_id": "depo-root",
            "kind": "reasoning",
            "content": question,
            "objective": "DEPO atomic decomposition root",
            "slot_id": None,
            "grounding": {"anchor_texts": [], "node_ids": [], "chunk_ids": [], "evidence": [], "notes": ["depo-root"]},
            "score": 0.0,
            "status": "root",
            "parent_ids": [],
            "metadata": {},
            "grounding_text": "",
        }
    }
    for result in subquestion_results:
        node = result.get("node", {})
        node_id = str(node.get("id", "") if isinstance(node, dict) else "")
        evidence = result.get("evidence", [])
        thoughts[f"depo-{node_id}"] = {
            "thought_id": f"depo-{node_id}",
            "kind": "reasoning",
            "content": result.get("executable_question", ""),
            "objective": result.get("question", ""),
            "slot_id": node_id,
            "grounding": {
                "anchor_texts": [],
                "node_ids": [],
                "chunk_ids": [item.get("chunk_id", "") for item in evidence if isinstance(item, dict) and item.get("chunk_id")],
                "evidence": evidence,
                "notes": [str(result.get("answer", "") or "")],
            },
            "score": 0.0,
            "status": "completed",
            "parent_ids": ["depo-root"],
            "metadata": {"answer": result.get("answer", ""), "hyperbranch_run_dir": result.get("hyperbranch_run_dir", "")},
            "grounding_text": " | ".join(short_text(item.get("content", ""), 160) for item in evidence[:3] if isinstance(item, dict)),
        }
    thoughts["depo-final"] = {
        "thought_id": "depo-final",
        "kind": "answer",
        "content": final_answer.get("answer", ""),
        "objective": question,
        "slot_id": "target-0",
        "grounding": {
            "anchor_texts": [],
            "node_ids": [],
            "chunk_ids": [item.get("chunk_id", "") for item in combined_evidence if item.get("chunk_id")],
            "evidence": combined_evidence,
            "notes": [final_answer.get("reasoning_summary", ""), *final_answer.get("remaining_gaps", [])],
        },
        "score": final_answer.get("confidence", 0.0),
        "status": "completed",
        "parent_ids": [key for key in thoughts if key.startswith("depo-q")],
        "metadata": final_answer,
        "grounding_text": " | ".join(short_text(item.get("content", ""), 160) for item in combined_evidence[:5]),
    }
    return {
        "question": question,
        "root_id": "depo-root",
        "frontier_ids": [],
        "status": "completed",
        "termination_reason": "depo_final_synthesis",
        "final_answer": final_answer,
        "thoughts": thoughts,
        "history": [],
    }


def _synthetic_task_frame(question: str, dag: AtomicQuestionDAG) -> dict[str, Any]:
    return {
        "question": question,
        "anchors": [],
        "target": question,
        "constraints": [],
        "bridges": [node.question for node in dag.nodes],
        "topic_entities": [],
        "answer_type_hint": "grounded short answer",
        "relation_intent": "answer via DEPO atomic subquestion DAG",
        "hard_constraints": [],
        "relation_skeleton": "",
        "initial_entity_ids": [],
        "initial_hyperedge_ids": [],
        "metadata": {"atomic_subquestion_count": len(dag.nodes)},
        "checklist": {},
    }


def _load_existing_record(question_entry: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    final_answer = json.loads((run_dir / "artifacts" / "final_answer.json").read_text(encoding="utf-8"))
    evidence_path = run_dir / "artifacts" / "evidence_subgraph.json"
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    evidence = evidence_payload.get("evidence", []) if isinstance(evidence_payload, dict) else []
    return {
        "question": question_entry["question"],
        "golden_answers": _gold_answers(question_entry),
        "context": list(question_entry.get("context", [])),
        "nhops": question_entry.get("nhops"),
        "run_dir": str(run_dir),
        "run_status": "success",
        "answer": final_answer.get("answer", ""),
        "generation": f"<answer>{final_answer.get('answer', '')}</answer>" if final_answer.get("answer") else "",
        "generation_explanation": str(final_answer.get("reasoning_summary", "") or ""),
        "retrieved": evidence,
        "retrieved_knowledge": "\n\n".join(item.get("content", "") for item in evidence if isinstance(item, dict)),
    }


def _write_failure(question_entry: dict[str, Any], dataset_index: int, run_dir: Path, exc: Exception) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "question": question_entry.get("question", ""),
        "dataset_index": dataset_index,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    _write_json(artifacts_dir / "error.json", failure)
    _write_json(artifacts_dir / "final_answer.json", {"answer": "", "reasoning_summary": "", "confidence": 0.0, "remaining_gaps": [str(exc)]})
    return failure


def _failed_aggregate_record(question_entry: dict[str, Any], run_dir: Path, failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": question_entry.get("question", ""),
        "golden_answers": _gold_answers(question_entry),
        "context": list(question_entry.get("context", [])),
        "nhops": question_entry.get("nhops"),
        "run_dir": str(run_dir),
        "run_status": "failed",
        "answer": "",
        "generation": "",
        "generation_explanation": "",
        "retrieved": [],
        "retrieved_knowledge": "",
        "run_error": failure.get("error_message", ""),
    }


def _is_fatal_online_error(exc: Exception) -> bool:
    text = str(exc).lower()
    fatal_markers = (
        "http 401",
        "invalid_api_key",
        "incorrect api key",
        "environment variable openai_api_key is required",
        "authentication",
    )
    return any(marker in text for marker in fatal_markers)


def _write_aggregate_files(
    output_dir: Path,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    args: argparse.Namespace,
    question_file: Path,
) -> None:
    generated = [
        {
            "question": record["question"],
            "golden_answers": record["golden_answers"],
            "context": record.get("context", []),
            "nhops": record.get("nhops"),
            "run_dir": record.get("run_dir"),
            "run_status": record.get("run_status"),
            "answer": record.get("answer", ""),
            "generation": record.get("generation", ""),
            "generation_explanation": record.get("generation_explanation", ""),
            "retrieved": record.get("retrieved", []),
            "retrieved_knowledge": record.get("retrieved_knowledge", ""),
        }
        for record in records
    ]
    summary = {
        "question_file": str(question_file),
        "output_dir": str(output_dir),
        "start_index": args.start_index,
        "limit": args.limit,
        "completed": len(records),
        "success": sum(1 for record in records if record.get("run_status") == "success"),
        "failed": sum(1 for record in records if record.get("run_status") == "failed"),
        "failures": failures,
    }
    _write_json(output_dir / "generated_answer.json", generated)
    _write_json(output_dir / "test_result.json", records)
    _write_json(output_dir / "run_summary.json", summary)


def _gold_answers(question_entry: dict[str, Any]) -> list[str]:
    for key in ("golden_answers", "answers"):
        value = question_entry.get(key)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    answer = question_entry.get("answer")
    if isinstance(answer, list):
        return [str(item).strip() for item in answer if str(item).strip()]
    if isinstance(answer, str) and answer.strip():
        return [answer.strip()]
    return []


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(payload), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    try:
        return asdict(value)
    except TypeError:
        pass
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        return cleaned
    parsed = parse.urlparse(cleaned)
    if parsed.path in {"", "/"}:
        return cleaned + "/v1"
    return cleaned


def _file_logger(name: str, path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
