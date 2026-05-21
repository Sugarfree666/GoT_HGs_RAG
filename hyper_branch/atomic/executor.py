from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from ..llm.service import AtomicLLMService
from ..utils import ensure_list, normalize_label, short_text
from .analyzer import AtomicQuestionAnalyzer
from .composer import FinalAnswerComposer
from .fusion import AtomicEvidenceFusion
from .models import (
    AtomicAnswerResult,
    AtomicQuestionAnalysis,
    AtomicQuestionNode,
    DagExecutionResult,
    FusedHyperedgeCandidate,
)
from .retriever import AtomicHyperedgeRetriever


class DagCycleError(ValueError):
    pass


class AtomicDagExecutor:
    def __init__(
        self,
        analyzer: AtomicQuestionAnalyzer,
        retriever: AtomicHyperedgeRetriever,
        fusion: AtomicEvidenceFusion,
        composer: FinalAnswerComposer,
        llm_service: AtomicLLMService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.retriever = retriever
        self.fusion = fusion
        self.composer = composer
        self.llm_service = llm_service
        self.logger = logger or logging.getLogger(__name__)

    def run(self, original_question: str, dag_payload: Any | None = None) -> DagExecutionResult:
        nodes = self.normalize_dag_payload(dag_payload, original_question=original_question)
        order = self.topological_sort(nodes)
        results_by_id: dict[str, AtomicAnswerResult] = {}
        analyses_artifact: list[dict[str, Any]] = []
        retrieval_artifact: list[dict[str, Any]] = []
        answer_artifact: list[dict[str, Any]] = []

        self.logger.info("Executing atomic DAG with %s node(s)", len(order))
        for node in order:
            dependency_answers = self._dependency_context(node.dependencies, results_by_id)
            analysis = self.analyzer.analyze(node.question, dependency_answers)
            branch_hits = self.retriever.retrieve(node.question, analysis)
            fused_evidence = self.fusion.fuse(node.question, analysis, branch_hits)
            answer_payload = self._answer_atomic_question(
                node=node,
                analysis=analysis,
                evidence=fused_evidence,
                dependency_answers=dependency_answers,
            )
            result = AtomicAnswerResult(
                node_id=node.node_id,
                question=node.question,
                analysis=analysis,
                evidence=fused_evidence,
                answer=str(answer_payload.get("answer", "") or ""),
                confidence=max(0.0, min(1.0, float(answer_payload.get("confidence", 0.0) or 0.0))),
                reasoning_summary=str(answer_payload.get("reasoning_summary", "") or ""),
                used_dependencies=list(node.dependencies),
                used_hyperedge_ids=self._used_hyperedge_ids(answer_payload, fused_evidence),
            )
            results_by_id[node.node_id] = result
            analyses_artifact.append(
                {
                    "node_id": node.node_id,
                    "question": node.question,
                    "dependency_answers": dependency_answers,
                    "analysis": analysis.to_dict(),
                }
            )
            retrieval_artifact.append(
                {
                    "node_id": node.node_id,
                    "question": node.question,
                    "branch_hits": [hit.to_dict() for hit in branch_hits],
                    "top_evidence": [candidate.to_dict() for candidate in fused_evidence],
                }
            )
            answer_artifact.append(result.to_dict())

        atomic_results = [results_by_id[node.node_id] for node in order]
        final_answer = self.composer.compose(original_question, atomic_results)
        artifacts = {
            "dag_input": [node.to_dict() for node in nodes],
            "execution_order": [node.node_id for node in order],
            "atomic_question_analyses": analyses_artifact,
            "atomic_retrieval": retrieval_artifact,
            "atomic_answers": answer_artifact,
            "final_answer": final_answer,
        }
        return DagExecutionResult(
            original_question=original_question,
            atomic_results=atomic_results,
            final_answer=final_answer,
            artifacts=artifacts,
        )

    @classmethod
    def normalize_dag_payload(cls, payload: Any | None, original_question: str | None = None) -> list[AtomicQuestionNode]:
        if payload is None:
            if original_question:
                return [AtomicQuestionNode(node_id="q1", question=original_question, metadata={"source": "single_node"})]
            raise ValueError("Atomic DAG payload is required when original_question is not provided.")

        payload = _to_plain_payload(payload)
        if isinstance(payload, dict):
            root_payload = payload
            payload = _to_plain_payload(payload.get("subquestion_dag") or payload.get("dag") or payload)
            if not isinstance(payload, dict):
                raise TypeError(f"Unsupported atomic DAG payload type: {type(payload).__name__}")
            nodes_payload = payload.get("nodes") or payload.get("subquestions") or payload.get("questions")
            if nodes_payload is None:
                nodes_payload = _nodes_from_mapping_payload(payload)
            edges_payload = payload.get("edges", [])
            variable_to_question = _clean_variable_map(
                payload.get("variable_to_question") or root_payload.get("variable_to_question", {})
            )
        elif isinstance(payload, list):
            nodes_payload = payload
            edges_payload = []
            variable_to_question = {}
        else:
            raise TypeError(f"Unsupported atomic DAG payload type: {type(payload).__name__}")

        nodes = cls._coerce_nodes(nodes_payload, variable_to_question=variable_to_question)
        cls._apply_edge_dependencies(nodes, edges_payload, variable_to_question=variable_to_question)
        if not nodes and original_question:
            return [AtomicQuestionNode(node_id="q1", question=original_question, metadata={"source": "single_node"})]
        if not nodes:
            raise ValueError("Atomic DAG payload did not contain any nodes.")
        return nodes

    @staticmethod
    def topological_sort(nodes: list[AtomicQuestionNode]) -> list[AtomicQuestionNode]:
        by_id = {node.node_id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise ValueError("Atomic DAG contains duplicate node IDs.")

        unknown_dependencies = sorted(
            {
                dependency
                for node in nodes
                for dependency in node.dependencies
                if dependency not in by_id
            }
        )
        if unknown_dependencies:
            raise ValueError(f"Atomic DAG contains unknown dependencies: {unknown_dependencies}")

        dependents: dict[str, list[str]] = {node.node_id: [] for node in nodes}
        indegree: dict[str, int] = {node.node_id: len(node.dependencies) for node in nodes}
        for node in nodes:
            for dependency in node.dependencies:
                dependents[dependency].append(node.node_id)

        ready = [node.node_id for node in nodes if indegree[node.node_id] == 0]
        order: list[AtomicQuestionNode] = []
        while ready:
            node_id = ready.pop(0)
            order.append(by_id[node_id])
            for dependent_id in dependents[node_id]:
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    ready.append(dependent_id)

        if len(order) != len(nodes):
            cycle_nodes = [node_id for node_id, degree in indegree.items() if degree > 0]
            raise DagCycleError(f"Atomic DAG contains a cycle involving: {cycle_nodes}")
        return order

    @staticmethod
    def _coerce_nodes(
        nodes_payload: Any,
        variable_to_question: dict[str, str] | None = None,
    ) -> list[AtomicQuestionNode]:
        variable_to_question = variable_to_question or {}
        raw_nodes: list[tuple[int, dict[str, Any], str]] = []
        for index, item in enumerate(ensure_list(nodes_payload), start=1):
            item = _to_plain_payload(item)
            if not isinstance(item, dict):
                raise TypeError(f"Expected DAG node object at position {index}, got {type(item).__name__}.")
            node_id = _node_id_from_payload(item, index)
            raw_nodes.append((index, item, node_id))

        node_ids = {node_id for _, _, node_id in raw_nodes}
        nodes: list[AtomicQuestionNode] = []
        for index, item, node_id in raw_nodes:
            question = str(item.get("question") or item.get("sub_question") or item.get("subquestion") or "").strip()
            if not node_id:
                raise ValueError(f"Atomic DAG node at position {index} has an empty node ID.")
            if not question:
                raise ValueError(f"Atomic DAG node {node_id} has an empty question.")

            explicit_dependencies = _first_present(
                item,
                ("dependencies", "depends_on", "parents", "prerequisites"),
            )
            if explicit_dependencies is _MISSING:
                dependencies = _dependencies_from_inputs(
                    item.get("inputs", []),
                    node_ids=node_ids,
                    variable_to_question=variable_to_question,
                )
            else:
                dependencies = _resolve_dependency_ids(
                    explicit_dependencies,
                    node_ids=node_ids,
                    variable_to_question=variable_to_question,
                    keep_unknown=True,
                )
            metadata = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "node_id",
                    "id",
                    "qid",
                    "step_id",
                    "question",
                    "sub_question",
                    "subquestion",
                    "dependencies",
                    "depends_on",
                    "parents",
                    "prerequisites",
                }
            }
            nodes.append(
                AtomicQuestionNode(
                    node_id=node_id,
                    question=question,
                    dependencies=dependencies,
                    metadata=metadata,
                )
            )
        return nodes

    @staticmethod
    def _apply_edge_dependencies(
        nodes: list[AtomicQuestionNode],
        edges_payload: Any,
        variable_to_question: dict[str, str] | None = None,
    ) -> None:
        variable_to_question = variable_to_question or {}
        by_id = {node.node_id: node for node in nodes}
        for edge in ensure_list(edges_payload):
            edge = _to_plain_payload(edge)
            if isinstance(edge, dict):
                source = str(edge.get("source") or edge.get("from") or edge.get("parent") or "").strip()
                target = str(edge.get("target") or edge.get("to") or edge.get("child") or "").strip()
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                source = str(edge[0]).strip()
                target = str(edge[1]).strip()
            else:
                continue
            source = _resolve_dependency_id(source, by_id.keys(), variable_to_question) or source
            target = _resolve_dependency_id(target, by_id.keys(), variable_to_question) or target
            if source and target and target in by_id and source not in by_id[target].dependencies:
                by_id[target].dependencies.append(source)

    def _dependency_context(
        self,
        dependency_ids: list[str],
        results_by_id: dict[str, AtomicAnswerResult],
    ) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        for dependency_id in dependency_ids:
            result = results_by_id[dependency_id]
            context.append(
                {
                    "node_id": result.node_id,
                    "question": result.question,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "reasoning_summary": result.reasoning_summary,
                    "used_hyperedge_ids": list(result.used_hyperedge_ids),
                    "evidence_summary": [
                        {
                            "hyperedge_id": evidence.hyperedge_id,
                            "fusion_score": evidence.fusion_score,
                            "branch_support": sorted(evidence.branch_support),
                            "evidence_texts": evidence.evidence_texts[:2],
                        }
                        for evidence in result.evidence[:3]
                    ],
                }
            )
        return context

    def _answer_atomic_question(
        self,
        node: AtomicQuestionNode,
        analysis: AtomicQuestionAnalysis,
        evidence: list[FusedHyperedgeCandidate],
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        evidence_payload = [item.to_dict() for item in evidence]
        if self.llm_service is not None:
            payload = self.llm_service.answer_atomic_question(
                atomic_question=node.question,
                dependency_answers=dependency_answers,
                evidence=evidence_payload,
            )
        else:
            payload = self._fallback_answer(node.question, analysis, evidence, dependency_answers)
        return self._coerce_answer_payload(payload, node.question, analysis, evidence, dependency_answers)

    def _fallback_answer(
        self,
        question: str,
        analysis: AtomicQuestionAnalysis,
        evidence: list[FusedHyperedgeCandidate],
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not evidence:
            return {
                "answer": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "reasoning_summary": "No evidence was retrieved for the atomic question.",
                "used_hyperedge_ids": [],
                "insufficient": True,
            }

        query_entities = {normalize_label(entity).lower() for entity in analysis.entities}
        answer = ""
        for candidate in evidence:
            for entity_id in candidate.entity_ids:
                label = normalize_label(entity_id)
                if label and label.lower() not in query_entities:
                    answer = label
                    break
            if answer:
                break
        if not answer:
            answer = short_text(evidence[0].hyperedge_text, 180)

        dependency_boost = 0.05 if dependency_answers else 0.0
        confidence = min(0.95, 0.35 + (0.1 * len(evidence[0].branch_support)) + dependency_boost)
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_summary": short_text(" ".join(evidence[0].evidence_texts or [evidence[0].hyperedge_text]), 420),
            "used_hyperedge_ids": [evidence[0].hyperedge_id],
            "insufficient": False,
        }

    def _coerce_answer_payload(
        self,
        payload: Any,
        question: str,
        analysis: AtomicQuestionAnalysis,
        evidence: list[FusedHyperedgeCandidate],
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = self._fallback_answer(question, analysis, evidence, dependency_answers)
        payload.setdefault("answer", "")
        payload.setdefault("confidence", 0.0)
        payload.setdefault("reasoning_summary", "")
        payload.setdefault("used_hyperedge_ids", [])
        insufficient = bool(payload.get("insufficient", False))
        if insufficient and not str(payload.get("answer", "")).strip():
            payload["answer"] = "INSUFFICIENT_EVIDENCE"
        payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
        return payload

    @staticmethod
    def _used_hyperedge_ids(payload: dict[str, Any], evidence: list[FusedHyperedgeCandidate]) -> list[str]:
        evidence_ids = {item.hyperedge_id for item in evidence}
        used = [
            str(item).strip()
            for item in ensure_list(payload.get("used_hyperedge_ids", []))
            if str(item).strip() and str(item).strip() in evidence_ids
        ]
        if used:
            return used
        return [item.hyperedge_id for item in evidence[:1]]


def _to_plain_payload(payload: Any) -> Any:
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        return payload.to_dict()
    if is_dataclass(payload):
        return asdict(payload)
    return payload


def _nodes_from_mapping_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if all(isinstance(value, dict) for value in payload.values()):
        nodes: list[dict[str, Any]] = []
        for key, value in payload.items():
            node = dict(value)
            node.setdefault("id", key)
            nodes.append(node)
        return nodes
    return []


_MISSING = object()


def _node_id_from_payload(item: dict[str, Any], index: int) -> str:
    return str(
        item.get("node_id")
        or item.get("id")
        or item.get("qid")
        or item.get("step_id")
        or f"q{index}"
    ).strip()


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return _MISSING


def _clean_variable_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    mapping: dict[str, str] = {}
    for variable, node_id in value.items():
        variable_text = str(variable).strip()
        node_text = str(node_id).strip()
        if variable_text and node_text:
            mapping[variable_text] = node_text
    return mapping


def _dependencies_from_inputs(
    value: Any,
    *,
    node_ids: Iterable[str],
    variable_to_question: dict[str, str],
) -> list[str]:
    return _resolve_dependency_ids(
        value,
        node_ids=node_ids,
        variable_to_question=variable_to_question,
        keep_unknown=False,
    )


def _resolve_dependency_ids(
    value: Any,
    *,
    node_ids: Iterable[str],
    variable_to_question: dict[str, str],
    keep_unknown: bool,
) -> list[str]:
    node_id_set = set(node_ids)
    dependencies: list[str] = []
    for item in ensure_list(value):
        raw = str(item).strip()
        if not raw:
            continue
        resolved = _resolve_dependency_id(raw, node_id_set, variable_to_question)
        if resolved:
            if resolved not in dependencies:
                dependencies.append(resolved)
        elif keep_unknown and raw not in dependencies:
            dependencies.append(raw)
    return dependencies


def _resolve_dependency_id(
    value: str,
    node_ids: Iterable[str],
    variable_to_question: dict[str, str],
) -> str:
    text = str(value).strip()
    if not text:
        return ""
    node_id_set = set(node_ids)
    if text in node_id_set:
        return text
    mapped = variable_to_question.get(text, "").strip()
    if mapped in node_id_set:
        return mapped
    return ""

