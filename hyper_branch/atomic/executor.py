from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from ..llm.service import AtomicLLMService
from ..utils import ensure_list, normalize_label, short_text
from .analyzer import AtomicQuestionAnalyzer
from .dependency_rewrite import resolve_dependency_question
from .models import (
    AtomicAnswerResult,
    AtomicQuestionAnalysis,
    AtomicQuestionNode,
    DagExecutionResult,
    FusedHyperedgeCandidate,
)
from .retriever import AtomicHyperedgeRetriever, LocalHyperedgeRetrievalResult


class DagCycleError(ValueError):
    pass


class AtomicDagExecutor:
    def __init__(
        self,
        analyzer: AtomicQuestionAnalyzer,
        retriever: AtomicHyperedgeRetriever,
        llm_service: AtomicLLMService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.retriever = retriever
        self.llm_service = llm_service
        self.logger = logger or logging.getLogger(__name__)

    def run(self, original_question: str, dag_payload: Any | None = None) -> DagExecutionResult:
        nodes = self.normalize_dag_payload(dag_payload, original_question=original_question)
        nodes, dag_repair = self.repair_dag_for_execution(nodes)
        order = self.topological_sort(nodes)
        self.validate_terminal_leaf(order)
        results_by_id: dict[str, AtomicAnswerResult] = {}
        analyses_artifact: list[dict[str, Any]] = []
        retrieval_artifact: list[dict[str, Any]] = []
        answer_artifact: list[dict[str, Any]] = []

        self.logger.info("Executing atomic DAG with %s node(s)", len(order))
        for node in order:
            dependency_answers = self._dependency_context(node.dependencies, results_by_id)
            dependency_rewrite = resolve_dependency_question(node.question, dependency_answers)
            resolved_question = dependency_rewrite.retrieval_question
            if dependency_rewrite.whether_rewritten:
                self.logger.info(
                    "Resolved dependency question for %s: %s -> %s",
                    node.node_id,
                    node.question,
                    resolved_question,
                )

            analysis = self.analyzer.analyze(
                resolved_question,
                self._analysis_dependency_context(dependency_answers),
            )
            primary_anchor_mention = _primary_anchor_mention(
                dependency_rewrite.primary_anchor_entities,
                analysis,
            )
            retrieval_result = self.retriever.retrieve_primary_anchor_local(
                question=resolved_question,
                analysis=analysis,
                primary_anchor_mention=primary_anchor_mention,
            )
            evidence = retrieval_result.evidence
            answer_payload = self._answer_atomic_question(
                atomic_question=resolved_question,
                analysis=analysis,
                evidence=evidence,
                dependency_answers=dependency_answers,
            )

            used_hyperedge_ids = self._used_hyperedge_ids(answer_payload, evidence)
            result = AtomicAnswerResult(
                node_id=node.node_id,
                question=resolved_question,
                analysis=analysis,
                evidence=evidence,
                answer=str(answer_payload.get("answer", "") or ""),
                confidence=max(0.0, min(1.0, float(answer_payload.get("confidence", 0.0) or 0.0))),
                reasoning_summary=str(answer_payload.get("reasoning_summary", "") or ""),
                used_dependencies=list(node.dependencies),
                used_hyperedge_ids=used_hyperedge_ids,
                insufficient=bool(answer_payload.get("insufficient", False)),
            )
            results_by_id[node.node_id] = result

            rewrite_payload = dependency_rewrite.to_dict()
            analyses_artifact.append(
                {
                    "node_id": node.node_id,
                    "question": resolved_question,
                    "original_question": node.question,
                    "resolved_question": resolved_question,
                    "retrieval_question": resolved_question,
                    "dependency_question_rewrite": rewrite_payload,
                    "dependency_replacements": rewrite_payload["dependency_replacements"],
                    "dependency_answers": dependency_answers,
                    "dependency_answers_used": dependency_rewrite.dependency_answers_used,
                    "unresolved_dependency": dependency_rewrite.unresolved_dependencies,
                    "primary_anchor_entities": dependency_rewrite.primary_anchor_entities,
                    "analysis": analysis.to_dict(),
                }
            )
            retrieval_record = self._retrieval_artifact(
                node=node,
                resolved_question=resolved_question,
                dependency_answers=dependency_answers,
                dependency_rewrite_payload=rewrite_payload,
                retrieval_result=retrieval_result,
                answer_payload=result.to_dict(),
            )
            retrieval_artifact.append(retrieval_record)
            answer_artifact.append(result.to_dict())

        atomic_results = [results_by_id[node.node_id] for node in order]
        final_answer = self._final_answer_from_terminal_node(atomic_results[-1], atomic_results)
        artifacts = {
            "dag_input": [node.to_dict() for node in nodes],
            "dag_repair": dag_repair,
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
    def repair_dag_for_execution(nodes: list[AtomicQuestionNode]) -> tuple[list[AtomicQuestionNode], dict[str, Any]]:
        if not nodes:
            return nodes, {"applied": False, "reason": "empty_dag"}

        dependents: dict[str, list[str]] = {node.node_id: [] for node in nodes}
        for node in nodes:
            for dependency in node.dependencies:
                if dependency in dependents:
                    dependents[dependency].append(node.node_id)

        leaf_ids = [node.node_id for node in nodes if not dependents[node.node_id]]
        terminal_id = nodes[-1].node_id
        if len(leaf_ids) <= 1:
            return nodes, {"applied": False, "leaf_ids": leaf_ids, "terminal_id": terminal_id}
        if terminal_id not in leaf_ids:
            return nodes, {
                "applied": False,
                "reason": "terminal_not_leaf",
                "leaf_ids": leaf_ids,
                "terminal_id": terminal_id,
            }

        extra_leaf_ids = [leaf_id for leaf_id in leaf_ids if leaf_id != terminal_id]
        repaired: list[AtomicQuestionNode] = []
        for node in nodes:
            if node.node_id != terminal_id:
                repaired.append(node)
                continue
            dependencies = list(node.dependencies)
            for leaf_id in extra_leaf_ids:
                if leaf_id not in dependencies:
                    dependencies.append(leaf_id)
            metadata = dict(node.metadata)
            metadata["dag_repair_added_dependencies"] = list(extra_leaf_ids)
            repaired.append(
                AtomicQuestionNode(
                    node_id=node.node_id,
                    question=node.question,
                    dependencies=dependencies,
                    metadata=metadata,
                )
            )
        return repaired, {
            "applied": True,
            "reason": "attached_extra_leaves_to_terminal",
            "terminal_id": terminal_id,
            "added_dependencies": extra_leaf_ids,
            "original_leaf_ids": leaf_ids,
        }

    @staticmethod
    def validate_terminal_leaf(nodes: list[AtomicQuestionNode]) -> None:
        if not nodes:
            raise ValueError("Atomic DAG did not contain any executable nodes.")

        dependents: dict[str, list[str]] = {node.node_id: [] for node in nodes}
        for node in nodes:
            for dependency in node.dependencies:
                dependents[dependency].append(node.node_id)

        leaf_ids = [node.node_id for node in nodes if not dependents[node.node_id]]
        if len(leaf_ids) != 1:
            raise ValueError(f"Atomic DAG must contain exactly one leaf node, found: {leaf_ids}")

        terminal_id = nodes[-1].node_id
        if leaf_ids[0] != terminal_id:
            raise ValueError(
                f"Atomic DAG terminal leaf must be the final topological node: leaf={leaf_ids[0]}, final={terminal_id}"
            )

        unreachable = [
            node.node_id
            for node in nodes
            if node.node_id != terminal_id and not _can_reach_terminal(node.node_id, terminal_id, dependents)
        ]
        if unreachable:
            raise ValueError(f"Every non-final atomic node must reach the terminal node: {unreachable}")

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
            metadata = _coerce_node_metadata(item)
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
                    "resolved_question": result.question,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "answer_type": result.analysis.answer_type,
                    "reasoning_summary": result.reasoning_summary,
                    "used_hyperedge_ids": list(result.used_hyperedge_ids),
                    "insufficient": result.insufficient,
                    "evidence_summary": [
                        {
                            "hyperedge_id": evidence.hyperedge_id,
                            "semantic_score": evidence.semantic_score,
                            "rank": evidence.rank,
                            "evidence_texts": evidence.evidence_texts[:2],
                        }
                        for evidence in result.evidence[:3]
                    ],
                }
            )
        return context

    def _answer_atomic_question(
        self,
        atomic_question: str,
        analysis: AtomicQuestionAnalysis,
        evidence: list[FusedHyperedgeCandidate],
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        answer_contract = self._answer_contract(atomic_question)
        answer_dependency_answers = self._answer_dependency_context(dependency_answers)
        evidence_payload = self._answer_evidence_payload(evidence)
        if self.llm_service is not None:
            payload = self.llm_service.answer_atomic_question(
                atomic_question=atomic_question,
                answer_contract=answer_contract,
                dependency_answers=answer_dependency_answers,
                evidence=evidence_payload,
            )
        else:
            payload = self._fallback_answer(atomic_question, analysis, evidence, dependency_answers)
        return self._coerce_answer_payload(payload, atomic_question, analysis, evidence, dependency_answers)

    @staticmethod
    def _analysis_dependency_context(dependency_answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in dependency_answers:
            compact.append(
                {
                    "node_id": str(item.get("node_id", "") or ""),
                    "question": str(item.get("question", "") or ""),
                    "resolved_question": str(item.get("resolved_question") or item.get("question", "") or ""),
                    "answer": str(item.get("answer", "") or ""),
                    "insufficient": bool(item.get("insufficient", False)),
                }
            )
        return compact

    @staticmethod
    def _answer_dependency_context(dependency_answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in dependency_answers:
            compact.append(
                {
                    "node_id": str(item.get("node_id", "") or ""),
                    "question": str(item.get("question", "") or ""),
                    "resolved_question": str(item.get("resolved_question") or item.get("question", "") or ""),
                    "answer": str(item.get("answer", "") or ""),
                    "insufficient": bool(item.get("insufficient", False)),
                }
            )
        return compact

    @staticmethod
    def _answer_evidence_payload(evidence: list[FusedHyperedgeCandidate]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        seen_chunk_texts: set[str] = set()
        for index, item in enumerate(evidence, start=1):
            evidence_id = f"E{index}"
            chunk_texts: list[str] = []
            for raw_text in item.chunk_texts:
                text = str(raw_text or "").strip()
                if not text or text in seen_chunk_texts:
                    continue
                seen_chunk_texts.add(text)
                chunk_texts.append(text)
            payload.append(
                {
                    "evidence_id": evidence_id,
                    "hyperedge_text": str(item.hyperedge_text or "").strip(),
                    "chunk_texts": chunk_texts,
                }
            )
        return payload

    @staticmethod
    def _answer_contract(question: str) -> dict[str, str]:
        output_format = _answer_output_format(question)
        return {"output_format": output_format} if output_format else {}

    def _fallback_answer(
        self,
        question: str,
        analysis: AtomicQuestionAnalysis,
        evidence: list[FusedHyperedgeCandidate],
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not evidence:
            dependency_payload = self._answer_from_dependencies(question, dependency_answers)
            if dependency_payload is not None:
                return dependency_payload
            return self._insufficient_answer_payload("no_local_evidence")

        query_entities = {normalize_label(entity).lower() for entity in analysis.entities}
        answer = ""
        for candidate in evidence:
            for entity_id in candidate.entity_ids:
                label = normalize_label(entity_id)
                if label and label.lower() not in query_entities and label.lower() not in normalize_label(question).lower():
                    answer = label
                    break
            if answer:
                break
        if not answer:
            answer = short_text(evidence[0].hyperedge_text, 180)

        return {
            "answer": answer or "INSUFFICIENT_EVIDENCE",
            "confidence": 0.7 if answer else 0.0,
            "reasoning_summary": short_text(" ".join(evidence[0].evidence_texts), 420),
            "used_hyperedge_ids": [evidence[0].hyperedge_id] if answer else [],
            "insufficient": not bool(answer),
        }

    @staticmethod
    def _answer_from_dependencies(question: str, dependency_answers: list[dict[str, Any]]) -> dict[str, Any] | None:
        usable = [
            item
            for item in dependency_answers
            if str(item.get("answer", "") or "").strip()
            and str(item.get("answer", "") or "").strip().upper() != "INSUFFICIENT_EVIDENCE"
            and not bool(item.get("insufficient", False))
        ]
        if not usable:
            return None

        lowered = question.lower()
        if lowered.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ")):
            if len(usable) >= 2:
                left = normalize_label(str(usable[-2].get("answer", ""))).lower()
                right = normalize_label(str(usable[-1].get("answer", ""))).lower()
                same_question = "same" in lowered or "share" in lowered or "both" in lowered
                different_question = "different" in lowered
                if same_question or different_question:
                    same = bool(left and right and left == right)
                    answer = "no" if same == different_question else "yes"
                    return {
                        "answer": answer,
                        "confidence": min(
                            0.85,
                            max(float(item.get("confidence", 0.0) or 0.0) for item in usable[-2:]),
                        ),
                        "reasoning_summary": "Answered from direct dependency answers.",
                        "used_hyperedge_ids": [],
                        "insufficient": False,
                    }
            if len(usable) == 1 and str(usable[-1].get("answer", "")).strip().lower() in {"yes", "no"}:
                return {
                    "answer": str(usable[-1]["answer"]).strip().lower(),
                    "confidence": max(0.0, min(0.85, float(usable[-1].get("confidence", 0.0) or 0.0))),
                    "reasoning_summary": "Mirrored the direct dependency yes/no answer.",
                    "used_hyperedge_ids": [],
                    "insufficient": False,
                }

        return None

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
        answer = str(payload.get("answer", "") or "").strip()
        if not answer:
            answer = "INSUFFICIENT_EVIDENCE"
        insufficient = answer.upper() == "INSUFFICIENT_EVIDENCE"
        coerced = {
            "answer": "INSUFFICIENT_EVIDENCE" if insufficient else answer,
            "confidence": 0.0 if insufficient else 1.0,
            "reasoning_summary": "",
            "used_hyperedge_ids": [],
            "insufficient": insufficient,
        }
        if insufficient:
            return coerced
        coerced["used_hyperedge_ids"] = _dedupe_strings(
            [
                str(item).strip()
                for item in ensure_list(payload.get("used_hyperedge_ids", []))
                if str(item).strip()
            ]
        )
        return coerced

    @staticmethod
    def _insufficient_answer_payload(reason: str) -> dict[str, Any]:
        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.0,
            "reasoning_summary": f"No local primary-anchor evidence was available: {reason}.",
            "used_hyperedge_ids": [],
            "insufficient": True,
        }

    @staticmethod
    def _used_hyperedge_ids(payload: dict[str, Any], evidence: list[FusedHyperedgeCandidate]) -> list[str]:
        evidence_ids = {item.hyperedge_id for item in evidence}
        used = [
            str(item).strip()
            for item in ensure_list(payload.get("used_hyperedge_ids", []))
            if str(item).strip() and str(item).strip() in evidence_ids
        ]
        if used:
            return _dedupe_strings(used)
        if bool(payload.get("insufficient", False)):
            return []
        return [item.hyperedge_id for item in evidence[:1]]

    @staticmethod
    def _retrieval_artifact(
        *,
        node: AtomicQuestionNode,
        resolved_question: str,
        dependency_answers: list[dict[str, Any]],
        dependency_rewrite_payload: dict[str, Any],
        retrieval_result: LocalHyperedgeRetrievalResult,
        answer_payload: dict[str, Any],
    ) -> dict[str, Any]:
        retrieval_payload = retrieval_result.to_artifact()
        return {
            "method": "two_hop_multi_anchor_topk",
            "node_id": node.node_id,
            "original_question": node.question,
            "resolved_question": resolved_question,
            "retrieval_question": resolved_question,
            "dependency_answers": dependency_answers,
            "dependency_question_rewrite": dependency_rewrite_payload,
            "dependency_replacements": dependency_rewrite_payload["dependency_replacements"],
            "dependency_answers_used": dependency_rewrite_payload["dependency_answers_used"],
            "unresolved_dependency": dependency_rewrite_payload["unresolved_dependencies"],
            "primary_anchor_mention": retrieval_payload["primary_anchor_mention"],
            "linked_entity_id": retrieval_payload["linked_entity_id"],
            "anchor_match": retrieval_payload["anchor_match"],
            "anchor_mentions": retrieval_payload.get("anchor_mentions", []),
            "linked_entities": retrieval_payload.get("linked_entities", []),
            "anchor_matches": retrieval_payload.get("anchor_matches", []),
            "unlinked_anchor_mentions": retrieval_payload.get("unlinked_anchor_mentions", []),
            "adjacent_hyperedge_ids": retrieval_payload["adjacent_hyperedge_ids"],
            "expansion_entity_ids": retrieval_payload["expansion_entity_ids"],
            "second_hop_hyperedge_ids": retrieval_payload["second_hop_hyperedge_ids"],
            "candidate_hyperedge_ids": retrieval_payload["candidate_hyperedge_ids"],
            "candidate_sources": retrieval_payload["candidate_sources"],
            "top_hyperedges": retrieval_payload["top_hyperedges"],
            "answerer_evidence": retrieval_payload["evidence"],
            "top_evidence": retrieval_payload["evidence"],
            "insufficient_reason": retrieval_payload["insufficient_reason"],
            "atomic_answer": answer_payload,
        }

    @staticmethod
    def _final_answer_from_terminal_node(
        terminal: AtomicAnswerResult,
        atomic_results: list[AtomicAnswerResult],
    ) -> dict[str, Any]:
        answer = terminal.answer
        return {
            "answer": answer,
            "candidate_answer": answer,
            "semantic_answer": answer,
            "judgment": _yes_no_judgment(answer),
            "confidence": terminal.confidence,
            "reasoning_summary": terminal.reasoning_summary,
            "source_node_id": terminal.node_id,
            "used_hyperedge_ids": list(terminal.used_hyperedge_ids),
            "insufficient": terminal.insufficient,
            "atomic_answer_trace": [
                {
                    "node_id": result.node_id,
                    "question": result.question,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "answer_type": result.analysis.answer_type,
                    "used_hyperedge_ids": list(result.used_hyperedge_ids),
                    "insufficient": result.insufficient,
                }
                for result in atomic_results
            ],
            "remaining_gaps": [
                result.node_id
                for result in atomic_results
                if result.insufficient or result.answer.strip().upper() == "INSUFFICIENT_EVIDENCE"
            ],
        }


def _primary_anchor_mention(primary_anchor_entities: Iterable[str], analysis: AtomicQuestionAnalysis) -> str:
    for entity in primary_anchor_entities:
        text = normalize_label(str(entity).strip())
        if text:
            return text
    for entity in analysis.entities:
        text = normalize_label(str(entity).strip())
        if text:
            return text
    return ""


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


def _coerce_node_metadata(item: dict[str, Any]) -> dict[str, Any]:
    reserved = {
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
    raw_metadata = item.get("metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    for key, value in item.items():
        if key in reserved or key == "metadata":
            continue
        metadata.setdefault(key, value)
    return metadata


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


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _can_reach_terminal(start_id: str, terminal_id: str, dependents: dict[str, list[str]]) -> bool:
    stack = list(dependents.get(start_id, []))
    seen: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id == terminal_id:
            return True
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(dependents.get(node_id, []))
    return False


def _answer_output_format(question: str) -> str:
    lowered = normalize_label(str(question or "")).lower()
    if _looks_like_candidate_selection(lowered):
        return "exact candidate surface from the question"
    if lowered.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ")):
        return "yes or no"
    if "how many" in lowered or lowered.startswith("number of ") or " count " in f" {lowered} ":
        return "number only"
    if lowered.startswith("what year"):
        return "year only"
    if lowered.startswith("when") or "date of" in lowered or "birthday" in lowered:
        return "supported date granularity"
    if "nationality" in lowered:
        return "full supported nationality expression"
    return "short answer only"


def _looks_like_candidate_selection(question: str) -> bool:
    if " or " not in question:
        return False
    selection_markers = (
        "which ",
        "who ",
        "born first",
        "born later",
        "born earlier",
        "died first",
        "died later",
        "released first",
        "released earlier",
        "came out first",
        "came out earlier",
        "older",
        "younger",
        "longer",
        "more recently",
    )
    return any(marker in question for marker in selection_markers)


def _yes_no_judgment(answer: str) -> str | None:
    normalized = str(answer or "").strip().lower()
    return normalized if normalized in {"yes", "no"} else None
