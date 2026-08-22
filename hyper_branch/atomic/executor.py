"""Execute a validated DEPO atomic-question DAG over local hypergraph evidence."""

from __future__ import annotations

from typing import Any, Iterable

from ..llm.service import AtomicLLMService
from ..utils import normalize_label
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
        llm_service: AtomicLLMService,
    ) -> None:
        self.analyzer = analyzer
        self.retriever = retriever
        self.llm_service = llm_service

    def run(
        self,
        original_question: str,
        dag_payload: dict[str, Any],
        original_question_entities: list[str] | None = None,
    ) -> DagExecutionResult:
        nodes = self.normalize_dag_payload(dag_payload)
        order = self.topological_sort(nodes)
        self.validate_terminal_leaf(order)

        original_analysis = self._original_question_analysis(
            original_question,
            original_question_entities,
        )
        shared_pool = self.retriever.build_original_question_candidate_pool(
            question=original_question,
            analysis=original_analysis,
            primary_anchor_mention=_primary_anchor_mention([], original_analysis),
        )
        results_by_id: dict[str, AtomicAnswerResult] = {}
        local_pools: dict[str, LocalHyperedgeRetrievalResult] = {}

        dependencies_by_id = {node.node_id: node.dependencies for node in order}
        ordered_ids = [node.node_id for node in order]
        for node in order:
            dependency_answers = self._dependency_context(node.dependencies, results_by_id)
            rewrite = resolve_dependency_question(node.question, dependency_answers)
            question = rewrite.retrieval_question
            analysis = self.analyzer.analyze(question, self._compact_dependencies(dependency_answers))
            local_pool = self.retriever.build_atomic_candidate_pool(
                question=question,
                analysis=analysis,
                primary_anchor_mention=_primary_anchor_mention(rewrite.primary_anchor_entities, analysis),
            )
            ancestor_ids = self._transitive_ancestor_node_ids(
                node.node_id,
                dependencies_by_id=dependencies_by_id,
                topological_node_ids=ordered_ids,
            )
            active_shared_pool = self._active_shared_candidate_pool(shared_pool, ancestor_ids, local_pools)
            retrieval = self.retriever.merge_candidate_pools(
                shared_pool=active_shared_pool,
                local_pool=local_pool,
            )
            retrieval = self.retriever.rank_candidate_pool(retrieval, question=question)
            answer_payload = self._answer_atomic_question(
                original_question=original_question,
                atomic_question=question,
                evidence=retrieval.evidence,
                dependency_answers=dependency_answers,
            )
            result = AtomicAnswerResult(
                node_id=node.node_id,
                question=question,
                answer=answer_payload["answer"],
                used_dependencies=list(node.dependencies),
                used_hyperedge_ids=self._used_hyperedge_ids(answer_payload, retrieval.evidence),
                insufficient=answer_payload["insufficient"],
            )
            results_by_id[node.node_id] = result
            local_pools[node.node_id] = local_pool

        atomic_results = [results_by_id[node.node_id] for node in order]
        final_answer = self._final_answer(atomic_results[-1], atomic_results)
        return DagExecutionResult(
            atomic_results=atomic_results,
            final_answer=final_answer,
        )

    @staticmethod
    def normalize_dag_payload(payload: dict[str, Any]) -> list[AtomicQuestionNode]:
        raw_nodes = payload["nodes"]
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError("Atomic DAG requires a non-empty nodes list.")
        nodes: list[AtomicQuestionNode] = []
        for item in raw_nodes:
            node_id = str(item["id"]).strip()
            question = str(item["question"]).strip()
            dependencies = [str(dependency).strip() for dependency in item["depends_on"]]
            if not node_id or not question:
                raise ValueError("Atomic DAG nodes require non-empty id and question fields.")
            nodes.append(
                AtomicQuestionNode(
                    node_id=node_id,
                    question=question,
                    dependencies=dependencies,
                    metadata={"operation": str(item["operation"])},
                )
            )
        return nodes

    @staticmethod
    def topological_sort(nodes: list[AtomicQuestionNode]) -> list[AtomicQuestionNode]:
        by_id = {node.node_id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise ValueError("Atomic DAG contains duplicate node IDs.")
        unknown = sorted(
            dependency
            for node in nodes
            for dependency in node.dependencies
            if dependency not in by_id
        )
        if unknown:
            raise ValueError(f"Atomic DAG contains unknown dependencies: {unknown}")

        dependents = {node.node_id: [] for node in nodes}
        indegree = {node.node_id: len(node.dependencies) for node in nodes}
        for node in nodes:
            for dependency in node.dependencies:
                dependents[dependency].append(node.node_id)
        ready = [node.node_id for node in nodes if not indegree[node.node_id]]
        order: list[AtomicQuestionNode] = []
        while ready:
            node_id = ready.pop(0)
            order.append(by_id[node_id])
            for dependent_id in dependents[node_id]:
                indegree[dependent_id] -= 1
                if not indegree[dependent_id]:
                    ready.append(dependent_id)
        if len(order) != len(nodes):
            raise DagCycleError("Atomic DAG contains a cycle.")
        return order

    @staticmethod
    def validate_terminal_leaf(nodes: list[AtomicQuestionNode]) -> None:
        dependents = {node.node_id: [] for node in nodes}
        for node in nodes:
            for dependency in node.dependencies:
                dependents[dependency].append(node.node_id)
        leaves = [node.node_id for node in nodes if not dependents[node.node_id]]
        if len(leaves) != 1 or leaves[0] != nodes[-1].node_id:
            raise ValueError("Atomic DAG must have one final terminal leaf.")
        terminal_id = leaves[0]
        disconnected = [
            node.node_id
            for node in nodes
            if node.node_id != terminal_id and not _can_reach_terminal(node.node_id, terminal_id, dependents)
        ]
        if disconnected:
            raise ValueError(f"Atomic DAG nodes must reach the terminal node: {disconnected}")

    def _original_question_analysis(
        self,
        question: str,
        original_question_entities: list[str] | None,
    ) -> AtomicQuestionAnalysis:
        entities = _clean_entity_mentions(original_question_entities or [])
        if entities:
            return AtomicQuestionAnalysis(entities=entities)
        return self.analyzer.analyze(question, [])

    @staticmethod
    def _dependency_context(
        dependency_ids: list[str],
        results_by_id: dict[str, AtomicAnswerResult],
    ) -> list[dict[str, Any]]:
        return [
            {
                "node_id": result.node_id,
                "question": result.question,
                "answer": result.answer,
                "insufficient": result.insufficient,
            }
            for dependency_id in dependency_ids
            for result in [results_by_id[dependency_id]]
        ]

    @staticmethod
    def _compact_dependencies(dependencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "node_id": item["node_id"],
                "question": item["question"],
                "answer": item["answer"],
                "insufficient": item["insufficient"],
            }
            for item in dependencies
        ]

    def _answer_atomic_question(
        self,
        *,
        original_question: str,
        atomic_question: str,
        evidence: list[FusedHyperedgeCandidate],
        dependency_answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = self.llm_service.answer_atomic_question(
            atomic_question=atomic_question,
            answer_contract=self._answer_contract(atomic_question),
            dependency_answers=self._compact_dependencies(dependency_answers),
            evidence=self._answer_evidence_payload(evidence),
            original_question=original_question,
        )
        answer = str(payload["answer"]).strip()
        return {
            "answer": answer or "INSUFFICIENT_EVIDENCE",
            "insufficient": answer.upper() == "INSUFFICIENT_EVIDENCE" or not answer,
        }

    @staticmethod
    def _answer_contract(question: str) -> dict[str, str]:
        lowered = question.strip().lower()
        if lowered.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ")):
            return {"output_format": "yes or no"}
        if lowered.startswith("what year"):
            return {"output_format": "year only"}
        if lowered.startswith(("when ", "date of")):
            return {"output_format": "supported date"}
        return {"output_format": "short answer"}

    def _answer_evidence_payload(
        self,
        evidence: list[FusedHyperedgeCandidate],
    ) -> dict[str, list[dict[str, Any]]]:
        blocks: list[dict[str, Any]] = []
        by_chunk_id: dict[str, dict[str, Any]] = {}
        for rank, candidate in enumerate(evidence, start=1):
            hyperedge = {"hyperedge_id": f"H{rank}", "hyperedge_text": candidate.hyperedge_text}
            first_hop_ids = candidate.score_breakdown.get("via_first_hyperedge_ids", [])
            if first_hop_ids:
                hyperedge["first_hop_hyperedge_text"] = self.retriever.dataset.graph.describe_hyperedge(
                    first_hop_ids[0]
                )["hyperedge_text"]
            chunks = _candidate_chunks(candidate) or [(f"__{rank}", "")]
            for chunk_id, text in chunks:
                block = by_chunk_id.get(chunk_id)
                if block is None:
                    title, _, body = text.partition("\n")
                    block = {
                        "chunk_id": f"C{len(blocks) + 1}",
                        "title": title.strip(),
                        "text": body.strip() or title.strip(),
                        "hyperedges": [],
                    }
                    by_chunk_id[chunk_id] = block
                    blocks.append(block)
                block["hyperedges"].append(hyperedge)
        return {"evidence_blocks": blocks}

    @staticmethod
    def _used_hyperedge_ids(
        answer_payload: dict[str, Any],
        evidence: list[FusedHyperedgeCandidate],
    ) -> list[str]:
        if answer_payload["insufficient"] or not evidence:
            return []
        return [evidence[0].hyperedge_id]

    def _active_shared_candidate_pool(
        self,
        original_pool: LocalHyperedgeRetrievalResult,
        ancestor_ids: list[str],
        local_pools: dict[str, LocalHyperedgeRetrievalResult],
    ) -> LocalHyperedgeRetrievalResult:
        pool = original_pool
        for node_id in ancestor_ids:
            pool = self.retriever.merge_candidate_pools(
                shared_pool=pool,
                local_pool=local_pools[node_id],
            )
        return pool

    @staticmethod
    def _transitive_ancestor_node_ids(
        node_id: str,
        *,
        dependencies_by_id: dict[str, list[str]],
        topological_node_ids: list[str],
    ) -> list[str]:
        ancestors: set[str] = set()

        def visit(current_id: str) -> None:
            for dependency in dependencies_by_id[current_id]:
                if dependency not in ancestors:
                    ancestors.add(dependency)
                    visit(dependency)

        visit(node_id)
        positions = {node_id: index for index, node_id in enumerate(topological_node_ids)}
        return sorted(ancestors, key=positions.__getitem__)

    @staticmethod
    def _final_answer(
        terminal: AtomicAnswerResult,
        results: list[AtomicAnswerResult],
    ) -> dict[str, Any]:
        return {
            "answer": terminal.answer,
            "source_node_id": terminal.node_id,
            "used_hyperedge_ids": terminal.used_hyperedge_ids,
            "insufficient": terminal.insufficient,
            "atomic_answer_trace": [
                {
                    "node_id": result.node_id,
                    "answer": result.answer,
                    "used_hyperedge_ids": result.used_hyperedge_ids,
                    "insufficient": result.insufficient,
                }
                for result in results
            ],
        }


def _primary_anchor_mention(primary_entities: Iterable[str], analysis: AtomicQuestionAnalysis) -> str:
    for entity in [*primary_entities, *analysis.entities]:
        text = normalize_label(str(entity).strip())
        if text:
            return text
    return ""


def _candidate_chunks(candidate: FusedHyperedgeCandidate) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for index, chunk_id in enumerate(candidate.chunk_ids):
        text = candidate.chunk_texts[index] if index < len(candidate.chunk_texts) else ""
        if chunk_id not in [item[0] for item in chunks]:
            chunks.append((chunk_id, text))
    return chunks


def _clean_entity_mentions(values: Iterable[str]) -> list[str]:
    entities: list[str] = []
    for value in values:
        entity = normalize_label(str(value).strip())
        if entity and entity.lower() not in {item.lower() for item in entities}:
            entities.append(entity)
    return entities


def _can_reach_terminal(start_id: str, terminal_id: str, dependents: dict[str, list[str]]) -> bool:
    pending = list(dependents[start_id])
    seen: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id == terminal_id:
            return True
        if node_id not in seen:
            seen.add(node_id)
            pending.extend(dependents[node_id])
    return False
