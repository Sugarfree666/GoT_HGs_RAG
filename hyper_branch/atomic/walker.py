from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..llm.service import AtomicLLMService
from ..utils import ensure_list, normalize_label, short_text
from .models import (
    AtomicQuestionAnalysis,
    AtomicWalkResult,
    HypergraphPathStep,
    HypergraphReasoningPath,
    PathLabel,
)
from .retriever import AtomicHyperedgeRetriever


MAX_HOPS = 2
_VALID_LABELS = {"ANSWER", "EXPAND", "DROP"}


class RoutedHypergraphWalker:
    """Execute a bounded two-hop, LLM-routed hypergraph walk for one atomic question."""

    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
        llm_service: AtomicLLMService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self.llm_service = llm_service
        self.logger = logger or logging.getLogger(__name__)
        self.walk_top_k = max(0, int(getattr(config, "walk_top_k", 5)))
        self._query_vector_cache: dict[str, Any] = {}
        self._anchor_resolver = AtomicHyperedgeRetriever(
            dataset=dataset,
            embedder=embedder,
            config=config,
            llm_service=llm_service,
            logger=self.logger,
        )

    def run_atomic_walk(
        self,
        atomic_question: str,
        analysis: AtomicQuestionAnalysis,
        dependency_answers: list[dict[str, Any]],
        *,
        node_id: str = "",
    ) -> AtomicWalkResult:
        """Run the two-hop state machine and return selected reasoning paths."""

        anchor_entities = self._resolve_anchor_entities(atomic_question, analysis)
        resolved_anchor_ids = _dedupe([str(anchor["entity_id"]) for anchor in anchor_entities])
        if not resolved_anchor_ids or self.walk_top_k <= 0:
            return AtomicWalkResult(
                selected_paths=[],
                evidence_mode="insufficient",
                answer_paths_found=False,
                insufficient=True,
                hop_artifacts=[],
                anchor_entities=anchor_entities,
                resolved_anchor_entity_ids=resolved_anchor_ids,
            )

        self.logger.info(
            "Atomic walk node=%s question=%s anchors=%s",
            node_id,
            short_text(atomic_question, 90),
            [anchor["label"] for anchor in anchor_entities],
        )

        frontier = [
            HypergraphReasoningPath(
                path_id=self._path_id([str(anchor["entity_id"])], []),
                anchor_entity_id=str(anchor["entity_id"]),
                entity_ids=[str(anchor["entity_id"])],
                hyperedge_ids=[],
                steps=[],
                hop_count=0,
            )
            for anchor in anchor_entities
        ]
        hop_artifacts: list[dict[str, Any]] = []

        for hop in range(1, MAX_HOPS + 1):
            candidate_paths, expansion_sources, adjacent_ids, semantic_top_k = self._expand_frontier(
                atomic_question=atomic_question,
                frontier=frontier,
                hop=hop,
            )
            candidate_paths = self._dedupe_paths(candidate_paths)
            hop_artifact: dict[str, Any] = {
                "hop": hop,
                "expansion_sources": expansion_sources,
                "adjacent_hyperedge_ids": adjacent_ids,
                "semantic_top_k": semantic_top_k,
                "candidate_paths": [self.path_payload(path) for path in candidate_paths],
                "router_raw_output": {},
                "router_labels": [],
                "answer_path_ids": [],
                "expand_path_ids": [],
                "drop_path_ids": [],
                "router_validation_errors": [],
            }

            self.logger.info(
                "Atomic walk node=%s hop=%s sources=%s adjacent=%s top_k=%s candidate_paths=%s",
                node_id,
                hop,
                len(expansion_sources),
                len(adjacent_ids),
                len(semantic_top_k),
                len(candidate_paths),
            )
            if not candidate_paths:
                hop_artifacts.append(hop_artifact)
                return AtomicWalkResult(
                    selected_paths=[],
                    evidence_mode="insufficient",
                    answer_paths_found=False,
                    insufficient=True,
                    hop_artifacts=hop_artifacts,
                    anchor_entities=anchor_entities,
                    resolved_anchor_entity_ids=resolved_anchor_ids,
                )

            raw_output = self._route_paths(atomic_question, dependency_answers, hop, candidate_paths)
            routed_paths, labels, validation_errors = self._apply_router_labels(candidate_paths, raw_output)
            hop_artifact["router_raw_output"] = raw_output if isinstance(raw_output, dict) else {"raw": raw_output}
            hop_artifact["router_labels"] = labels
            hop_artifact["router_validation_errors"] = validation_errors
            answer_paths = [path for path in routed_paths if path.label == "ANSWER"]
            expand_paths = [path for path in routed_paths if path.label == "EXPAND"]
            drop_paths = [path for path in routed_paths if path.label == "DROP"]
            hop_artifact["answer_path_ids"] = [path.path_id for path in answer_paths]
            hop_artifact["expand_path_ids"] = [path.path_id for path in expand_paths]
            hop_artifact["drop_path_ids"] = [path.path_id for path in drop_paths]
            hop_artifacts.append(hop_artifact)

            self.logger.info(
                "Atomic walk node=%s hop=%s labels ANSWER=%s EXPAND=%s DROP=%s",
                node_id,
                hop,
                len(answer_paths),
                len(expand_paths),
                len(drop_paths),
            )

            if answer_paths:
                self.logger.info("Atomic walk node=%s stopped with routed_answer at hop=%s", node_id, hop)
                return AtomicWalkResult(
                    selected_paths=answer_paths,
                    evidence_mode="routed_answer",
                    answer_paths_found=True,
                    insufficient=False,
                    hop_artifacts=hop_artifacts,
                    anchor_entities=anchor_entities,
                    resolved_anchor_entity_ids=resolved_anchor_ids,
                )

            if not expand_paths:
                return AtomicWalkResult(
                    selected_paths=[],
                    evidence_mode="insufficient",
                    answer_paths_found=False,
                    insufficient=True,
                    hop_artifacts=hop_artifacts,
                    anchor_entities=anchor_entities,
                    resolved_anchor_entity_ids=resolved_anchor_ids,
                )

            if hop == MAX_HOPS:
                self.logger.info("Atomic walk node=%s using second-hop EXPAND fallback", node_id)
                return AtomicWalkResult(
                    selected_paths=expand_paths,
                    evidence_mode="second_hop_expand_fallback",
                    answer_paths_found=False,
                    insufficient=False,
                    hop_artifacts=hop_artifacts,
                    anchor_entities=anchor_entities,
                    resolved_anchor_entity_ids=resolved_anchor_ids,
                )
            frontier = expand_paths

        return AtomicWalkResult(
            selected_paths=[],
            evidence_mode="insufficient",
            answer_paths_found=False,
            insufficient=True,
            hop_artifacts=hop_artifacts,
            anchor_entities=anchor_entities,
            resolved_anchor_entity_ids=resolved_anchor_ids,
        )

    def local_semantic_top_hyperedges(
        self,
        query: str,
        current_entity_id: str,
        *,
        exclude_hyperedge_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return semantic top-k hyperedges from the current entity adjacency only."""

        exclude_hyperedge_ids = exclude_hyperedge_ids or set()
        adjacent_ids = self._adjacent_hyperedge_ids(current_entity_id, exclude_hyperedge_ids)
        if not adjacent_ids or self.walk_top_k <= 0:
            return []
        scores = self._hyperedge_similarity_scores(query, adjacent_ids)
        ranked = [
            {
                "current_entity_id": current_entity_id,
                "hyperedge_id": hyperedge_id,
                "semantic_score": float(scores.get(hyperedge_id, 0.0)),
                "rank": 0,
            }
            for hyperedge_id in adjacent_ids
        ]
        ranked.sort(key=lambda item: (-float(item["semantic_score"]), str(item["hyperedge_id"])))
        selected = ranked[: self.walk_top_k]
        for index, item in enumerate(selected, start=1):
            item["rank"] = index
        return selected

    def path_payload(self, path: HypergraphReasoningPath) -> dict[str, Any]:
        payload = path.to_dict()
        payload["entity_path"] = [self._entity_payload(entity_id) for entity_id in path.entity_ids]
        payload["hyperedges"] = [
            {
                "hyperedge_id": step.hyperedge_id,
                "hyperedge_text": step.hyperedge_text,
                "from_entity_id": step.from_entity_id,
                "to_entity_id": step.to_entity_id,
                "semantic_score": step.semantic_score,
                "semantic_rank": step.semantic_rank,
                "entity_ids": list(step.entity_ids),
                "chunk_ids": list(step.chunk_ids),
                "chunk_texts": list(step.chunk_texts),
            }
            for step in path.steps
        ]
        payload["current_tail_entity"] = self._entity_payload(path.tail_entity_id)
        return payload

    def _resolve_anchor_entities(self, question: str, analysis: AtomicQuestionAnalysis) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()
        for query_index, entity in enumerate(analysis.entities):
            mention = normalize_label(str(entity).strip())
            if not mention:
                continue
            matches = self._anchor_resolver._resolve_anchor_entity_matches(
                question=question,
                entity=mention,
                analysis=analysis,
                query_index=query_index,
            )
            for match in matches:
                key = (query_index, match.entity_id)
                if key in seen:
                    continue
                seen.add(key)
                anchors.append(
                    {
                        "query_index": query_index,
                        "mention": mention,
                        "entity_id": match.entity_id,
                        "label": normalize_label(match.entity_id),
                        "match_type": match.match_type,
                        "link_score": match.link_score,
                        "vector_score": match.vector_score,
                        "llm_confidence": match.llm_confidence,
                    }
                )
        return anchors

    def _expand_frontier(
        self,
        *,
        atomic_question: str,
        frontier: list[HypergraphReasoningPath],
        hop: int,
    ) -> tuple[list[HypergraphReasoningPath], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        candidate_paths: list[HypergraphReasoningPath] = []
        expansion_sources: list[dict[str, Any]] = []
        adjacent_hyperedge_ids: list[str] = []
        semantic_top_k: list[dict[str, Any]] = []

        for path in frontier:
            current_entity_id = path.tail_entity_id
            query = atomic_question if hop == 1 else self._path_conditioned_query(atomic_question, path)
            adjacent = self._adjacent_hyperedge_ids(current_entity_id, set(path.hyperedge_ids))
            _append_unique(adjacent_hyperedge_ids, adjacent)
            top_hyperedges = self.local_semantic_top_hyperedges(
                query,
                current_entity_id,
                exclude_hyperedge_ids=set(path.hyperedge_ids),
            )
            semantic_top_k.extend(top_hyperedges)
            expansion_sources.append(
                {
                    "path_id": path.path_id,
                    "current_entity_id": current_entity_id,
                    "current_entity": normalize_label(current_entity_id),
                    "query": query,
                    "adjacent_hyperedge_count": len(adjacent),
                    "semantic_top_k_count": len(top_hyperedges),
                }
            )
            for hyperedge_record in top_hyperedges:
                hyperedge_id = str(hyperedge_record["hyperedge_id"])
                for tail_entity_id in self._tail_entity_ids(hyperedge_id, current_entity_id, path.entity_ids):
                    step = self._path_step(
                        from_entity_id=current_entity_id,
                        hyperedge_id=hyperedge_id,
                        to_entity_id=tail_entity_id,
                        semantic_score=float(hyperedge_record["semantic_score"]),
                        semantic_rank=int(hyperedge_record["rank"]),
                    )
                    new_entity_ids = [*path.entity_ids, tail_entity_id]
                    new_hyperedge_ids = [*path.hyperedge_ids, hyperedge_id]
                    candidate_paths.append(
                        HypergraphReasoningPath(
                            path_id=self._path_id(new_entity_ids, new_hyperedge_ids),
                            anchor_entity_id=path.anchor_entity_id,
                            entity_ids=new_entity_ids,
                            hyperedge_ids=new_hyperedge_ids,
                            steps=[*path.steps, step],
                            hop_count=hop,
                        )
                    )
        return candidate_paths, expansion_sources, adjacent_hyperedge_ids, semantic_top_k

    def _route_paths(
        self,
        atomic_question: str,
        dependency_answers: list[dict[str, Any]],
        hop: int,
        candidate_paths: list[HypergraphReasoningPath],
    ) -> dict[str, Any]:
        candidate_payload = [self.path_payload(path) for path in candidate_paths]
        if self.llm_service is None or not hasattr(self.llm_service, "route_reasoning_paths"):
            return self._fallback_route(atomic_question, hop, candidate_payload)
        try:
            response = self.llm_service.route_reasoning_paths(
                atomic_question=atomic_question,
                dependency_answers=dependency_answers,
                hop=hop,
                candidate_paths=candidate_payload,
            )
            return response if isinstance(response, dict) else {"labels": []}
        except Exception as exc:  # pragma: no cover - defensive guard for external LLM failures
            self.logger.warning("Atomic path router failed at hop=%s: %s", hop, exc)
            return {"labels": []}

    def _apply_router_labels(
        self,
        candidate_paths: list[HypergraphReasoningPath],
        raw_output: dict[str, Any],
    ) -> tuple[list[HypergraphReasoningPath], list[dict[str, Any]], list[dict[str, Any]]]:
        candidates_by_id = {path.path_id: path for path in candidate_paths}
        assigned: dict[str, HypergraphReasoningPath] = {}
        labels_artifact: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        raw_labels = raw_output.get("labels", []) if isinstance(raw_output, dict) else []
        for index, item in enumerate(ensure_list(raw_labels)):
            if not isinstance(item, dict):
                errors.append({"index": index, "error": "label_record_not_object"})
                continue
            path_id = str(item.get("path_id", "") or "").strip()
            if path_id not in candidates_by_id:
                errors.append({"index": index, "path_id": path_id, "error": "unknown_path_id"})
                continue
            if path_id in assigned:
                errors.append({"index": index, "path_id": path_id, "error": "duplicate_label_ignored"})
                continue
            label = str(item.get("label", "") or "").strip().upper()
            if label not in _VALID_LABELS:
                errors.append({"index": index, "path_id": path_id, "label": label, "error": "invalid_label"})
                continue
            answer_entity_ids = [str(value).strip() for value in ensure_list(item.get("answer_entity_ids", [])) if str(value).strip()]
            path_entity_set = set(candidates_by_id[path_id].entity_ids)
            if label == "ANSWER":
                invalid_entities = [entity_id for entity_id in answer_entity_ids if entity_id not in path_entity_set]
                if invalid_entities:
                    errors.append(
                        {
                            "index": index,
                            "path_id": path_id,
                            "answer_entity_ids": answer_entity_ids,
                            "error": "answer_entity_outside_path",
                        }
                    )
                    continue
            elif answer_entity_ids:
                errors.append(
                    {
                        "index": index,
                        "path_id": path_id,
                        "answer_entity_ids": answer_entity_ids,
                        "error": "non_answer_label_has_answer_entities",
                    }
                )
                answer_entity_ids = []
            routed = replace(
                candidates_by_id[path_id],
                label=label,  # type: ignore[arg-type]
                label_reason=short_text(str(item.get("reason", "") or ""), 300),
                answer_entity_ids=answer_entity_ids if label == "ANSWER" else [],
            )
            assigned[path_id] = routed
            labels_artifact.append(
                {
                    "path_id": path_id,
                    "label": label,
                    "answer_entity_ids": list(routed.answer_entity_ids),
                    "reason": routed.label_reason,
                }
            )

        for path in candidate_paths:
            if path.path_id in assigned:
                continue
            errors.append({"path_id": path.path_id, "error": "missing_or_invalid_label_fallback_expand"})
            routed = replace(path, label="EXPAND", label_reason="Router output missing or invalid; conservative EXPAND fallback.")
            assigned[path.path_id] = routed
            labels_artifact.append(
                {
                    "path_id": path.path_id,
                    "label": "EXPAND",
                    "answer_entity_ids": [],
                    "reason": routed.label_reason,
                }
            )

        routed_paths = [assigned[path.path_id] for path in candidate_paths]
        return routed_paths, labels_artifact, errors

    def _fallback_route(self, atomic_question: str, hop: int, candidate_paths: list[dict[str, Any]]) -> dict[str, Any]:
        del atomic_question, hop
        labels: list[dict[str, Any]] = []
        for path in candidate_paths:
            label = "EXPAND"
            labels.append(
                {
                    "path_id": str(path.get("path_id", "")),
                    "label": label,
                    "answer_entity_ids": [],
                    "reason": "Programmatic conservative routing fallback.",
                }
            )
        return {"labels": labels}

    def _adjacent_hyperedge_ids(self, current_entity_id: str, exclude_hyperedge_ids: set[str]) -> list[str]:
        if not current_entity_id or not hasattr(self.dataset.graph, "entity_hyperedge_ids"):
            return []
        seen: set[str] = set()
        result: list[str] = []
        for hyperedge_id in self.dataset.graph.entity_hyperedge_ids(current_entity_id):
            text = str(hyperedge_id)
            if not text or text in exclude_hyperedge_ids or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _hyperedge_similarity_scores(self, query: str, hyperedge_ids: list[str]) -> dict[str, float]:
        if not query.strip() or not hyperedge_ids:
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        store = getattr(self.dataset, "hyperedge_store", None)
        if store is None or not hasattr(store, "similarities"):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        try:
            if query in self._query_vector_cache:
                query_vector = self._query_vector_cache[query]
            else:
                vectors = self.embedder.embed_texts([query], stage="atomic_walk_local_hyperedge_retrieval")
                if not vectors:
                    return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
                query_vector = np.asarray(vectors[0], dtype=np.float32)
                self._query_vector_cache[query] = query_vector
            scores = dict(store.similarities(query_vector, list(hyperedge_ids)))
        except (TypeError, ValueError, AttributeError):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        return {hyperedge_id: float(scores.get(hyperedge_id, 0.0)) for hyperedge_id in hyperedge_ids}

    def _tail_entity_ids(
        self,
        hyperedge_id: str,
        current_entity_id: str,
        visited_entity_ids: list[str],
    ) -> list[str]:
        if not hasattr(self.dataset.graph, "hyperedge_entity_ids"):
            return []
        visited = set(visited_entity_ids)
        tail_ids: list[str] = []
        for entity_id in self.dataset.graph.hyperedge_entity_ids(hyperedge_id):
            text = str(entity_id)
            if not text or text == current_entity_id or text in visited:
                continue
            if text not in tail_ids:
                tail_ids.append(text)
        return tail_ids

    def _path_step(
        self,
        *,
        from_entity_id: str,
        hyperedge_id: str,
        to_entity_id: str,
        semantic_score: float,
        semantic_rank: int,
    ) -> HypergraphPathStep:
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        chunk_ids = [str(item) for item in description.get("chunk_ids", []) if str(item)]
        chunk_pairs = _dedupe_chunk_pairs(
            [
                (chunk_id, short_text(self.dataset.get_chunk_text(chunk_id), 900))
                for chunk_id in chunk_ids
            ]
        )
        return HypergraphPathStep(
            from_entity_id=from_entity_id,
            hyperedge_id=hyperedge_id,
            hyperedge_text=normalize_label(str(description.get("hyperedge_text") or hyperedge_id)),
            to_entity_id=to_entity_id,
            semantic_score=float(semantic_score),
            semantic_rank=int(semantic_rank),
            entity_ids=[str(item) for item in description.get("entity_ids", [])],
            chunk_ids=[chunk_id for chunk_id, _ in chunk_pairs],
            chunk_texts=[text for _, text in chunk_pairs],
        )

    def _path_conditioned_query(self, atomic_question: str, path: HypergraphReasoningPath) -> str:
        known_path_parts: list[str] = []
        for index, entity_id in enumerate(path.entity_ids):
            if index > 0:
                step = path.steps[index - 1]
                known_path_parts.append(f"[{step.hyperedge_text}]")
            known_path_parts.append(normalize_label(entity_id))
        return "\n".join(
            [
                "Original atomic question:",
                atomic_question,
                "",
                "Known path:",
                " -> ".join(known_path_parts),
                "",
                "Current entity:",
                normalize_label(path.tail_entity_id),
            ]
        )

    def _entity_payload(self, entity_id: str) -> dict[str, Any]:
        nodes = getattr(self.dataset.graph, "nodes", {})
        node = nodes.get(entity_id) if hasattr(nodes, "get") else None
        return {
            "entity_id": entity_id,
            "label": normalize_label(entity_id),
            "entity_type": getattr(node, "entity_type", None),
            "description": short_text(str(getattr(node, "description", "") or ""), 500),
        }

    def _path_id(self, entity_ids: list[str], hyperedge_ids: list[str]) -> str:
        signature = "|".join([*entity_ids, "=>", *hyperedge_ids])
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:14]
        return f"p{len(hyperedge_ids)}_{digest}"

    def _dedupe_paths(self, paths: list[HypergraphReasoningPath]) -> list[HypergraphReasoningPath]:
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        result: list[HypergraphReasoningPath] = []
        for path in paths:
            signature = (tuple(path.entity_ids), tuple(path.hyperedge_ids))
            if signature in seen:
                continue
            seen.add(signature)
            result.append(path)
        result.sort(key=lambda path: (path.hop_count, path.hyperedge_ids, path.entity_ids, path.path_id))
        return result

    def _path_payload_text(self, path: dict[str, Any]) -> str:
        texts: list[str] = []
        texts.extend(str(item) for item in ensure_list(path.get("entity_ids", [])))
        for step in ensure_list(path.get("steps", [])):
            if not isinstance(step, dict):
                continue
            texts.append(str(step.get("hyperedge_text", "")))
            texts.extend(str(item) for item in ensure_list(step.get("chunk_texts", [])))
        return " ".join(text for text in texts if text)


def _append_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        text = str(value).strip()
        if text and text not in target:
            target.append(text)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _dedupe_chunk_pairs(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk_id, chunk_text in values:
        text_id = str(chunk_id).strip()
        if not text_id or text_id in seen:
            continue
        seen.add(text_id)
        result.append((text_id, str(chunk_text or "")))
    return result
