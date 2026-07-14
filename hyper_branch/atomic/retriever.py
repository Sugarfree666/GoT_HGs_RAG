from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..llm.service import AtomicLLMService
from ..models import VectorMatch
from ..utils import normalize_label, short_text
from .models import AtomicQuestionAnalysis, EvidencePathCandidate, FusedHyperedgeCandidate


_ANCHOR_ENTITY_TOP_K = 30
_ANCHOR_ENTITY_LLM_MIN_CONFIDENCE = 0.6
_ANCHOR_ENTITY_AUTO_ACCEPT_SCORE = 0.98
_ANCHOR_ENTITY_CANDIDATE_LIMIT = 40
_CHUNK_MENTION_ENTITY_LIMIT = 20
_LOCAL_RETRIEVAL_METHOD = "two_hop_multi_anchor_topk"
_SHARED_RETRIEVAL_METHOD = "shared_original_question_augmented_topk"
_CHUNK_ENTITY_EXCLUDED_TYPES = {
    "CATEGORY",
    "CONCEPT",
    "CONDITION",
    "DATE",
    "NUMBER",
    "RELATION",
    "ROLE",
    "TITLE",
    "TYPE",
}
_CHUNK_ENTITY_GENERIC_LABELS = {
    "ACTOR",
    "ACTRESS",
    "ALBUM",
    "ARTIST",
    "AUTHOR",
    "BAND",
    "CHILD",
    "CITY",
    "COMPANY",
    "COMPOSER",
    "COUNTRY",
    "DAUGHTER",
    "DIRECTOR",
    "FATHER",
    "FILM",
    "HUSBAND",
    "LOCATION",
    "MAN",
    "MOTHER",
    "ORGANIZATION",
    "PEOPLE",
    "PERSON",
    "PLACE",
    "PRODUCER",
    "SINGER",
    "SON",
    "SONG",
    "SONGWRITER",
    "SPOUSE",
    "UNIVERSITY",
    "WIFE",
    "WOMAN",
    "WORK",
    "WRITER",
}


@dataclass(slots=True)
class AnchorEntityMatch:
    query_index: int
    query_entity: str
    entity_id: str
    match_type: str
    link_score: float
    vector_score: float | None = None
    llm_confidence: float | None = None
    candidate_rank: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "query_entity": self.query_entity,
            "matched_entity": normalize_label(self.entity_id),
            "matched_entity_id": self.entity_id,
            "match_type": self.match_type,
            "link_score": self.link_score,
            "vector_score": self.vector_score,
            "llm_confidence": self.llm_confidence,
            "candidate_rank": self.candidate_rank,
        }


@dataclass(slots=True)
class LocalHyperedgeRetrievalResult:
    method: str = _LOCAL_RETRIEVAL_METHOD
    primary_anchor_mention: str = ""
    linked_entity_id: str = ""
    anchor_match: dict[str, Any] = field(default_factory=dict)
    anchor_mentions: list[str] = field(default_factory=list)
    linked_entities: list[dict[str, Any]] = field(default_factory=list)
    anchor_matches: list[dict[str, Any]] = field(default_factory=list)
    unlinked_anchor_mentions: list[str] = field(default_factory=list)
    adjacent_hyperedge_ids: list[str] = field(default_factory=list)
    expansion_entity_ids: list[str] = field(default_factory=list)
    second_hop_hyperedge_ids: list[str] = field(default_factory=list)
    candidate_hyperedge_ids: list[str] = field(default_factory=list)
    shared_candidate_hyperedge_ids: list[str] = field(default_factory=list)
    local_candidate_hyperedge_ids: list[str] = field(default_factory=list)
    candidate_sources: list[dict[str, Any]] = field(default_factory=list)
    top_hyperedges: list[dict[str, Any]] = field(default_factory=list)
    candidate_paths: list[EvidencePathCandidate] = field(default_factory=list)
    top_paths: list[EvidencePathCandidate] = field(default_factory=list)
    evidence: list[EvidencePathCandidate] = field(default_factory=list)
    candidate_path_count: int = 0
    deduplicated_path_count: int = 0
    selected_path_count: int = 0
    insufficient_reason: str = ""
    local_insufficient_reason: str = ""
    shared_insufficient_reason: str = ""
    fallback_reason: str = ""

    @property
    def insufficient(self) -> bool:
        return bool(self.insufficient_reason) or not self.evidence

    def to_artifact(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "primary_anchor_mention": self.primary_anchor_mention,
            "linked_entity_id": self.linked_entity_id,
            "anchor_match": dict(self.anchor_match),
            "anchor_mentions": list(self.anchor_mentions),
            "linked_entities": [dict(item) for item in self.linked_entities],
            "anchor_matches": [dict(item) for item in self.anchor_matches],
            "unlinked_anchor_mentions": list(self.unlinked_anchor_mentions),
            "adjacent_hyperedge_ids": list(self.adjacent_hyperedge_ids),
            "expansion_entity_ids": list(self.expansion_entity_ids),
            "second_hop_hyperedge_ids": list(self.second_hop_hyperedge_ids),
            "candidate_hyperedge_ids": list(self.candidate_hyperedge_ids),
            "shared_candidate_hyperedge_ids": list(self.shared_candidate_hyperedge_ids),
            "local_candidate_hyperedge_ids": list(self.local_candidate_hyperedge_ids),
            "candidate_sources": [dict(item) for item in self.candidate_sources],
            "top_hyperedges": [dict(item) for item in self.top_hyperedges],
            "candidate_paths": [item.to_dict() for item in self.candidate_paths],
            "top_paths": [item.to_dict() for item in self.top_paths],
            "evidence": [item.to_answer_payload() for item in self.evidence],
            "candidate_path_count": self.candidate_path_count,
            "deduplicated_path_count": self.deduplicated_path_count,
            "selected_path_count": self.selected_path_count,
            "insufficient_reason": self.insufficient_reason,
            "local_insufficient_reason": self.local_insufficient_reason,
            "shared_insufficient_reason": self.shared_insufficient_reason,
            "fallback_reason": self.fallback_reason,
        }


class AtomicHyperedgeRetriever:
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
        self._entity_ids = [
            node_id for node_id, node in self.dataset.graph.nodes.items() if getattr(node, "role", "") == "entity"
        ]
        self._entity_lookup = self._build_entity_lookup(self._entity_ids)
        self._hyperedge_ids = [
            node_id for node_id, node in self.dataset.graph.nodes.items() if getattr(node, "role", "") == "hyperedge"
        ]
        self._hyperedge_lookup = self._normalized_lookup(self._hyperedge_ids)

    def retrieve_primary_anchor_local(
        self,
        *,
        question: str,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str,
    ) -> LocalHyperedgeRetrievalResult:
        result = self.build_atomic_candidate_pool(
            question=question,
            analysis=analysis,
            primary_anchor_mention=primary_anchor_mention,
        )
        return self.rank_candidate_pool(result, question=question)

    def build_original_question_candidate_pool(
        self,
        *,
        question: str,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str = "",
    ) -> LocalHyperedgeRetrievalResult:
        result = self._build_anchor_candidate_pool(
            question=question,
            analysis=analysis,
            primary_anchor_mention=primary_anchor_mention,
            method=_SHARED_RETRIEVAL_METHOD,
            pool_source="original_question_shared_pool",
            use_descriptive_fallback=False,
        )
        result.shared_candidate_hyperedge_ids = list(result.candidate_hyperedge_ids)
        result.shared_insufficient_reason = result.insufficient_reason
        return result

    def build_atomic_candidate_pool(
        self,
        *,
        question: str,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str,
    ) -> LocalHyperedgeRetrievalResult:
        result = self._build_anchor_candidate_pool(
            question=question,
            analysis=analysis,
            primary_anchor_mention=primary_anchor_mention,
            method=_LOCAL_RETRIEVAL_METHOD,
            pool_source="atomic_node_local_pool",
            use_descriptive_fallback=True,
        )
        result.local_candidate_hyperedge_ids = list(result.candidate_hyperedge_ids)
        result.local_insufficient_reason = result.insufficient_reason
        return result

    def _build_anchor_candidate_pool(
        self,
        *,
        question: str,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str,
        method: str,
        pool_source: str,
        use_descriptive_fallback: bool,
    ) -> LocalHyperedgeRetrievalResult:
        anchor_mentions = self._anchor_mentions(primary_anchor_mention, analysis)
        primary_mention = anchor_mentions[0] if anchor_mentions else ""
        result = LocalHyperedgeRetrievalResult(
            method=method,
            primary_anchor_mention=primary_mention,
            anchor_mentions=list(anchor_mentions),
        )
        if not anchor_mentions:
            result.insufficient_reason = "missing_primary_anchor"
            if use_descriptive_fallback:
                return self._try_descriptive_fallback_candidates(result, question=question, pool_source=pool_source)
            return result

        linked_matches: list[AnchorEntityMatch] = []
        seen_linked_entity_ids: set[str] = set()
        for query_index, mention in enumerate(anchor_mentions):
            match = self.link_anchor_entity(
                question=question,
                mention=mention,
                analysis=analysis,
                query_index=query_index,
            )
            if match is None:
                result.unlinked_anchor_mentions.append(mention)
                continue
            if match.entity_id in seen_linked_entity_ids:
                continue
            seen_linked_entity_ids.add(match.entity_id)
            linked_matches.append(match)

        if not linked_matches:
            result.insufficient_reason = "unlinked_primary_anchor"
            if use_descriptive_fallback:
                return self._try_descriptive_fallback_candidates(result, question=question, pool_source=pool_source)
            return result

        result.linked_entity_id = linked_matches[0].entity_id
        result.anchor_match = linked_matches[0].to_metadata()
        result.anchor_matches = [match.to_metadata() for match in linked_matches]
        result.linked_entities = [
            {
                "mention": match.query_entity,
                "entity_id": match.entity_id,
                "match_type": match.match_type,
                "link_score": match.link_score,
            }
            for match in linked_matches
        ]

        candidate_pool = self._multi_anchor_candidate_pool(linked_matches)
        result.adjacent_hyperedge_ids = list(candidate_pool["adjacent_hyperedge_ids"])
        if not result.adjacent_hyperedge_ids:
            result.insufficient_reason = "primary_anchor_has_no_adjacent_hyperedges"
            if use_descriptive_fallback:
                return self._try_descriptive_fallback_candidates(result, question=question, pool_source=pool_source)
            return result

        result.expansion_entity_ids = list(candidate_pool["expansion_entity_ids"])
        result.second_hop_hyperedge_ids = list(candidate_pool["second_hop_hyperedge_ids"])
        result.candidate_hyperedge_ids = list(candidate_pool["candidate_hyperedge_ids"])
        result.candidate_sources = [dict(item) for item in candidate_pool["candidate_sources"]]
        self._tag_candidate_pool_sources(result, pool_source)
        if not result.candidate_hyperedge_ids:
            result.insufficient_reason = "no_local_candidate_hyperedges"
            if use_descriptive_fallback:
                return self._try_descriptive_fallback_candidates(result, question=question, pool_source=pool_source)
            return result

        return result

    def _try_descriptive_fallback_candidates(
        self,
        result: LocalHyperedgeRetrievalResult,
        *,
        question: str,
        pool_source: str,
    ) -> LocalHyperedgeRetrievalResult:
        original_reason = result.insufficient_reason
        if _has_unresolved_dependency_reference(question):
            return result
        candidate_pool = self._descriptive_candidate_pool(question)
        if not candidate_pool["candidate_hyperedge_ids"]:
            return result

        result.fallback_reason = original_reason or "descriptive_fallback"
        result.insufficient_reason = ""
        result.candidate_hyperedge_ids = list(candidate_pool["candidate_hyperedge_ids"])
        result.candidate_sources = [dict(item) for item in candidate_pool["candidate_sources"]]
        self._tag_candidate_pool_sources(result, pool_source)
        result.expansion_entity_ids = _dedupe_strings(
            [*result.expansion_entity_ids, *candidate_pool["expansion_entity_ids"]]
        )
        result.second_hop_hyperedge_ids = _dedupe_strings(
            [*result.second_hop_hyperedge_ids, *candidate_pool["second_hop_hyperedge_ids"]]
        )
        return result

    def merge_candidate_pools(
        self,
        *,
        shared_pool: LocalHyperedgeRetrievalResult,
        local_pool: LocalHyperedgeRetrievalResult,
    ) -> LocalHyperedgeRetrievalResult:
        result = LocalHyperedgeRetrievalResult(
            method=_SHARED_RETRIEVAL_METHOD,
            primary_anchor_mention=local_pool.primary_anchor_mention,
            linked_entity_id=local_pool.linked_entity_id,
            anchor_match=dict(local_pool.anchor_match),
            anchor_mentions=list(local_pool.anchor_mentions),
            linked_entities=[dict(item) for item in local_pool.linked_entities],
            anchor_matches=[dict(item) for item in local_pool.anchor_matches],
            unlinked_anchor_mentions=list(local_pool.unlinked_anchor_mentions),
            adjacent_hyperedge_ids=list(local_pool.adjacent_hyperedge_ids),
            expansion_entity_ids=_dedupe_strings(
                [*shared_pool.expansion_entity_ids, *local_pool.expansion_entity_ids]
            ),
            second_hop_hyperedge_ids=_dedupe_strings(
                [*shared_pool.second_hop_hyperedge_ids, *local_pool.second_hop_hyperedge_ids]
            ),
            shared_candidate_hyperedge_ids=list(shared_pool.candidate_hyperedge_ids),
            local_candidate_hyperedge_ids=list(local_pool.candidate_hyperedge_ids),
            local_insufficient_reason=local_pool.insufficient_reason,
            shared_insufficient_reason=shared_pool.insufficient_reason,
            fallback_reason=local_pool.fallback_reason,
        )
        candidate_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}
        for pool in (shared_pool, local_pool):
            for hyperedge_id in pool.candidate_hyperedge_ids:
                if hyperedge_id not in candidate_ids:
                    candidate_ids.append(hyperedge_id)
            for source in pool.candidate_sources:
                self._merge_candidate_source(source_by_id, dict(source))

        result.candidate_hyperedge_ids = candidate_ids
        result.candidate_sources = [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids if hyperedge_id in source_by_id]
        if not result.candidate_hyperedge_ids:
            result.insufficient_reason = local_pool.insufficient_reason or shared_pool.insufficient_reason
        return result

    def rank_candidate_pool(
        self,
        result: LocalHyperedgeRetrievalResult,
        *,
        question: str,
    ) -> LocalHyperedgeRetrievalResult:
        result = self.rank_seed_hyperedges(result, question=question)
        return self.reconstruct_and_rank_paths(result, question=question)

    def rank_seed_hyperedges(
        self,
        result: LocalHyperedgeRetrievalResult,
        *,
        question: str,
    ) -> LocalHyperedgeRetrievalResult:
        source_by_id: dict[str, dict[str, Any]] = {}
        for source in result.candidate_sources:
            self._merge_candidate_source(source_by_id, dict(source))
        scores = self._hyperedge_similarity_scores(question, result.candidate_hyperedge_ids)
        ranked = [
            {
                "hyperedge_id": hyperedge_id,
                "semantic_score": float(scores.get(hyperedge_id, 0.0)),
                "rank": 0,
                "hop": int(source_by_id.get(hyperedge_id, {}).get("hop", 1)),
                "via_entity_ids": list(source_by_id.get(hyperedge_id, {}).get("via_entity_ids", [])),
                "via_first_hyperedge_ids": list(source_by_id.get(hyperedge_id, {}).get("via_first_hyperedge_ids", [])),
                "expansion_sources": list(source_by_id.get(hyperedge_id, {}).get("expansion_sources", [])),
                "via_chunk_ids": list(source_by_id.get(hyperedge_id, {}).get("via_chunk_ids", [])),
                "pool_sources": list(source_by_id.get(hyperedge_id, {}).get("pool_sources", [])),
            }
            for hyperedge_id in result.candidate_hyperedge_ids
        ]
        ranked.sort(key=lambda item: (-float(item["semantic_score"]), str(item["hyperedge_id"])))
        top_k = max(0, int(getattr(self.config, "local_hyperedge_top_k", 3)))
        selected = ranked[:top_k]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        result.top_hyperedges = selected
        return result

    def reconstruct_and_rank_paths(
        self,
        result: LocalHyperedgeRetrievalResult,
        *,
        question: str,
    ) -> LocalHyperedgeRetrievalResult:
        candidate_paths, raw_path_count = self.reconstruct_paths_from_seeds(result)
        result.candidate_path_count = raw_path_count
        result.deduplicated_path_count = len(candidate_paths)
        result.candidate_paths = candidate_paths
        result.top_paths = self.rank_evidence_paths(candidate_paths, question=question)
        result.selected_path_count = len(result.top_paths)
        result.evidence = list(result.top_paths)
        if not result.evidence and not result.insufficient_reason:
            result.insufficient_reason = "no_valid_evidence_paths"
        return result

    def reconstruct_paths_from_seeds(
        self,
        result: LocalHyperedgeRetrievalResult,
    ) -> tuple[list[EvidencePathCandidate], int]:
        source_by_id: dict[str, dict[str, Any]] = {}
        for source in result.candidate_sources:
            self._merge_candidate_source(source_by_id, dict(source))

        raw_paths: list[EvidencePathCandidate] = []
        for seed in result.top_hyperedges:
            seed_hyperedge_id = str(seed.get("hyperedge_id", "") or "")
            if not seed_hyperedge_id:
                continue
            source = source_by_id.get(seed_hyperedge_id, {})
            traces = [
                dict(trace)
                for trace in source.get("path_traces", [])
                if isinstance(trace, dict)
                and str(trace.get("terminal_hyperedge_id") or trace.get("hyperedge_id") or "") == seed_hyperedge_id
            ]
            seed_paths: list[EvidencePathCandidate] = []
            for trace in traces:
                path = self._path_from_trace(trace, seed)
                if path is not None:
                    seed_paths.append(path)
            if not seed_paths:
                fallback = self._seed_only_path(
                    seed_hyperedge_id=seed_hyperedge_id,
                    seed=seed,
                    source=source,
                    fallback_reason="missing_or_invalid_precise_path_trace",
                )
                if fallback is not None:
                    seed_paths.append(fallback)
            raw_paths.extend(seed_paths)

        deduped_by_key: dict[tuple[str, ...], EvidencePathCandidate] = {}
        for path in raw_paths:
            key = path.structural_key or self._path_structural_key(path)
            if key not in deduped_by_key:
                deduped_by_key[key] = path
                continue
            existing = deduped_by_key[key]
            existing.provenance = self._merge_path_provenance(existing.provenance, path.provenance)

        return list(deduped_by_key.values()), len(raw_paths)

    def rank_evidence_paths(
        self,
        paths: list[EvidencePathCandidate],
        *,
        question: str,
    ) -> list[EvidencePathCandidate]:
        if not paths:
            return []

        scores = self._path_similarity_scores(question, paths)
        for index, path in enumerate(paths):
            path.path_score = float(scores.get(index, 0.0))
        ranked = sorted(
            paths,
            key=lambda item: (
                -float(item.path_score),
                int(item.seed_hyperedge_rank or 0),
                tuple(str(value) for value in item.structural_key),
            ),
        )
        for rank, path in enumerate(ranked, start=1):
            path.path_rank = rank
        top_k = max(0, int(getattr(self.config, "local_path_top_k", 5)))
        return ranked[:top_k]

    def _path_from_trace(self, trace: dict[str, Any], seed: dict[str, Any]) -> EvidencePathCandidate | None:
        path_kind = str(trace.get("path_kind", "") or "").strip().lower()
        seed_hyperedge_id = str(seed.get("hyperedge_id", "") or trace.get("terminal_hyperedge_id", "") or "")
        seed_rank = int(seed.get("rank", 0) or 0)
        seed_score = float(seed.get("semantic_score", 0.0) or 0.0)
        anchor_entity_id = str(trace.get("anchor_entity_id", "") or "")
        first_hyperedge_id = str(trace.get("first_hyperedge_id", "") or "")
        terminal_hyperedge_id = str(trace.get("terminal_hyperedge_id", "") or seed_hyperedge_id)
        bridge_entity_id = str(trace.get("bridge_entity_id", "") or "")
        context_id = str(trace.get("context_id", "") or "")

        if path_kind == "1he":
            h1 = first_hyperedge_id or terminal_hyperedge_id
            if not h1 or h1 != terminal_hyperedge_id:
                return None
            if anchor_entity_id and anchor_entity_id not in self._hyperedge_entity_ids(h1):
                return None
            key = ("1he", h1)
            return self._make_path_candidate(
                path_type="1he",
                path_texts=[self._hyperedge_text(h1)],
                hyperedge_ids=[h1],
                context_ids=[],
                anchor_entity_id=anchor_entity_id,
                bridge_entity_id="",
                seed_hyperedge_id=seed_hyperedge_id,
                seed_hyperedge_rank=seed_rank,
                seed_hyperedge_score=seed_score,
                provenance=trace,
                structural_key=key,
            )

        if path_kind == "2he":
            if not first_hyperedge_id or not terminal_hyperedge_id or not bridge_entity_id:
                return None
            first_entities = set(self._hyperedge_entity_ids(first_hyperedge_id))
            terminal_entities = set(self._hyperedge_entity_ids(terminal_hyperedge_id))
            if anchor_entity_id and anchor_entity_id not in first_entities:
                return None
            if bridge_entity_id not in first_entities or bridge_entity_id not in terminal_entities:
                return None
            key = ("2he", first_hyperedge_id, bridge_entity_id, terminal_hyperedge_id)
            return self._make_path_candidate(
                path_type="2he",
                path_texts=[self._hyperedge_text(first_hyperedge_id), self._hyperedge_text(terminal_hyperedge_id)],
                hyperedge_ids=[first_hyperedge_id, terminal_hyperedge_id],
                context_ids=[],
                anchor_entity_id=anchor_entity_id,
                bridge_entity_id=bridge_entity_id,
                seed_hyperedge_id=seed_hyperedge_id,
                seed_hyperedge_rank=seed_rank,
                seed_hyperedge_score=seed_score,
                provenance=trace,
                structural_key=key,
            )

        if path_kind == "3he":
            if not first_hyperedge_id or not terminal_hyperedge_id or not bridge_entity_id or not context_id:
                return None
            first_entities = set(self._hyperedge_entity_ids(first_hyperedge_id))
            terminal_entities = set(self._hyperedge_entity_ids(terminal_hyperedge_id))
            if anchor_entity_id and anchor_entity_id not in first_entities:
                return None
            if context_id not in self._hyperedge_chunk_ids(first_hyperedge_id):
                return None
            if bridge_entity_id not in self._chunk_entity_ids(context_id):
                return None
            if bridge_entity_id not in terminal_entities:
                return None
            context_text = self.dataset.get_chunk_text(context_id)
            if not str(context_text or "").strip():
                return None
            key = ("3he", first_hyperedge_id, context_id, bridge_entity_id, terminal_hyperedge_id)
            return self._make_path_candidate(
                path_type="3he",
                path_texts=[
                    self._hyperedge_text(first_hyperedge_id),
                    str(context_text or "").strip(),
                    self._hyperedge_text(terminal_hyperedge_id),
                ],
                hyperedge_ids=[first_hyperedge_id, terminal_hyperedge_id],
                context_ids=[context_id],
                anchor_entity_id=anchor_entity_id,
                bridge_entity_id=bridge_entity_id,
                seed_hyperedge_id=seed_hyperedge_id,
                seed_hyperedge_rank=seed_rank,
                seed_hyperedge_score=seed_score,
                provenance=trace,
                structural_key=key,
            )

        return None

    def _seed_only_path(
        self,
        *,
        seed_hyperedge_id: str,
        seed: dict[str, Any],
        source: dict[str, Any],
        fallback_reason: str,
    ) -> EvidencePathCandidate | None:
        if not seed_hyperedge_id:
            return None
        provenance = {
            "path_kind": "seed_only",
            "terminal_hyperedge_id": seed_hyperedge_id,
            "fallback_reason": fallback_reason,
            "expansion_sources": list(source.get("expansion_sources", [])),
            "pool_sources": list(source.get("pool_sources", [])),
        }
        return self._make_path_candidate(
            path_type="seed_only",
            path_texts=[self._hyperedge_text(seed_hyperedge_id)],
            hyperedge_ids=[seed_hyperedge_id],
            context_ids=[],
            anchor_entity_id=str(source.get("anchor_entity_ids", [""])[0] if source.get("anchor_entity_ids") else ""),
            bridge_entity_id="",
            seed_hyperedge_id=seed_hyperedge_id,
            seed_hyperedge_rank=int(seed.get("rank", 0) or 0),
            seed_hyperedge_score=float(seed.get("semantic_score", 0.0) or 0.0),
            provenance=provenance,
            structural_key=("seed_only", seed_hyperedge_id),
        )

    @staticmethod
    def _make_path_candidate(
        *,
        path_type: str,
        path_texts: list[str],
        hyperedge_ids: list[str],
        context_ids: list[str],
        anchor_entity_id: str,
        bridge_entity_id: str,
        seed_hyperedge_id: str,
        seed_hyperedge_rank: int,
        seed_hyperedge_score: float,
        provenance: dict[str, Any],
        structural_key: tuple[str, ...],
    ) -> EvidencePathCandidate:
        clean_texts = [str(text or "").strip() for text in path_texts if str(text or "").strip()]
        payload = dict(provenance)
        payload["structural_key"] = list(structural_key)
        return EvidencePathCandidate(
            path_type=path_type,
            path_texts=clean_texts,
            hyperedge_ids=_dedupe_strings(hyperedge_ids),
            context_ids=_dedupe_strings(context_ids),
            anchor_entity_id=anchor_entity_id,
            bridge_entity_id=bridge_entity_id,
            seed_hyperedge_id=seed_hyperedge_id,
            seed_hyperedge_rank=seed_hyperedge_rank,
            seed_hyperedge_score=seed_hyperedge_score,
            provenance=payload,
            structural_key=structural_key,
        )

    @staticmethod
    def _path_structural_key(path: EvidencePathCandidate) -> tuple[str, ...]:
        if path.structural_key:
            return path.structural_key
        if path.path_type == "1he" and path.hyperedge_ids:
            return ("1he", path.hyperedge_ids[0])
        if path.path_type == "2he" and len(path.hyperedge_ids) >= 2:
            return ("2he", path.hyperedge_ids[0], path.bridge_entity_id, path.hyperedge_ids[1])
        if path.path_type == "3he" and len(path.hyperedge_ids) >= 2 and path.context_ids:
            return ("3he", path.hyperedge_ids[0], path.context_ids[0], path.bridge_entity_id, path.hyperedge_ids[1])
        return (path.path_type, path.seed_hyperedge_id)

    @staticmethod
    def _merge_path_provenance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        merged = dict(left)
        for key, value in right.items():
            if key not in merged:
                merged[key] = value
                continue
            if isinstance(merged[key], list):
                values = list(merged[key])
                incoming_values = value if isinstance(value, list) else [value]
                for item in incoming_values:
                    if item not in values:
                        values.append(item)
                merged[key] = values
        return merged

    def _path_similarity_scores(self, question: str, paths: list[EvidencePathCandidate]) -> dict[int, float]:
        if not question.strip() or not paths:
            return {index: 0.0 for index in range(len(paths))}
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return {index: 0.0 for index in range(len(paths))}
        texts = [question, *[path.serialized_text() for path in paths]]
        try:
            vectors = self.embedder.embed_texts(texts, stage="atomic_evidence_path_rerank")
        except (TypeError, ValueError, AttributeError):
            return {index: 0.0 for index in range(len(paths))}
        if len(vectors) < len(paths) + 1:
            return {index: 0.0 for index in range(len(paths))}
        question_vector = np.asarray(vectors[0], dtype=np.float32)
        scores: dict[int, float] = {}
        for index, vector in enumerate(vectors[1: len(paths) + 1]):
            scores[index] = _cosine_similarity(question_vector, np.asarray(vector, dtype=np.float32))
        return scores

    def _hyperedge_text(self, hyperedge_id: str) -> str:
        if not hyperedge_id:
            return ""
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        return normalize_label(str(description.get("hyperedge_text") or hyperedge_id))

    def _descriptive_candidate_pool(self, question: str) -> dict[str, Any]:
        candidate_ids: list[str] = []
        expansion_entity_ids: list[str] = []
        second_hop_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}

        for rank, hyperedge_id in enumerate(self._query_hyperedge_ids(question), start=1):
            self._add_descriptive_hyperedge_source(
                hyperedge_id=hyperedge_id,
                candidate_ids=candidate_ids,
                source_by_id=source_by_id,
                expansion_source="descriptive_hyperedge",
                rank=rank,
            )

        for rank, chunk_id in enumerate(self._query_chunk_ids(question), start=1):
            for hyperedge_id in self._hyperedge_ids_for_chunk(chunk_id):
                self._add_descriptive_hyperedge_source(
                    hyperedge_id=hyperedge_id,
                    candidate_ids=candidate_ids,
                    source_by_id=source_by_id,
                    expansion_source="descriptive_chunk_hyperedge",
                    rank=rank,
                    via_chunk_ids=[chunk_id],
                )
            for entity_id in self._chunk_entity_ids(chunk_id):
                if entity_id not in expansion_entity_ids:
                    expansion_entity_ids.append(entity_id)
                for hyperedge_id in self._adjacent_hyperedge_ids(entity_id):
                    if hyperedge_id not in second_hop_ids:
                        second_hop_ids.append(hyperedge_id)
                    self._add_descriptive_hyperedge_source(
                        hyperedge_id=hyperedge_id,
                        candidate_ids=candidate_ids,
                        source_by_id=source_by_id,
                        expansion_source="descriptive_chunk_entity",
                        rank=rank,
                        via_entity_ids=[entity_id],
                        via_chunk_ids=[chunk_id],
                    )

        return {
            "expansion_entity_ids": expansion_entity_ids,
            "second_hop_hyperedge_ids": second_hop_ids,
            "candidate_hyperedge_ids": candidate_ids,
            "candidate_sources": [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids],
        }

    def _add_descriptive_hyperedge_source(
        self,
        *,
        hyperedge_id: str,
        candidate_ids: list[str],
        source_by_id: dict[str, dict[str, Any]],
        expansion_source: str,
        rank: int,
        via_entity_ids: list[str] | None = None,
        via_chunk_ids: list[str] | None = None,
    ) -> None:
        if not hyperedge_id:
            return
        if hyperedge_id not in candidate_ids:
            candidate_ids.append(hyperedge_id)
        source = source_by_id.setdefault(
            hyperedge_id,
            {
                "hyperedge_id": hyperedge_id,
                "hop": 0,
                "via_entity_ids": [],
                "via_first_hyperedge_ids": [],
                "expansion_sources": [],
                "via_chunk_ids": [],
                "descriptive_seed_ranks": [],
            },
        )
        if expansion_source not in source["expansion_sources"]:
            source["expansion_sources"].append(expansion_source)
        if rank not in source["descriptive_seed_ranks"]:
            source["descriptive_seed_ranks"].append(rank)
        for entity_id in via_entity_ids or []:
            if entity_id not in source["via_entity_ids"]:
                source["via_entity_ids"].append(entity_id)
        for chunk_id in via_chunk_ids or []:
            if chunk_id not in source["via_chunk_ids"]:
                source["via_chunk_ids"].append(chunk_id)

    @staticmethod
    def _tag_candidate_pool_sources(result: LocalHyperedgeRetrievalResult, pool_source: str) -> None:
        if not pool_source:
            return
        for source in result.candidate_sources:
            pool_sources = source.setdefault("pool_sources", [])
            if pool_source not in pool_sources:
                pool_sources.append(pool_source)
            for trace in source.get("path_traces", []):
                if not isinstance(trace, dict):
                    continue
                trace_pool_sources = trace.setdefault("pool_sources", [])
                if pool_source not in trace_pool_sources:
                    trace_pool_sources.append(pool_source)

    @staticmethod
    def _anchor_mentions(primary_anchor_mention: str, analysis: AtomicQuestionAnalysis) -> list[str]:
        raw_mentions = [primary_anchor_mention, *analysis.entities]
        mentions: list[str] = []
        seen: set[str] = set()
        for raw_mention in raw_mentions:
            mention = normalize_label(str(raw_mention or "").strip())
            key = mention.lower()
            if mention and key not in seen:
                seen.add(key)
                mentions.append(mention)
        return mentions

    def link_primary_anchor(
        self,
        *,
        question: str,
        mention: str,
        analysis: AtomicQuestionAnalysis,
    ) -> AnchorEntityMatch | None:
        return self.link_anchor_entity(question=question, mention=mention, analysis=analysis, query_index=0)

    def link_anchor_entity(
        self,
        *,
        question: str,
        mention: str,
        analysis: AtomicQuestionAnalysis,
        query_index: int,
    ) -> AnchorEntityMatch | None:
        matches = self._resolve_anchor_entity_matches(
            question=question,
            entity=mention,
            analysis=analysis,
            query_index=query_index,
        )
        if not matches:
            return None
        matches.sort(
            key=lambda item: (
                -float(item.link_score),
                int(item.candidate_rank if item.candidate_rank is not None else 0),
                str(item.entity_id),
            )
        )
        return matches[0]

    def _adjacent_hyperedge_ids(self, entity_id: str) -> list[str]:
        if not entity_id or not hasattr(self.dataset.graph, "entity_hyperedge_ids"):
            return []
        return _dedupe_strings([str(item) for item in self.dataset.graph.entity_hyperedge_ids(entity_id)])

    def _multi_anchor_candidate_pool(self, matches: list[AnchorEntityMatch]) -> dict[str, Any]:
        adjacent_ids: list[str] = []
        expansion_entity_ids: list[str] = []
        second_hop_ids: list[str] = []
        candidate_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}

        for match in matches:
            first_hop_ids = self._adjacent_hyperedge_ids(match.entity_id)
            adjacent_ids.extend(hyperedge_id for hyperedge_id in first_hop_ids if hyperedge_id not in adjacent_ids)
            candidate_pool = self._local_candidate_pool(match.entity_id, first_hop_ids)
            expansion_entity_ids.extend(
                entity_id
                for entity_id in candidate_pool["expansion_entity_ids"]
                if entity_id not in expansion_entity_ids
            )
            second_hop_ids.extend(
                hyperedge_id
                for hyperedge_id in candidate_pool["second_hop_hyperedge_ids"]
                if hyperedge_id not in second_hop_ids
            )
            for hyperedge_id in candidate_pool["candidate_hyperedge_ids"]:
                if hyperedge_id not in candidate_ids:
                    candidate_ids.append(hyperedge_id)
            for source in candidate_pool["candidate_sources"]:
                enriched = dict(source)
                enriched["anchor_mentions"] = [match.query_entity]
                enriched["anchor_entity_ids"] = [match.entity_id]
                enriched["anchor_query_indices"] = [match.query_index]
                self._merge_candidate_source(source_by_id, enriched)

        return {
            "adjacent_hyperedge_ids": adjacent_ids,
            "expansion_entity_ids": expansion_entity_ids,
            "second_hop_hyperedge_ids": second_hop_ids,
            "candidate_hyperedge_ids": candidate_ids,
            "candidate_sources": [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids],
        }

    def _local_candidate_pool(self, primary_anchor_entity_id: str, first_hop_ids: list[str]) -> dict[str, Any]:
        candidate_ids = _dedupe_strings(list(first_hop_ids))
        expansion_entity_ids: list[str] = []
        second_hop_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}
        for hyperedge_id in first_hop_ids:
            source_by_id[hyperedge_id] = {
                "hyperedge_id": hyperedge_id,
                "hop": 1,
                "via_entity_ids": [primary_anchor_entity_id],
                "via_first_hyperedge_ids": [],
                "expansion_sources": ["direct"],
                "via_chunk_ids": [],
                "path_traces": [
                    {
                        "path_kind": "1he",
                        "anchor_entity_id": primary_anchor_entity_id,
                        "first_hyperedge_id": hyperedge_id,
                        "bridge_entity_id": "",
                        "context_id": "",
                        "terminal_hyperedge_id": hyperedge_id,
                        "expansion_source": "direct",
                        "pool_sources": [],
                    }
                ],
            }

        max_hops = max(1, int(getattr(self.config, "local_hyperedge_hops", 2)))
        if max_hops >= 2:
            for first_hop_id in first_hop_ids:
                for entity_id in self._hyperedge_entity_ids(first_hop_id):
                    if entity_id == primary_anchor_entity_id:
                        continue
                    self._add_second_hop_candidates(
                        anchor_entity_id=primary_anchor_entity_id,
                        entity_id=entity_id,
                        first_hop_id=first_hop_id,
                        first_hop_ids=first_hop_ids,
                        candidate_ids=candidate_ids,
                        expansion_entity_ids=expansion_entity_ids,
                        second_hop_ids=second_hop_ids,
                        source_by_id=source_by_id,
                        expansion_source="hyperedge_entity",
                    )
                for chunk_id in self._hyperedge_chunk_ids(first_hop_id):
                    for entity_id in self._chunk_entity_ids(chunk_id):
                        if entity_id == primary_anchor_entity_id:
                            continue
                        self._add_second_hop_candidates(
                            anchor_entity_id=primary_anchor_entity_id,
                            entity_id=entity_id,
                            first_hop_id=first_hop_id,
                            first_hop_ids=first_hop_ids,
                            candidate_ids=candidate_ids,
                            expansion_entity_ids=expansion_entity_ids,
                            second_hop_ids=second_hop_ids,
                            source_by_id=source_by_id,
                            expansion_source="chunk_entity",
                            via_chunk_ids=[chunk_id],
                        )

        return {
            "expansion_entity_ids": expansion_entity_ids,
            "second_hop_hyperedge_ids": second_hop_ids,
            "candidate_hyperedge_ids": candidate_ids,
            "candidate_sources": [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids],
        }

    def _add_second_hop_candidates(
        self,
        *,
        anchor_entity_id: str,
        entity_id: str,
        first_hop_id: str,
        first_hop_ids: list[str],
        candidate_ids: list[str],
        expansion_entity_ids: list[str],
        second_hop_ids: list[str],
        source_by_id: dict[str, dict[str, Any]],
        expansion_source: str,
        via_chunk_ids: list[str] | None = None,
    ) -> None:
        if entity_id not in expansion_entity_ids:
            expansion_entity_ids.append(entity_id)
        for second_hop_id in self._adjacent_hyperedge_ids(entity_id):
            if second_hop_id in first_hop_ids:
                continue
            if second_hop_id not in candidate_ids:
                candidate_ids.append(second_hop_id)
            if second_hop_id not in second_hop_ids:
                second_hop_ids.append(second_hop_id)
            source = source_by_id.setdefault(
                second_hop_id,
                {
                    "hyperedge_id": second_hop_id,
                    "hop": 2,
                    "via_entity_ids": [],
                    "via_first_hyperedge_ids": [],
                    "expansion_sources": [],
                    "via_chunk_ids": [],
                    "path_traces": [],
                },
            )
            if int(source.get("hop", 2)) > 1:
                source["hop"] = 2
            if entity_id not in source["via_entity_ids"]:
                source["via_entity_ids"].append(entity_id)
            if first_hop_id not in source["via_first_hyperedge_ids"]:
                source["via_first_hyperedge_ids"].append(first_hop_id)
            if expansion_source not in source["expansion_sources"]:
                source["expansion_sources"].append(expansion_source)
            for chunk_id in via_chunk_ids or []:
                if chunk_id not in source["via_chunk_ids"]:
                    source["via_chunk_ids"].append(chunk_id)
            source["path_traces"].append(
                {
                    "path_kind": "3he" if expansion_source == "chunk_entity" else "2he",
                    "anchor_entity_id": anchor_entity_id,
                    "first_hyperedge_id": first_hop_id,
                    "bridge_entity_id": entity_id,
                    "context_id": str((via_chunk_ids or [""])[0]) if expansion_source == "chunk_entity" else "",
                    "terminal_hyperedge_id": second_hop_id,
                    "expansion_source": expansion_source,
                    "pool_sources": [],
                }
            )

    @staticmethod
    def _merge_candidate_source(source_by_id: dict[str, dict[str, Any]], source: dict[str, Any]) -> None:
        hyperedge_id = str(source.get("hyperedge_id", "") or "")
        if not hyperedge_id:
            return
        existing = source_by_id.setdefault(
            hyperedge_id,
            {
                "hyperedge_id": hyperedge_id,
                "hop": int(source.get("hop", 1) or 1),
                "via_entity_ids": [],
                "via_first_hyperedge_ids": [],
                "expansion_sources": [],
                "via_chunk_ids": [],
                "anchor_mentions": [],
                "anchor_entity_ids": [],
                "anchor_query_indices": [],
                "pool_sources": [],
                "path_traces": [],
            },
        )
        existing["hop"] = min(int(existing.get("hop", 1) or 1), int(source.get("hop", 1) or 1))
        for key in (
            "via_entity_ids",
            "via_first_hyperedge_ids",
            "expansion_sources",
            "via_chunk_ids",
            "anchor_mentions",
            "anchor_entity_ids",
            "pool_sources",
        ):
            for value in source.get(key, []):
                if value not in existing[key]:
                    existing[key].append(value)
        for value in source.get("anchor_query_indices", []):
            if value not in existing["anchor_query_indices"]:
                existing["anchor_query_indices"].append(value)
        trace_by_key = {
            AtomicHyperedgeRetriever._path_trace_key(trace): trace
            for trace in existing.get("path_traces", [])
            if isinstance(trace, dict)
        }
        for trace in source.get("path_traces", []):
            if not isinstance(trace, dict):
                continue
            payload = dict(trace)
            inherited_pool_sources = source.get("pool_sources", [])
            if inherited_pool_sources:
                trace_pool_sources = payload.setdefault("pool_sources", [])
                for pool_source in inherited_pool_sources:
                    if pool_source not in trace_pool_sources:
                        trace_pool_sources.append(pool_source)
            key = AtomicHyperedgeRetriever._path_trace_key(payload)
            existing_trace = trace_by_key.get(key)
            if existing_trace is None:
                existing["path_traces"].append(payload)
                trace_by_key[key] = payload
            else:
                AtomicHyperedgeRetriever._merge_path_trace(existing_trace, payload)

    @staticmethod
    def _path_trace_key(trace: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(trace.get("path_kind", "") or ""),
            str(trace.get("first_hyperedge_id", "") or ""),
            str(trace.get("context_id", "") or ""),
            str(trace.get("bridge_entity_id", "") or ""),
            str(trace.get("terminal_hyperedge_id", "") or ""),
        )

    @staticmethod
    def _merge_path_trace(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
        for key, value in incoming.items():
            if key not in existing:
                existing[key] = value
                continue
            if isinstance(existing[key], list):
                values = existing[key]
                incoming_values = value if isinstance(value, list) else [value]
                for item in incoming_values:
                    if item not in values:
                        values.append(item)

    def _hyperedge_entity_ids(self, hyperedge_id: str) -> list[str]:
        if hasattr(self.dataset.graph, "hyperedge_entity_ids"):
            entity_ids = self.dataset.graph.hyperedge_entity_ids(hyperedge_id)
            if entity_ids:
                return _dedupe_strings([str(item) for item in entity_ids])
        if hasattr(self.dataset.graph, "describe_hyperedge"):
            description = self.dataset.graph.describe_hyperedge(hyperedge_id)
            return _dedupe_strings([str(item) for item in description.get("entity_ids", [])])
        return []

    def _hyperedge_chunk_ids(self, hyperedge_id: str) -> list[str]:
        if hasattr(self.dataset.graph, "hyperedge_chunk_ids"):
            chunk_ids = self.dataset.graph.hyperedge_chunk_ids(hyperedge_id)
            if chunk_ids:
                return _dedupe_strings([str(item) for item in chunk_ids])
        if hasattr(self.dataset.graph, "describe_hyperedge"):
            description = self.dataset.graph.describe_hyperedge(hyperedge_id)
            return _dedupe_strings([str(item) for item in description.get("chunk_ids", [])])
        return []

    def _hyperedge_ids_for_chunk(self, chunk_id: str) -> list[str]:
        source_to_nodes = getattr(self.dataset.graph, "source_to_nodes", None)
        if source_to_nodes is None or not hasattr(source_to_nodes, "get"):
            return []
        nodes = getattr(self.dataset.graph, "nodes", {})
        hyperedge_ids: list[str] = []
        for node_id in source_to_nodes.get(chunk_id, []):
            hyperedge_id = str(node_id)
            node = nodes.get(hyperedge_id) if hasattr(nodes, "get") else None
            if node is None or getattr(node, "role", "") != "hyperedge":
                continue
            if hyperedge_id not in hyperedge_ids:
                hyperedge_ids.append(hyperedge_id)
        return hyperedge_ids

    def _chunk_entity_ids(self, chunk_id: str) -> list[str]:
        source_to_nodes = getattr(self.dataset.graph, "source_to_nodes", None)
        if source_to_nodes is None or not hasattr(source_to_nodes, "get"):
            return []
        nodes = getattr(self.dataset.graph, "nodes", {})
        entity_ids: list[str] = []
        for node_id in source_to_nodes.get(chunk_id, []):
            entity_id = str(node_id)
            node = nodes.get(entity_id) if hasattr(nodes, "get") else None
            if node is None or getattr(node, "role", "") != "entity":
                continue
            if not self._is_concrete_chunk_entity(entity_id):
                continue
            if entity_id not in entity_ids:
                entity_ids.append(entity_id)
        return entity_ids

    def _chunk_entity_ids_for_hyperedge(self, hyperedge_id: str) -> list[str]:
        source_to_nodes = getattr(self.dataset.graph, "source_to_nodes", None)
        if source_to_nodes is None or not hasattr(source_to_nodes, "get"):
            return []
        entity_ids: list[str] = []
        for chunk_id in self._hyperedge_chunk_ids(hyperedge_id):
            for entity_id in self._chunk_entity_ids(chunk_id):
                if entity_id not in entity_ids:
                    entity_ids.append(entity_id)
        return entity_ids

    def _is_concrete_chunk_entity(self, entity_id: str) -> bool:
        nodes = getattr(self.dataset.graph, "nodes", {})
        node = nodes.get(entity_id) if hasattr(nodes, "get") else None
        label = normalize_label(entity_id).upper()
        if label in _CHUNK_ENTITY_GENERIC_LABELS:
            return False
        entity_type = normalize_label(str(getattr(node, "entity_type", "") or "")).upper()
        if entity_type in _CHUNK_ENTITY_EXCLUDED_TYPES:
            return False
        return True

    def _hyperedge_similarity_scores(self, query: str, hyperedge_ids: list[str]) -> dict[str, float]:
        if not query.strip() or not hyperedge_ids:
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        store = getattr(self.dataset, "hyperedge_store", None)
        if store is None or not hasattr(store, "similarities"):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        try:
            vectors = self.embedder.embed_texts([query], stage="atomic_local_hyperedge_retrieval")
            if not vectors:
                return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
            query_vector = np.asarray(vectors[0], dtype=np.float32)
            scores = dict(store.similarities(query_vector, list(hyperedge_ids)))
        except (TypeError, ValueError, AttributeError):
            return {hyperedge_id: 0.0 for hyperedge_id in hyperedge_ids}
        return {hyperedge_id: float(scores.get(hyperedge_id, 0.0)) for hyperedge_id in hyperedge_ids}

    def _query_hyperedge_ids(self, query: str) -> list[str]:
        top_k = max(0, int(getattr(self.config, "descriptive_fallback_hyperedge_top_k", 80)))
        if top_k <= 0:
            return []
        matches = self._query_vector_store(
            query=query,
            store=getattr(self.dataset, "hyperedge_store", None),
            top_k=top_k,
            stage="atomic_descriptive_hyperedge_fallback",
        )
        hyperedge_ids: list[str] = []
        for match in matches:
            hyperedge_id = self._resolve_hyperedge_id_from_vector_match(match)
            if hyperedge_id and hyperedge_id not in hyperedge_ids:
                hyperedge_ids.append(hyperedge_id)
        return hyperedge_ids

    def _query_chunk_ids(self, query: str) -> list[str]:
        top_k = max(0, int(getattr(self.config, "descriptive_fallback_chunk_top_k", 20)))
        if top_k <= 0:
            return []
        matches = self._query_vector_store(
            query=query,
            store=getattr(self.dataset, "chunk_store", None),
            top_k=top_k,
            stage="atomic_descriptive_chunk_fallback",
        )
        chunk_ids: list[str] = []
        for match in matches:
            chunk_id = self._resolve_chunk_id_from_vector_match(match)
            if chunk_id and chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
        return chunk_ids

    def _query_vector_store(self, *, query: str, store: Any, top_k: int, stage: str) -> list[VectorMatch]:
        if top_k <= 0 or not query.strip():
            return []
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return []
        if store is None or not hasattr(store, "query"):
            return []
        try:
            vectors = self.embedder.embed_texts([query], stage=stage)
            if not vectors:
                return []
            return list(store.query(vectors[0], top_k=top_k))
        except (TypeError, ValueError, AttributeError):
            return []

    def _resolve_hyperedge_id_from_vector_match(self, match: VectorMatch) -> str | None:
        metadata = match.metadata if isinstance(match.metadata, dict) else {}
        raw_candidates = [
            match.label,
            match.item_id,
            metadata.get("hyperedge_name"),
            metadata.get("__id__"),
            metadata.get("name"),
        ]
        nodes = getattr(self.dataset.graph, "nodes", {})
        for raw_candidate in raw_candidates:
            if raw_candidate is None:
                continue
            candidate = str(raw_candidate).strip()
            if not candidate:
                continue
            node = nodes.get(candidate) if hasattr(nodes, "get") else None
            if node is not None and getattr(node, "role", "") == "hyperedge":
                return candidate
            normalized = normalize_label(candidate).lower()
            mapped = self._hyperedge_lookup.get(normalized, [])
            if not mapped:
                mapped = self._hyperedge_lookup.get(_canonical_entity_key(candidate), [])
            if mapped:
                return mapped[0]
        return None

    def _resolve_chunk_id_from_vector_match(self, match: VectorMatch) -> str | None:
        metadata = match.metadata if isinstance(match.metadata, dict) else {}
        raw_candidates = [
            match.item_id,
            metadata.get("__id__"),
            metadata.get("chunk_id"),
            metadata.get("id"),
            match.label,
        ]
        text_chunks = getattr(self.dataset, "text_chunks", {})
        for raw_candidate in raw_candidates:
            if raw_candidate is None:
                continue
            candidate = str(raw_candidate).strip()
            if not candidate:
                continue
            if isinstance(text_chunks, dict) and candidate in text_chunks:
                return candidate
        return None

    def _evidence_from_hyperedge(
        self,
        *,
        hyperedge_id: str,
        semantic_score: float,
        rank: int,
        primary_anchor_mention: str,
        primary_anchor_entity_id: str,
        candidate_source: dict[str, Any],
        selection_source: str = _LOCAL_RETRIEVAL_METHOD,
    ) -> FusedHyperedgeCandidate:
        description = self.dataset.graph.describe_hyperedge(hyperedge_id)
        hyperedge_text = normalize_label(str(description.get("hyperedge_text") or hyperedge_id))
        entity_ids = _dedupe_strings([str(item) for item in description.get("entity_ids", [])])
        if not entity_ids and hasattr(self.dataset.graph, "hyperedge_entity_ids"):
            entity_ids = _dedupe_strings([str(item) for item in self.dataset.graph.hyperedge_entity_ids(hyperedge_id)])
        chunk_ids = _dedupe_strings([str(item) for item in description.get("chunk_ids", [])])
        chunk_texts = [self.dataset.get_chunk_text(chunk_id) for chunk_id in chunk_ids]
        entity_records = [self._entity_payload(entity_id) for entity_id in entity_ids]
        candidate_hop = int(candidate_source.get("hop", 1) or 1)
        return FusedHyperedgeCandidate(
            hyperedge_id=hyperedge_id,
            hyperedge_text=hyperedge_text,
            branch_support={"local_primary_anchor", f"hop{candidate_hop}"},
            semantic_score=float(semantic_score),
            fusion_score=float(semantic_score),
            entity_ids=entity_ids,
            entity_records=entity_records,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            evidence_texts=[text for text in [hyperedge_text, *chunk_texts] if text],
            rank=int(rank),
            score_breakdown={
                "selection_source": selection_source,
                "semantic_rank": int(rank),
                "semantic_score": float(semantic_score),
                "primary_anchor_mention": primary_anchor_mention,
                "primary_anchor_entity_id": primary_anchor_entity_id,
                "candidate_hop": candidate_hop,
                "via_entity_ids": list(candidate_source.get("via_entity_ids", [])),
                "via_first_hyperedge_ids": list(candidate_source.get("via_first_hyperedge_ids", [])),
                "expansion_sources": list(candidate_source.get("expansion_sources", [])),
                "via_chunk_ids": list(candidate_source.get("via_chunk_ids", [])),
                "pool_sources": list(candidate_source.get("pool_sources", [])),
                "anchor_mentions": list(candidate_source.get("anchor_mentions", [])),
                "anchor_entity_ids": list(candidate_source.get("anchor_entity_ids", [])),
                "anchor_query_indices": list(candidate_source.get("anchor_query_indices", [])),
            },
        )

    def _entity_payload(self, entity_id: str) -> dict[str, Any]:
        nodes = getattr(self.dataset.graph, "nodes", {})
        node = nodes.get(entity_id) if hasattr(nodes, "get") else None
        payload = {
            "entity_id": entity_id,
            "label": normalize_label(entity_id),
            "entity_type": getattr(node, "entity_type", None),
            "description": str(getattr(node, "description", "") or ""),
        }
        if node is not None and hasattr(node, "to_dict"):
            payload["metadata"] = node.to_dict()
        return payload

    def _resolve_anchor_entity_matches(
        self,
        question: str,
        entity: str,
        analysis: AtomicQuestionAnalysis,
        query_index: int,
    ) -> list[AnchorEntityMatch]:
        exact_candidates = self._entity_lookup_candidates(entity)
        if len(exact_candidates) == 1:
            return [self._anchor_match_from_candidate(exact_candidates[0], query_index=query_index, query_entity=entity)]
        if len(exact_candidates) > 1:
            selected_exact = self._select_or_auto_anchor_candidate(
                question=question,
                entity=entity,
                analysis=analysis,
                candidates=exact_candidates,
            )
            if selected_exact is not None:
                return [self._anchor_match_from_candidate(selected_exact, query_index=query_index, query_entity=entity)]
            return [
                self._anchor_match_from_candidate(candidate, query_index=query_index, query_entity=entity)
                for candidate in exact_candidates
            ]

        candidates = self._merge_anchor_candidates(
            [
                *self._anchor_entity_vector_candidates(entity),
                *self._chunk_mention_entity_candidates(entity),
            ]
        )
        if not candidates:
            return []

        selected = self._select_or_auto_anchor_candidate(
            question=question,
            entity=entity,
            analysis=analysis,
            candidates=candidates,
        )
        if selected is None:
            return []

        return [self._anchor_match_from_candidate(selected, query_index=query_index, query_entity=entity)]

    def _resolve_entity_ids(self, entity: str) -> list[str]:
        return [candidate["entity_id"] for candidate in self._entity_lookup_candidates(entity)]

    def _entity_lookup_candidates(self, entity: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, (key, match_type, score) in enumerate(_lookup_keys_for_mention(entity), start=1):
            for entity_id in self._entity_lookup.get(key, []):
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                candidates.append(
                    {
                        "entity_id": entity_id,
                        "label": normalize_label(entity_id),
                        "link_score": float(score),
                        "vector_score": float(score),
                        "candidate_rank": rank,
                        "source_label": normalize_label(entity_id),
                        "source_item_id": entity_id,
                        "match_type": match_type,
                    }
                )
        return candidates

    def _anchor_entity_vector_candidates(self, entity: str) -> list[dict[str, Any]]:
        top_k = int(getattr(self.config, "entity_link_top_k", _ANCHOR_ENTITY_TOP_K))
        matches = self._query_entity_store(entity, max(top_k, _ANCHOR_ENTITY_TOP_K))
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, match in enumerate(matches, start=1):
            entity_id = self._resolve_entity_id_from_vector_match(match)
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            candidates.append(
                {
                    "entity_id": entity_id,
                    "label": normalize_label(entity_id),
                    "link_score": float(match.score),
                    "vector_score": float(match.score),
                    "candidate_rank": rank,
                    "source_label": normalize_label(match.label),
                    "source_item_id": match.item_id,
                    "match_type": "vector",
                }
            )
        return candidates

    def _chunk_mention_entity_candidates(self, entity: str) -> list[dict[str, Any]]:
        mention_key = _canonical_entity_key(entity)
        if not mention_key:
            return []
        source_to_nodes = getattr(self.dataset.graph, "source_to_nodes", None)
        if source_to_nodes is None or not hasattr(source_to_nodes, "get"):
            return []
        text_chunks = getattr(self.dataset, "text_chunks", {})
        if not isinstance(text_chunks, dict):
            return []

        candidate_by_id: dict[str, dict[str, Any]] = {}
        for chunk_id, record in text_chunks.items():
            if not isinstance(record, dict):
                continue
            content = str(record.get("content", "") or "")
            if not _text_contains_entity_key(content, mention_key):
                continue
            source_title = _chunk_source_title(content)
            for node_id in source_to_nodes.get(str(chunk_id), []):
                entity_id = str(node_id)
                node = self.dataset.graph.nodes.get(entity_id)
                if node is None or getattr(node, "role", "") != "entity":
                    continue
                if not self._is_concrete_chunk_entity(entity_id):
                    continue
                label_key = _canonical_entity_key(entity_id)
                source_title_key = _canonical_entity_key(source_title)
                score = 0.82
                if label_key == mention_key:
                    score = 0.99
                elif source_title_key == mention_key:
                    score = 0.97
                elif mention_key in label_key or label_key in mention_key:
                    score = 0.9
                existing = candidate_by_id.get(entity_id)
                if existing is not None and float(existing["link_score"]) >= score:
                    continue
                snippet = short_text(content, 240)
                candidate_by_id[entity_id] = {
                    "entity_id": entity_id,
                    "label": normalize_label(entity_id),
                    "link_score": float(score),
                    "vector_score": float(score),
                    "candidate_rank": 0,
                    "source_label": source_title or normalize_label(entity_id),
                    "source_item_id": str(chunk_id),
                    "match_type": "chunk_mention",
                    "source_title": source_title,
                    "chunk_snippet": snippet,
                    "adjacent_hyperedge_count": len(self._adjacent_hyperedge_ids(entity_id)),
                }

        candidates = list(candidate_by_id.values())
        candidates.sort(
            key=lambda item: (
                -float(item.get("link_score", 0.0)),
                -int(item.get("adjacent_hyperedge_count", 0) or 0),
                str(item.get("entity_id", "")),
            )
        )
        for rank, candidate in enumerate(candidates[:_CHUNK_MENTION_ENTITY_LIMIT], start=1):
            candidate["candidate_rank"] = rank
        return candidates[:_CHUNK_MENTION_ENTITY_LIMIT]

    def _merge_anchor_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            entity_id = str(candidate.get("entity_id", "") or "")
            if not entity_id:
                continue
            existing = merged.get(entity_id)
            if existing is None or float(candidate.get("link_score", 0.0)) > float(existing.get("link_score", 0.0)):
                merged[entity_id] = dict(candidate)
        result = list(merged.values())
        result.sort(
            key=lambda item: (
                -float(item.get("link_score", item.get("vector_score", 0.0))),
                int(item.get("candidate_rank", 0) or 0),
                str(item.get("entity_id", "")),
            )
        )
        for rank, candidate in enumerate(result[:_ANCHOR_ENTITY_CANDIDATE_LIMIT], start=1):
            candidate["candidate_rank"] = rank
        return result[:_ANCHOR_ENTITY_CANDIDATE_LIMIT]

    def _select_or_auto_anchor_candidate(
        self,
        *,
        question: str,
        entity: str,
        analysis: AtomicQuestionAnalysis,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        selected = self._select_anchor_entity_with_llm(
            question=question,
            entity=entity,
            analysis=analysis,
            candidates=candidates,
        )
        if selected is not None:
            return selected
        return self._auto_select_anchor_candidate(entity, candidates)

    def _auto_select_anchor_candidate(self, entity: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        mention_key = _canonical_entity_key(entity)
        for candidate in candidates:
            label_key = _canonical_entity_key(str(candidate.get("label", "") or candidate.get("entity_id", "")))
            source_label_key = _canonical_entity_key(str(candidate.get("source_label", "") or ""))
            score = float(candidate.get("link_score", candidate.get("vector_score", 0.0)) or 0.0)
            if mention_key and mention_key in {label_key, source_label_key}:
                selected = dict(candidate)
                selected["llm_confidence"] = 0.0
                return selected
            if score >= _ANCHOR_ENTITY_AUTO_ACCEPT_SCORE and str(candidate.get("match_type", "")).startswith("alias"):
                selected = dict(candidate)
                selected["llm_confidence"] = 0.0
                return selected
        return None

    @staticmethod
    def _anchor_match_from_candidate(
        candidate: dict[str, Any],
        *,
        query_index: int,
        query_entity: str,
    ) -> AnchorEntityMatch:
        link_score = float(candidate.get("link_score", candidate.get("vector_score", 0.0)) or 0.0)
        return AnchorEntityMatch(
            query_index=query_index,
            query_entity=query_entity,
            entity_id=str(candidate["entity_id"]),
            match_type=str(candidate.get("match_type", "entity_link")),
            link_score=link_score,
            vector_score=float(candidate.get("vector_score", link_score) or link_score),
            llm_confidence=float(candidate.get("llm_confidence", 0.0) or 0.0),
            candidate_rank=int(candidate.get("candidate_rank", 0) or 0),
        )

    def _query_entity_store(self, entity: str, top_k: int) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        if self.embedder is None or not hasattr(self.embedder, "embed_texts"):
            return []
        store = getattr(self.dataset, "entity_store", None)
        if store is None or not hasattr(store, "query"):
            return []
        try:
            vectors = self.embedder.embed_texts([entity], stage="atomic_anchor_entity_retrieval")
            if not vectors:
                return []
            return list(store.query(vectors[0], top_k=top_k))
        except (TypeError, ValueError):
            return []

    def _resolve_entity_id_from_vector_match(self, match: VectorMatch) -> str | None:
        metadata = match.metadata if isinstance(match.metadata, dict) else {}
        raw_candidates = [
            match.label,
            match.item_id,
            metadata.get("entity_name"),
            metadata.get("__id__"),
            metadata.get("name"),
        ]
        for raw_candidate in raw_candidates:
            if raw_candidate is None:
                continue
            candidate = str(raw_candidate).strip()
            if not candidate:
                continue
            node = self.dataset.graph.nodes.get(candidate)
            if node is not None and getattr(node, "role", "") == "entity":
                return candidate
            normalized = normalize_label(candidate).lower()
            mapped = self._entity_lookup.get(normalized, [])
            if not mapped:
                mapped = self._entity_lookup.get(_canonical_entity_key(candidate), [])
            if mapped:
                return mapped[0]
        return None

    def _select_anchor_entity_with_llm(
        self,
        question: str,
        entity: str,
        analysis: AtomicQuestionAnalysis,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if self.llm_service is None or not hasattr(self.llm_service, "select_anchor_entity"):
            return None
        try:
            response = self.llm_service.select_anchor_entity(question, entity, analysis, candidates)
        except Exception as exc:  # pragma: no cover - defensive guard for external LLM failures
            self.logger.warning("Anchor entity selector failed for %s: %s", short_text(entity, 80), exc)
            return None

        if not isinstance(response, dict):
            return None
        selected_entity_id = str(response.get("selected_entity_id", "NONE")).strip()
        if not selected_entity_id or selected_entity_id.upper() == "NONE":
            return None

        try:
            confidence = float(response.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if confidence < _ANCHOR_ENTITY_LLM_MIN_CONFIDENCE:
            return None

        candidates_by_id = {str(candidate["entity_id"]): candidate for candidate in candidates}
        selected = candidates_by_id.get(selected_entity_id)
        if selected is None:
            return None

        payload = dict(selected)
        payload["llm_confidence"] = confidence
        payload["link_score"] = float(payload.get("link_score", payload.get("vector_score", 0.0)) or 0.0)
        payload["vector_score"] = float(payload.get("vector_score", payload["link_score"]) or payload["link_score"])
        payload["candidate_rank"] = int(payload.get("candidate_rank", 0) or 0)
        payload.setdefault("match_type", "vector_llm")
        return payload

    @staticmethod
    def _build_entity_lookup(values: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for value in values:
            for key in _lookup_keys_for_entity(value):
                if key and value not in lookup[key]:
                    lookup[key].append(value)
        return lookup

    @staticmethod
    def _normalized_lookup(values: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for value in values:
            for key in _lookup_keys_from_variants([value]):
                if key and value not in lookup[key]:
                    lookup[key].append(value)
        return lookup


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _lookup_keys_for_entity(entity_id: str) -> list[str]:
    label = normalize_label(entity_id)
    variants = [label]
    without_parenthetical = re.sub(r"\s*\([^)]*\)\s*", " ", label).strip()
    if without_parenthetical and without_parenthetical != label:
        variants.append(without_parenthetical)
    if "," in label:
        before_comma = label.split(",", 1)[0].strip()
        if before_comma:
            variants.append(before_comma)
    return _lookup_keys_from_variants(variants)


def _lookup_keys_for_mention(mention: str) -> list[tuple[str, str, float]]:
    label = normalize_label(mention)
    keys: list[tuple[str, str, float]] = []
    for key in _lookup_keys_from_variants([label]):
        keys.append((key, "exact", 1.0))
    without_article = re.sub(r"^(the|a|an)\s+", "", label, flags=re.IGNORECASE).strip()
    for key in _lookup_keys_from_variants([without_article]):
        keys.append((key, "alias_article", 0.99))
    without_parenthetical = re.sub(r"\s*\([^)]*\)\s*", " ", label).strip()
    for key in _lookup_keys_from_variants([without_parenthetical]):
        keys.append((key, "alias_parenthetical", 0.97))
    if "," in label:
        before_comma = label.split(",", 1)[0].strip()
        for key in _lookup_keys_from_variants([before_comma]):
            keys.append((key, "alias_comma", 0.95))

    deduped: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for key, match_type, score in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append((key, match_type, score))
    return deduped


def _lookup_keys_from_variants(variants: list[str]) -> list[str]:
    keys: list[str] = []
    for variant in variants:
        normalized = normalize_label(str(variant or "")).lower()
        canonical = _canonical_entity_key(variant)
        for key in (normalized, canonical):
            if key and key not in keys:
                keys.append(key)
    return keys


def _canonical_entity_key(text: str) -> str:
    normalized = normalize_label(str(text or ""))
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"'s\b", "s", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("'", "")
    normalized = re.sub(r"[^0-9A-Za-z]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _text_contains_entity_key(text: str, entity_key: str) -> bool:
    if not entity_key:
        return False
    canonical_text = f" {_canonical_entity_key(text)} "
    return f" {entity_key} " in canonical_text


def _chunk_source_title(text: str) -> str:
    for line in str(text or "").splitlines():
        title = normalize_label(line)
        if title:
            return title
    return ""


def _has_unresolved_dependency_reference(text: str) -> bool:
    return bool(re.search(r"\bq\d+'s\s+answer\b", str(text or ""), flags=re.IGNORECASE))
