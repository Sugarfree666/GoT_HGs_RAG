from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..models import VectorMatch
from ..utils import normalize_label, short_text
from .models import AtomicQuestionAnalysis, FusedHyperedgeCandidate


_ANCHOR_ENTITY_VECTOR_TOP_K = 1
_ANCHOR_ENTITY_VECTOR_MIN_SCORE = 0.6
_CHUNK_MENTION_ENTITY_LIMIT = 20
_LOCAL_RETRIEVAL_METHOD = "two_hop_multi_anchor_topk"
_SHARED_RETRIEVAL_METHOD = "shared_original_question_augmented_topk"
_CHUNK_ENTITY_EXCLUDED_TYPES = {
    "CATEGORY",
    "CONCEPT",
    "CONDITION",
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
_GENERIC_MENTION_TYPE_WORDS = {
    "album",
    "band",
    "book",
    "city",
    "college",
    "company",
    "country",
    "film",
    "magazine",
    "movie",
    "organization",
    "person",
    "place",
    "school",
    "song",
    "university",
    "work",
}
_MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
_QUALIFIER_STOPWORDS = {"a", "an", "of", "the"}
_PARENTHETICAL_ENTITY_TYPE_HINTS = {
    "film": {"FILM", "MOVIE", "WORK"},
    "journal": {"JOURNAL", "MAGAZINE", "PUBLICATION", "WORK"},
    "magazine": {"JOURNAL", "MAGAZINE", "PUBLICATION", "WORK"},
    "noble": {"PERSON", "TITLE"},
    "song": {"MUSIC", "SONG", "TRACK", "WORK"},
}
_QUESTION_ENTITY_TYPE_HINTS = {
    "album": {"ALBUM", "MUSIC", "WORK"},
    "book": {"BOOK", "NOVEL", "PUBLICATION", "TEXT", "WORK"},
    "film": {"FILM", "MOVIE", "WORK"},
    "journal": {"JOURNAL", "MAGAZINE", "PUBLICATION", "WORK"},
    "magazine": {"JOURNAL", "MAGAZINE", "PUBLICATION", "WORK"},
    "movie": {"FILM", "MOVIE", "WORK"},
    "novel": {"BOOK", "NOVEL", "TEXT", "WORK"},
    "song": {"MUSIC", "SONG", "TRACK", "WORK"},
}
_QUESTION_WORK_TYPE_CONFLICT_TYPES = {
    "CITY",
    "COUNTRY",
    "LOCATION",
    "ORGANIZATION",
    "PERSON",
    "PLACE",
    "REGION",
}
_PERSON_TITLE_WORDS = {"baron", "count", "duke", "earl", "emperor", "empress", "king", "lama", "prince", "queen"}
_LOCATION_ENTITY_TYPES = {"CITY", "COUNTRY", "LOCATION", "PLACE", "REGION"}
_INSTITUTION_HEAD_WORDS = {"academy", "college", "institute", "institution", "museum", "school", "university"}
_UNICODE_ASCII_EQUIVALENTS = str.maketrans(
    {
        "Đ": "D",
        "đ": "d",
        "Ł": "L",
        "ł": "l",
        "Ø": "O",
        "ø": "o",
        "Æ": "AE",
        "æ": "ae",
        "Œ": "OE",
        "œ": "oe",
        "Þ": "Th",
        "þ": "th",
        "Ð": "D",
        "ð": "d",
        "ß": "ss",
    }
)


@dataclass(slots=True)
class AnchorEntityMatch:
    query_index: int
    query_entity: str
    entity_id: str
    match_type: str
    link_score: float
    vector_score: float | None = None
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
    evidence: list[FusedHyperedgeCandidate] = field(default_factory=list)
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
            "evidence": [item.to_dict() for item in self.evidence],
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
        llm_service: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        # Kept only so older callers do not break; entity linking never calls an LLM.
        del llm_service
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._entity_ids = [
            node_id for node_id, node in self.dataset.graph.nodes.items() if getattr(node, "role", "") == "entity"
        ]
        self._entity_lookup = self._build_entity_lookup(self._entity_ids)
        self._entity_base_lookup = self._build_entity_base_lookup(self._entity_ids)
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
        hyperedge_query: str | None = None,
    ) -> LocalHyperedgeRetrievalResult:
        retrieval_query = self._atomic_hyperedge_query(
            question=question,
            analysis=analysis,
            hyperedge_query=hyperedge_query,
        )
        result = self.build_atomic_candidate_pool(
            question=question,
            analysis=analysis,
            primary_anchor_mention=primary_anchor_mention,
            hyperedge_query=retrieval_query,
        )
        return self.rank_candidate_pool(result, question=retrieval_query)

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
        hyperedge_query: str | None = None,
    ) -> LocalHyperedgeRetrievalResult:
        retrieval_query = self._atomic_hyperedge_query(
            question=question,
            analysis=analysis,
            hyperedge_query=hyperedge_query,
        )
        result = self._build_anchor_candidate_pool(
            question=question,
            hyperedge_query=retrieval_query,
            analysis=analysis,
            primary_anchor_mention=primary_anchor_mention,
            method=_LOCAL_RETRIEVAL_METHOD,
            pool_source="atomic_node_local_pool",
            use_descriptive_fallback=True,
        )
        result.local_candidate_hyperedge_ids = list(result.candidate_hyperedge_ids)
        result.local_insufficient_reason = result.insufficient_reason
        return result

    def _atomic_hyperedge_query(
        self,
        *,
        question: str,
        analysis: AtomicQuestionAnalysis,
        hyperedge_query: str | None = None,
    ) -> str:
        del analysis
        if hyperedge_query and hyperedge_query.strip():
            return hyperedge_query
        return question

    def _build_anchor_candidate_pool(
        self,
        *,
        question: str,
        hyperedge_query: str | None = None,
        analysis: AtomicQuestionAnalysis,
        primary_anchor_mention: str,
        method: str,
        pool_source: str,
        use_descriptive_fallback: bool,
    ) -> LocalHyperedgeRetrievalResult:
        retrieval_query = hyperedge_query if hyperedge_query and hyperedge_query.strip() else question
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
                return self._try_descriptive_fallback_candidates(
                    result,
                    question=question,
                    hyperedge_query=retrieval_query,
                    pool_source=pool_source,
                )
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

        mention_seed_pool = self._mention_seed_candidate_pool(result.unlinked_anchor_mentions)
        if not linked_matches:
            if mention_seed_pool["candidate_hyperedge_ids"]:
                result.expansion_entity_ids = list(mention_seed_pool["expansion_entity_ids"])
                result.second_hop_hyperedge_ids = list(mention_seed_pool["second_hop_hyperedge_ids"])
                result.candidate_hyperedge_ids = list(mention_seed_pool["candidate_hyperedge_ids"])
                result.candidate_sources = [dict(item) for item in mention_seed_pool["candidate_sources"]]
                self._tag_candidate_pool_sources(result, pool_source)
                return result

            result.insufficient_reason = "unlinked_primary_anchor"
            if use_descriptive_fallback:
                return self._try_descriptive_fallback_candidates(
                    result,
                    question=question,
                    hyperedge_query=retrieval_query,
                    pool_source=pool_source,
                )
            return result

        expansion_matches = self._expansion_anchor_matches(linked_matches)
        expansion_entity_ids = {match.entity_id for match in expansion_matches}
        result.linked_entity_id = expansion_matches[0].entity_id
        result.anchor_match = expansion_matches[0].to_metadata()
        result.anchor_matches = [match.to_metadata() for match in linked_matches]
        result.linked_entities = [
            {
                "mention": match.query_entity,
                "entity_id": match.entity_id,
                "match_type": match.match_type,
                "link_score": match.link_score,
                "used_for_expansion": match.entity_id in expansion_entity_ids,
            }
            for match in linked_matches
        ]

        candidate_pool = self._multi_anchor_candidate_pool(expansion_matches)
        if mention_seed_pool["candidate_hyperedge_ids"]:
            candidate_pool = self._merge_local_candidate_payloads(candidate_pool, mention_seed_pool)
        result.adjacent_hyperedge_ids = list(candidate_pool["adjacent_hyperedge_ids"])
        result.expansion_entity_ids = list(candidate_pool["expansion_entity_ids"])
        result.second_hop_hyperedge_ids = list(candidate_pool["second_hop_hyperedge_ids"])
        result.candidate_hyperedge_ids = list(candidate_pool["candidate_hyperedge_ids"])
        result.candidate_sources = [dict(item) for item in candidate_pool["candidate_sources"]]
        self._tag_candidate_pool_sources(result, pool_source)
        if not result.candidate_hyperedge_ids:
            result.insufficient_reason = (
                "primary_anchor_has_no_adjacent_hyperedges"
                if not result.adjacent_hyperedge_ids
                else "no_local_candidate_hyperedges"
            )
            if use_descriptive_fallback:
                return self._try_descriptive_fallback_candidates(
                    result,
                    question=question,
                    hyperedge_query=retrieval_query,
                    pool_source=pool_source,
                )
            return result

        return result

    def _try_descriptive_fallback_candidates(
        self,
        result: LocalHyperedgeRetrievalResult,
        *,
        question: str,
        hyperedge_query: str,
        pool_source: str,
    ) -> LocalHyperedgeRetrievalResult:
        original_reason = result.insufficient_reason
        if _has_unresolved_dependency_reference(question):
            return result
        candidate_pool = self._descriptive_candidate_pool(hyperedge_query)
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
        result.evidence = [
            self._evidence_from_hyperedge(
                hyperedge_id=str(item["hyperedge_id"]),
                semantic_score=float(item["semantic_score"]),
                rank=int(item["rank"]),
                primary_anchor_mention=result.primary_anchor_mention,
                primary_anchor_entity_id=result.linked_entity_id,
                candidate_source=source_by_id.get(str(item["hyperedge_id"]), {}),
                selection_source=result.method,
            )
            for item in selected
        ]
        result.evidence = [item for item in result.evidence if item.hyperedge_id]
        if not result.evidence and not result.insufficient_reason:
            result.insufficient_reason = "no_valid_local_evidence"
        return result

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

    def _expansion_anchor_matches(self, matches: list[AnchorEntityMatch]) -> list[AnchorEntityMatch]:
        named_matches = [match for match in matches if not self._is_date_or_number_match(match)]
        return named_matches or list(matches)

    def _is_date_or_number_match(self, match: AnchorEntityMatch) -> bool:
        node = getattr(self.dataset.graph, "nodes", {}).get(match.entity_id)
        entity_type = normalize_label(str(getattr(node, "entity_type", "") or "")).upper()
        return entity_type in {"DATE", "NUMBER"} or _is_date_or_number_label(match.query_entity)

    def _multi_anchor_candidate_pool(self, matches: list[AnchorEntityMatch]) -> dict[str, Any]:
        adjacent_ids: list[str] = []
        expansion_entity_ids: list[str] = []
        second_hop_ids: list[str] = []
        candidate_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}

        for match in matches:
            first_hop_ids = self._qualified_adjacent_hyperedge_ids(match)
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

    def _qualified_adjacent_hyperedge_ids(self, match: AnchorEntityMatch) -> list[str]:
        adjacent_ids = self._adjacent_hyperedge_ids(match.entity_id)
        qualifiers = _parenthetical_qualifier_tokens(match.query_entity)
        if not qualifiers or not adjacent_ids:
            return adjacent_ids

        qualified_ids = [
            hyperedge_id
            for hyperedge_id in adjacent_ids
            if _text_contains_qualifiers(self._hyperedge_context_text(hyperedge_id), qualifiers)
        ]
        return qualified_ids or adjacent_ids

    def _hyperedge_context_text(self, hyperedge_id: str) -> str:
        texts: list[str] = []
        if hasattr(self.dataset.graph, "describe_hyperedge"):
            description = self.dataset.graph.describe_hyperedge(hyperedge_id)
            texts.append(str(description.get("hyperedge_text", "") or ""))
        for chunk_id in self._hyperedge_chunk_ids(hyperedge_id):
            texts.append(self.dataset.get_chunk_text(chunk_id))
        return " ".join(text for text in texts if text)

    def _mention_seed_candidate_pool(self, mentions: list[str]) -> dict[str, Any]:
        expansion_entity_ids: list[str] = []
        second_hop_ids: list[str] = []
        candidate_ids: list[str] = []
        source_by_id: dict[str, dict[str, Any]] = {}

        for mention in mentions:
            for rank, chunk_id in enumerate(self._literal_mention_chunk_ids(mention), start=1):
                for hyperedge_id in self._hyperedge_ids_for_chunk(chunk_id):
                    self._add_descriptive_hyperedge_source(
                        hyperedge_id=hyperedge_id,
                        candidate_ids=candidate_ids,
                        source_by_id=source_by_id,
                        expansion_source="mention_chunk_hyperedge_seed",
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
                            expansion_source="mention_chunk_entity_seed",
                            rank=rank,
                            via_entity_ids=[entity_id],
                            via_chunk_ids=[chunk_id],
                        )

        return {
            "adjacent_hyperedge_ids": [],
            "expansion_entity_ids": expansion_entity_ids,
            "second_hop_hyperedge_ids": second_hop_ids,
            "candidate_hyperedge_ids": candidate_ids,
            "candidate_sources": [source_by_id[hyperedge_id] for hyperedge_id in candidate_ids],
        }

    def _literal_mention_chunk_ids(self, mention: str) -> list[str]:
        if not _is_specific_mention_seed(mention):
            return []
        mention_key = _canonical_entity_key(mention)
        if not mention_key:
            return []
        text_chunks = getattr(self.dataset, "text_chunks", {})
        if not isinstance(text_chunks, dict):
            return []

        chunk_ids: list[str] = []
        for chunk_id, record in text_chunks.items():
            if not isinstance(record, dict):
                continue
            content = str(record.get("content", "") or "")
            if _text_contains_entity_key(content, mention_key):
                chunk_ids.append(str(chunk_id))
            if len(chunk_ids) >= _CHUNK_MENTION_ENTITY_LIMIT:
                break
        return chunk_ids

    def _merge_local_candidate_payloads(
        self,
        primary: dict[str, Any],
        additional: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_ids = _dedupe_strings(
            [*primary.get("candidate_hyperedge_ids", []), *additional.get("candidate_hyperedge_ids", [])]
        )
        source_by_id: dict[str, dict[str, Any]] = {}
        for source in [*primary.get("candidate_sources", []), *additional.get("candidate_sources", [])]:
            self._merge_candidate_source(source_by_id, dict(source))
        return {
            "adjacent_hyperedge_ids": _dedupe_strings(
                [*primary.get("adjacent_hyperedge_ids", []), *additional.get("adjacent_hyperedge_ids", [])]
            ),
            "expansion_entity_ids": _dedupe_strings(
                [*primary.get("expansion_entity_ids", []), *additional.get("expansion_entity_ids", [])]
            ),
            "second_hop_hyperedge_ids": _dedupe_strings(
                [*primary.get("second_hop_hyperedge_ids", []), *additional.get("second_hop_hyperedge_ids", [])]
            ),
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
            }

        max_hops = max(1, int(getattr(self.config, "local_hyperedge_hops", 2)))
        if max_hops >= 2:
            for first_hop_id in first_hop_ids:
                for entity_id in self._hyperedge_entity_ids(first_hop_id):
                    if entity_id == primary_anchor_entity_id:
                        continue
                    self._add_second_hop_candidates(
                        entity_id=entity_id,
                        first_hop_id=first_hop_id,
                        first_hop_ids=first_hop_ids,
                        candidate_ids=candidate_ids,
                        expansion_entity_ids=expansion_entity_ids,
                        second_hop_ids=second_hop_ids,
                        source_by_id=source_by_id,
                        expansion_source="hyperedge_entity",
                    )
                for entity_id in self._chunk_entity_ids_for_hyperedge(first_hop_id):
                    if entity_id == primary_anchor_entity_id:
                        continue
                    self._add_second_hop_candidates(
                        entity_id=entity_id,
                        first_hop_id=first_hop_id,
                        first_hop_ids=first_hop_ids,
                        candidate_ids=candidate_ids,
                        expansion_entity_ids=expansion_entity_ids,
                        second_hop_ids=second_hop_ids,
                        source_by_id=source_by_id,
                        expansion_source="chunk_entity",
                        via_chunk_ids=self._hyperedge_chunk_ids(first_hop_id),
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
        del analysis
        exact_candidates = self._entity_lookup_candidates(entity, question=question)
        if len(exact_candidates) == 1:
            return [self._anchor_match_from_candidate(exact_candidates[0], query_index=query_index, query_entity=entity)]
        vector_candidates = self._anchor_entity_vector_candidates(entity, question=question)
        if not vector_candidates:
            return []
        return [self._anchor_match_from_candidate(vector_candidates[0], query_index=query_index, query_entity=entity)]

    def _entity_lookup_candidates(self, entity: str, *, question: str = "") -> list[dict[str, Any]]:
        direct_candidates = [
            candidate
            for candidate in self._entity_lookup_candidates_for_label(entity)
            if not _question_entity_type_conflict(
                question,
                entity,
                str(candidate["entity_id"]),
                self.dataset.graph,
            )
        ]
        if direct_candidates:
            return direct_candidates

        # The analysis prompt often preserves a disambiguating suffix such as
        # ``(1948 Film)`` or ``(91.5 FM)``, while construction stores the same
        # entity under its base title.  Treat that as an exact-name fallback,
        # but keep the suffix as a hard identity constraint so a same-title
        # entity from another year/type is never accepted.
        base_label = _without_trailing_parenthetical(entity)
        fallback_candidates = [
            *(
                self._entity_lookup_candidates_for_label(base_label)
                if base_label
                else []
            ),
            *self._entity_base_alias_candidates(base_label or entity),
        ]
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in fallback_candidates:
            entity_id = str(candidate["entity_id"])
            if entity_id in seen:
                continue
            seen.add(entity_id)
            if _vector_candidate_constraint_conflict(
                entity,
                entity_id,
                self.dataset.graph,
                question=question,
            ):
                continue
            candidates.append(candidate)
        return candidates

    def _entity_lookup_candidates_for_label(self, label: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for key in _lookup_keys_from_variants([label]):
            for entity_id in self._entity_lookup.get(key, []):
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                candidate = {
                    "entity_id": entity_id,
                    "label": normalize_label(entity_id),
                    "link_score": 1.0,
                    "vector_score": 1.0,
                    "candidate_rank": 1,
                    "source_label": normalize_label(entity_id),
                    "source_item_id": entity_id,
                    "match_type": "exact",
                }
                candidates.append(candidate)
        return candidates

    def _entity_base_alias_candidates(self, label: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for key in _lookup_keys_from_variants([label]):
            for entity_id in self._entity_base_lookup.get(key, []):
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                candidates.append(
                    {
                        "entity_id": entity_id,
                        "label": normalize_label(entity_id),
                        "link_score": 1.0,
                        "vector_score": 1.0,
                        "candidate_rank": 1,
                        "source_label": normalize_label(entity_id),
                        "source_item_id": entity_id,
                        "match_type": "exact",
                    }
                )
        return candidates

    def _anchor_entity_vector_candidates(self, entity: str, *, question: str = "") -> list[dict[str, Any]]:
        matches = self._query_entity_store(entity, _ANCHOR_ENTITY_VECTOR_TOP_K)
        if not matches:
            return []
        match = matches[0]
        score = float(match.score)
        if score < _ANCHOR_ENTITY_VECTOR_MIN_SCORE:
            return []
        entity_id = self._resolve_entity_id_from_vector_match(match)
        if not entity_id:
            return []
        if _vector_candidate_constraint_conflict(entity, entity_id, self.dataset.graph, question=question):
            return []
        return [
            {
                "entity_id": entity_id,
                "label": normalize_label(entity_id),
                "link_score": score,
                "vector_score": score,
                "candidate_rank": 1,
                "source_label": normalize_label(match.label),
                "source_item_id": match.item_id,
                "match_type": "vector",
            }
        ]

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

    @staticmethod
    def _build_entity_lookup(values: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for value in values:
            for key in _lookup_keys_from_variants([value]):
                if key and value not in lookup[key]:
                    lookup[key].append(value)
        return lookup

    @staticmethod
    def _build_entity_base_lookup(values: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for value in values:
            base_label = _without_trailing_parenthetical(value)
            if not base_label:
                continue
            for key in _lookup_keys_from_variants([base_label]):
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
    normalized = normalized.translate(_UNICODE_ASCII_EQUIVALENTS)
    for apostrophe in ("‘", "’", "ʼ", "`", "´"):
        normalized = normalized.replace(apostrophe, "'")
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"'s\b", "s", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("'", "")
    normalized = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^0-9A-Za-z]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _without_trailing_parenthetical(text: str) -> str:
    label = normalize_label(str(text or "")).strip()
    base = re.sub(r"(?:\s*\([^()]*\)\s*)+$", "", label).strip()
    if not base or base == label:
        return ""
    return base


def _vector_candidate_constraint_conflict(
    mention: str,
    entity_id: str,
    graph: Any,
    *,
    question: str = "",
) -> str:
    """Return a general mention-constraint conflict that should veto vector Top-1.

    Vector similarity is still the only ranking signal. These checks only reject a
    candidate that contradicts explicit identity constraints already present in
    the mention; they never promote a lower-ranked candidate.
    """

    node = getattr(graph, "nodes", {}).get(entity_id)
    entity_type = normalize_label(str(getattr(node, "entity_type", "") or "")).upper().strip('"')
    candidate_text = " ".join(
        [
            normalize_label(entity_id),
            normalize_label(str(getattr(node, "description", "") or "")),
        ]
    )
    candidate_key = _canonical_entity_key(candidate_text)

    if _question_entity_type_conflict(question, mention, entity_id, graph):
        return "question_entity_type_mismatch"

    parenthetical_parts = re.findall(r"\(([^)]*)\)", normalize_label(mention))
    for part in parenthetical_parts:
        part_key = _canonical_entity_key(part)
        numeric_constraints = re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", part)
        for number in numeric_constraints:
            if re.search(rf"(?<!\d){re.escape(number)}(?!\d)", candidate_text, flags=re.IGNORECASE) is None:
                return "parenthetical_number_mismatch"

        part_tokens = set(part_key.split())
        for type_hint, allowed_types in _PARENTHETICAL_ENTITY_TYPE_HINTS.items():
            if type_hint in part_tokens and entity_type not in allowed_types:
                return f"parenthetical_{type_hint}_type_mismatch"
        if "serial" in part_tokens and "serial" not in candidate_key.split():
            return "parenthetical_serial_mismatch"

    mention_tokens = set(_canonical_entity_key(mention).split())
    if mention_tokens & _PERSON_TITLE_WORDS and entity_type in _LOCATION_ENTITY_TYPES:
        return "person_title_location_mismatch"

    institution_location = _institution_location_qualifier(mention)
    if institution_location:
        location_tokens = _canonical_entity_key(institution_location).split()
        candidate_tokens = set(candidate_key.split())
        if location_tokens and not all(token in candidate_tokens for token in location_tokens):
            return "institution_location_mismatch"

    return ""


def _question_entity_type_conflict(question: str, mention: str, entity_id: str, graph: Any) -> bool:
    allowed_types = _question_entity_type_hints(question, mention)
    if not allowed_types:
        return False
    node = getattr(graph, "nodes", {}).get(entity_id)
    entity_type = normalize_label(str(getattr(node, "entity_type", "") or "")).upper().strip('"')
    # Graph construction may assign a broad or conflated work type (for
    # example PROJECT or SONG to a node that also carries film facts).  The
    # question hint is therefore only strong enough to reject an unmistakable
    # cross-category entity such as a place or person, not to enforce strict
    # type equality.
    return bool(entity_type) and entity_type not in allowed_types and entity_type in _QUESTION_WORK_TYPE_CONFLICT_TYPES


def _question_entity_type_hints(question: str, mention: str) -> set[str]:
    question_key = _canonical_entity_key(question)
    mention_key = _canonical_entity_key(mention)
    if not question_key or not mention_key:
        return set()
    escaped_mention = re.escape(mention_key)
    for type_hint, allowed_types in _QUESTION_ENTITY_TYPE_HINTS.items():
        pattern = rf"\b{type_hint}(?:\s+(?:named|called|titled)(?:\s+after)?)?\s+{escaped_mention}\b"
        if re.search(pattern, question_key):
            return set(allowed_types)
    return set()


def _institution_location_qualifier(mention: str) -> str:
    label = normalize_label(mention)
    before_in, separator, after_in = label.rpartition(" in ")
    if not separator or not after_in.strip():
        return ""
    head_tokens = set(_canonical_entity_key(before_in).split())
    if not head_tokens.intersection(_INSTITUTION_HEAD_WORDS):
        return ""
    return after_in.strip()


def _is_date_or_number_label(text: str) -> bool:
    label = normalize_label(str(text or "")).strip('"').strip().lower()
    if not label:
        return False
    tokens = set(re.findall(r"[a-z]+", label))
    if tokens & _MONTH_NAMES and re.search(r"\d", label):
        return True
    numeric_pattern = (
        r"[+-]?\d[\d\s,.:/%\-–—]*"
        r"(?:st|nd|rd|th)?"
        r"(?:\s+(?:hundred|thousand|million|billion|trillion|percent|percentage|years?|months?|days?))?"
    )
    return re.fullmatch(numeric_pattern, label, flags=re.IGNORECASE) is not None


def _is_generic_descriptive_mention(text: str) -> bool:
    label = normalize_label(str(text or "")).strip()
    if not label or "(" in label:
        return False
    words = re.findall(r"[A-Za-z]+", label)
    if not words:
        return False
    if len(words) == 1:
        return words[0].lower() in _GENERIC_MENTION_TYPE_WORDS
    last_word = words[-1]
    return last_word.islower() and last_word in _GENERIC_MENTION_TYPE_WORDS


def _is_specific_mention_seed(text: str) -> bool:
    if _is_date_or_number_label(text) or _is_generic_descriptive_mention(text):
        return False
    tokens = _canonical_entity_key(text).split()
    return bool(tokens) and (len(tokens) > 1 or len(tokens[0]) >= 3)


def _parenthetical_qualifier_tokens(text: str) -> list[str]:
    qualifiers: list[str] = []
    for content in re.findall(r"\(([^)]*)\)", normalize_label(str(text or ""))):
        for token in _canonical_entity_key(content).split():
            if token not in _QUALIFIER_STOPWORDS and token not in qualifiers:
                qualifiers.append(token)
    return qualifiers


def _text_contains_qualifiers(text: str, qualifiers: list[str]) -> bool:
    text_tokens = set(_canonical_entity_key(text).split())
    return bool(text_tokens) and all(token in text_tokens for token in qualifiers)


def _text_contains_entity_key(text: str, entity_key: str) -> bool:
    if not entity_key:
        return False
    canonical_text = f" {_canonical_entity_key(text)} "
    return f" {entity_key} " in canonical_text


def _has_unresolved_dependency_reference(text: str) -> bool:
    return bool(re.search(r"\bq\d+'s\s+answer\b", str(text or ""), flags=re.IGNORECASE))
