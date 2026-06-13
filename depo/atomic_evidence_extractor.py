from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import asdict, is_dataclass
from typing import Any

from models import AtomicEvidence, CoreNLPViewAnnotation, DeclarativeView, ExplicitEntityResult, MaskMapping


ROLE_WORDS = {
    "actor",
    "artist",
    "author",
    "batsman",
    "book",
    "captain",
    "ceo",
    "city",
    "company",
    "composer",
    "country",
    "county",
    "date",
    "director",
    "film",
    "grandfather",
    "highway",
    "mother",
    "nationality",
    "performer",
    "place",
    "player",
    "screenplay",
    "singer",
    "song",
    "team",
    "town",
    "university",
    "winner",
}

OPERATOR_CUES = {
    "after",
    "before",
    "both",
    "earlier",
    "either",
    "first",
    "how many",
    "largest",
    "later",
    "older",
    "same",
    "smallest",
    "younger",
}

GENERIC_RELATIONS = {"be", "is", "are", "was", "were", "been", "being", "have", "has", "had"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "both",
    "did",
    "do",
    "does",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
}


class AtomicEvidenceExtractor:
    """Build a unified evidence pool from masking, CoreNLP, and OpenIE outputs."""

    def extract(
        self,
        *,
        masked_question: str,
        explicit_entities: ExplicitEntityResult | dict[str, Any] | None = None,
        mask_mappings: list[MaskMapping] | list[dict[str, Any]] | None = None,
        declarative_views: list[DeclarativeView] | list[dict[str, Any]] | None = None,
        corenlp_annotations: list[CoreNLPViewAnnotation] | list[dict[str, Any]] | None = None,
        operator_intent: dict[str, Any] | None = None,
    ) -> list[AtomicEvidence]:
        self._counters: Counter[str] = Counter()
        self._seen: set[tuple[Any, ...]] = set()

        mappings = [_to_dict(item) for item in (mask_mappings or [])]
        views = [_to_dict(item) for item in (declarative_views or [])]
        annotations = [_annotation_to_dict(item) for item in (corenlp_annotations or [])]
        operator_intent = operator_intent or {}

        evidence: list[AtomicEvidence] = []
        evidence.extend(self._candidate_entity_evidence(mappings))
        evidence.extend(self._answer_focus_evidence(masked_question, views, mappings))
        evidence.extend(self._operator_cue_evidence(masked_question, views, operator_intent, mappings))

        for annotation in annotations:
            evidence.extend(self._corenlp_evidence(annotation, mappings))
            evidence.extend(self._openie_evidence(annotation, mappings))

        return [item for item in evidence if item.id]

    def _candidate_entity_evidence(self, mappings: list[dict[str, Any]]) -> list[AtomicEvidence]:
        result: list[AtomicEvidence] = []
        for mapping in mappings:
            placeholder = str(mapping.get("placeholder") or "").strip()
            original_text = str(mapping.get("original_text") or "").strip()
            if not placeholder and not original_text:
                continue
            result.append(
                self._make(
                    prefix="entity",
                    type="candidate_entity",
                    source="masking",
                    text=f"{placeholder} -> {original_text}" if original_text else placeholder,
                    aligned_entities=[placeholder] if placeholder else [],
                    semantic_hint=str(mapping.get("semantic_type_hint") or ""),
                    metadata={"mask_mapping": mapping},
                )
            )
        return result

    def _answer_focus_evidence(
        self,
        masked_question: str,
        views: list[dict[str, Any]],
        mappings: list[dict[str, Any]],
    ) -> list[AtomicEvidence]:
        text = " ".join([masked_question, *[str(view.get("sentence") or "") for view in views]])
        cues = _answer_focus_cues(text)
        return [
            self._make(
                prefix="focus",
                type="answer_focus",
                source="surface",
                text=cue,
                semantic_hint=_answer_type_hint(cue),
                aligned_entities=_aligned_entities(cue, mappings),
            )
            for cue in cues
        ]

    def _operator_cue_evidence(
        self,
        masked_question: str,
        views: list[dict[str, Any]],
        operator_intent: dict[str, Any],
        mappings: list[dict[str, Any]],
    ) -> list[AtomicEvidence]:
        text = " ".join([masked_question, *[str(view.get("sentence") or "") for view in views]]).lower()
        cues = []
        for cue in OPERATOR_CUES:
            if cue in text:
                cues.append(cue)
        cues.extend(str(cue).strip() for cue in operator_intent.get("cues", []) or [] if str(cue).strip())
        result: list[AtomicEvidence] = []
        for cue in _unique(cues):
            result.append(
                self._make(
                    prefix="operator",
                    type="operator_cue",
                    source="surface",
                    text=cue,
                    operator_hint=str(operator_intent.get("type") or ""),
                    aligned_entities=_aligned_entities(cue, mappings),
                    metadata={"operator_intent": operator_intent},
                )
            )
        if operator_intent:
            result.append(
                self._make(
                    prefix="question_type",
                    type="question_type",
                    source="surface",
                    text=str(operator_intent.get("type") or "unknown"),
                    operator_hint=str(operator_intent.get("type") or ""),
                    metadata={"operator_intent": operator_intent},
                )
            )
        return result

    def _corenlp_evidence(
        self,
        annotation: dict[str, Any],
        mappings: list[dict[str, Any]],
    ) -> list[AtomicEvidence]:
        view_id = str(annotation.get("view_id") or "")
        tokens = [_to_dict(token) for token in annotation.get("tokens", []) or []]
        edges = [_to_dict(edge) for edge in annotation.get("edges", []) or []]
        result: list[AtomicEvidence] = []

        for token in tokens:
            word = str(token.get("word") or "").strip()
            if not word:
                continue
            lower = word.lower()
            pos = str(token.get("pos") or "")
            if lower in ROLE_WORDS or (pos.startswith("NN") and lower not in STOPWORDS and not _is_placeholder(word)):
                result.append(
                    self._make(
                        prefix="role",
                        type="role_or_attribute",
                        source="corenlp",
                        view_id=view_id,
                        text=word,
                        head=word,
                        semantic_hint=_semantic_hint_for_role(lower),
                        aligned_entities=_aligned_entities(word, mappings),
                        metadata={"token": token},
                    )
                )

        for edge in edges:
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            relation = str(edge.get("relation") or "").strip()
            if not source or not target or not relation:
                continue
            result.append(
                self._make(
                    prefix="corenlp_dep",
                    type="dependency_edge",
                    source="corenlp",
                    view_id=view_id,
                    text=f"{source} --{relation}--> {target}",
                    head=source,
                    dependent=target,
                    dependency_relation=relation,
                    aligned_entities=_aligned_entities(f"{source} {target}", mappings),
                    metadata={"edge": edge},
                )
            )

        result.extend(self._dependency_path_evidence(view_id, tokens, edges, mappings))
        result.extend(self._coordination_evidence(view_id, edges, mappings))
        result.extend(self._constraint_evidence(view_id, tokens, edges, mappings))
        for phrase in annotation.get("phrase_spans", []) or []:
            if isinstance(phrase, dict):
                result.append(
                    self._make(
                        prefix="phrase",
                        type="phrase_boundary",
                        source="corenlp",
                        view_id=view_id,
                        text=str(phrase.get("text") or phrase),
                        span=_int_list(phrase.get("span")),
                        metadata={"phrase": phrase},
                    )
                )
        return result

    def _dependency_path_evidence(
        self,
        view_id: str,
        tokens: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        mappings: list[dict[str, Any]],
    ) -> list[AtomicEvidence]:
        token_by_index = {_int_value(token.get("index")): str(token.get("word") or "") for token in tokens}
        placeholder_indices = {
            index
            for index, word in token_by_index.items()
            if any(word == str(mapping.get("placeholder") or "") for mapping in mappings)
        }
        role_indices = {
            _int_value(token.get("index"))
            for token in tokens
            if str(token.get("word") or "").lower() in ROLE_WORDS
        }
        if not placeholder_indices or not role_indices:
            return []

        adjacency: dict[int, list[tuple[int, str]]] = {}
        for edge in edges:
            source_index = _int_value(edge.get("source_index"))
            target_index = _int_value(edge.get("target_index"))
            relation = str(edge.get("relation") or "")
            if source_index <= 0 or target_index <= 0:
                continue
            adjacency.setdefault(source_index, []).append((target_index, relation))
            adjacency.setdefault(target_index, []).append((source_index, relation))

        result: list[AtomicEvidence] = []
        for start in sorted(placeholder_indices):
            for target in sorted(role_indices):
                if start == target:
                    continue
                path = _shortest_path(adjacency, start, target, max_edges=3)
                if not path:
                    continue
                words = [token_by_index.get(index, str(index)) for index in path]
                result.append(
                    self._make(
                        prefix="corenlp_path",
                        type="dependency_path",
                        source="corenlp",
                        view_id=view_id,
                        text=" -> ".join(words),
                        aligned_entities=_aligned_entities(" ".join(words), mappings),
                        metadata={"token_indices": path},
                    )
                )
        return result

    def _coordination_evidence(
        self,
        view_id: str,
        edges: list[dict[str, Any]],
        mappings: list[dict[str, Any]],
    ) -> list[AtomicEvidence]:
        result: list[AtomicEvidence] = []
        for edge in edges:
            relation = str(edge.get("relation") or "")
            if not relation.startswith("conj") and relation not in {"cc", "cc:preconj"}:
                continue
            text = f"{edge.get('source')} {relation} {edge.get('target')}"
            result.append(
                self._make(
                    prefix="coord",
                    type="coordination_candidate_set",
                    source="corenlp",
                    view_id=view_id,
                    text=text,
                    dependency_relation=relation,
                    aligned_entities=_aligned_entities(text, mappings),
                    metadata={"edge": edge},
                )
            )
        return result

    def _constraint_evidence(
        self,
        view_id: str,
        tokens: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        mappings: list[dict[str, Any]],
    ) -> list[AtomicEvidence]:
        result: list[AtomicEvidence] = []
        for edge in edges:
            relation = str(edge.get("relation") or "")
            if not (relation.startswith("obl:") or relation.startswith("nmod:") or relation in {"amod", "acl", "advcl"}):
                continue
            text = f"{edge.get('source')} --{relation}--> {edge.get('target')}"
            result.append(
                self._make(
                    prefix="constraint",
                    type="constraint",
                    source="corenlp",
                    view_id=view_id,
                    text=text,
                    dependency_relation=relation,
                    aligned_entities=_aligned_entities(text, mappings),
                    metadata={"edge": edge},
                )
            )
        for token in tokens:
            word = str(token.get("word") or "")
            if re.fullmatch(r"\d{3,4}", word):
                result.append(
                    self._make(
                        prefix="constraint",
                        type="constraint",
                        source="corenlp",
                        view_id=view_id,
                        text=word,
                        semantic_hint="date_or_number",
                        metadata={"token": token},
                    )
                )
        return result

    def _openie_evidence(
        self,
        annotation: dict[str, Any],
        mappings: list[dict[str, Any]],
    ) -> list[AtomicEvidence]:
        view_id = str(annotation.get("view_id") or "")
        result: list[AtomicEvidence] = []
        for triple in annotation.get("openie_triples", []) or []:
            triple_dict = _to_dict(triple)
            subject = str(triple_dict.get("subject") or "").strip()
            relation = str(triple_dict.get("relation") or "").strip()
            object_value = str(triple_dict.get("object") or "").strip()
            if not subject or not relation or not object_value:
                continue
            if _triple_inside_single_masked_entity(subject, relation, object_value, mappings):
                continue
            aligned = _aligned_entities(f"{subject} {relation} {object_value}", mappings)
            metadata = {"triple": triple_dict}
            if relation.lower() in GENERIC_RELATIONS:
                metadata["surface_relation_hint"] = True
            confidence = _float_value(triple_dict.get("confidence"), default=1.0)
            result.append(
                self._make(
                    prefix="openie_triple",
                    type="openie_triple",
                    source="openie",
                    view_id=view_id,
                    text=f"({subject}, {relation}, {object_value})",
                    subject=subject,
                    relation=relation,
                    object=object_value,
                    aligned_entities=aligned,
                    confidence=confidence,
                    metadata=metadata,
                )
            )
            result.append(
                self._make(
                    prefix="relation_phrase",
                    type="relation_phrase",
                    source="openie",
                    view_id=view_id,
                    text=relation,
                    relation=relation,
                    aligned_entities=aligned,
                    confidence=confidence,
                    metadata=metadata,
                )
            )
            result.append(
                self._make(
                    prefix="relation_direction",
                    type="relation_direction",
                    source="openie",
                    view_id=view_id,
                    text=f"{subject} -> {object_value}",
                    subject=subject,
                    relation=relation,
                    object=object_value,
                    aligned_entities=aligned,
                    confidence=confidence,
                    metadata=metadata,
                )
            )
            result.append(
                self._make(
                    prefix="argument_boundary",
                    type="argument_boundary",
                    source="openie",
                    view_id=view_id,
                    text=f"subject={subject}; object={object_value}",
                    subject=subject,
                    object=object_value,
                    span=_int_list(triple_dict.get("subject_span") or triple_dict.get("subjectSpan")),
                    aligned_entities=aligned,
                    confidence=confidence,
                    metadata=metadata,
                )
            )
            if aligned and relation.lower() not in GENERIC_RELATIONS:
                result.append(
                    self._make(
                        prefix="openie_constraint",
                        type="openie_constraint",
                        source="openie",
                        view_id=view_id,
                        text=f"{subject} {relation} {object_value}",
                        subject=subject,
                        relation=relation,
                        object=object_value,
                        aligned_entities=aligned,
                        confidence=confidence,
                        metadata=metadata,
                    )
                )
        return result

    def _make(self, *, prefix: str, type: str, source: str, text: str, **kwargs: Any) -> AtomicEvidence:
        key = (
            type,
            source,
            kwargs.get("view_id"),
            _norm(text),
            _norm(str(kwargs.get("subject") or "")),
            _norm(str(kwargs.get("relation") or "")),
            _norm(str(kwargs.get("object") or "")),
            _norm(str(kwargs.get("dependency_relation") or "")),
        )
        if key in self._seen:
            return AtomicEvidence(id="", type=type, source=source, text=text)
        self._seen.add(key)
        self._counters[prefix] += 1
        return AtomicEvidence(
            id=f"{prefix}_{self._counters[prefix]}",
            type=type,
            source=source,
            text=text,
            kind=type,
            **kwargs,
        )


def _annotation_to_dict(annotation: Any) -> dict[str, Any]:
    data = _to_dict(annotation)
    if "openie_triples" not in data and "openie" in data:
        data["openie_triples"] = data.get("openie")
    return data


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return converted if isinstance(converted, dict) else {}
    return {}


def _answer_focus_cues(text: str) -> list[str]:
    lower = text.lower()
    cues: list[str] = []
    for cue in ("how many", "what nationality", "what year", "what date", "where", "when", "who", "which", "what"):
        if cue in lower:
            cues.append(cue)
    return _unique(cues)


def _answer_type_hint(cue: str) -> str:
    cue = cue.lower()
    if "nationality" in cue:
        return "Nationality"
    if "where" in cue:
        return "Location"
    if "when" in cue or "year" in cue or "date" in cue:
        return "Date"
    if "who" in cue:
        return "Person"
    if "how many" in cue:
        return "Number"
    return "EntityOrAttribute"


def _semantic_hint_for_role(role: str) -> str:
    if role in {"nationality"}:
        return "attribute"
    if role in {"date", "place", "county", "city", "country", "university"}:
        return "value_slot"
    return "role"


def _aligned_entities(text: str, mappings: list[dict[str, Any]]) -> list[str]:
    normalized = _norm(text)
    aligned: list[str] = []
    for mapping in mappings:
        placeholder = str(mapping.get("placeholder") or "").strip()
        original = str(mapping.get("original_text") or "").strip()
        if placeholder and _norm(placeholder) in normalized:
            aligned.append(placeholder)
        elif original and _norm(original) in normalized:
            aligned.append(placeholder or original)
    return _unique(aligned)


def _triple_inside_single_masked_entity(subject: str, relation: str, object_value: str, mappings: list[dict[str, Any]]) -> bool:
    parts = [_norm(subject), _norm(relation), _norm(object_value)]
    for mapping in mappings:
        original = _norm(str(mapping.get("original_text") or ""))
        placeholder = _norm(str(mapping.get("placeholder") or ""))
        if not original and not placeholder:
            continue
        if placeholder and all(part == placeholder or part in placeholder for part in parts):
            return True
        if original and all(part and part in original for part in parts):
            return True
    return False


def _is_placeholder(word: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z]+[A-Z0-9][A-Za-z0-9]*", word))


def _shortest_path(adjacency: dict[int, list[tuple[int, str]]], start: int, target: int, *, max_edges: int) -> list[int]:
    queue: deque[tuple[int, list[int]]] = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if len(path) - 1 >= max_edges:
            continue
        for neighbor, _relation in adjacency.get(node, []):
            if neighbor in seen:
                continue
            next_path = [*path, neighbor]
            if neighbor == target:
                return next_path
            seen.add(neighbor)
            queue.append((neighbor, next_path))
    return []


def _unique(values: list[str] | Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _int_value(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _int_list(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for item in raw:
        value = _int_value(item)
        if value:
            result.append(value)
    return result


def _float_value(raw: Any, *, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default
