from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Any

from models import ExplicitEntity, ExplicitEntityResult, MaskSpan, MaskSpanResult
from prompts import (
    EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
    build_explicit_entity_extraction_prompt,
)

if TYPE_CHECKING:
    from llm_client import LLMClient


SIMPLE_TYPE_VARIABLES = {
    "actor",
    "ai",
    "age",
    "ceo",
    "city",
    "company",
    "country",
    "director",
    "nationality",
    "population",
    "region",
    "university",
}

LEADING_TYPE_SPAN_WORDS = {"a", "an", "the"}
WH_SPAN_WORDS = {"how", "what", "whatever", "when", "where", "which", "who", "whom", "whose"}
NAMED_ENTITY_SEMANTIC_TYPES = {
    "album",
    "book",
    "city",
    "company",
    "country",
    "event",
    "film",
    "institution",
    "location",
    "movie",
    "organization",
    "organisation",
    "person",
    "place",
    "product",
    "region",
    "series",
    "song",
    "university",
    "work",
}

TYPE_PHRASE_PATTERNS = [
    (r"\blocal food distribution network\b", "type_variable", "Network", "multi-word functional noun phrase"),
    (r"\bfood distribution network\b", "type_variable", "Network", "multi-word functional noun phrase"),
    (r"\bdistribution network\b", "type_variable", "Network", "multi-word functional noun phrase"),
    (r"\bartificial intelligence company\b", "type_variable", "Company", "multi-word company type phrase"),
    (r"\bmixed-use space\b", "type_variable", "Space", "hyphenated multi-word type phrase"),
]

TITLE_HEADS = {
    "album": "Album",
    "book": "Book",
    "film": "Film",
    "game": "Game",
    "movie": "Film",
    "novel": "Book",
    "play": "Work",
    "series": "Series",
    "song": "Song",
    "work": "Work",
}

HUMAN_CONTEXT_CUES = {
    "actor",
    "actress",
    "age",
    "author",
    "born",
    "ceo",
    "director",
    "elder",
    "eldest",
    "founder",
    "older",
    "oldest",
    "people",
    "person",
    "player",
    "president",
    "singer",
    "who",
    "whom",
    "whose",
    "younger",
    "youngest",
}

NON_PERSON_NAME_WORDS = {
    "academy",
    "album",
    "association",
    "book",
    "city",
    "college",
    "company",
    "corporation",
    "country",
    "film",
    "foundation",
    "inc",
    "institute",
    "ltd",
    "movie",
    "network",
    "organization",
    "organisation",
    "school",
    "song",
    "university",
}

PERSON_NAME_PARTICLES = {"al", "bin", "da", "de", "del", "der", "di", "la", "le", "van", "von"}
UNICODE_WORD_TOKEN_PATTERN = r"[^\W\d_][\w'.-]*"
CAPITALIZED_ENTITY_TOKEN = r"(?:[A-Z][A-Za-z0-9']+|[A-Z]\.|[A-Z]{2,}(?:\.)?)"
CAPITALIZED_ENTITY_CONNECTORS = {"de", "for", "la", "of", "the"}
CONTEXT_SEMANTIC_TYPE_CUES = [
    ("film", "Film"),
    ("movie", "Film"),
    ("song", "Song"),
    ("album", "Album"),
    ("book", "Book"),
    ("game", "Game"),
    ("university", "University"),
    ("institution", "Institution"),
    ("school", "Institution"),
    ("company", "Company"),
    ("organization", "Organization"),
    ("organisation", "Organization"),
    ("city", "City"),
    ("country", "Country"),
    ("region", "Region"),
    ("province", "Region"),
    ("state", "Region"),
    ("location", "Location"),
    ("place", "Location"),
]

COORDINATED_DESIGNATION_HEAD_TYPES = {
    "battle": "Event",
    "battles": "Event",
    "campaign": "Event",
    "conference": "Event",
    "congress": "Event",
    "council": "Event",
    "operation": "Event",
    "rebellion": "Event",
    "revolt": "Event",
    "siege": "Event",
    "treaty": "Event",
    "war": "Event",
}
COORDINATED_DESIGNATION_PREPOSITIONS = {"at", "between", "for", "in", "near", "of", "on"}
INTERNAL_NAME_CONNECTORS = {
    "al",
    "bin",
    "da",
    "de",
    "del",
    "der",
    "di",
    "el",
    "ibn",
    "la",
    "le",
    "mac",
    "mc",
    "of",
    "saint",
    "st",
    "the",
    "van",
    "von",
}

CLAUSE_BOUNDARY = {
    "and",
    "or",
    "share",
    "shares",
    "shared",
    "have",
    "has",
    "had",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "which",
    "who",
    "that",
}


class ExplicitEntityExtractor:
    """Step 2 extractor for explicit named entities only."""

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.llm_client = llm_client

    def extract(self, question: str) -> ExplicitEntityResult:
        warnings: list[str] = []
        raw_payload: dict[str, Any] | None = None
        deterministic_candidates = _generate_explicit_entity_candidates(question)
        candidate_payloads = _entity_candidate_payloads(deterministic_candidates)
        candidates_by_id = {
            str(candidate["candidate_id"]): deterministic_candidates[index]
            for index, candidate in enumerate(candidate_payloads)
        }
        if self.llm_client is not None:
            try:
                raw = self.llm_client.chat_json(
                    EXPLICIT_ENTITY_EXTRACTION_SYSTEM,
                    build_explicit_entity_extraction_prompt(question, candidate_payloads),
                )
                raw_payload = raw if isinstance(raw, dict) else {}
                raw_payload.setdefault("deterministic_candidates", candidate_payloads)
                normalized_question, normalization_changed, normalization_note = self._parse_normalization_payload(
                    question,
                    raw_payload,
                    warnings,
                )
                llm_entities = self._parse_payload(
                    question,
                    raw_payload,
                    warnings,
                    candidates_by_id=candidates_by_id,
                )
                candidate_decisions_present = any(
                    key in raw_payload for key in ("verified_entities", "candidate_entities")
                )
                if llm_entities or candidate_decisions_present:
                    base_entities = llm_entities
                else:
                    warnings.append("LLM returned no verified explicit entities; using deterministic candidates.")
                    base_entities = deterministic_candidates
                return ExplicitEntityResult(
                    entities=_merge_explicit_entities(
                        _with_structural_explicit_entities(
                            question,
                            base_entities,
                            warnings,
                        ),
                        warnings,
                    ),
                    warnings=warnings,
                    raw_payload=raw_payload,
                    normalized_question=normalized_question,
                    normalization_changed=normalization_changed,
                    normalization_note=normalization_note,
                )
            except Exception as exc:
                warnings.append(f"Explicit entity LLM failed; using heuristic fallback: {exc}")

        return ExplicitEntityResult(
            entities=_merge_explicit_entities(
                _with_structural_explicit_entities(
                    question,
                    deterministic_candidates,
                    warnings,
                ),
                warnings,
            ),
            warnings=warnings,
            raw_payload=raw_payload,
            normalized_question=question,
            normalization_changed=False,
            normalization_note="",
        )

    @staticmethod
    def _parse_normalization_payload(
        question: str,
        payload: dict[str, Any],
        warnings: list[str],
    ) -> tuple[str, bool, str]:
        raw_normalized = payload.get("normalized_question")
        if isinstance(raw_normalized, str) and raw_normalized.strip():
            normalized_question = raw_normalized.strip()
        else:
            normalized_question = question
            if "normalized_question" in payload and raw_normalized not in (None, ""):
                warnings.append("Ignored invalid normalized_question from explicit entity payload.")

        changed = _coerce_bool(
            payload.get("normalization_changed"),
            default=normalized_question != question,
        )
        if normalized_question == question:
            changed = False
        elif "normalization_changed" not in payload:
            changed = True

        note = str(payload.get("normalization_note") or "").strip()
        return normalized_question, changed, note

    @staticmethod
    def _parse_payload(
        question: str,
        payload: dict[str, Any],
        warnings: list[str],
        candidates_by_id: dict[str, ExplicitEntity] | None = None,
    ) -> list[ExplicitEntity]:
        candidates_by_id = candidates_by_id or {}
        raw_entities = payload.get("verified_entities", payload.get("candidate_entities"))
        if raw_entities is None:
            raw_entities = payload.get("entities", payload.get("explicit_entities"))
        if raw_entities is None:
            raw_entities = payload.get("mask_spans", payload.get("maskSpans", []))
        if not isinstance(raw_entities, list):
            warnings.append("Explicit entity payload did not contain a list entities field.")
            return []

        entities: list[ExplicitEntity] = []
        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue
            candidate_id = str(raw.get("candidate_id", raw.get("id", ""))).strip()
            if candidate_id:
                if candidate_id not in candidates_by_id:
                    warnings.append(f"Dropped unknown explicit entity candidate_id={candidate_id!r}.")
                    continue
                if not _coerce_bool(
                    raw.get("is_entity", raw.get("selected", raw.get("keep", True))),
                    default=True,
                ):
                    continue
                candidate = candidates_by_id[candidate_id]
                semantic_type_hint = _normalize_entity_type(
                    raw.get("type", raw.get("semantic_type_hint", raw.get("semantic_type", candidate.semantic_type_hint)))
                )
                confidence = _clamp_float(raw.get("confidence", candidate.confidence), 0.0, 1.0)
                entities.append(
                    ExplicitEntity(
                        text=candidate.text,
                        start_char=candidate.start_char,
                        end_char=candidate.end_char,
                        semantic_type_hint=semantic_type_hint,
                        confidence=confidence,
                        reason=str(raw.get("reason", candidate.reason)).strip(),
                    )
                )
                continue
            if _normalize_kind_hint(raw.get("kind_hint", raw.get("kind", "entity"))) != "entity":
                warnings.append(f"Dropped non-entity explicit entity item: {raw!r}.")
                continue
            text = str(raw.get("surface", raw.get("text", ""))).strip()
            start = _coerce_int(raw.get("start_char", raw.get("start")))
            end = _coerce_int(raw.get("end_char", raw.get("end")))
            if not text:
                continue
            start, end = _resolve_explicit_entity_span(question, text, start, end, warnings)
            if start is None or end is None:
                warnings.append(f"Could not resolve explicit entity text={text!r}.")
                continue
            entity_text = question[start:end]
            if _is_forbidden_explicit_entity(entity_text):
                warnings.append(f"Dropped forbidden non-entity span text={entity_text!r}.")
                continue
            semantic_type_hint = _normalize_entity_type(raw.get("type", raw.get("semantic_type_hint", raw.get("semantic_type", ""))))
            entities.append(
                ExplicitEntity(
                    text=question[start:end],
                    start_char=start,
                    end_char=end,
                    semantic_type_hint=semantic_type_hint,
                    confidence=_clamp_float(raw.get("confidence", 1.0), 0.0, 1.0),
                    reason=str(raw.get("reason", "")).strip(),
                )
            )
        return _merge_explicit_entities(entities, warnings)


class MaskSpanExtractor:
    """Compatibility wrapper for the new explicit-entity Step 2."""

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self.explicit_extractor = ExplicitEntityExtractor(llm_client)

    def extract(self, question: str) -> MaskSpanResult:
        result = self.explicit_extractor.extract(question)
        return MaskSpanResult(
            mask_spans=_mask_spans_from_explicit_entities(result.entities),
            warnings=result.warnings,
            raw_payload=result.raw_payload,
        )


def _mask_spans_from_explicit_entities(entities: list[ExplicitEntity]) -> list[MaskSpan]:
    return [
        MaskSpan(
            text=entity.text,
            start_char=entity.start_char,
            end_char=entity.end_char,
            kind_hint="entity",
            semantic_type_hint=entity.semantic_type_hint or "Entity",
            reason=entity.reason,
        )
        for entity in entities
    ]


def _heuristic_explicit_entities(question: str) -> list[ExplicitEntity]:
    spans = _heuristic_mask_spans(question)
    entities = [
        ExplicitEntity(
            text=span.text,
            start_char=span.start_char,
            end_char=span.end_char,
            semantic_type_hint=_normalize_entity_type(span.semantic_type_hint),
            confidence=0.55,
            reason=span.reason or "deterministic explicit entity fallback",
        )
        for span in spans
        if span.kind_hint == "entity" and not _is_forbidden_explicit_entity(span.text)
    ]
    return _merge_explicit_entities(entities, [])


def _generate_explicit_entity_candidates(question: str) -> list[ExplicitEntity]:
    """Generate deterministic candidate spans before LLM verification.

    The LLM should classify these candidates instead of inventing character
    offsets. Candidate generation prioritizes recall for explicit named
    entities while leaving final yes/no filtering to the LLM.
    """
    candidates: list[ExplicitEntity] = []
    candidates.extend(_heuristic_explicit_entities(question))
    candidates.extend(_single_token_explicit_entity_candidates(question))
    return _dedupe_explicit_entity_candidates(question, candidates)


def _dedupe_explicit_entity_candidates(question: str, candidates: list[ExplicitEntity]) -> list[ExplicitEntity]:
    by_span: dict[tuple[int, int], ExplicitEntity] = {}
    for candidate in candidates:
        start, end = _trim_explicit_entity_boundary(question, candidate.start_char, candidate.end_char)
        if start < 0 or end <= start:
            continue
        text = question[start:end]
        if _is_forbidden_explicit_entity(text):
            continue
        normalized = ExplicitEntity(
            text=text.strip(),
            start_char=start,
            end_char=end,
            semantic_type_hint=_normalize_entity_type(candidate.semantic_type_hint),
            confidence=_clamp_float(candidate.confidence, 0.0, 1.0),
            reason=candidate.reason,
        )
        key = (normalized.start_char, normalized.end_char)
        existing = by_span.get(key)
        if existing is None or normalized.confidence > existing.confidence:
            by_span[key] = normalized
    return sorted(by_span.values(), key=lambda item: (item.start_char, -(item.end_char - item.start_char)))


def _entity_candidate_payloads(candidates: list[ExplicitEntity]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        payloads.append(
            {
                "candidate_id": f"c{index}",
                "text": candidate.text,
                "start_char": candidate.start_char,
                "end_char": candidate.end_char,
                "reason": candidate.reason,
            }
        )
    return payloads


def _single_token_explicit_entity_candidates(question: str) -> list[ExplicitEntity]:
    candidates: list[ExplicitEntity] = []
    for match in re.finditer(UNICODE_WORD_TOKEN_PATTERN, question):
        start, end = _trim_explicit_entity_boundary(question, match.start(), match.end())
        if end <= start:
            continue
        text = question[start:end]
        if (
            _is_forbidden_explicit_entity(text)
            or _starts_sentence_only(question, start, text)
            or not _looks_like_single_token_entity_candidate(question, text, start, end)
        ):
            continue
        candidates.append(
            ExplicitEntity(
                text=text,
                start_char=start,
                end_char=end,
                semantic_type_hint=_infer_semantic_type(text, "entity", question, start, end),
                confidence=0.5,
                reason="single-token proper-name candidate",
            )
        )
    return candidates


def _looks_like_single_token_entity_candidate(
    question: str,
    text: str,
    start: int,
    end: int,
) -> bool:
    del start, end
    stripped = text.strip()
    lowered = stripped.lower().strip(".")
    if len(stripped) < 2 or lowered in INTERNAL_NAME_CONNECTORS or lowered in PERSON_NAME_PARTICLES:
        return False
    if _looks_like_acronym(stripped) or _looks_like_mixedcase_name(stripped):
        return True
    if any(ord(char) > 127 for char in stripped) and stripped[:1].isupper():
        return True
    if stripped[:1].isupper() and re.search(r"[A-Za-z]", stripped):
        return True
    return False


def _with_structural_explicit_entities(
    question: str,
    entities: list[ExplicitEntity],
    warnings: list[str],
) -> list[ExplicitEntity]:
    entities = _with_coordinated_designation_entities(question, entities, warnings)
    return _with_typed_coordinate_title_entities(question, entities, warnings)


def _with_coordinated_designation_entities(
    question: str,
    entities: list[ExplicitEntity],
    warnings: list[str],
) -> list[ExplicitEntity]:
    additions = _coordinated_designation_entities(question)
    if not additions:
        return entities
    existing_spans = {(entity.start_char, entity.end_char) for entity in entities}
    for entity in additions:
        if (entity.start_char, entity.end_char) not in existing_spans:
            warnings.append(f"Added complete coordinated named designation entity text={entity.text!r}.")
            entities.append(entity)
    return entities


def _with_typed_coordinate_title_entities(
    question: str,
    entities: list[ExplicitEntity],
    warnings: list[str],
) -> list[ExplicitEntity]:
    additions = _typed_coordinate_title_entities(question)
    if not additions:
        return entities
    existing_spans = {(entity.start_char, entity.end_char) for entity in entities}
    for entity in additions:
        if (entity.start_char, entity.end_char) in existing_spans:
            continue
        warnings.append(f"Added typed coordinate title candidate entity text={entity.text!r}.")
        entities.append(entity)
        existing_spans.add((entity.start_char, entity.end_char))
    return entities


def _typed_coordinate_title_entities(question: str) -> list[ExplicitEntity]:
    return [
        ExplicitEntity(
            text=span.text,
            start_char=span.start_char,
            end_char=span.end_char,
            semantic_type_hint=span.semantic_type_hint,
            confidence=0.82,
            reason=span.reason,
        )
        for span in _typed_coordinate_title_spans(question)
    ]


def _coordinated_designation_entities(question: str) -> list[ExplicitEntity]:
    """Find official names like 'Battle of X and Y' where 'and' is internal."""
    entities: list[ExplicitEntity] = []
    head_pattern = "|".join(re.escape(head) for head in sorted(COORDINATED_DESIGNATION_HEAD_TYPES, key=len, reverse=True))
    prep_pattern = "|".join(re.escape(prep) for prep in sorted(COORDINATED_DESIGNATION_PREPOSITIONS, key=len, reverse=True))
    pattern = re.compile(rf"\b(?P<head>{head_pattern})\s+(?P<prep>{prep_pattern})\s+", flags=re.IGNORECASE)
    for match in pattern.finditer(question):
        start = match.start()
        body_start = match.end()
        end = _find_internal_coordinated_name_end(question, body_start)
        if end is None or end <= body_start:
            continue
        text = question[start:end].strip()
        if not _is_complete_coordinated_designation(text):
            continue
        semantic_type = COORDINATED_DESIGNATION_HEAD_TYPES.get(match.group("head").lower(), "Entity")
        entities.append(
            ExplicitEntity(
                text=text,
                start_char=start,
                end_char=end,
                semantic_type_hint=semantic_type,
                confidence=0.9,
                reason="complete coordinated named designation",
            )
        )
    return entities


def _find_internal_coordinated_name_end(question: str, start: int) -> int | None:
    token_matches = list(re.finditer(r"\S+", question[start:]))
    if not token_matches:
        return None
    content_count = 0
    saw_and = False
    saw_content_after_and = False
    expecting_after_and = False
    last_content_end: int | None = None
    current_end = start

    for match in token_matches:
        token_start = start + match.start()
        token_end = start + match.end()
        raw_token = match.group(0)
        cleaned = raw_token.strip(" \t\r\n?.,;:!\"'()[]{}")
        lowered = cleaned.lower().strip(".")
        if not cleaned:
            break
        if lowered == "and":
            if content_count > 0 and _looks_like_title_continuation(question, token_end):
                saw_and = True
                expecting_after_and = True
                current_end = token_end
                continue
            break
        if _is_internal_name_content_token(cleaned):
            content_count += 1
            if expecting_after_and:
                saw_content_after_and = True
            current_end = token_end
            last_content_end = token_end
            continue
        if lowered in INTERNAL_NAME_CONNECTORS and content_count > 0:
            current_end = token_end
            continue
        break

    del current_end
    if saw_and and saw_content_after_and and content_count >= 2:
        return last_content_end
    return None


def _is_internal_name_content_token(token: str) -> bool:
    return bool(token[:1].isupper() or token.isupper() or re.search(r"\d", token))


def _is_complete_coordinated_designation(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    if " and " not in lowered:
        return False
    for head in COORDINATED_DESIGNATION_HEAD_TYPES:
        for prep in COORDINATED_DESIGNATION_PREPOSITIONS:
            if lowered.startswith(f"{head} {prep} "):
                return True
    return False


def _resolve_explicit_entity_span(
    question: str,
    text: str,
    start: int | None,
    end: int | None,
    warnings: list[str],
) -> tuple[int | None, int | None]:
    if _valid_bounds(question, start, end) and question[start or 0 : end or 0] == text:
        return _trim_explicit_entity_boundary(question, start or 0, end or 0)

    matches = list(re.finditer(re.escape(text), question))
    if len(matches) == 1:
        match = matches[0]
        repaired_start, repaired_end = _trim_explicit_entity_boundary(question, match.start(), match.end())
        if start is not None or end is not None:
            warnings.append(
                f"Corrected explicit entity span for text={text!r} from ({start}, {end}) "
                f"to ({repaired_start}, {repaired_end})."
            )
        return repaired_start, repaired_end
    if len(matches) > 1:
        warnings.append(f"Dropped ambiguous repeated explicit entity text={text!r}.")
        return None, None

    normalized_match = _find_unique_normalized_match(question, text)
    if normalized_match is not None:
        repaired_start, repaired_end = _trim_explicit_entity_boundary(
            question,
            normalized_match[0],
            normalized_match[1],
        )
        warnings.append(
            f"Corrected explicit entity span by normalized surface match for text={text!r} "
            f"to ({repaired_start}, {repaired_end})."
        )
        return repaired_start, repaired_end

    return None, None


def _trim_explicit_entity_boundary(question: str, start: int, end: int) -> tuple[int, int]:
    while start < end and question[start].isspace():
        start += 1
    while end > start and question[end - 1].isspace():
        end -= 1
    while end > start and question[end - 1] in {"?", "!", ".", ",", ";", ":", "\""}:
        end -= 1

    surface = question[start:end]
    lowered = surface.lower()
    for suffix in ("'s", "’s", "‘s", "`s"):
        if lowered.endswith(suffix):
            end -= len(suffix)
            break
    return start, end


def _find_unique_normalized_match(question: str, text: str) -> tuple[int, int] | None:
    target = _canonical_surface(text)
    if not target:
        return None
    matches: list[tuple[int, int]] = []
    for start in range(len(question)):
        for end in range(start + 1, len(question) + 1):
            candidate = question[start:end]
            if len(_canonical_surface(candidate)) > len(target) + 4:
                break
            if _canonical_surface(candidate) == target:
                matches.append((start, end))
    if len(matches) == 1:
        return matches[0]
    return None


def _canonical_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("’", "'").replace("‘", "'").replace("`", "'")
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _valid_bounds(question: str, start: int | None, end: int | None) -> bool:
    return start is not None and end is not None and 0 <= start < end <= len(question)


def _normalize_entity_type(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Entity"
    words = re.findall(r"[A-Za-z0-9]+", raw)
    normalized = "".join(word[:1].upper() + word[1:] for word in words) if words else "Entity"
    aliases = {
        "Movie": "Film",
        "Place": "Location",
        "Organisation": "Organization",
        "Creativework": "Work",
        "CreativeWork": "Work",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "Album",
        "Book",
        "City",
        "Company",
        "Country",
        "Entity",
        "Event",
        "Film",
        "Game",
        "Institution",
        "Location",
        "Organization",
        "Product",
        "Region",
        "Series",
        "Song",
        "University",
        "Work",
        "Person",
    }
    return normalized if normalized in allowed else "Entity"


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = maximum
    return min(max(number, minimum), maximum)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "selected", "keep"}:
        return True
    if lowered in {"0", "false", "no", "n", "reject", "drop"}:
        return False
    return default


FORBIDDEN_EXPLICIT_ENTITY_SURFACES = {
    *SIMPLE_TYPE_VARIABLES,
    "actor",
    "album",
    "artificial intelligence company",
    "author",
    "birth date",
    "book",
    "cause",
    "ceo",
    "chief operating officer",
    "date",
    "death date",
    "father",
    "film",
    "founder",
    "husband",
    "movie",
    "reason",
    "research institute",
    "song",
    "spouse",
    "wife",
}
FORBIDDEN_PREFIXES = {
    "born in",
    "ceo of",
    "company that",
    "developed by",
    "director of",
    "graduated from",
    "located in",
    "released first",
    "wife of",
}


def _is_forbidden_explicit_entity(text: str) -> bool:
    stripped = text.strip().strip("?.!,;:")
    lowered = re.sub(r"\s+", " ", stripped.lower())
    if not lowered:
        return True
    if lowered in WH_SPAN_WORDS or lowered in FORBIDDEN_EXPLICIT_ENTITY_SURFACES:
        return True
    if lowered in {
        "a",
        "an",
        "and",
        "both",
        "different",
        "first",
        "of",
        "or",
        "same",
        "share",
        "the",
        "older",
        "younger",
    }:
        return True
    if any(lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return True
    if lowered.startswith(("which ", "what ", "who ", "whose ", "when ", "where ", "why ")):
        return not _looks_like_official_title(stripped)
    if lowered.startswith(("was ", "were ", "is ", "are ", "has ", "have ", "had ", "from ")):
        return True
    return False


def _looks_like_official_title(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text)
    if len(tokens) < 3:
        return False
    title_like = sum(1 for token in tokens if token[:1].isupper() or token.isdigit())
    return title_like >= max(2, len(tokens) - 1)


def _merge_explicit_entities(
    entities: list[ExplicitEntity],
    warnings: list[str],
) -> list[ExplicitEntity]:
    by_span: dict[tuple[int, int], ExplicitEntity] = {}
    for entity in entities:
        if entity.start_char < 0 or entity.end_char <= entity.start_char:
            continue
        if _is_forbidden_explicit_entity(entity.text):
            warnings.append(f"Dropped forbidden non-entity span text={entity.text!r}.")
            continue
        normalized = ExplicitEntity(
            text=entity.text.strip(),
            start_char=entity.start_char,
            end_char=entity.end_char,
            semantic_type_hint=_normalize_entity_type(entity.semantic_type_hint),
            confidence=_clamp_float(entity.confidence, 0.0, 1.0),
            reason=entity.reason,
        )
        key = (normalized.start_char, normalized.end_char)
        existing = by_span.get(key)
        if existing is None or normalized.confidence > existing.confidence:
            by_span[key] = normalized

    ordered = sorted(by_span.values(), key=_explicit_entity_merge_key)
    kept: list[ExplicitEntity] = []
    occupied: list[tuple[int, int]] = []
    for entity in ordered:
        if any(not (entity.end_char <= start or entity.start_char >= end) for start, end in occupied):
            warnings.append(f"Dropped overlapping non-minimal entity span text={entity.text!r}.")
            continue
        kept.append(entity)
        occupied.append((entity.start_char, entity.end_char))
    return sorted(kept, key=lambda item: item.start_char)


def _explicit_entity_merge_key(entity: ExplicitEntity) -> tuple[int, int, float, int]:
    length = entity.end_char - entity.start_char
    if _is_complete_coordinated_designation(entity.text):
        return (0, -length, -entity.confidence, entity.start_char)
    if _token_count(entity.text) >= 2:
        return (1, -length, -entity.confidence, entity.start_char)
    return (2, length, -entity.confidence, entity.start_char)


def _heuristic_mask_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    spans.extend(_coordinated_designation_mask_spans(question))
    spans.extend(_colon_title_entity_spans(question))
    spans.extend(_title_spans_after_type_heads(question))
    spans.extend(_typed_coordinate_title_spans(question))
    spans.extend(_parenthetical_entity_spans(question))
    spans.extend(_quoted_spans(question))
    spans.extend(_person_name_token_spans(question))
    spans.extend(_capitalized_entity_spans(question))
    return _merge_spans(question, spans, [])


def _colon_title_entity_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    connector_pattern = "|".join(re.escape(item) for item in sorted(CAPITALIZED_ENTITY_CONNECTORS))
    token_pattern = (
        rf"{CAPITALIZED_ENTITY_TOKEN}"
        rf"(?:\s+(?:{CAPITALIZED_ENTITY_TOKEN}|{connector_pattern}|\d+))*"
    )
    pattern = re.compile(
        rf"\b(?P<title>{token_pattern})\s*:\s*(?P<subtitle>{token_pattern}(?:\s*\([^)]*\))?)",
    )
    for match in pattern.finditer(question):
        start, end = _trim_explicit_entity_boundary(question, match.start(), match.end())
        if end <= start:
            continue
        text = question[start:end]
        if _is_mask_worthy(text):
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=start,
                    end_char=end,
                    kind_hint="entity",
                    semantic_type_hint="Work",
                    reason="colon-connected title or subtitle",
                )
            )
    return spans


def _coordinated_designation_mask_spans(question: str) -> list[MaskSpan]:
    return [
        MaskSpan(
            text=entity.text,
            start_char=entity.start_char,
            end_char=entity.end_char,
            kind_hint="entity",
            semantic_type_hint=entity.semantic_type_hint,
            reason=entity.reason,
        )
        for entity in _coordinated_designation_entities(question)
    ]


def _title_spans_after_type_heads(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    for match in re.finditer(
        r"\b(?P<head>film|movie|book|album|song|novel|play|series|work|game)\s+",
        question,
        flags=re.IGNORECASE,
    ):
        head = match.group("head").lower()
        start = match.end()
        end = _find_title_end(question, start)
        if end <= start:
            continue
        text = question[start:end].strip()
        leading_ws = len(question[start:end]) - len(question[start:end].lstrip())
        trailing_ws = len(question[start:end]) - len(question[start:end].rstrip())
        start += leading_ws
        end -= trailing_ws
        text = question[start:end]
        if _is_mask_worthy(text):
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=start,
                    end_char=end,
                    kind_hint="entity",
                    semantic_type_hint=TITLE_HEADS.get(head, "Entity"),
                    reason="complex title after explicit type head",
                )
            )
    return spans


def _typed_coordinate_title_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    type_pattern = "|".join(re.escape(head) for head in sorted(TITLE_HEADS, key=len, reverse=True))
    for comma in re.finditer(",", question):
        prefix = question[: comma.start()]
        type_matches = list(re.finditer(rf"\b(?P<head>{type_pattern})\b", prefix, flags=re.IGNORECASE))
        if not type_matches or not re.search(r"\b(?:which|what)\b", prefix, flags=re.IGNORECASE):
            continue
        head = type_matches[-1].group("head").lower()
        tail_start = comma.end()
        tail_end = _coordinate_tail_end(question, tail_start)
        for start, end in _or_coordinate_parts(question, tail_start, tail_end):
            start, end = _trim_explicit_entity_boundary(question, start, end)
            if end <= start:
                continue
            text = question[start:end]
            if not _looks_like_typed_coordinate_title_candidate(text):
                continue
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=start,
                    end_char=end,
                    kind_hint="entity",
                    semantic_type_hint=TITLE_HEADS.get(head, "Work"),
                    reason="typed coordinate title candidate",
                )
            )
    return spans


def _coordinate_tail_end(question: str, start: int) -> int:
    match = re.search(r"[?;]", question[start:])
    if match:
        return start + match.start()
    return len(question)


def _or_coordinate_parts(question: str, start: int, end: int) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(r"\s+or\s+", question[start:end], flags=re.IGNORECASE):
        split_start = start + match.start()
        split_end = start + match.end()
        parts.append((cursor, split_start))
        cursor = split_end
    parts.append((cursor, end))
    return parts


def _looks_like_typed_coordinate_title_candidate(text: str) -> bool:
    stripped = text.strip()
    if _is_forbidden_explicit_entity(stripped):
        return False
    if _starts_with_wh_span_word(stripped) and not _looks_like_official_title(stripped):
        return False
    if _token_count(stripped) < 2:
        return False
    if not re.search(r"[A-Z0-9:()\"']", stripped):
        return False
    lowered_tokens = {token.lower() for token in re.findall(UNICODE_WORD_TOKEN_PATTERN, stripped)}
    if lowered_tokens and lowered_tokens <= CLAUSE_BOUNDARY:
        return False
    return True


def _find_title_end(question: str, start: int) -> int:
    token_matches = list(re.finditer(r"\S+", question[start:]))
    if not token_matches:
        return start
    end = start
    previous_end = start
    for index, match in enumerate(token_matches):
        token_start = start + match.start()
        token_end = start + match.end()
        cleaned = match.group(0).strip("?,.;:")
        lowered = cleaned.lower()
        if index > 0 and not (
            _is_capitalized_entity_token(cleaned)
            or lowered in CAPITALIZED_ENTITY_CONNECTORS
            or re.search(r"\d|[:()\"']", cleaned)
        ):
            break
        if index > 0 and lowered in CLAUSE_BOUNDARY:
            if lowered in {"and", "or"} and _looks_like_title_continuation(question, token_end):
                previous_end = token_end
                continue
            break
        end = token_end
        previous_end = token_end
    return end or previous_end


def _looks_like_title_continuation(question: str, position: int) -> bool:
    next_match = re.search(r"\S+", question[position:])
    if not next_match:
        return False
    token = next_match.group(0).strip("?,.;:")
    return bool(token[:1].isupper() or re.search(r"\d|[:()\"']", token))


def _parenthetical_entity_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    pattern = re.compile(
        rf"\b{CAPITALIZED_ENTITY_TOKEN}(?:\s+{CAPITALIZED_ENTITY_TOKEN}){{0,5}}\s*\([^)]*\)"
    )
    for match in pattern.finditer(question):
        text = match.group(0)
        if _is_mask_worthy(text):
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=match.start(),
                    end_char=match.end(),
                    kind_hint="entity",
                    semantic_type_hint=_infer_semantic_type(text, "entity", question, match.start(), match.end()),
                    reason="entity with parenthetical qualifier",
                )
            )
    return spans


def _quoted_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    for match in re.finditer(r"[\"“”']([^\"“”']{3,})[\"“”']", question):
        text = match.group(1).strip()
        start = match.start(1) + (len(match.group(1)) - len(match.group(1).lstrip()))
        end = start + len(text)
        if _is_mask_worthy(text):
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=start,
                    end_char=end,
                    kind_hint="entity",
                    semantic_type_hint=_infer_semantic_type(text, "entity", question, start, end),
                    reason="quoted complex title/name",
                )
            )
    return spans


def _person_name_token_spans(question: str) -> list[MaskSpan]:
    if not _question_has_human_context(question, None, None):
        return []
    spans: list[MaskSpan] = []
    token_matches = list(re.finditer(UNICODE_WORD_TOKEN_PATTERN, question))
    index = 0
    while index < len(token_matches):
        match = token_matches[index]
        token = match.group(0).strip()
        if not _is_person_name_token(token) or _starts_sentence_only(question, match.start(), token):
            index += 1
            continue
        start = match.start()
        end = match.end()
        content_count = 1 if token.lower().strip(".") not in PERSON_NAME_PARTICLES else 0
        cursor = index + 1
        while cursor < len(token_matches):
            next_match = token_matches[cursor]
            between = question[end : next_match.start()]
            if not between.isspace():
                break
            next_token = next_match.group(0).strip()
            lowered = next_token.lower().strip(".")
            if lowered in {"and", "or"}:
                break
            if not _is_person_name_token(next_token):
                break
            end = next_match.end()
            if lowered not in PERSON_NAME_PARTICLES:
                content_count += 1
            cursor += 1
        text = question[start:end]
        if content_count >= 2 and _looks_like_person_name(text):
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=start,
                    end_char=end,
                    kind_hint="entity",
                    semantic_type_hint="Person",
                    reason="multi-token person name in human context",
                )
            )
            index = cursor
            continue
        index += 1
    return spans


def _is_person_name_token(token: str) -> bool:
    stripped = token.strip()
    lowered = stripped.lower().strip(".")
    if lowered in PERSON_NAME_PARTICLES:
        return True
    if re.fullmatch(r"[A-Z]\.", stripped):
        return True
    return bool(stripped[:1].isupper() and re.fullmatch(UNICODE_WORD_TOKEN_PATTERN, stripped))


def _is_capitalized_entity_token(token: str) -> bool:
    stripped = token.strip()
    return bool(
        (stripped[:1].isupper() and re.fullmatch(UNICODE_WORD_TOKEN_PATTERN, stripped))
        or re.fullmatch(r"[A-Z]\.", stripped)
        or re.fullmatch(r"[A-Z]{2,}(?:\.)?", stripped.strip("."))
    )


def _capitalized_entity_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    token_matches = list(re.finditer(UNICODE_WORD_TOKEN_PATTERN, question))
    index = 0
    while index < len(token_matches):
        match = token_matches[index]
        token = match.group(0).strip()
        if not _is_capitalized_entity_token(token):
            index += 1
            continue

        start = match.start()
        end = match.end()
        last_content_end = end
        content_count = 1
        cursor = index + 1
        while cursor < len(token_matches):
            next_match = token_matches[cursor]
            between = question[end : next_match.start()]
            if not between.isspace():
                break
            next_token = next_match.group(0).strip()
            lowered = next_token.lower().strip(".")
            if _is_capitalized_entity_token(next_token):
                end = next_match.end()
                last_content_end = end
                content_count += 1
                cursor += 1
                continue
            if lowered in CAPITALIZED_ENTITY_CONNECTORS:
                end = next_match.end()
                cursor += 1
                continue
            break

        end = last_content_end
        text = question[start:end].strip()
        if (
            content_count < 2
            or _starts_sentence_only(question, start, text)
            or not _is_mask_worthy(text)
        ):
            index += 1
            continue
        spans.append(
            MaskSpan(
                text=text,
                start_char=start,
                end_char=end,
                kind_hint="entity",
                semantic_type_hint=_infer_semantic_type(text, "entity", question, start, end),
                reason="continuous multi-word named entity",
            )
        )
        index = max(cursor, index + 1)
    return spans


def _type_phrase_spans(question: str) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    for pattern, kind, semantic_type, reason in TYPE_PHRASE_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            text = match.group(0)
            if _is_simple_type_variable(text):
                continue
            spans.append(
                MaskSpan(
                    text=text,
                    start_char=match.start(),
                    end_char=match.end(),
                    kind_hint=kind,
                    semantic_type_hint=semantic_type,
                    reason=reason,
                )
            )
    return spans


def _merge_spans(
    question: str,
    spans: list[MaskSpan],
    warnings: list[str],
) -> list[MaskSpan]:
    normalized: list[MaskSpan] = []
    seen: set[tuple[int, int]] = set()
    for span in spans:
        start, end = _resolve_span(question, span.text, span.start_char, span.end_char)
        if start is None or end is None:
            continue
        kind_hint = _normalize_kind_hint(span.kind_hint)
        start, end = _trim_mask_span(
            question=question,
            start=start,
            end=end,
            kind_hint=kind_hint,
            semantic_type_hint=span.semantic_type_hint,
        )
        text = question[start:end]
        if not _is_mask_worthy(
            text,
            kind_hint=kind_hint,
            semantic_type_hint=span.semantic_type_hint,
            question=question,
            start=start,
            end=end,
        ):
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            MaskSpan(
                text=text,
                start_char=start,
                end_char=end,
                kind_hint=kind_hint,
                semantic_type_hint=_refine_semantic_type(
                    question=question,
                    text=text,
                    kind_hint=span.kind_hint,
                    existing=span.semantic_type_hint,
                    start=start,
                    end=end,
                ),
                reason=span.reason,
            )
        )

    ordered = sorted(normalized, key=lambda item: (item.start_char, -(item.end_char - item.start_char)))
    result: list[MaskSpan] = []
    occupied: list[tuple[int, int]] = []
    for span in ordered:
        if any(not (span.end_char <= start or span.start_char >= end) for start, end in occupied):
            warnings.append(f"Dropped overlapping mask span text={span.text!r}.")
            continue
        result.append(span)
        occupied.append((span.start_char, span.end_char))
    return result


def _filter_llm_mask_spans(
    question: str,
    spans: list[MaskSpan],
    warnings: list[str],
) -> list[MaskSpan]:
    """Validate LLM spans without rewriting their accepted boundaries."""
    result: list[MaskSpan] = []
    occupied: list[tuple[int, int]] = []
    for span in sorted(spans, key=lambda item: (item.start_char, -(item.end_char - item.start_char))):
        kind_hint = _normalize_kind_hint(span.kind_hint)
        if not _is_llm_mask_span_allowed(
            span.text,
            kind_hint=kind_hint,
            semantic_type_hint=span.semantic_type_hint,
        ):
            warnings.append(f"Dropped invalid mask span text={span.text!r}.")
            continue
        if any(not (span.end_char <= start or span.start_char >= end) for start, end in occupied):
            warnings.append(f"Dropped overlapping mask span text={span.text!r}.")
            continue
        result.append(
            MaskSpan(
                text=span.text,
                start_char=span.start_char,
                end_char=span.end_char,
                kind_hint=kind_hint,
                semantic_type_hint=_refine_semantic_type(
                    question=question,
                    text=span.text,
                    kind_hint=kind_hint,
                    existing=span.semantic_type_hint,
                    start=span.start_char,
                    end=span.end_char,
                ),
                reason=span.reason,
            )
        )
        occupied.append((span.start_char, span.end_char))
    return result


def _is_llm_mask_span_allowed(
    text: str,
    kind_hint: str,
    semantic_type_hint: str | None,
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _starts_with_wh_span_word(stripped) and not _looks_like_official_title(stripped):
        return False
    if _token_count(stripped) < 2:
        return False
    if kind_hint == "type_variable":
        content_text = _drop_leading_type_span_words(stripped)
        if _token_count(content_text) < 2:
            return False
        if _is_simple_type_variable(content_text):
            return False
        return True
    if _has_named_entity_semantic_type(semantic_type_hint):
        return True
    return _token_count(stripped) >= 2


def _trim_mask_span(
    question: str,
    start: int,
    end: int,
    kind_hint: str,
    semantic_type_hint: str | None,
) -> tuple[int, int]:
    del semantic_type_hint
    while start < end and question[start].isspace():
        start += 1
    while end > start and question[end - 1].isspace():
        end -= 1

    if kind_hint == "type_variable":
        while True:
            match = re.match(r"\s*([A-Za-z][A-Za-z0-9'-]*)(\s+)", question[start:end])
            if not match or match.group(1).lower() not in LEADING_TYPE_SPAN_WORDS:
                break
            start += match.end()
    return start, end


def _is_mask_worthy(
    text: str,
    kind_hint: str = "entity",
    semantic_type_hint: str | None = None,
    question: str = "",
    start: int | None = None,
    end: int | None = None,
) -> bool:
    stripped = text.strip()
    if _is_simple_type_variable(stripped):
        return False
    token_count = _token_count(stripped)
    if _starts_with_wh_span_word(stripped) and not _is_wh_leading_title_candidate(stripped, question, start):
        return False
    if token_count < 2:
        return False

    if kind_hint == "type_variable":
        content_text = _drop_leading_type_span_words(stripped)
        return _token_count(content_text) >= 2 and not _is_simple_type_variable(content_text)
    return True


def _is_wh_leading_title_candidate(text: str, question: str = "", start: int | None = None) -> bool:
    if not _looks_like_official_title(text):
        return False
    if question and start is not None and _starts_sentence_only(question, start, text):
        return False
    return True


def _is_simple_type_variable(text: str) -> bool:
    return text.strip().lower() in SIMPLE_TYPE_VARIABLES


def _starts_with_wh_span_word(text: str) -> bool:
    first = _first_token(text)
    return first in WH_SPAN_WORDS


def _drop_leading_type_span_words(text: str) -> str:
    result = text.strip()
    while True:
        match = re.match(r"([A-Za-z][A-Za-z0-9'-]*)(\s+)", result)
        if not match:
            return result
        if match.group(1).lower() not in LEADING_TYPE_SPAN_WORDS:
            return result
        result = result[match.end() :].strip()


def _first_token(text: str) -> str:
    match = re.match(rf"\s*({UNICODE_WORD_TOKEN_PATTERN})", text)
    return match.group(1).lower() if match else ""


def _looks_like_acronym(text: str) -> bool:
    stripped = text.strip(".")
    return bool(
        re.fullmatch(r"(?:[A-Z]\.){2,}[A-Z]?\.?", text)
        or re.fullmatch(r"[A-Z]{2,}[A-Z0-9]*", stripped)
    )


def _looks_like_mixedcase_name(text: str) -> bool:
    return bool(
        re.fullmatch(r"[^\W_]*[^\W\d_A-Z][A-Z][\w'.-]*", text)
        or re.fullmatch(r"[^\W\d_]+[0-9][\w'.-]*", text)
    )


def _looks_like_single_token_proper_name(text: str) -> bool:
    return bool(text[:1].isupper() and re.fullmatch(r"[\w'._-]+", text))


def _has_named_entity_semantic_type(value: str | None) -> bool:
    normalized = re.sub(r"[^A-Za-z]+", " ", value or "").strip().lower()
    if not normalized:
        return False
    return any(word in NAMED_ENTITY_SEMANTIC_TYPES for word in normalized.split())


def _starts_sentence_only(question: str, start: int, text: str) -> bool:
    if start != 0:
        return False
    first = re.match(r"\w+", text)
    if not first:
        return False
    return first.group(0).lower() in {
        "are",
        "did",
        "do",
        "does",
        "how",
        "is",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }


def _infer_semantic_type(
    text: str,
    kind_hint: str,
    question: str = "",
    start: int | None = None,
    end: int | None = None,
) -> str:
    lowered = text.lower()
    if "film" in lowered or "movie" in lowered:
        return "Film"
    if "network" in lowered:
        return "Network"
    if "company" in lowered:
        return "Company"
    if "university" in lowered:
        return "University"
    if "city" in lowered:
        return "City"
    context_type = _semantic_type_from_local_context(question, start, end)
    if context_type is not None:
        return context_type
    if (
        kind_hint == "entity"
        and question
        and _looks_like_person_name(text)
        and _question_has_human_context(question, start, end)
    ):
        return "Person"
    return _normalize_entity_type(text) if kind_hint == "entity" else _normalize_type_variable(text)


def _normalize_type_variable(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.strip())
    if not words:
        return "Variable"
    return "".join(word[:1].upper() + word[1:] for word in words)


def _semantic_type_from_local_context(
    question: str,
    start: int | None,
    end: int | None,
) -> str | None:
    if not question or start is None or end is None:
        return None
    window_start = max(0, start - 80)
    window_end = min(len(question), end + 80)
    local_text = question[window_start:window_end].lower()
    best: tuple[int, str] | None = None
    for cue, semantic_type in CONTEXT_SEMANTIC_TYPE_CUES:
        for match in re.finditer(rf"\b{re.escape(cue)}\b", local_text):
            cue_center = window_start + match.start() + (match.end() - match.start()) // 2
            distance = min(abs(cue_center - start), abs(cue_center - end))
            if best is None or distance < best[0]:
                best = (distance, semantic_type)
    return best[1] if best is not None else None


def _refine_semantic_type(
    question: str,
    text: str,
    kind_hint: str,
    existing: str | None,
    start: int | None,
    end: int | None,
) -> str:
    inferred = _infer_semantic_type(text, kind_hint, question, start, end)
    existing = (existing or "").strip()
    if not existing:
        return inferred
    if inferred == "Person" and _is_generic_or_surface_semantic_type(existing, text):
        return inferred
    return existing


def _is_generic_or_surface_semantic_type(value: str, text: str) -> bool:
    normalized_value = re.sub(r"[^A-Za-z0-9]+", "", value).lower()
    surface_value = re.sub(r"[^A-Za-z0-9]+", "", text).lower()
    return normalized_value in {
        "",
        "entity",
        "someentity",
        "namedentity",
        "unknown",
        "thing",
        surface_value,
    }


def _looks_like_person_name(text: str) -> bool:
    if re.search(r"\d|[:()\[\]{}\"']", text):
        return False
    words = re.findall(UNICODE_WORD_TOKEN_PATTERN, text)
    if len(words) < 2 or len(words) > 5:
        return False
    lowered_words = {word.lower().strip("'") for word in words}
    if lowered_words & NON_PERSON_NAME_WORDS:
        return False
    content_words = [word for word in words if word.lower() not in PERSON_NAME_PARTICLES]
    if len(content_words) < 2:
        return False
    return all(word[:1].isupper() or word.isupper() for word in content_words)


def _question_has_human_context(
    question: str,
    start: int | None,
    end: int | None,
) -> bool:
    lowered_words = set(re.findall(r"[A-Za-z]+", question.lower()))
    if lowered_words & HUMAN_CONTEXT_CUES:
        return True
    if start is None or end is None:
        return False
    local_left = question[max(0, start - 40) : start].lower()
    local_right = question[end : min(len(question), end + 40)].lower()
    local_words = set(re.findall(r"[A-Za-z]+", f"{local_left} {local_right}"))
    return bool(local_words & HUMAN_CONTEXT_CUES)


def _normalize_kind_hint(value: object) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {
        "functional_noun_phrase",
        "function_noun_phrase",
        "multi_word_type",
        "noun_phrase",
        "type",
        "type_phrase",
        "type_variable",
        "type-variable",
        "variable",
    }:
        return "type_variable"
    return "entity"


def _resolve_span(
    question: str,
    text: str,
    start: int | None,
    end: int | None,
) -> tuple[int | None, int | None]:
    if start is not None and end is not None and 0 <= start < end <= len(question):
        if question[start:end].strip().lower() == text.strip().lower():
            leading = len(question[start:end]) - len(question[start:end].lstrip())
            trailing = len(question[start:end]) - len(question[start:end].rstrip())
            return start + leading, end - trailing
    matches = list(re.finditer(re.escape(text.strip()), question, flags=re.IGNORECASE))
    if not matches:
        return None, None
    match = matches[0]
    return match.start(), match.end()


def _token_count(text: str) -> int:
    return len(re.findall(r"[^\W_]+", text))


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
