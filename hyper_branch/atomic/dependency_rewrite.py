"""在检索依赖节点前，将已回答的 DAG 引用替换为具体值。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..utils import normalize_label


_INSUFFICIENT_ANSWERS = {
    "",
    "insufficient_evidence",
    "insufficient evidence",
    "unknown",
    "none",
    "n/a",
    "yes",
    "no",
    "true",
    "false",
}
_MONTHS = {
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
_NON_ENTITY_SINGLE_WORDS = {
    "american",
    "athenian",
    "british",
    "canadian",
    "chinese",
    "dutch",
    "english",
    "french",
    "german",
    "greek",
    "indian",
    "irish",
    "italian",
    "japanese",
    "polish",
    "russian",
    "spanish",
}
_NON_ENTITY_ANSWER_TYPES = {
    "age",
    "boolean",
    "count",
    "date",
    "duration",
    "nationality",
    "number",
    "time",
    "year",
}


@dataclass(slots=True)
class DependencyReplacement:
    dependency_node_id: str
    replacement_span: str
    replacement_answer: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dependency_node_id": self.dependency_node_id,
            "replacement_span": self.replacement_span,
            "replacement_answer": self.replacement_answer,
        }


@dataclass(slots=True)
class DependencyQuestionRewrite:
    original_question: str
    retrieval_question: str
    whether_rewritten: bool
    replacement_span: str = ""
    replacement_answer: str = ""
    replacements: list[DependencyReplacement] = field(default_factory=list)
    unresolved_dependencies: list[dict[str, Any]] = field(default_factory=list)
    primary_anchor_entities: list[str] = field(default_factory=list)
    dependency_answers_used: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "resolved_question": self.retrieval_question,
            "retrieval_question": self.retrieval_question,
            "whether_rewritten": self.whether_rewritten,
            "replacement_span": self.replacement_span,
            "replacement_answer": self.replacement_answer,
            "replacements": [item.to_dict() for item in self.replacements],
            "dependency_replacements": [item.to_dict() for item in self.replacements],
            "unresolved_dependencies": list(self.unresolved_dependencies),
            "primary_anchor_entities": list(self.primary_anchor_entities),
            "dependency_answers_used": list(self.dependency_answers_used),
        }


def resolve_dependency_question(
    question: str,
    dependency_answers: list[dict[str, Any]],
) -> DependencyQuestionRewrite:
    """Resolve explicit dependency-answer variables before legacy phrase rewriting."""

    retrieval_question = question
    replacements: list[DependencyReplacement] = []
    unresolved_dependencies: list[dict[str, Any]] = []
    primary_anchor_entities: list[str] = []
    dependency_answers_used: list[dict[str, Any]] = []

    for dependency in dependency_answers:
        # 显式 qN_answer 引用没有歧义，因此优先处理。
        dependency_node_id = str(dependency.get("node_id", "") or "").strip()
        if not dependency_node_id:
            continue
        matches = _find_dependency_variable_matches(retrieval_question, dependency_node_id)
        if not matches:
            continue
        answer = str(dependency.get("answer", "") or "").strip()
        if (
            not answer
            or normalize_label(answer).lower() in _INSUFFICIENT_ANSWERS
        ):
            unresolved_dependencies.append(
                {
                    "node_id": dependency_node_id,
                    "reason": "missing_or_insufficient_answer",
                    "answer": answer,
                }
            )
            continue
        for match in reversed(matches):
            retrieval_question = (
                retrieval_question[: match.start()]
                + answer
                + retrieval_question[match.end() :]
            )
            replacements.insert(
                0,
                DependencyReplacement(
                    dependency_node_id=dependency_node_id,
                    replacement_span=match.group(0),
                    replacement_answer=answer,
                ),
            )
        _append_anchor_mentions(primary_anchor_entities, answer)
        dependency_answers_used.append(_dependency_answer_summary(dependency))

    if replacements:
        first_replacement = replacements[0]
        return DependencyQuestionRewrite(
            original_question=question,
            retrieval_question=retrieval_question,
            whether_rewritten=True,
            replacement_span=first_replacement.replacement_span,
            replacement_answer=first_replacement.replacement_answer,
            replacements=replacements,
            unresolved_dependencies=unresolved_dependencies,
            primary_anchor_entities=primary_anchor_entities,
            dependency_answers_used=dependency_answers_used,
        )

    # 旧版 DAG 没有 qN_answer 引用时，才使用保守的短语匹配回退。
    for dependency in dependency_answers:
        answer = str(dependency.get("answer", "") or "").strip()
        if not answer or normalize_label(answer).lower() in _INSUFFICIENT_ANSWERS:
            continue
        if not is_entity_like_answer(answer, dependency.get("answer_type")):
            continue
        dependency_question = str(dependency.get("question", "") or "")
        match = _find_dependency_reference_match(retrieval_question, dependency_question)
        if match is None:
            continue
        retrieval_question = (
            retrieval_question[: match.start()]
            + answer
            + retrieval_question[match.end() :]
        )
        replacements.append(
            DependencyReplacement(
                dependency_node_id=str(dependency.get("node_id", "") or ""),
                replacement_span=match.group(0),
                replacement_answer=answer,
            )
        )
        _append_anchor_mentions(primary_anchor_entities, answer)
        dependency_answers_used.append(_dependency_answer_summary(dependency))

    first_replacement = replacements[0] if replacements else None
    return DependencyQuestionRewrite(
        original_question=question,
        retrieval_question=retrieval_question,
        whether_rewritten=bool(replacements),
        replacement_span=first_replacement.replacement_span if first_replacement else "",
        replacement_answer=first_replacement.replacement_answer if first_replacement else "",
        replacements=replacements,
        unresolved_dependencies=unresolved_dependencies,
        primary_anchor_entities=primary_anchor_entities,
        dependency_answers_used=dependency_answers_used,
    )


def _find_dependency_reference_match(question: str, dependency_question: str) -> re.Match[str] | None:
    """在旧式下游问题中定位可被依赖答案替换的描述短语。"""

    for span in _dependency_reference_spans(dependency_question):
        match = _find_span_match(question, span)
        if match is not None:
            return match
    return _find_generic_role_match(question, dependency_question)


def _find_dependency_variable_matches(question: str, dependency_node_id: str) -> list[re.Match[str]]:
    qid = re.escape(dependency_node_id)
    patterns = (
        rf"\{{\s*{qid}\.answer\s*\}}",
        rf"\b{qid}\s*['\u2019]s\s+answer\b",
        rf"\b{qid}\s+answer\b",
        rf"\banswer\s+(?:of|to)\s+{qid}\b",
    )
    matches: list[re.Match[str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            span = (match.start(), match.end())
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in occupied):
                continue
            matches.append(match)
            occupied.append(span)
    return sorted(matches, key=lambda item: item.start())


def _dependency_answer_summary(dependency: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": str(dependency.get("node_id", "") or ""),
        "question": str(dependency.get("question", "") or ""),
        "answer": str(dependency.get("answer", "") or ""),
        "answer_type": str(dependency.get("answer_type", "") or ""),
    }


def is_entity_like_answer(answer: str, answer_type: Any = None) -> bool:
    """判断一个依赖答案是否适合作为后续问题的实体锚点。"""

    text = normalize_label(answer).strip()
    lowered = text.lower()
    if lowered in _INSUFFICIENT_ANSWERS:
        return False
    if any(answer_type_name in normalize_label(str(answer_type)).lower() for answer_type_name in _NON_ENTITY_ANSWER_TYPES):
        return False
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", text)
    if not tokens or len(tokens) > 10 or len(text) > 100:
        return False
    token_lowers = {token.lower().strip(".") for token in tokens}
    if token_lowers & _MONTHS:
        return False
    if re.fullmatch(r"(?:c\.\s*)?\d{1,4}(?:[-/]\d{1,2})?(?:[-/]\d{1,4})?", lowered):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?%?", lowered):
        return False
    if len(tokens) == 1 and lowered in _NON_ENTITY_SINGLE_WORDS:
        return False
    if re.search(r"[.!?]\s+\w", text):
        return False
    if len(tokens) > 4 and re.search(r"\b(?:is|are|was|were|because|therefore)\b", lowered):
        return False
    return any(any(char.isupper() for char in token) for token in tokens)


def _resolved_intermediate_span(dependency_question: str) -> str:
    text = normalize_label(dependency_question).strip().rstrip("?")
    patterns = (
        r"^(?:who|what) (?:is|was|are|were) (?P<span>.+)$",
        r"^(?:who|what) (?:did|does|do) (?P<span>.+?) (?:become|be)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        span = match.group("span").strip()
        if _is_intermediate_span(span):
            return span
    return ""


def _dependency_reference_spans(dependency_question: str) -> list[str]:
    text = normalize_label(dependency_question).strip().rstrip("?")
    spans: list[str] = []
    base_span = _resolved_intermediate_span(text)
    if base_span:
        spans.append(base_span)

    role_object_patterns = (
        r"^(?:who|what) (?:is|was|are|were) (?P<span>(?:the\s+)?(?:paternal\s+grandfather|maternal\s+grandfather|father-in-law|mother-in-law|director|composer|performer|author|writer|producer|father|mother|husband|wife|spouse|child|son|daughter|parent|grandfather|grandmother) of .+)$",
        r"^(?:who|what) (?:is|was|are|were) (?P<object>.+?)(?:'s|’s|鈥檚|鈥橲) (?P<role>paternal grandfather|maternal grandfather|father-in-law|mother-in-law|father|mother|husband|wife|spouse|child|son|daughter|parent|grandfather|grandmother)$",
    )
    for pattern in role_object_patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if "span" in match.groupdict():
            spans.append(match.group("span"))
        else:
            obj = match.group("object").strip()
            role = match.group("role").strip()
            spans.append(f"{obj}'s {role}")

    action_patterns = (
        (r"^who (?:directed|directs) (?P<object>.+)$", "director"),
        (r"^who (?:composed|composes|scored|scores) (?P<object>.+)$", "composer"),
        (r"^who (?:performed|performs|sang|sings) (?P<object>.+)$", "performer"),
        (r"^who (?:wrote|writes|authored|authors) (?P<object>.+)$", "writer"),
        (r"^who (?:produced|produces) (?P<object>.+)$", "producer"),
    )
    for pattern, role in action_patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        obj = match.group("object").strip()
        spans.extend(
            [
                f"{role} of {obj}",
                f"the {role} of {obj}",
            ]
        )

    return _dedupe_by_lowercase(spans)


def _dependency_roles(dependency_question: str) -> list[str]:
    roles: list[str] = []
    for span in _dependency_reference_spans(dependency_question):
        lowered = span.lower()
        role_match = re.match(
            r"^(?:the\s+)?(?P<role>paternal\s+grandfather|maternal\s+grandfather|father-in-law|mother-in-law|director|composer|performer|author|writer|producer|father|mother|husband|wife|spouse|child|son|daughter|parent|grandfather|grandmother)\b",
            lowered,
        )
        if role_match:
            roles.append(role_match.group("role"))
            continue
        possessive_match = re.search(
            r"(?:'s|’s|鈥檚|鈥橲) (?P<role>paternal grandfather|maternal grandfather|father-in-law|mother-in-law|father|mother|husband|wife|spouse|child|son|daughter|parent|grandfather|grandmother)$",
            lowered,
        )
        if possessive_match:
            roles.append(possessive_match.group("role"))
    return _dedupe_by_lowercase(roles)


def _find_generic_role_match(question: str, dependency_question: str) -> re.Match[str] | None:
    for role in _dependency_roles(dependency_question):
        pattern = rf"\b(?:the\s+)?{re.escape(role)}\b(?!\s+of\b)"
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match
    return None


def _is_intermediate_span(span: str) -> bool:
    lowered = span.lower()
    if " of " in lowered:
        return True
    if "'s " in lowered or "’s " in lowered:
        return True
    return False


def _find_span_match(question: str, span: str) -> re.Match[str] | None:
    for variant in _span_variants(span):
        pattern = r"(?<!\w)" + re.escape(variant) + r"(?!\w)"
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match
    return None


def _span_variants(span: str) -> list[str]:
    variants = [normalize_label(span).strip()]
    replacements = (
        (" of the song ", " of song "),
        (" of the song ", " of "),
        (" of song ", " of "),
        (" of the film ", " of film "),
        (" of the film ", " of "),
        (" of film ", " of "),
        (" of the movie ", " of movie "),
        (" of the movie ", " of "),
        (" of movie ", " of "),
    )
    queue = list(variants)
    while queue:
        current = queue.pop(0)
        article_variant = current[4:] if current.lower().startswith("the ") else f"the {current}"
        if article_variant not in variants:
            variants.append(article_variant)
            queue.append(article_variant)
        for source, target in replacements:
            variant = re.sub(
                re.escape(source.strip()),
                target.strip(),
                current,
                flags=re.IGNORECASE,
            )
            if variant != current and variant not in variants:
                variant = normalize_label(variant)
                variants.append(variant)
                queue.append(variant)
    return sorted(
        [variant for index, variant in enumerate(variants) if variant and variant not in variants[:index]],
        key=len,
        reverse=True,
    )


def _append_anchor_mentions(target: list[str], answer: str) -> None:
    for mention in _anchor_mentions_from_answer(answer):
        if mention not in target:
            target.append(mention)


def _anchor_mentions_from_answer(answer: str) -> list[str]:
    text = normalize_label(answer).strip()
    if not text:
        return []
    parts = [
        part.strip(" ,;")
        for part in re.split(r"\s+(?:and|or)\s+|;", text)
        if part.strip(" ,;")
    ]
    if len(parts) > 1:
        return _dedupe_by_lowercase(parts + [text])
    return [text]


def _dedupe_by_lowercase(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_label(value).strip()
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result
