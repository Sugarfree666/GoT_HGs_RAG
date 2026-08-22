"""Small text and serialization helpers shared by HyperBranch."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "in", "is", "it", "its", "known", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def normalize_label(text: str) -> str:
    cleaned = text.strip()
    for prefix in ("<hyperedge>", "<synonyms>"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    cleaned = cleaned.replace("<SEP>", " / ").strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    return re.sub(r"\s+", " ", cleaned).strip()


def split_source_ids(source_text: str) -> list[str]:
    return [part.strip() for part in source_text.split("<SEP>") if part.strip()]


def slugify(text: str, max_length: int = 64) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value[:max_length].strip("-") or "run"


def short_text(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else f"{text[:limit - 3].rstrip()}..."


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def content_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(normalize_label(text).lower()) if token not in STOPWORDS]
