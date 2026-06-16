from __future__ import annotations

import argparse
import importlib
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL_NAME = "EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE"


@dataclass(frozen=True)
class ProbeEntity:
    text: str
    start_char: int
    end_char: int
    semantic_type_hint: str = "Entity"
    reason: str = ""


@dataclass(frozen=True)
class ProbeMaskMapping:
    original_text: str
    placeholder: str
    semantic_type_hint: str = "Entity"
    original_char_span: list[int] = field(default_factory=list)
    masked_char_span: list[int] = field(default_factory=list)


@dataclass
class ProbeMaskingResult:
    masked_sentence: str
    entities: list[ProbeEntity] = field(default_factory=list)
    mappings: list[ProbeMaskMapping] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = "depo"


@dataclass(frozen=True)
class SdpEdge:
    head_index: int
    dependent_index: int
    relation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe local HanLP SDP output after DEPO explicit entity masking."
    )
    parser.add_argument("--sentence", required=True, help="Natural-language question to parse.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=(
            "HanLP model constant name from hanlp.pretrained.mtl/sdp, or a local model path. "
            f"Default: {DEFAULT_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show dependency warnings from HanLP/PyTorch. Hidden by default for cleaner probe output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sentence = args.sentence.strip()
    if not sentence:
        print("Missing --sentence text.", file=sys.stderr)
        return 2

    configure_warnings(show_warnings=args.show_warnings)

    print("If this is the first run, HanLP may download the model automatically.")
    print("You can set HANLP_HOME to control the cache directory.")
    print()

    masking = detect_and_mask(sentence)
    print_masking_result(sentence, masking)

    hanlp = import_hanlp()
    if hanlp is None:
        return 2

    try:
        model_ref, model_label = resolve_model_reference(hanlp, args.model)
        HanLP = hanlp.load(model_ref)
    except Exception as exc:
        print("[HanLP Model]", file=sys.stderr)
        print(f"Failed to load model: {args.model}", file=sys.stderr)
        print(f"Resolved as: {model_label if 'model_label' in locals() else args.model}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print(
            "Confirm that the network can download the model, or download it ahead of time "
            "and pass the local model path with --model.",
            file=sys.stderr,
        )
        return 1

    print("[HanLP Model]")
    print(model_label)
    print()

    try:
        document = HanLP(masking.masked_sentence)
    except Exception as exc:
        print("HanLP parsing failed.", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1

    payload = document_to_mapping(document)
    token_sentences = extract_token_sentences(payload, masking.masked_sentence)
    print_tokens(token_sentences)
    print_available_keys(payload)
    print_sdp_graphs(payload, token_sentences)
    return 0


def configure_warnings(*, show_warnings: bool) -> None:
    if show_warnings:
        return
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\.|$)")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"hanlp(\.|$)")
    warnings.filterwarnings("ignore", message=r".*pynvml package is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*Sparse invariant checks are implicitly disabled.*")


def detect_and_mask(sentence: str) -> ProbeMaskingResult:
    try:
        from mask_span_extractor import ExplicitEntityExtractor, _mask_spans_from_explicit_entities
        from models import MaskSpanResult
        from placeholder import selective_entity_masking

        explicit_entities = ExplicitEntityExtractor(None).extract(sentence)
        mask_span_result = MaskSpanResult(
            mask_spans=_mask_spans_from_explicit_entities(explicit_entities.entities),
            warnings=list(explicit_entities.warnings),
            raw_payload=explicit_entities.raw_payload,
        )
        replacement = selective_entity_masking(question=sentence, mask_spans=mask_span_result)
        entities = [
            ProbeEntity(
                text=entity.text,
                start_char=entity.start_char,
                end_char=entity.end_char,
                semantic_type_hint=entity.semantic_type_hint or "Entity",
                reason=entity.reason,
            )
            for entity in explicit_entities.entities
        ]
        mappings = [
            ProbeMaskMapping(
                original_text=mapping.original_text,
                placeholder=mapping.placeholder,
                semantic_type_hint=mapping.semantic_type_hint or "Entity",
                original_char_span=list(mapping.original_char_span),
                masked_char_span=list(mapping.masked_char_span),
            )
            for mapping in replacement.mask_mappings
        ]
        return ProbeMaskingResult(
            masked_sentence=replacement.masked_question,
            entities=entities,
            mappings=mappings,
            warnings=list(explicit_entities.warnings),
            source="depo",
        )
    except Exception as exc:
        fallback = fallback_detect_and_mask(sentence)
        fallback.warnings.insert(
            0,
            f"DEPO explicit entity masking failed; using simple capitalized-entity fallback: {exc}",
        )
        return fallback


def fallback_detect_and_mask(sentence: str) -> ProbeMaskingResult:
    entities = fallback_entities(sentence)
    counters: dict[str, int] = {}
    mappings: list[ProbeMaskMapping] = []
    replacements: list[tuple[int, int, str]] = []
    for entity in entities:
        base = placeholder_base(entity.semantic_type_hint)
        index = counters.get(base, 0)
        placeholder = f"{base}{letter_suffix(index)}"
        counters[base] = index + 1
        mappings.append(
            ProbeMaskMapping(
                original_text=entity.text,
                placeholder=placeholder,
                semantic_type_hint=entity.semantic_type_hint,
                original_char_span=[entity.start_char, entity.end_char],
                masked_char_span=[],
            )
        )
        replacements.append((entity.start_char, entity.end_char, placeholder))

    masked = sentence
    for start, end, placeholder in sorted(replacements, key=lambda item: item[0], reverse=True):
        masked = masked[:start] + placeholder + masked[end:]
    return ProbeMaskingResult(
        masked_sentence=masked,
        entities=entities,
        mappings=mappings,
        warnings=["Fallback is only used because the existing DEPO masking path failed."],
        source="fallback",
    )


def fallback_entities(sentence: str) -> list[ProbeEntity]:
    token_matches = list(re.finditer(r"[^\W\d_][\w'.-]*|,", sentence, flags=re.UNICODE))
    entities: list[ProbeEntity] = []
    index = 0
    while index < len(token_matches):
        match = token_matches[index]
        token = match.group(0)
        if not is_name_token(token) or is_sentence_initial_question_word(sentence, match.start(), token):
            index += 1
            continue

        start = match.start()
        end = match.end()
        content_count = 1
        cursor = index + 1
        while cursor < len(token_matches):
            next_match = token_matches[cursor]
            next_token = next_match.group(0)
            between = sentence[end : next_match.start()]
            if next_token != "," and not between.isspace():
                break
            lowered = next_token.lower().strip(".")
            if lowered in {"and", "or"}:
                break
            if next_token == ",":
                if cursor + 1 < len(token_matches) and is_name_token(token_matches[cursor + 1].group(0)):
                    end = next_match.end()
                    cursor += 1
                    continue
                break
            if is_name_token(next_token):
                end = next_match.end()
                content_count += 1
                cursor += 1
                continue
            if lowered in {"de", "del", "der", "di", "la", "le", "of", "the", "van", "von"}:
                if cursor + 1 < len(token_matches) and is_name_token(token_matches[cursor + 1].group(0)):
                    end = next_match.end()
                    cursor += 1
                    continue
            break

        text = sentence[start:end].strip(" \t\r\n,.;:?!")
        adjusted_end = start + len(sentence[start:end].rstrip(" \t\r\n,.;:?!"))
        if content_count >= 2 or looks_like_existing_placeholder(text):
            entities.append(
                ProbeEntity(
                    text=text,
                    start_char=start,
                    end_char=adjusted_end,
                    semantic_type_hint=infer_fallback_semantic_type(sentence, text, start, adjusted_end),
                    reason="capitalized entity fallback",
                )
            )
            index = max(cursor, index + 1)
            continue
        index += 1
    return remove_overlapping_entities(entities)


def is_name_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped or stripped == ",":
        return False
    return bool(stripped[:1].isupper() or re.fullmatch(r"[A-Z]{2,}(?:\.)?", stripped.strip(".")))


def is_sentence_initial_question_word(sentence: str, start: int, token: str) -> bool:
    if start != 0:
        return False
    return token.lower().strip(".") in {
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


def looks_like_existing_placeholder(text: str) -> bool:
    return bool(re.fullmatch(r"(?:Entity|Film|Person|Song|Book|Location|Country|City|Work)[A-Z]+", text))


def infer_fallback_semantic_type(sentence: str, text: str, start: int, end: int) -> str:
    window = sentence[max(0, start - 80) : min(len(sentence), end + 80)].lower()
    lowered = text.lower()
    if re.search(r"\b(film|films|movie|movies)\b", window):
        return "Film"
    if re.search(r"\b(song|songs)\b", window):
        return "Song"
    if re.search(r"\b(book|books|novel|novels)\b", window):
        return "Book"
    if re.search(r"\b(country|countries)\b", window):
        return "Country"
    if re.search(r"\b(city|cities)\b", window):
        return "City"
    if (
        re.search(r"\b(who|older|younger|director|performer|person|people|born|died|death)\b", window)
        and len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)) <= 5
        and not re.search(r"\d|[:()\[\]{}\"']", text)
    ):
        return "Person"
    if lowered.startswith("mr. "):
        return "Person"
    return "Entity"


def remove_overlapping_entities(entities: list[ProbeEntity]) -> list[ProbeEntity]:
    ordered = sorted(entities, key=lambda item: (item.start_char, -(item.end_char - item.start_char)))
    result: list[ProbeEntity] = []
    occupied: list[tuple[int, int]] = []
    for entity in ordered:
        if any(not (entity.end_char <= start or entity.start_char >= end) for start, end in occupied):
            continue
        result.append(entity)
        occupied.append((entity.start_char, entity.end_char))
    return result


def placeholder_base(semantic_type: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", semantic_type or "")
    if normalized in {
        "Book",
        "City",
        "Country",
        "Entity",
        "Film",
        "Location",
        "Person",
        "Song",
        "Work",
    }:
        return normalized
    return "Entity"


def letter_suffix(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    current = index
    while True:
        label = alphabet[current % len(alphabet)] + label
        current = current // len(alphabet) - 1
        if current < 0:
            return label


def import_hanlp() -> Any | None:
    try:
        import hanlp

        return hanlp
    except ModuleNotFoundError:
        print("Missing dependency: hanlp", file=sys.stderr)
        print("Install it with:", file=sys.stderr)
        print("  pip install hanlp", file=sys.stderr)
        return None


def resolve_model_reference(hanlp: Any, model: str) -> tuple[Any, str]:
    model = model.strip()
    if not model:
        model = DEFAULT_MODEL_NAME

    for module_name in ("mtl", "sdp"):
        module_path = f"hanlp.pretrained.{module_name}"
        try:
            module = importlib.import_module(module_path)
        except Exception:
            module = getattr(getattr(hanlp, "pretrained", None), module_name, None)
        if module is not None and hasattr(module, model):
            return getattr(module, model), f"hanlp.pretrained.{module_name}.{model}"

    return model, model


def document_to_mapping(document: Any) -> dict[str, Any]:
    if isinstance(document, dict):
        return dict(document)
    if hasattr(document, "to_dict"):
        data = document.to_dict()
        if isinstance(data, dict):
            return data
    if hasattr(document, "items"):
        return {str(key): value for key, value in document.items()}
    try:
        data = dict(document)
    except Exception:
        return {"_raw": document}
    return {str(key): value for key, value in data.items()}


def extract_token_sentences(payload: dict[str, Any], fallback_text: str) -> list[list[str]]:
    token_key = find_token_key(payload)
    if token_key:
        normalized = normalize_token_value(payload[token_key])
        if normalized:
            return normalized
    fallback_tokens = re.findall(r"\w+|[^\w\s]", fallback_text, flags=re.UNICODE)
    return [[str(token) for token in fallback_tokens]] if fallback_tokens else [[]]


def find_token_key(payload: dict[str, Any]) -> str | None:
    if "tok" in payload:
        return "tok"
    for key in payload:
        if key.startswith("tok/"):
            return key
    for key in payload:
        if "tok" in key.lower():
            return key
    return None


def normalize_token_value(value: Any) -> list[list[str]]:
    if is_string_sequence(value):
        return [[str(item) for item in value]]
    if is_non_string_sequence(value) and all(is_string_sequence(item) for item in value):
        return [[str(token) for token in sentence] for sentence in value]
    return []


def print_masking_result(original: str, masking: ProbeMaskingResult) -> None:
    print("[Original]")
    print(original)
    print()

    print("[Detected Entities]")
    if masking.entities:
        for entity in masking.entities:
            type_text = f" [{entity.semantic_type_hint}]" if entity.semantic_type_hint else ""
            print(f"{entity.text}{type_text}")
    else:
        print("(none)")
    print()

    print("[Mask Mapping]")
    if masking.mappings:
        for mapping in masking.mappings:
            print(f"{mapping.original_text} -> {mapping.placeholder}")
    else:
        print("(none)")
    print()

    print("[Masked Sentence]")
    print(masking.masked_sentence)
    print()

    if masking.source == "fallback" and masking.warnings:
        print("[Masking Warnings]")
        for warning in masking.warnings:
            print(warning)
        print()


def print_tokens(token_sentences: list[list[str]]) -> None:
    print("[HanLP Tokens]")
    if len(token_sentences) <= 1:
        for index, token in enumerate(token_sentences[0] if token_sentences else [], start=1):
            print(f"{index} {token}")
    else:
        for sentence_index, tokens in enumerate(token_sentences, start=1):
            print(f"Sentence {sentence_index}:")
            for index, token in enumerate(tokens, start=1):
                print(f"{index} {token}")
    print()


def print_available_keys(payload: dict[str, Any]) -> None:
    print("[Available HanLP Keys]")
    if payload:
        for key in payload:
            print(key)
    else:
        print("(none)")
    print()


def print_sdp_graphs(payload: dict[str, Any], token_sentences: list[list[str]]) -> None:
    sdp_items = [(key, payload[key]) for key in payload if "sdp" in key.lower()]

    print("[SDP Graphs]")
    if sdp_items:
        for key, _value in sdp_items:
            print(key)
    else:
        print("(none)")
        print("No SDP field found. Inspect [Available HanLP Keys] above to debug the loaded model output.")
    print()

    print("[Readable SDP Edges]")
    if not sdp_items:
        print("(none)")
        print()
        return

    for key, value in sdp_items:
        print(f"[SDP: {key}]")
        sentence_graphs = split_graph_by_sentence(value, token_sentences)
        printed_any = False
        for sentence_index, (tokens, graph) in enumerate(zip(token_sentences, sentence_graphs), start=1):
            edges = edges_from_graph(graph, tokens)
            if len(token_sentences) > 1:
                print(f"Sentence {sentence_index}:")
            if edges:
                for edge in edges:
                    print(format_sdp_edge(edge, tokens))
                    printed_any = True
            else:
                print("(no readable edges)")
        if not printed_any and not sentence_graphs:
            print("(no readable edges)")
        print()


def split_graph_by_sentence(value: Any, token_sentences: list[list[str]]) -> list[Any]:
    sentence_count = len(token_sentences)
    if sentence_count <= 1:
        return [value]
    if is_non_string_sequence(value) and len(value) == sentence_count:
        return list(value)

    flattened_token_count = sum(len(tokens) for tokens in token_sentences)
    if is_non_string_sequence(value) and len(value) == flattened_token_count:
        graphs: list[Any] = []
        cursor = 0
        for tokens in token_sentences:
            graphs.append(list(value)[cursor : cursor + len(tokens)])
            cursor += len(tokens)
        return graphs
    return [value]


def edges_from_graph(graph: Any, tokens: list[str]) -> list[SdpEdge]:
    if graph is None:
        return []
    if isinstance(graph, dict):
        return edges_from_dict_graph(graph, tokens)
    if is_non_string_sequence(graph):
        items = list(graph)
        if len(items) == len(tokens) and all(looks_like_token_head_entry(item) for item in items):
            edges: list[SdpEdge] = []
            for dependent_index, entry in enumerate(items, start=1):
                edges.extend(edges_from_token_head_entry(entry, dependent_index))
            return edges

        edges = []
        for item in items:
            edge = edge_from_edge_record(item)
            if edge is not None:
                edges.append(edge)
                continue
            if is_non_string_sequence(item):
                for nested in item:
                    nested_edge = edge_from_edge_record(nested)
                    if nested_edge is not None:
                        edges.append(nested_edge)
        return edges
    return []


def edges_from_dict_graph(graph: dict[Any, Any], tokens: list[str]) -> list[SdpEdge]:
    for key in ("edges", "arcs", "dependencies"):
        value = graph.get(key)
        if value is not None:
            return edges_from_graph(value, tokens)

    edges: list[SdpEdge] = []
    for dependent, entry in graph.items():
        dependent_index = coerce_int(dependent)
        if dependent_index is None:
            continue
        edges.extend(edges_from_token_head_entry(entry, dependent_index))
    return edges


def looks_like_token_head_entry(entry: Any) -> bool:
    if entry in (None, "", []):
        return True
    if isinstance(entry, dict):
        return bool({"head", "head_id", "head_index", "heads"} & set(entry))
    if is_head_relation_pair(entry):
        return True
    if is_non_string_sequence(entry):
        return all(is_head_relation_pair(item) or isinstance(item, dict) for item in entry)
    return False


def edges_from_token_head_entry(entry: Any, dependent_index: int) -> list[SdpEdge]:
    if entry in (None, "", []):
        return []
    if isinstance(entry, dict):
        return edges_from_token_head_dict(entry, dependent_index)
    if is_head_relation_pair(entry):
        head, relation = parse_head_relation_pair(entry)
        return [SdpEdge(head, dependent_index, relation)] if head is not None else []
    if is_non_string_sequence(entry):
        edges: list[SdpEdge] = []
        for item in entry:
            if isinstance(item, dict):
                edges.extend(edges_from_token_head_dict(item, dependent_index))
                continue
            if is_head_relation_pair(item):
                head, relation = parse_head_relation_pair(item)
                if head is not None:
                    edges.append(SdpEdge(head, dependent_index, relation))
        return edges
    return []


def edges_from_token_head_dict(entry: dict[Any, Any], dependent_index: int) -> list[SdpEdge]:
    if "heads" in entry and is_non_string_sequence(entry["heads"]):
        edges: list[SdpEdge] = []
        for item in entry["heads"]:
            if isinstance(item, dict):
                edges.extend(edges_from_token_head_dict(item, dependent_index))
            elif is_head_relation_pair(item):
                head, relation = parse_head_relation_pair(item)
                if head is not None:
                    edges.append(SdpEdge(head, dependent_index, relation))
        return edges

    head = coerce_int(entry.get("head", entry.get("head_id", entry.get("head_index"))))
    relation = str(
        entry.get(
            "relation",
            entry.get("rel", entry.get("label", entry.get("deprel", entry.get("arc", "")))),
        )
        or ""
    )
    return [SdpEdge(head, dependent_index, relation)] if head is not None else []


def edge_from_edge_record(record: Any) -> SdpEdge | None:
    if isinstance(record, dict):
        dependent = coerce_int(
            record.get(
                "dependent",
                record.get("dep", record.get("dependent_index", record.get("target", record.get("to")))),
            )
        )
        head = coerce_int(record.get("head", record.get("source", record.get("from"))))
        relation = str(
            record.get(
                "relation",
                record.get("rel", record.get("label", record.get("deprel", record.get("arc", "")))),
            )
            or ""
        )
        if head is not None and dependent is not None:
            return SdpEdge(head, dependent, relation)
        return None

    if not is_non_string_sequence(record):
        return None
    items = list(record)
    if len(items) >= 3:
        first = coerce_int(items[0])
        second = coerce_int(items[1])
        if first is not None and second is not None:
            return SdpEdge(first, second, str(items[2]))
    return None


def is_head_relation_pair(value: Any) -> bool:
    if not is_non_string_sequence(value):
        return False
    items = list(value)
    if len(items) != 2:
        return False
    return coerce_int(items[0]) is not None and coerce_int(items[1]) is None


def parse_head_relation_pair(value: Any) -> tuple[int | None, str]:
    items = list(value)
    return coerce_int(items[0]), str(items[1])


def format_sdp_edge(edge: SdpEdge, tokens: list[str]) -> str:
    head = "ROOT[0]" if edge.head_index == 0 else token_label(tokens, edge.head_index)
    dependent = token_label(tokens, edge.dependent_index)
    return f"{head} --{edge.relation}--> {dependent}"


def token_label(tokens: list[str], index: int) -> str:
    if 1 <= index <= len(tokens):
        return f"{tokens[index - 1]}[{index}]"
    return f"?[{index}]"


def is_string_sequence(value: Any) -> bool:
    return is_non_string_sequence(value) and all(isinstance(item, str) for item in value)


def is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


def coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
