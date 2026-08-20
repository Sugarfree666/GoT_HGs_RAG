"""加载 HanLP SDP 模型，并将不同版本的输出统一为 DEPO 稳定图结构。"""

from __future__ import annotations

import importlib
import re
import warnings
from typing import Any

from models import HanLPSDPEdge, HanLPSDPResult


DEFAULT_MODEL_CANDIDATES = [
    "EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE",
    "EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_BASE",
    "UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE",
    "UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6",
]


class HanLPSDPParser:
    """按需加载一个 HanLP 管线，并以 ``HanLPSDPResult`` 暴露 PAS 边。"""

    def __init__(self, model_name_or_path: str | None = None) -> None:
        self.model_name_or_path = (model_name_or_path or "").strip() or None
        self.model_label = ""
        self._hanlp: Any | None = None
        self._pipeline: Any | None = None

    def parse(self, text: str, placeholders: list[str] | None = None) -> HanLPSDPResult:
        """解析一条掩码问题，同时保留 PAS 证据和分词诊断结果。"""

        pipeline = self._load_pipeline()
        document = pipeline(text)
        # HanLP 版本可能返回映射或文档对象；先统一为映射，后续逻辑不依赖版本。
        payload = _document_to_mapping(document)
        token_sentences = _extract_token_sentences(payload, text)
        tokens = [token for sentence in token_sentences for token in sentence]
        available_keys = list(payload.keys())
        sdp_graphs = {key: payload[key] for key in available_keys if _is_pas_sdp_key(key)}
        syntax_heads, syntax_head_source = _extract_syntax_heads(payload, token_sentences)
        parse_warnings: list[str] = []
        if not sdp_graphs:
            parse_warnings.append("HanLP result did not contain an sdp/pas field.")

        # 将句内索引展平为全局索引，供 Step4 在单一 token 图上推理。
        edges = _collect_sdp_edges(sdp_graphs, token_sentences)
        result = HanLPSDPResult(
            text=text,
            tokens=tokens,
            available_keys=available_keys,
            sdp_graphs=sdp_graphs,
            edges=edges,
            raw=payload,
            warnings=parse_warnings,
            model=self.model_label,
            syntax_heads=syntax_heads,
            syntax_head_source=syntax_head_source,
        )
        # 若 ENTITY 占位符被拆分，实体掩码阶段的前提不再成立。
        check_mask_tokens(result, placeholders or [])
        return result

    def _load_pipeline(self) -> Any:
        """只加载一次指定模型；未指定时按顺序尝试兼容的 PAS 默认模型。"""

        if self._pipeline is not None:
            return self._pipeline

        _quiet_dependency_warnings()
        hanlp = _import_hanlp()
        self._hanlp = hanlp
        attempts: list[str] = []

        if self.model_name_or_path:
            model_ref, label = _resolve_model_reference(hanlp, self.model_name_or_path)
            try:
                self._pipeline = hanlp.load(model_ref)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load HanLP model {self.model_name_or_path!r} resolved as {label!r}: {exc}"
                ) from exc
            self.model_label = label
            return self._pipeline

        for model_name in DEFAULT_MODEL_CANDIDATES:
            try:
                model_ref, label = _resolve_model_reference(hanlp, model_name)
            except AttributeError as exc:
                attempts.append(f"{model_name}: unavailable ({exc})")
                continue
            try:
                self._pipeline = hanlp.load(model_ref)
            except Exception as exc:
                attempts.append(f"{label}: load failed ({exc})")
                continue
            self.model_label = label
            return self._pipeline

        detail = "; ".join(attempts) if attempts else "no model candidates were tried"
        raise RuntimeError(
            "Failed to load a default HanLP SDP model. "
            "Confirm that the network can download the model, or pass a local model path with --hanlp-model. "
            f"Attempts: {detail}"
        )


def check_mask_tokens(result: HanLPSDPResult, placeholders: list[str]) -> dict[str, str]:
    """记录每个 DEPO 占位符是否被 HanLP 保持为单个 token。"""

    token_set = set(result.tokens)
    checks: dict[str, str] = {}
    for placeholder in placeholders:
        if placeholder in token_set:
            checks[placeholder] = "OK"
            continue
        split_hint = _placeholder_split_hint(placeholder, result.tokens)
        if split_hint:
            message = (
                f"FAILED, placeholder was split by HanLP tokenization as {split_hint}. "
                "This SDP graph may be unreliable because entity masking was split by HanLP tokenization."
            )
        else:
            message = (
                "FAILED, placeholder was not found as a single HanLP token. "
                "This SDP graph may be unreliable because entity masking was split by HanLP tokenization."
            )
        checks[placeholder] = message
        result.warnings.append(f"{placeholder}: {message}")
    result.mask_token_checks = checks
    return checks


def _quiet_dependency_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\.|$)")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"hanlp(\.|$)")
    warnings.filterwarnings("ignore", message=r".*pynvml package is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*Sparse invariant checks are implicitly disabled.*")


def _import_hanlp() -> Any:
    try:
        import hanlp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency: hanlp. Run: pip install hanlp") from exc
    return hanlp


def _resolve_model_reference(hanlp: Any, model_name_or_path: str) -> tuple[Any, str]:
    for module_name in ("mtl", "sdp"):
        module_path = f"hanlp.pretrained.{module_name}"
        try:
            module = importlib.import_module(module_path)
        except Exception:
            module = getattr(getattr(hanlp, "pretrained", None), module_name, None)
        if module is not None and hasattr(module, model_name_or_path):
            return getattr(module, model_name_or_path), f"hanlp.pretrained.{module_name}.{model_name_or_path}"
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", model_name_or_path):
        raise AttributeError(f"HanLP pretrained constant {model_name_or_path!r} was not found")
    return model_name_or_path, model_name_or_path


def _document_to_mapping(document: Any) -> dict[str, Any]:
    """将随 HanLP 版本变化的文档对象转换为普通映射。"""

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


def _extract_token_sentences(payload: dict[str, Any], fallback_text: str) -> list[list[str]]:
    """读取 HanLP token；缺失时才使用透明的正则分词回退。"""

    token_key = _find_token_key(payload)
    if token_key:
        normalized = _normalize_token_value(payload[token_key])
        if normalized:
            return normalized
    fallback_tokens = re.findall(r"\w+|[^\w\s]", fallback_text, flags=re.UNICODE)
    return [[str(token) for token in fallback_tokens]] if fallback_tokens else [[]]


def _find_token_key(payload: dict[str, Any]) -> str | None:
    if "tok" in payload:
        return "tok"
    for key in payload:
        if key.startswith("tok/"):
            return key
    for key in payload:
        if "tok" in key.lower():
            return key
    return None


def _normalize_token_value(value: Any) -> list[list[str]]:
    if _is_string_sequence(value):
        return [[str(item) for item in value]]
    if _is_non_string_sequence(value) and all(_is_string_sequence(item) for item in value):
        return [[str(token) for token in sentence] for sentence in value]
    return []


def _collect_sdp_edges(
    sdp_graphs: dict[str, Any],
    token_sentences: list[list[str]],
) -> list[HanLPSDPEdge]:
    """将句内 SDP 弧转换为全局索引、可展示的边记录。"""

    edges: list[HanLPSDPEdge] = []
    sentence_offsets: list[int] = []
    offset = 0
    for tokens in token_sentences:
        sentence_offsets.append(offset)
        offset += len(tokens)

    flat_tokens = [token for sentence in token_sentences for token in sentence]
    for formalism, graph_payload in sdp_graphs.items():
        sentence_graphs = _split_graph_by_sentence(graph_payload, token_sentences)
        for sentence_index, graph in enumerate(sentence_graphs):
            tokens = token_sentences[min(sentence_index, len(token_sentences) - 1)] if token_sentences else []
            token_offset = sentence_offsets[min(sentence_index, len(sentence_offsets) - 1)] if sentence_offsets else 0
            for edge in _edges_from_graph(graph, tokens):
                head_idx = 0 if edge.head_index == 0 else edge.head_index + token_offset
                dep_idx = edge.dependent_index + token_offset
                edges.append(
                    HanLPSDPEdge(
                        formalism=formalism,
                        head_idx=head_idx,
                        head="ROOT" if head_idx == 0 else _token_at(flat_tokens, head_idx),
                        relation=edge.relation,
                        dep_idx=dep_idx,
                        dep=_token_at(flat_tokens, dep_idx),
                    )
                )
    return edges


def _extract_syntax_heads(payload: dict[str, Any], token_sentences: list[list[str]]) -> tuple[dict[str, int], str]:
    syntax_key = _find_syntax_dependency_key(payload)
    if syntax_key is None:
        return {}, ""
    return _collect_dependency_heads(payload[syntax_key], token_sentences), syntax_key


def _find_syntax_dependency_key(payload: dict[str, Any]) -> str | None:
    keys = list(payload.keys())
    for key in keys:
        if _is_udep_key(str(key)):
            return str(key)
    for key in keys:
        if _is_dep_key(str(key)):
            return str(key)
    return None


def _is_udep_key(key: str) -> bool:
    normalized = key.lower()
    return normalized == "udep" or normalized.endswith("/udep") or normalized.endswith("_udep")


def _is_dep_key(key: str) -> bool:
    normalized = key.lower()
    if "sdp" in normalized:
        return False
    return (
        normalized == "dep"
        or normalized == "dependencies"
        or normalized.endswith("/dep")
        or normalized.endswith("_dep")
    )


def _collect_dependency_heads(graph_payload: Any, token_sentences: list[list[str]]) -> dict[str, int]:
    syntax_heads: dict[str, int] = {}
    sentence_offsets: list[int] = []
    offset = 0
    for tokens in token_sentences:
        sentence_offsets.append(offset)
        offset += len(tokens)

    sentence_graphs = _split_graph_by_sentence(graph_payload, token_sentences)
    for sentence_index, graph in enumerate(sentence_graphs):
        tokens = token_sentences[min(sentence_index, len(token_sentences) - 1)] if token_sentences else []
        token_offset = sentence_offsets[min(sentence_index, len(sentence_offsets) - 1)] if sentence_offsets else 0
        for edge in _edges_from_graph(graph, tokens):
            if edge.dependent_index <= 0:
                continue
            dep_idx = edge.dependent_index + token_offset
            head_idx = 0 if edge.head_index == 0 else edge.head_index + token_offset
            if dep_idx <= 0:
                continue
            syntax_heads[str(dep_idx)] = head_idx
    return syntax_heads


def _is_pas_sdp_key(key: str) -> bool:
    normalized = str(key).lower()
    return normalized == "sdp/pas" or normalized.endswith("/pas") and "sdp" in normalized


def _split_graph_by_sentence(value: Any, token_sentences: list[list[str]]) -> list[Any]:
    sentence_count = len(token_sentences)
    if sentence_count <= 1:
        return [value]
    if _is_non_string_sequence(value) and len(value) == sentence_count:
        return list(value)

    flattened_token_count = sum(len(tokens) for tokens in token_sentences)
    if _is_non_string_sequence(value) and len(value) == flattened_token_count:
        graphs: list[Any] = []
        cursor = 0
        values = list(value)
        for tokens in token_sentences:
            graphs.append(values[cursor : cursor + len(tokens)])
            cursor += len(tokens)
        return graphs
    return [value]


class _RawEdge:
    def __init__(self, head_index: int, dependent_index: int, relation: str) -> None:
        self.head_index = head_index
        self.dependent_index = dependent_index
        self.relation = relation


def _edges_from_graph(graph: Any, tokens: list[str]) -> list[_RawEdge]:
    if graph is None:
        return []
    if isinstance(graph, dict):
        return _edges_from_dict_graph(graph, tokens)
    if _is_non_string_sequence(graph):
        items = list(graph)
        if len(items) == len(tokens) and all(_looks_like_token_head_entry(item) for item in items):
            edges: list[_RawEdge] = []
            for dependent_index, entry in enumerate(items, start=1):
                edges.extend(_edges_from_token_head_entry(entry, dependent_index))
            return edges

        edges = []
        for item in items:
            edge = _edge_from_edge_record(item)
            if edge is not None:
                edges.append(edge)
                continue
            if _is_non_string_sequence(item):
                for nested in item:
                    nested_edge = _edge_from_edge_record(nested)
                    if nested_edge is not None:
                        edges.append(nested_edge)
        return edges
    return []


def _edges_from_dict_graph(graph: dict[Any, Any], tokens: list[str]) -> list[_RawEdge]:
    for key in ("edges", "arcs", "dependencies"):
        value = graph.get(key)
        if value is not None:
            return _edges_from_graph(value, tokens)

    edges: list[_RawEdge] = []
    for dependent, entry in graph.items():
        dependent_index = _coerce_int(dependent)
        if dependent_index is None:
            continue
        edges.extend(_edges_from_token_head_entry(entry, dependent_index))
    return edges


def _looks_like_token_head_entry(entry: Any) -> bool:
    if entry in (None, "", []):
        return True
    if isinstance(entry, dict):
        return bool({"head", "head_id", "head_index", "heads"} & set(entry))
    if _is_head_relation_pair(entry):
        return True
    if _is_non_string_sequence(entry):
        return all(_is_head_relation_pair(item) or isinstance(item, dict) for item in entry)
    return False


def _edges_from_token_head_entry(entry: Any, dependent_index: int) -> list[_RawEdge]:
    if entry in (None, "", []):
        return []
    if isinstance(entry, dict):
        return _edges_from_token_head_dict(entry, dependent_index)
    if _is_head_relation_pair(entry):
        head, relation = _parse_head_relation_pair(entry)
        return [_RawEdge(head, dependent_index, relation)] if head is not None else []
    if _is_non_string_sequence(entry):
        edges: list[_RawEdge] = []
        for item in entry:
            if isinstance(item, dict):
                edges.extend(_edges_from_token_head_dict(item, dependent_index))
                continue
            if _is_head_relation_pair(item):
                head, relation = _parse_head_relation_pair(item)
                if head is not None:
                    edges.append(_RawEdge(head, dependent_index, relation))
        return edges
    return []


def _edges_from_token_head_dict(entry: dict[Any, Any], dependent_index: int) -> list[_RawEdge]:
    if "heads" in entry and _is_non_string_sequence(entry["heads"]):
        edges: list[_RawEdge] = []
        for item in entry["heads"]:
            if isinstance(item, dict):
                edges.extend(_edges_from_token_head_dict(item, dependent_index))
            elif _is_head_relation_pair(item):
                head, relation = _parse_head_relation_pair(item)
                if head is not None:
                    edges.append(_RawEdge(head, dependent_index, relation))
        return edges

    head = _coerce_int(entry.get("head", entry.get("head_id", entry.get("head_index"))))
    relation = str(
        entry.get(
            "relation",
            entry.get("rel", entry.get("label", entry.get("deprel", entry.get("arc", "")))),
        )
        or ""
    )
    return [_RawEdge(head, dependent_index, relation)] if head is not None else []


def _edge_from_edge_record(record: Any) -> _RawEdge | None:
    if isinstance(record, dict):
        dependent = _coerce_int(
            record.get(
                "dependent",
                record.get("dep", record.get("dependent_index", record.get("target", record.get("to")))),
            )
        )
        head = _coerce_int(record.get("head", record.get("source", record.get("from"))))
        relation = str(
            record.get(
                "relation",
                record.get("rel", record.get("label", record.get("deprel", record.get("arc", "")))),
            )
            or ""
        )
        if head is not None and dependent is not None:
            return _RawEdge(head, dependent, relation)
        return None

    if not _is_non_string_sequence(record):
        return None
    items = list(record)
    if len(items) >= 3:
        first = _coerce_int(items[0])
        second = _coerce_int(items[1])
        if first is not None and second is not None:
            return _RawEdge(first, second, str(items[2]))
    return None


def _is_head_relation_pair(value: Any) -> bool:
    if not _is_non_string_sequence(value):
        return False
    items = list(value)
    if len(items) != 2:
        return False
    return _coerce_int(items[0]) is not None and _coerce_int(items[1]) is None


def _parse_head_relation_pair(value: Any) -> tuple[int | None, str]:
    items = list(value)
    return _coerce_int(items[0]), str(items[1])


def _placeholder_split_hint(placeholder: str, tokens: list[str]) -> str:
    lower = placeholder.lower()
    compact = ""
    pieces: list[str] = []
    for token in tokens:
        if lower.startswith((compact + token).lower()):
            compact += token
            pieces.append(token)
            if compact.lower() == lower and len(pieces) > 1:
                return " / ".join(pieces)
            continue
        compact = token
        pieces = [token]
    return ""


def _token_at(tokens: list[str], index: int) -> str:
    if 1 <= index <= len(tokens):
        return tokens[index - 1]
    return "?"


def _is_string_sequence(value: Any) -> bool:
    return _is_non_string_sequence(value) and all(isinstance(item, str) for item in value)


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
