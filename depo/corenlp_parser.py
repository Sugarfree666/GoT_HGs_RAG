from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from models import CoreNLPToken, CoreNLPViewAnnotation, DependencyEdge, DependencyParse, OpenIETriple


class CoreNLPConnectionError(RuntimeError):
    pass


class CoreNLPParser:
    def __init__(
        self,
        url: str = "http://localhost:9000",
        timeout_ms: int = 60000,
        memory: str = "4G",
        be_quiet: bool = True,
        corenlp_home: str | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.memory = memory
        self.be_quiet = be_quiet
        self.corenlp_home = corenlp_home
        self.client: Any | None = None
        self._client_manager: Any | None = None
        self.properties = {
            "depparse.extradependencies": "MAXIMAL",
        }

    def __enter__(self) -> "CoreNLPParser":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self.client is not None:
            return
        try:
            with _suppress_info_logs():
                from stanza.server import CoreNLPClient

                corenlp_home = self._resolve_corenlp_home()
                client_kwargs: dict[str, Any] = {}
                if corenlp_home is not None:
                    client_kwargs["classpath"] = self._build_classpath(corenlp_home)

                self._client_manager = CoreNLPClient(
                    endpoint=self.url,
                    annotators="tokenize,ssplit,pos,lemma,depparse",
                    output_format="json",
                    properties=self.properties,
                    timeout=self.timeout_ms,
                    memory=self.memory,
                    be_quiet=self.be_quiet,
                    **client_kwargs,
                )
                self.client = self._client_manager.__enter__()
        except ModuleNotFoundError:
            raise
        except Exception as exc:
            self.client = None
            self._client_manager = None
            raise CoreNLPConnectionError(
                "CoreNLPClient could not start Stanford CoreNLP. "
                "Make sure Java is installed and CoreNLP is installed with "
                "`python -c \"import stanza; stanza.install_corenlp()\"` "
                "or pass --corenlp-home / set CORENLP_HOME to a valid CoreNLP directory. "
                f"Endpoint: {self.url}. Original error: {exc}"
            ) from exc

    def stop(self) -> None:
        if self._client_manager is None:
            return
        try:
            self._client_manager.__exit__(None, None, None)
        finally:
            self.client = None
            self._client_manager = None

    def parse(self, text: str) -> DependencyParse:
        if self.client is None:
            self.start()

        try:
            payload = self.client.annotate(text)
        except Exception as exc:
            raise CoreNLPConnectionError(
                f"CoreNLPClient failed to annotate text through endpoint {self.url}: {exc}"
            ) from exc

        payload = self._coerce_json_payload(payload)

        return self._parse_payload(payload)

    def annotate_view(
        self,
        text: str,
        *,
        view_id: str = "view_1",
        enable_openie: bool = True,
        include_constituency: bool = False,
    ) -> CoreNLPViewAnnotation:
        """Annotate one parser-facing declarative view.

        OpenIE is opportunistic: if the local CoreNLP server or stanza wrapper
        rejects natlog/openie annotators, this method returns structural
        dependency evidence with a warning instead of failing the pipeline.
        """

        warnings: list[str] = []
        annotators = "tokenize,ssplit,pos,lemma,ner,depparse"
        if include_constituency:
            annotators += ",parse"
        if enable_openie:
            annotators += ",natlog,openie"

        try:
            payload = self.annotate_raw(
                text,
                annotators=annotators,
                properties={
                    **self.properties,
                    "openie.triple.strict": "false",
                },
            )
        except CoreNLPConnectionError as exc:
            if not enable_openie:
                raise
            warnings.append(f"OpenIE annotation unavailable; using CoreNLP structural evidence only: {exc}")
            payload = self.annotate_raw(
                text,
                annotators="tokenize,ssplit,pos,lemma,ner,depparse",
                properties=self.properties,
            )

        annotation = self._parse_view_payload(payload, view_id=view_id, text=text)
        annotation.warnings.extend(warnings)
        if enable_openie and not annotation.openie_triples:
            annotation.warnings.append("CoreNLP returned no OpenIE triples for this view.")
        return annotation

    def annotate_views(
        self,
        views: list[dict[str, Any]],
        *,
        enable_openie: bool = True,
        include_constituency: bool = False,
    ) -> list[CoreNLPViewAnnotation]:
        annotations: list[CoreNLPViewAnnotation] = []
        for index, view in enumerate(views, start=1):
            view_id = str(view.get("id") or f"view_{index}")
            sentence = str(view.get("sentence") or "").strip()
            if not sentence:
                annotations.append(
                    CoreNLPViewAnnotation(
                        view_id=view_id,
                        text="",
                        warnings=["Skipped empty declarative view."],
                    )
                )
                continue
            annotations.append(
                self.annotate_view(
                    sentence,
                    view_id=view_id,
                    enable_openie=enable_openie,
                    include_constituency=include_constituency,
                )
            )
        return annotations

    def annotate_raw(
        self,
        text: str,
        *,
        annotators: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.client is None:
            self.start()
        assert self.client is not None
        try:
            if annotators is None and properties is None:
                payload = self.client.annotate(text)
            else:
                payload = self.client.annotate(
                    text,
                    annotators=annotators,
                    output_format="json",
                    properties=properties or {},
                )
        except TypeError:
            if annotators is None and properties is None:
                raise
            try:
                payload = self.client.annotate(text)
            except Exception as exc:
                raise CoreNLPConnectionError(
                    f"CoreNLPClient failed to annotate text through endpoint {self.url}: {exc}"
                ) from exc
        except Exception as exc:
            raise CoreNLPConnectionError(
                f"CoreNLPClient failed to annotate text through endpoint {self.url}: {exc}"
            ) from exc
        return self._coerce_json_payload(payload)

    def _resolve_corenlp_home(self) -> Path | None:
        explicit_home = self.corenlp_home or os.getenv("CORENLP_HOME")
        if explicit_home:
            home = Path(explicit_home).expanduser()
            if not self._is_valid_corenlp_home(home):
                raise CoreNLPConnectionError(
                    f"CoreNLP home does not contain Stanford CoreNLP jar files: {home}"
                )
            return home

        for candidate in self._candidate_corenlp_homes():
            if self._is_valid_corenlp_home(candidate):
                return candidate
        return None

    @staticmethod
    def _candidate_corenlp_homes() -> list[Path]:
        candidates: list[Path] = []
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            cache_root = Path(local_appdata) / "StanfordNLP" / "stanza" / "Cache"
            if cache_root.exists():
                candidates.extend(sorted(cache_root.glob("*/corenlp"), reverse=True))
        candidates.append(Path.home() / "stanza_corenlp")
        return candidates

    @staticmethod
    def _is_valid_corenlp_home(path: Path) -> bool:
        return path.exists() and any(path.glob("stanford-corenlp*.jar"))

    @staticmethod
    def _build_classpath(corenlp_home: Path) -> str:
        jars = sorted(str(path) for path in corenlp_home.glob("*.jar"))
        if not jars:
            raise CoreNLPConnectionError(
                f"CoreNLP home does not contain jar files: {corenlp_home}"
            )
        return os.pathsep.join(jars)

    @staticmethod
    def _coerce_json_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except ValueError as exc:
                raise CoreNLPConnectionError("CoreNLPClient did not return valid JSON.") from exc
            if isinstance(parsed, dict):
                return parsed
        raise CoreNLPConnectionError(
            "CoreNLPClient returned an unsupported annotation payload. "
            "Expected JSON dict output from Stanza CoreNLPClient."
        )

    def _parse_payload(self, payload: dict[str, Any]) -> DependencyParse:
        tokens: list[CoreNLPToken] = []
        edges: list[DependencyEdge] = []
        token_offset = 0

        for sentence in payload.get("sentences", []):
            local_to_global: dict[int, int] = {}
            for token in sentence.get("tokens", []):
                local_index = int(token["index"])
                global_index = token_offset + local_index
                local_to_global[local_index] = global_index
                tokens.append(
                    CoreNLPToken(
                        index=global_index,
                        word=token.get("word", ""),
                        lemma=token.get("lemma"),
                        pos=token.get("pos"),
                        ner=token.get("ner"),
                        character_offset_begin=int(token.get("characterOffsetBegin", -1)),
                        character_offset_end=int(token.get("characterOffsetEnd", -1)),
                    )
                )

            dependencies = sentence.get("enhancedPlusPlusDependencies")
            if dependencies is None:
                raise CoreNLPConnectionError(
                    "CoreNLP response did not contain enhancedPlusPlusDependencies. "
                    "Make sure the depparse annotator is enabled."
                )

            for dep in dependencies:
                governor = int(dep.get("governor", 0))
                dependent = int(dep.get("dependent", 0))
                if governor == 0 or dependent == 0:
                    continue
                if governor not in local_to_global or dependent not in local_to_global:
                    continue
                edges.append(
                    DependencyEdge(
                        source=dep.get("governorGloss", ""),
                        relation=dep.get("dep", ""),
                        target=dep.get("dependentGloss", ""),
                        source_index=local_to_global[governor],
                        target_index=local_to_global[dependent],
                    )
                )

            token_offset += len(sentence.get("tokens", []))

        return DependencyParse(tokens=tokens, edges=edges, raw=payload)

    def _parse_view_payload(self, payload: dict[str, Any], *, view_id: str, text: str) -> CoreNLPViewAnnotation:
        dependency_parse = self._parse_payload(payload)
        triples: list[OpenIETriple] = []
        constituency_parse: str | None = None
        phrase_spans: list[dict[str, Any]] = []

        for sentence in payload.get("sentences", []):
            if constituency_parse is None and isinstance(sentence.get("parse"), str):
                constituency_parse = sentence.get("parse")
            for triple in sentence.get("openie", []) or []:
                if not isinstance(triple, dict):
                    continue
                subject = str(triple.get("subject") or "").strip()
                relation = str(triple.get("relation") or "").strip()
                object_value = str(triple.get("object") or "").strip()
                if not subject or not relation or not object_value:
                    continue
                triples.append(
                    OpenIETriple(
                        subject=subject,
                        relation=relation,
                        object=object_value,
                        confidence=_float_value(triple.get("confidence"), default=1.0),
                        subject_span=_int_list(triple.get("subjectSpan")),
                        relation_span=_int_list(triple.get("relationSpan")),
                        object_span=_int_list(triple.get("objectSpan")),
                        metadata={key: value for key, value in triple.items() if key not in {"subject", "relation", "object"}},
                    )
                )
            phrase_spans.extend(_phrase_spans_from_sentence(sentence))

        return CoreNLPViewAnnotation(
            view_id=view_id,
            text=text,
            tokens=dependency_parse.tokens,
            edges=dependency_parse.edges,
            openie_triples=triples,
            constituency_parse=constituency_parse,
            phrase_spans=phrase_spans,
            raw=payload,
        )


@contextmanager
def _suppress_info_logs() -> Any:
    previous_disable = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(previous_disable)


def _int_list(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for item in raw:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _float_value(raw: Any, *, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _phrase_spans_from_sentence(sentence: dict[str, Any]) -> list[dict[str, Any]]:
    # CoreNLP's JSON constituency output is primarily a bracketed parse string.
    # We keep this as an extension point and expose an empty list unless a caller
    # provides precomputed phrase spans in a mocked payload.
    raw_spans = sentence.get("phraseSpans") or sentence.get("phrase_spans")
    if not isinstance(raw_spans, list):
        return []
    return [item for item in raw_spans if isinstance(item, dict)]
