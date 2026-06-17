from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

from io_utils import read_questions
from models import (
    HanLPSDPPreprocessResult,
    HanLPSDPResult,
    ExplicitEntityResult,
    QuestionRecord,
)

if TYPE_CHECKING:
    from corenlp_parser import CoreNLPParser
    from hanlp_sdp_parser import HanLPSDPParser
    from hanlp_sdp_preprocessor import HanLPSDPPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEPO parsing with explicit entity masking and parser output."
    )
    parser.add_argument(
        "--pipeline",
        choices=("hanlp_sdp", "corenlp_dependency"),
        default="corenlp_dependency",
        help="Pipeline mode. Default: corenlp_dependency.",
    )
    parser.add_argument("--question", help="Run one manually supplied question instead of questions.json.")
    parser.add_argument("--questions-file", default="questions.json", help="Path to questions.json.")
    parser.add_argument("--api-key", help="OpenAI API key. Used only if OPENAI_API_KEY is not set.")
    parser.add_argument("--base-url", help="OpenAI base URL. Used only if OPENAI_BASE_URL is not set.")
    parser.add_argument(
        "--hanlp-model",
        help="HanLP pretrained constant name from hanlp.pretrained.mtl/sdp, or a local model path.",
    )
    parser.add_argument(
        "--corenlp-url",
        default="http://localhost:9000",
        help="Endpoint used by Stanza CoreNLPClient for the managed CoreNLP server.",
    )
    parser.add_argument("--corenlp-memory", default="4G", help="Java heap memory for managed CoreNLP.")
    parser.add_argument(
        "--corenlp-home",
        help="Path to a Stanford CoreNLP directory containing stanford-corenlp*.jar files.",
    )
    parser.add_argument(
        "--corenlp-timeout-ms",
        type=int,
        default=60000,
        help="CoreNLP annotation timeout in milliseconds.",
    )
    parser.add_argument("--debug", action="store_true", help="Print detailed intermediate structures.")
    parser.add_argument(
        "--debug-dir",
        default="debug/hanlp_sdp",
        help="Directory for HanLP Tri-SDP debug JSON files when --debug is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [QuestionRecord(question=args.question)] if args.question else read_questions(args.questions_file)
    if args.pipeline == "hanlp_sdp":
        return _run_hanlp_sdp_cli(args, records)
    return _run_corenlp_dependency_cli(args, records)


def _run_hanlp_sdp_cli(args: argparse.Namespace, records: list[QuestionRecord]) -> int:
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print(
            "This HanLP SDP branch requires one LLM call for explicit entity masking.",
            file=sys.stderr,
        )
        print("Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        from hanlp_sdp_parser import HanLPSDPParser
        from hanlp_sdp_preprocessor import HanLPSDPPreprocessor
        from llm_client import LLMClient

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
        preprocessor = HanLPSDPPreprocessor(llm_client)
        parser = HanLPSDPParser(args.hanlp_model)

        print("If this is the first run, HanLP may download the model automatically.")
        print("You can set HANLP_HOME to control the cache directory.")
        print()

        for index, record in enumerate(records, start=1):
            result = run_hanlp_sdp_pipeline(
                record=record,
                index=index,
                preprocessor=preprocessor,
                parser=parser,
                debug=args.debug,
                debug_dir=args.debug_dir,
            )
            print_hanlp_sdp_result(index, record, result, debug=args.debug)
    except ModuleNotFoundError as exc:
        if "hanlp" in str(exc).lower() or getattr(exc, "name", "") == "hanlp":
            print("Missing dependency: hanlp", file=sys.stderr)
            print("Run: pip install hanlp", file=sys.stderr)
            return 2
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def _run_corenlp_dependency_cli(args: argparse.Namespace, records: list[QuestionRecord]) -> int:
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        from corenlp_parser import CoreNLPConnectionError, CoreNLPParser
        from hanlp_sdp_preprocessor import HanLPSDPPreprocessor
        from llm_client import LLMClient

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
        preprocessor = HanLPSDPPreprocessor(llm_client)

        with CoreNLPParser(
            args.corenlp_url,
            timeout_ms=args.corenlp_timeout_ms,
            memory=args.corenlp_memory,
            corenlp_home=args.corenlp_home,
        ) as parser:
            for index, record in enumerate(records, start=1):
                result = run_corenlp_dependency_pipeline(
                    record=record,
                    index=index,
                    preprocessor=preprocessor,
                    parser=parser,
                    debug=args.debug,
                )
                print_corenlp_dependency_result(index, record, result, debug=args.debug)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (CoreNLPConnectionError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def run_hanlp_sdp_pipeline(
    record: QuestionRecord,
    index: int,
    preprocessor: "HanLPSDPPreprocessor",
    parser: "HanLPSDPParser",
    debug: bool = False,
    debug_dir: str | None = None,
) -> dict[str, Any]:
    from tri_sdp_reasoning_compiler import compile_token_reasoning_structure

    preprocess_result = preprocessor.preprocess(record.question)
    hanlp_input_sentence = preprocess_result.masked_question
    explicit_entities = [mapping.placeholder for mapping in preprocess_result.mask_mappings]
    hanlp_sdp_result = parser.parse(
        hanlp_input_sentence,
        placeholders=explicit_entities,
    )
    token_reasoning_structure = compile_token_reasoning_structure(
        hanlp_sdp_result,
        explicit_entities=explicit_entities,
        masked_question=preprocess_result.masked_question,
        question_id=record.qid or f"q{index}",
        debug=debug,
        debug_dir=debug_dir,
    )
    return {
        "preprocess_result": preprocess_result,
        "explicit_entities": preprocess_result.explicit_entities,
        "explicit_entity_payload": preprocess_result.explicit_entities.raw_payload,
        "masked_question": preprocess_result.masked_question,
        "sdp_input_sentence": preprocess_result.sdp_input_sentence,
        "hanlp_input_sentence": hanlp_input_sentence,
        "entity_mask_mappings": preprocess_result.mask_mappings,
        "hanlp_sdp_result": hanlp_sdp_result,
        "token_reasoning_structure": token_reasoning_structure,
    }


def run_corenlp_dependency_pipeline(
    record: QuestionRecord,
    index: int,
    preprocessor: "HanLPSDPPreprocessor",
    parser: "CoreNLPParser",
    debug: bool = False,
) -> dict[str, Any]:
    del index, debug
    preprocess_result = preprocessor.preprocess(record.question)
    corenlp_input_sentence = preprocess_result.masked_question
    dependency_parse = parser.parse(corenlp_input_sentence)
    return {
        "preprocess_result": preprocess_result,
        "explicit_entities": preprocess_result.explicit_entities,
        "explicit_entity_payload": preprocess_result.explicit_entities.raw_payload,
        "masked_question": preprocess_result.masked_question,
        "sdp_input_sentence": preprocess_result.sdp_input_sentence,
        "corenlp_input_sentence": corenlp_input_sentence,
        "entity_mask_mappings": preprocess_result.mask_mappings,
        "dependency_parse": dependency_parse,
    }


def run_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return run_corenlp_dependency_pipeline(*args, **kwargs)


def print_hanlp_sdp_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    preprocess_result: HanLPSDPPreprocessResult = result["preprocess_result"]
    explicit_entities: ExplicitEntityResult = preprocess_result.explicit_entities
    hanlp_result: HanLPSDPResult = result["hanlp_sdp_result"]
    token_reasoning_structure = result["token_reasoning_structure"]

    separator = "=" * 60
    print(separator)
    title = f"Question {index}"
    if record.qid:
        title += f" ({record.qid})"
    print(title)
    print(separator)
    print()

    print("[Original Question]")
    print(record.question)
    print()

    print("[1. Explicit Entities]")
    if explicit_entities.entities:
        for entity in explicit_entities.entities:
            semantic_type = entity.semantic_type_hint or "Entity"
            print(f" - {entity.text} [{semantic_type}]")
    else:
        print(" (none)")
    print()

    print("[2. Entity Masking]")
    if preprocess_result.mask_mappings:
        for mapping in preprocess_result.mask_mappings:
            print(f" - {mapping.placeholder} -> {mapping.original_text}")
    else:
        print(" (none)")
    print(f"Masked question: {preprocess_result.masked_question}")
    print()

    print("[3. HanLP SDP Parsing]")
    print(f" Model: {hanlp_result.model or '(unknown)'}")
    print(f" HanLP input sentence: {result.get('hanlp_input_sentence') or preprocess_result.masked_question}")
    print()

    print("[Mask Token Check]")
    if hanlp_result.mask_token_checks:
        for placeholder, status in hanlp_result.mask_token_checks.items():
            print(f"{placeholder}: {status}")
    else:
        print("(none)")
    print()

    print("[Raw SDP Edges]")
    _print_all_hanlp_sdp_edges(hanlp_result)
    print()

    print("[4. Token Reasoning Structure]")
    print("[Graph]")
    if token_reasoning_structure.edges:
        for edge in token_reasoning_structure.edges:
            print(f"{edge.source_text} ---- {edge.target_text}")
    else:
        print("(none)")
    print()
    print("[Paths]")
    if token_reasoning_structure.paths:
        for path in token_reasoning_structure.paths:
            print(f"{path.path_id}: {' ---- '.join(path.nodes)}")
    else:
        print("(none)")
    if token_reasoning_structure.answer_anchor:
        print(f"answer_anchor: {token_reasoning_structure.answer_anchor}")
    if token_reasoning_structure.entity_anchors:
        print(f"entity_anchors: {', '.join(token_reasoning_structure.entity_anchors)}")
    if token_reasoning_structure.constraints:
        print(f"constraints: {_format_token_reasoning_constraints(token_reasoning_structure.constraints)}")
    if token_reasoning_structure.candidate_sets:
        print(f"candidate_sets: {_format_candidate_sets(token_reasoning_structure.candidate_sets)}")
    if getattr(token_reasoning_structure, "debug_file", None):
        print(f"Debug file: {token_reasoning_structure.debug_file}")
    combined_warnings = [*preprocess_result.warnings, *hanlp_result.warnings]
    if debug and combined_warnings:
        print()
        print("[HanLP SDP Warnings]")
        for warning in combined_warnings:
            print(f" - {warning}")
    print()


def print_corenlp_dependency_result(index: int, record: QuestionRecord, result: dict[str, Any], debug: bool = False) -> None:
    preprocess_result: HanLPSDPPreprocessResult = result["preprocess_result"]
    explicit_entities: ExplicitEntityResult = preprocess_result.explicit_entities
    dependency_parse = result["dependency_parse"]

    separator = "=" * 60
    print(separator)
    title = f"Question {index}"
    if record.qid:
        title += f" ({record.qid})"
    print(title)
    print(separator)
    print()

    print("[Original Question]")
    print(record.question)
    print()

    print("[1. Explicit Entities]")
    if explicit_entities.entities:
        for entity in explicit_entities.entities:
            semantic_type = entity.semantic_type_hint or "Entity"
            print(f" - {entity.text} [{semantic_type}]")
    else:
        print(" (none)")
    print()

    print("[2. Entity Masking]")
    if preprocess_result.mask_mappings:
        for mapping in preprocess_result.mask_mappings:
            print(f" - {mapping.placeholder} -> {mapping.original_text}")
    else:
        print(" (none)")
    print(f"Masked question: {preprocess_result.masked_question}")
    print()

    print("[3. CoreNLP Dependency Parsing]")
    print(f"CoreNLP input sentence: {result.get('corenlp_input_sentence') or preprocess_result.masked_question}")
    print("[CoreNLP Dependency Edges]")
    _print_dependency_parse_edges(dependency_parse.edges)
    if debug and preprocess_result.warnings:
        print()
        print("[CoreNLP Dependency Warnings]")
        for warning in preprocess_result.warnings:
            print(f" - {warning}")
    print()


def _print_hanlp_edges_for_formalism(hanlp_result: HanLPSDPResult, formalism: str) -> None:
    if formalism not in hanlp_result.sdp_graphs:
        print("(none)")
        return
    print(f"[SDP: {formalism}]")
    edges = [edge for edge in hanlp_result.edges if edge.formalism == formalism]
    if not edges:
        print("(no readable edges)")
        return
    for edge in edges:
        print(edge.display())


def _print_all_hanlp_sdp_edges(hanlp_result: HanLPSDPResult) -> None:
    if not hanlp_result.sdp_graphs:
        print("(none)")
        return
    for formalism in sorted(hanlp_result.sdp_graphs, key=_hanlp_sdp_formalism_sort_key):
        _print_hanlp_edges_for_formalism(hanlp_result, formalism)


def _hanlp_sdp_formalism_sort_key(formalism: str) -> tuple[int, str]:
    order = {"sdp/dm": 0, "sdp/pas": 1, "sdp/psd": 2}
    return (order.get(formalism, 100), formalism)


def _format_token_reasoning_constraints(constraints: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for constraint in constraints:
        text = str(constraint.get("text") or "")
        target = str(constraint.get("target") or "")
        constraint_type = str(constraint.get("type") or "")
        if target:
            rendered.append(f"{constraint_type}:{text}->{target}")
        else:
            rendered.append(f"{constraint_type}:{text}")
    return "; ".join(rendered)


def _format_candidate_sets(candidate_sets: list[list[str]]) -> str:
    return "; ".join(", ".join(candidate_set) for candidate_set in candidate_sets)


def _print_dependency_parse_edges(edges: list[Any]) -> None:
    if not edges:
        print("  (none)")
        return
    for edge in edges:
        print(f"  - {edge.display()}")


if __name__ == "__main__":
    raise SystemExit(main())
