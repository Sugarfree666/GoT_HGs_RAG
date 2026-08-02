from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from hyper_branch.atomic import AtomicHyperedgeRetriever, AtomicQuestionAnalyzer  # noqa: E402
from hyper_branch.atomic.models import AtomicQuestionAnalysis  # noqa: E402
from hyper_branch.config import load_config  # noqa: E402
from hyper_branch.data.loaders import HypergraphDatasetLoader  # noqa: E402
from hyper_branch.llm import OpenAIAtomicLLMService, OpenAICompatibleClient  # noqa: E402
from hyper_branch.llm.prompts import PromptManager  # noqa: E402
from hyper_branch.utils import normalize_label  # noqa: E402


DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
GENERIC_HEADS = {
    "actor",
    "actress",
    "artist",
    "author",
    "brother",
    "city",
    "company",
    "composer",
    "country",
    "daughter",
    "director",
    "father",
    "film",
    "group",
    "husband",
    "mother",
    "performer",
    "person",
    "place",
    "region",
    "song",
    "son",
    "wife",
}


class AuditPromptManager:
    def __init__(self, prompt_dir: Path, atomic_question_analysis_prompt: str) -> None:
        self.prompt_dir = prompt_dir
        self.atomic_question_analysis_prompt = atomic_question_analysis_prompt
        self.fallback = PromptManager(prompt_dir)

    def get(self, name: str) -> str:
        if name == "atomic_question_analysis":
            return self.atomic_question_analysis_prompt
        return self.fallback.get(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B audit atomic entity recognition prompts and current entity linking on cached DEPO final_test runs."
    )
    parser.add_argument("--source-root", default="runs/depo_hyperbranch")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--per-dataset", type=int, default=5)
    parser.add_argument("--sample-size", type=int, help="Sample this many rows across all datasets instead of --per-dataset.")
    parser.add_argument("--old-prompt-ref", default="HEAD")
    parser.add_argument("--output-dir", help="Exact output directory. Defaults under runs/entity_link_prompt_ab/.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--model", help="Override chat model from dataset config.")
    return parser.parse_args()


def main() -> int:
    _configure_stdout()
    args = parse_args()
    source_root = _repo_path(args.source_root)
    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _sample_rows(source_root=source_root, datasets=args.datasets, seed=args.seed, per_dataset=args.per_dataset, sample_size=args.sample_size)
    if not rows:
        print("No sample rows found.", file=sys.stderr)
        return 2

    current_prompt = (PROJECT_ROOT / "prompts" / "atomic_question_analysis.md").read_text(encoding="utf-8")
    old_prompt = _git_show_prompt(args.old_prompt_ref, "prompts/atomic_question_analysis.md")

    variants = {
        "old": {
            "label": f"{args.old_prompt_ref}:prompts/atomic_question_analysis.md",
            "prompt": old_prompt,
        },
        "new": {
            "label": "working-tree:prompts/atomic_question_analysis.md",
            "prompt": current_prompt,
        },
    }

    logger = logging.getLogger("entity_link_prompt_ab")
    logger.setLevel(logging.WARNING)
    logger.addHandler(logging.NullHandler())

    dataset_contexts: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        dataset = str(row["dataset"])
        context = dataset_contexts.get(dataset)
        if context is None:
            context = _build_dataset_context(dataset, args, logger, variants)
            dataset_contexts[dataset] = context

        print(f"[audit {row_index}/{len(rows)}] {dataset} #{row.get('index')} {row.get('question')}")
        records.append(_audit_row(row, context))

    summary = _build_summary(records, variants, args)
    payload = {
        "summary": summary,
        "samples": records,
    }
    json_path = output_dir / "audit.json"
    md_path = output_dir / "audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(summary, records), encoding="utf-8")

    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"audit_json={json_path}")
    print(f"audit_md={md_path}")
    return 0


def _configure_stdout() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _build_dataset_context(dataset: str, args: argparse.Namespace, logger: logging.Logger, variants: dict[str, dict[str, str]]) -> dict[str, Any]:
    config = load_config(_repo_path(args.config_dir) / f"{dataset}.yaml", PROJECT_ROOT)
    if args.model:
        config.llm.model = args.model
    bundle = HypergraphDatasetLoader(config.dataset, logger).load()
    clients: dict[str, OpenAICompatibleClient] = {}
    analyzers: dict[str, AtomicQuestionAnalyzer] = {}
    retrievers: dict[str, AtomicHyperedgeRetriever] = {}
    for variant_name, variant in variants.items():
        prompt_manager = AuditPromptManager(config.prompts.directory, variant["prompt"])
        client = OpenAICompatibleClient(config.llm)
        service = OpenAIAtomicLLMService(client=client, prompts=prompt_manager)
        clients[variant_name] = client
        analyzers[variant_name] = AtomicQuestionAnalyzer(llm_service=service)
        retrievers[variant_name] = AtomicHyperedgeRetriever(
            dataset=bundle,
            embedder=client,
            config=config.retrieval,
            logger=logger,
        )
    return {
        "config": config,
        "bundle": bundle,
        "clients": clients,
        "analyzers": analyzers,
        "retrievers": retrievers,
    }


def _audit_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    atomic_nodes = _load_atomic_nodes(row)
    row_record: dict[str, Any] = {
        "dataset": row.get("dataset"),
        "index": row.get("index"),
        "qid": row.get("qid"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "final_answer": row.get("final_answer"),
        "output_dir": row.get("output_dir"),
        "atomic_nodes": [],
    }
    for node in atomic_nodes:
        variants: dict[str, Any] = {}
        for variant_name in ("old", "new"):
            analyzer: AtomicQuestionAnalyzer = context["analyzers"][variant_name]
            retriever: AtomicHyperedgeRetriever = context["retrievers"][variant_name]
            question = str(node.get("resolved_question") or node.get("question") or "")
            dependency_answers = [item for item in node.get("dependency_answers", []) if isinstance(item, dict)]
            try:
                analysis = analyzer.analyze(question, dependency_answers)
                links = _link_entities(retriever, question, analysis)
                variants[variant_name] = {
                    "status": "ok",
                    "entities": list(analysis.entities),
                    "answer_type": analysis.answer_type,
                    "links": links,
                    "slot_like_mentions": [entity for entity in analysis.entities if _looks_slot_like(entity)],
                }
            except Exception as exc:
                variants[variant_name] = {
                    "status": "error",
                    "entities": [],
                    "answer_type": "",
                    "links": [],
                    "slot_like_mentions": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
        row_record["atomic_nodes"].append(
            {
                "node_id": node.get("node_id"),
                "original_question": node.get("original_question"),
                "resolved_question": node.get("resolved_question") or node.get("question"),
                "dependency_answers": node.get("dependency_answers", []),
                "variants": variants,
                "diff": _diff_variants(variants.get("old", {}), variants.get("new", {})),
            }
        )
    return row_record


def _link_entities(retriever: AtomicHyperedgeRetriever, question: str, analysis: AtomicQuestionAnalysis) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for index, mention in enumerate(analysis.entities):
        try:
            match = retriever.link_anchor_entity(question=question, mention=mention, analysis=analysis, query_index=index)
        except Exception as exc:
            links.append(
                {
                    "mention": mention,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if match is None:
            links.append(
                {
                    "mention": mention,
                    "status": "unlinked",
                    "matched_entity_id": "",
                    "matched_entity": "",
                    "match_type": "NONE",
                    "link_score": 0.0,
                    "same_canonical_label": False,
                }
            )
            continue
        metadata = match.to_metadata()
        matched = str(metadata.get("matched_entity", "") or "")
        links.append(
            {
                "mention": mention,
                "status": "linked",
                "matched_entity_id": metadata.get("matched_entity_id", ""),
                "matched_entity": matched,
                "match_type": metadata.get("match_type", ""),
                "link_score": metadata.get("link_score", 0.0),
                "vector_score": metadata.get("vector_score", 0.0),
                "llm_confidence": metadata.get("llm_confidence", 0.0),
                "candidate_rank": metadata.get("candidate_rank", 0),
                "same_canonical_label": _canonical(mention) == _canonical(matched),
            }
        )
    return links


def _diff_variants(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_entities = [str(item) for item in old.get("entities", [])]
    new_entities = [str(item) for item in new.get("entities", [])]
    old_keys = {_canonical(item) for item in old_entities}
    new_keys = {_canonical(item) for item in new_entities}
    return {
        "added_entities": [item for item in new_entities if _canonical(item) not in old_keys],
        "removed_entities": [item for item in old_entities if _canonical(item) not in new_keys],
        "same_entities": old_keys == new_keys,
    }


def _build_summary(records: list[dict[str, Any]], variants: dict[str, dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    metrics = {
        "questions": len(records),
        "atomic_nodes": sum(len(row.get("atomic_nodes", [])) for row in records),
        "old": _variant_metrics(records, "old"),
        "new": _variant_metrics(records, "new"),
        "changed_atomic_nodes": 0,
        "new_added_entities": 0,
        "new_removed_entities": 0,
    }
    for row in records:
        for node in row.get("atomic_nodes", []):
            diff = node.get("diff", {})
            added = diff.get("added_entities", [])
            removed = diff.get("removed_entities", [])
            if added or removed:
                metrics["changed_atomic_nodes"] += 1
            metrics["new_added_entities"] += len(added)
            metrics["new_removed_entities"] += len(removed)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "source_root": str(_repo_path(args.source_root)),
        "datasets": list(args.datasets),
        "per_dataset": args.per_dataset,
        "sample_size": args.sample_size,
        "prompt_variants": {
            name: {
                "label": item["label"],
                "sha256_12": hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()[:12],
            }
            for name, item in variants.items()
        },
        "metrics": metrics,
    }


def _variant_metrics(records: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    extracted = 0
    linked = 0
    unlinked = 0
    errors = 0
    exact = 0
    nonexact = 0
    same_label = 0
    slot_like = 0
    empty_nodes = 0
    for row in records:
        for node in row.get("atomic_nodes", []):
            variant = node.get("variants", {}).get(variant_name, {})
            entities = variant.get("entities", [])
            if not entities:
                empty_nodes += 1
            slot_like += len(variant.get("slot_like_mentions", []))
            extracted += len(entities)
            for link in variant.get("links", []):
                status = link.get("status")
                if status == "linked":
                    linked += 1
                    if link.get("match_type") == "exact":
                        exact += 1
                    else:
                        nonexact += 1
                    if link.get("same_canonical_label"):
                        same_label += 1
                elif status == "unlinked":
                    unlinked += 1
                else:
                    errors += 1
    return {
        "extracted_mentions": extracted,
        "linked_mentions": linked,
        "unlinked_mentions": unlinked,
        "link_errors": errors,
        "exact_links": exact,
        "nonexact_links": nonexact,
        "same_canonical_label_links": same_label,
        "slot_like_mentions": slot_like,
        "empty_entity_nodes": empty_nodes,
    }


def _render_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Atomic Entity Recognition + Linking Prompt A/B",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Source root: `{summary['source_root']}`",
        f"- Questions: `{metrics['questions']}`",
        f"- Atomic nodes: `{metrics['atomic_nodes']}`",
        "",
        "## Prompt Variants",
        "",
    ]
    for name, item in summary["prompt_variants"].items():
        lines.append(f"- `{name}`: {item['label']} (`{item['sha256_12']}`)")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Variant | Extracted | Linked | Unlinked | Link errors | Exact | Non-exact | Same-label | Slot-like | Empty nodes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("old", "new"):
        item = metrics[name]
        lines.append(
            f"| {name} | {item['extracted_mentions']} | {item['linked_mentions']} | {item['unlinked_mentions']} | "
            f"{item['link_errors']} | {item['exact_links']} | {item['nonexact_links']} | "
            f"{item['same_canonical_label_links']} | {item['slot_like_mentions']} | {item['empty_entity_nodes']} |"
        )
    lines.extend(
        [
            "",
            f"- Changed atomic nodes: `{metrics['changed_atomic_nodes']}`",
            f"- New-only added entities: `{metrics['new_added_entities']}`",
            f"- Old-only removed entities: `{metrics['new_removed_entities']}`",
            "",
            "## Samples",
            "",
        ]
    )
    for row in records:
        lines.extend(
            [
                f"### {row.get('dataset')} #{row.get('index')}",
                "",
                str(row.get("question") or ""),
                "",
                f"- Gold answer: `{row.get('gold_answer')}`",
                f"- Previous final answer: `{row.get('final_answer')}`",
                "",
            ]
        )
        for node in row.get("atomic_nodes", []):
            lines.append(f"#### {node.get('node_id')}: {node.get('resolved_question')}")
            lines.append("")
            diff = node.get("diff", {})
            if diff.get("added_entities") or diff.get("removed_entities"):
                lines.append(f"- New added: `{diff.get('added_entities')}`")
                lines.append(f"- New removed: `{diff.get('removed_entities')}`")
            old = node.get("variants", {}).get("old", {})
            new = node.get("variants", {}).get("new", {})
            lines.append(f"- Old entities: `{old.get('entities', [])}`")
            lines.append(f"- New entities: `{new.get('entities', [])}`")
            lines.append(f"- Old links: {_compact_links(old.get('links', []))}")
            lines.append(f"- New links: {_compact_links(new.get('links', []))}")
            for variant_name, variant in (("old", old), ("new", new)):
                if variant.get("status") == "error":
                    lines.append(f"- {variant_name} error: `{variant.get('error')}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _compact_links(links: list[dict[str, Any]]) -> str:
    if not links:
        return "`[]`"
    parts: list[str] = []
    for link in links:
        mention = link.get("mention")
        if link.get("status") == "linked":
            parts.append(
                f"{mention} -> {link.get('matched_entity')} ({link.get('match_type')})"
            )
        elif link.get("status") == "unlinked":
            parts.append(f"{mention} -> NONE")
        else:
            parts.append(f"{mention} -> ERROR")
    return "`" + "; ".join(parts) + "`"


def _load_atomic_nodes(row: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(str(row.get("hyperbranch_run_dir") or ""))
    path = run_dir / "artifacts" / "atomic_question_analyses.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing atomic_question_analyses.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [item for item in payload if isinstance(item, dict)]


def _sample_rows(
    *,
    source_root: Path,
    datasets: list[str],
    seed: int,
    per_dataset: int,
    sample_size: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        manifest = source_root / dataset / "final_test" / "manifest.jsonl"
        rows = _read_manifest(manifest)
        rows = [
            row
            for row in rows
            if row.get("status") == "ok"
            and Path(str(row.get("hyperbranch_run_dir") or "")).exists()
        ]
        by_dataset[dataset] = rows
    if sample_size is not None:
        all_rows = [row for rows in by_dataset.values() for row in rows]
        return sorted(rng.sample(all_rows, min(sample_size, len(all_rows))), key=lambda row: (str(row.get("dataset")), int(row.get("index", 0))))
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = by_dataset[dataset]
        selected.extend(rng.sample(rows, min(per_dataset, len(rows))))
    return sorted(selected, key=lambda row: (str(row.get("dataset")), int(row.get("index", 0))))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _git_show_prompt(ref: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{relative_path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _looks_slot_like(entity: str) -> bool:
    lowered = normalize_label(entity).lower().strip("`'\".,:;!?")
    if not lowered:
        return False
    for prefix in ("which ", "what ", "who ", "where ", "when ", "how "):
        if lowered.startswith(prefix):
            return True
    tokens = lowered.split()
    if not tokens:
        return False
    if tokens[0] in {"the", "a", "an"} and len(tokens) > 1:
        return tokens[1] in GENERIC_HEADS
    return lowered in GENERIC_HEADS


def _canonical(value: str) -> str:
    text = normalize_label(value).lower()
    text = text.strip("`'\" ")
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return _repo_path(args.output_dir)
    sample_label = args.sample_size if args.sample_size is not None else args.per_dataset * len(args.datasets)
    return PROJECT_ROOT / "runs" / "entity_link_prompt_ab" / f"sample_{sample_label}_seed_{args.seed}"


if __name__ == "__main__":
    raise SystemExit(main())
