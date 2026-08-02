from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from hyper_branch.atomic.retriever import (  # noqa: E402
    AtomicHyperedgeRetriever,
    _canonical_entity_key,
    _vector_candidate_constraint_conflict,
)
from hyper_branch.config import load_config  # noqa: E402
from hyper_branch.data.graph import KnowledgeHypergraph  # noqa: E402
from hyper_branch.data.vector_store import VectorStore  # noqa: E402
from hyper_branch.llm.client import OpenAICompatibleClient  # noqa: E402
from hyper_branch.utils import normalize_label, short_text  # noqa: E402


DEFAULT_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
DEFAULT_RUN_NAME = "top50_blocks_concise_prompt"
DEFAULT_THRESHOLD = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only entity linking by replaying cached atomic entity mentions. "
            "The script never calls a chat/completions endpoint."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--run-root", default="runs/depo_hyperbranch")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", default="runs/entity_link_eval/top50_current")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--report-top-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-cache", help="Defaults to <output-dir>/embedding_cache.json.")
    parser.add_argument("--labels", help="Optional human labels JSON generated from review_template.json.")
    parser.add_argument("--offline", action="store_true", help="Fail instead of requesting missing embeddings.")
    return parser.parse_args()


class PersistentEmbeddingClient:
    """Embedding-only wrapper with a disk cache; it exposes no chat method."""

    def __init__(
        self,
        *,
        config: Any,
        cache_path: Path,
        batch_size: int,
        offline: bool,
    ) -> None:
        self.config = config
        self.cache_path = cache_path
        self.batch_size = max(1, int(batch_size))
        self.offline = bool(offline)
        self.vectors: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.is_file():
            return
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if payload.get("embedding_model") != self.config.embedding_model:
            return
        for text, vector in payload.get("vectors", {}).items():
            self.vectors[str(text)] = np.asarray(vector, dtype=np.float32)

    def prefetch(self, texts: Iterable[str]) -> None:
        missing = list(dict.fromkeys(str(text) for text in texts if str(text) not in self.vectors))
        if not missing:
            return
        if self.offline:
            preview = ", ".join(repr(text) for text in missing[:10])
            raise RuntimeError(
                f"Embedding cache is missing {len(missing)} mention(s) in offline mode: {preview}"
            )
        client = OpenAICompatibleClient(self.config)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = client.embed_texts(batch, stage="entity_link_eval_embedding_only")
            for text, vector in zip(batch, vectors, strict=True):
                self.vectors[text] = np.asarray(vector, dtype=np.float32)
            self._save()

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.config.embedding_model,
            "vectors": {text: vector.astype(float).tolist() for text, vector in self.vectors.items()},
        }
        temp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.cache_path)

    def embed_texts(self, texts: list[str], stage: str | None = None) -> list[np.ndarray]:
        del stage
        self.prefetch(texts)
        return [self.vectors[text] for text in texts]


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.report_top_k <= 0:
        raise ValueError("--report-top-k must be positive.")

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _resolve(args.embedding_cache) if args.embedding_cache else output_dir / "embedding_cache.json"
    labels = _load_labels(_resolve(args.labels) if args.labels else None)

    all_records: list[dict[str, Any]] = []
    dataset_summaries: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        print(f"[entity-link-eval] loading cached mentions: {dataset}", flush=True)
        cached_rows = _load_cached_mentions(
            dataset=dataset,
            run_dir=_resolve(Path(args.run_root) / dataset / args.run_name),
        )
        config = load_config(_resolve(Path(args.config_dir) / f"{dataset}.yaml"), PROJECT_ROOT)
        graph = KnowledgeHypergraph.from_graphml(config.dataset.root / (config.dataset.graphml_file or "graph_chunk_entity_relation.graphml"))
        dataset_stub = SimpleNamespace(graph=graph, entity_store=None)
        exact_retriever = AtomicHyperedgeRetriever(
            dataset=dataset_stub,
            embedder=None,
            config=config.retrieval,
            logger=logging.getLogger("entity_link_eval"),
        )

        unresolved_mentions: list[str] = []
        for row in cached_rows:
            exact_candidates = exact_retriever._entity_lookup_candidates(
                row["mention"],
                question=row["atomic_question"],
            )
            row["exact_candidate_ids"] = [str(item["entity_id"]) for item in exact_candidates]
            if len(exact_candidates) != 1:
                unresolved_mentions.append(row["mention"])

        embedder = PersistentEmbeddingClient(
            config=config.llm,
            cache_path=cache_path,
            batch_size=args.embedding_batch_size,
            offline=args.offline,
        )
        embedder.prefetch(unresolved_mentions)

        print(f"[entity-link-eval] loading entity-name vectors: {dataset}", flush=True)
        entity_store = VectorStore.from_json(
            config.dataset.root / config.dataset.entity_vdb_file,
            name="entity_names",
            label_fields=("entity_name",),
        )
        dataset_stub.entity_store = entity_store
        retriever = AtomicHyperedgeRetriever(
            dataset=dataset_stub,
            embedder=embedder,
            config=config.retrieval,
            logger=logging.getLogger("entity_link_eval"),
        )

        dataset_records: list[dict[str, Any]] = []
        for row in cached_rows:
            record = _evaluate_mention(
                row=row,
                retriever=retriever,
                graph=graph,
                entity_store=entity_store,
                embedder=embedder,
                threshold=float(args.threshold),
                report_top_k=int(args.report_top_k),
                label=labels.get(row["record_id"]),
            )
            dataset_records.append(record)
            all_records.append(record)

        dataset_summaries[dataset] = _summarize(dataset_records, threshold=float(args.threshold))
        del retriever, exact_retriever, entity_store, graph, dataset_stub
        gc.collect()

    summary = {
        "run_name": args.run_name,
        "threshold": float(args.threshold),
        "embedding_model": "text-embedding-3-small",
        "chat_llm_calls": 0,
        "datasets": dataset_summaries,
        "overall": _summarize(all_records, threshold=float(args.threshold)),
        "threshold_sweep": _threshold_sweep(all_records),
        "metric_notes": {
            "accepted_link_precision": "Identity precision among accepted links. Unique exact-name links are presumed correct unless manually reviewed; every non-exact case is human-labeled.",
            "decision_accuracy": "Correct link/reject decisions. Every non-exact case and exact links that disagreed with the cached linker were reviewed; remaining unique exact-name links are presumed correct.",
            "cached_evidence_reachable_2hop": "Whether any linked mention can reach a cached used hyperedge through the current two-hop entity expansion.",
            "legacy_agreement": "Agreement with the older cached linker; this is diagnostic silver data, not ground truth.",
        },
    }

    (output_dir / "records.json").write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_markdown(summary, all_records), encoding="utf-8")
    _write_review_template(output_dir / "review_template.json", all_records, labels)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"report={output_dir / 'report.md'}")
    return 0


def _load_cached_mentions(*, dataset: str, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name[:5].isdigit()):
        pipeline_path = question_dir / "pipeline.json"
        analyses_path = question_dir / "hyperbranch_run" / "artifacts" / "atomic_question_analyses.json"
        retrieval_path = question_dir / "hyperbranch_run" / "artifacts" / "atomic_retrieval.json"
        answers_path = question_dir / "hyperbranch_run" / "artifacts" / "atomic_answers.json"
        if not all(path.is_file() for path in (pipeline_path, analyses_path, retrieval_path, answers_path)):
            continue
        pipeline = _read_json(pipeline_path)
        analyses = _read_json(analyses_path)
        retrievals = {str(item.get("node_id")): item for item in _read_json(retrieval_path)}
        answers = {str(item.get("node_id")): item for item in _read_json(answers_path)}
        index = int(pipeline.get("index", int(question_dir.name[:5])))
        original_question = str(pipeline.get("question", ""))

        for analysis_item in analyses:
            node_id = str(analysis_item.get("node_id", ""))
            analysis = analysis_item.get("analysis", {}) if isinstance(analysis_item.get("analysis"), dict) else {}
            entities = [str(item).strip() for item in analysis.get("entities", []) if str(item).strip()]
            retrieval = retrievals.get(node_id, {})
            answer = answers.get(node_id, {})
            legacy_by_mention: dict[str, list[str]] = defaultdict(list)
            for match in retrieval.get("anchor_matches", []):
                mention_key = _canonical_entity_key(str(match.get("query_entity", "")))
                entity_id = str(match.get("matched_entity_id", "") or "")
                if mention_key and entity_id and entity_id not in legacy_by_mention[mention_key]:
                    legacy_by_mention[mention_key].append(entity_id)

            for mention_index, mention in enumerate(entities):
                record_id = f"{dataset}:{index:05d}:{node_id}:{mention_index}"
                rows.append(
                    {
                        "record_id": record_id,
                        "dataset": dataset,
                        "index": index,
                        "question_dir": question_dir.name,
                        "node_id": node_id,
                        "mention_index": mention_index,
                        "original_question": original_question,
                        "atomic_question": str(analysis_item.get("resolved_question") or analysis_item.get("question") or ""),
                        "mention": mention,
                        "atomic_answer": str(answer.get("answer", "") or ""),
                        "used_hyperedge_ids": [str(item) for item in answer.get("used_hyperedge_ids", []) if str(item)],
                        "legacy_entity_ids": legacy_by_mention.get(_canonical_entity_key(mention), []),
                    }
                )
    return rows


def _evaluate_mention(
    *,
    row: dict[str, Any],
    retriever: AtomicHyperedgeRetriever,
    graph: KnowledgeHypergraph,
    entity_store: VectorStore,
    embedder: PersistentEmbeddingClient,
    threshold: float,
    report_top_k: int,
    label: dict[str, Any] | None,
) -> dict[str, Any]:
    exact_ids = list(row.get("exact_candidate_ids", []))
    candidates: list[dict[str, Any]] = []
    vector_score: float | None = None
    top1_entity_id = ""
    constraint_conflict = ""
    match_type = "unlinked"
    linked_entity_id = ""

    if len(exact_ids) == 1:
        linked_entity_id = exact_ids[0]
        top1_entity_id = linked_entity_id
        match_type = "exact"
    else:
        vector = embedder.embed_texts([row["mention"]], stage="entity_link_eval_embedding_only")[0]
        for rank, match in enumerate(entity_store.query(vector, top_k=report_top_k), start=1):
            entity_id = retriever._resolve_entity_id_from_vector_match(match) or ""
            candidates.append(
                _candidate_payload(graph=graph, entity_id=entity_id, label=match.label, score=float(match.score), rank=rank)
            )
        if candidates:
            top1_entity_id = str(candidates[0].get("entity_id", ""))
            vector_score = float(candidates[0]["score"])
            constraint_conflict = _vector_candidate_constraint_conflict(
                row["mention"],
                top1_entity_id,
                graph,
                question=row["atomic_question"],
            )
            if top1_entity_id and vector_score >= threshold and not constraint_conflict:
                linked_entity_id = top1_entity_id
                match_type = "vector"

    direct_hits, two_hop_hits = _used_hyperedge_reachability(
        graph=graph,
        entity_id=linked_entity_id,
        used_hyperedge_ids=set(row.get("used_hyperedge_ids", [])),
    )
    entity_payload = _entity_payload(graph, linked_entity_id)
    legacy_ids = list(row.get("legacy_entity_ids", []))
    legacy_agreement = bool(linked_entity_id and linked_entity_id in legacy_ids) if legacy_ids else None
    human_expected_entity = "" if label is None else str(label.get("expected_entity", "") or "")
    human_top1_correct = None if label is None else label.get("top1_correct")
    if human_top1_correct is not None:
        human_top1_correct = bool(human_top1_correct)
    # ``top1_correct`` describes the candidate shown during the original
    # review.  A later general rule may replace that candidate with the
    # reviewer-provided target, in which case the current result is correct
    # even if the old candidate was marked false.
    if (
        human_expected_entity
        and top1_entity_id
        and _canonical_entity_key(human_expected_entity) == _canonical_entity_key(top1_entity_id)
    ):
        human_top1_correct = True

    link_correct: bool | None
    if not linked_entity_id:
        link_correct = None
    elif human_top1_correct is not None:
        link_correct = human_top1_correct
    elif match_type == "exact":
        link_correct = True
    else:
        link_correct = None

    decision_correct: bool | None
    if human_top1_correct is None and match_type != "exact":
        decision_correct = None
    elif match_type == "exact":
        decision_correct = bool(link_correct)
    else:
        decision_correct = bool(linked_entity_id) == human_top1_correct

    return {
        **{key: value for key, value in row.items() if key != "exact_candidate_ids"},
        "exact_candidate_ids": exact_ids,
        "match_type": match_type,
        "linked_entity_id": linked_entity_id,
        "linked_entity": normalize_label(linked_entity_id),
        "linked_entity_type": entity_payload["entity_type"],
        "linked_entity_description": entity_payload["description"],
        "top1_entity_id": top1_entity_id,
        "top1_entity": normalize_label(top1_entity_id),
        "vector_score": vector_score,
        "threshold": threshold,
        "constraint_conflict": constraint_conflict,
        "vector_candidates": candidates,
        "legacy_agreement": legacy_agreement,
        "used_hyperedge_hit_1hop": direct_hits,
        "used_hyperedge_hit_2hop": two_hop_hits,
        "human_top1_correct": human_top1_correct,
        "human_expected_entity": human_expected_entity,
        "human_note": "" if label is None else str(label.get("note", "") or ""),
        "link_correct": link_correct,
        "decision_correct": decision_correct,
        "linked_context": _linked_context(graph, linked_entity_id),
    }


def _candidate_payload(*, graph: KnowledgeHypergraph, entity_id: str, label: str, score: float, rank: int) -> dict[str, Any]:
    payload = _entity_payload(graph, entity_id)
    return {
        "rank": rank,
        "entity_id": entity_id,
        "entity": normalize_label(entity_id or label),
        "score": score,
        "entity_type": payload["entity_type"],
        "description": payload["description"],
    }


def _entity_payload(graph: KnowledgeHypergraph, entity_id: str) -> dict[str, str]:
    node = graph.nodes.get(entity_id)
    return {
        "entity_type": normalize_label(str(getattr(node, "entity_type", "") or "")),
        "description": short_text(normalize_label(str(getattr(node, "description", "") or "")), 400),
    }


def _linked_context(graph: KnowledgeHypergraph, entity_id: str, limit: int = 5) -> list[str]:
    if not entity_id:
        return []
    return [
        short_text(normalize_label(str(graph.nodes[hyperedge_id].description or hyperedge_id)), 320)
        for hyperedge_id in graph.entity_hyperedge_ids(entity_id)[:limit]
        if hyperedge_id in graph.nodes
    ]


def _used_hyperedge_reachability(
    *,
    graph: KnowledgeHypergraph,
    entity_id: str,
    used_hyperedge_ids: set[str],
) -> tuple[list[str], list[str]]:
    if not entity_id or not used_hyperedge_ids:
        return [], []
    first_hop = set(graph.entity_hyperedge_ids(entity_id))
    direct_hits = sorted(first_hop & used_hyperedge_ids)
    two_hop_hits = set(direct_hits)
    remaining = used_hyperedge_ids - two_hop_hits
    if remaining:
        for hyperedge_id in first_hop:
            for neighbor_entity_id in graph.hyperedge_entity_ids(hyperedge_id):
                hits = set(graph.entity_hyperedge_ids(neighbor_entity_id)) & remaining
                two_hop_hits.update(hits)
                remaining.difference_update(hits)
                if not remaining:
                    break
            if not remaining:
                break
    return direct_hits, sorted(two_hop_hits)


def _summarize(records: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    exact = [item for item in records if item["match_type"] == "exact"]
    vector_attempts = [item for item in records if len(item.get("exact_candidate_ids", [])) != 1]
    vector_accepted = [item for item in vector_attempts if item["match_type"] == "vector"]
    unlinked = [item for item in records if item["match_type"] == "unlinked"]
    labeled = [item for item in vector_attempts if item.get("human_top1_correct") is not None]
    accepted_links = [item for item in records if item.get("linked_entity_id") and item.get("link_correct") is not None]
    correct_accepted = [item for item in accepted_links if item["link_correct"]]
    decisions = [item for item in records if item.get("decision_correct") is not None]
    correct_decisions = [item for item in decisions if item["decision_correct"]]

    legacy_comparable = [item for item in records if item.get("legacy_entity_ids") and item.get("linked_entity_id")]
    used_by_node: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        if item.get("used_hyperedge_ids"):
            used_by_node[(item["dataset"], int(item["index"]), item["node_id"])].append(item)
    reachable_nodes = sum(
        any(item.get("used_hyperedge_hit_2hop") for item in items)
        for items in used_by_node.values()
    )

    return {
        "mention_count": len(records),
        "exact_count": len(exact),
        "non_exact_count": len(vector_attempts),
        "vector_accepted_count": len(vector_accepted),
        "unlinked_count": len(unlinked),
        "link_coverage": _ratio(len(exact) + len(vector_accepted), len(records)),
        "threshold": threshold,
        "human_labeled_non_exact_count": len(labeled),
        "vector_top1_correct_rate": _ratio(sum(bool(item["human_top1_correct"]) for item in labeled), len(labeled)),
        "accepted_link_precision": _ratio(len(correct_accepted), len(accepted_links)),
        "decision_accuracy": _ratio(len(correct_decisions), len(decisions)),
        "legacy_comparable_count": len(legacy_comparable),
        "legacy_agreement": _ratio(sum(bool(item["legacy_agreement"]) for item in legacy_comparable), len(legacy_comparable)),
        "nodes_with_cached_used_evidence": len(used_by_node),
        "cached_evidence_reachable_2hop_nodes": reachable_nodes,
        "cached_evidence_reachable_2hop_rate": _ratio(reachable_nodes, len(used_by_node)),
    }


def _threshold_sweep(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = [
        item
        for item in records
        if len(item.get("exact_candidate_ids", [])) != 1
        and item.get("human_top1_correct") is not None
        and item.get("vector_score") is not None
    ]
    rows: list[dict[str, Any]] = []
    for threshold_int in range(40, 91, 2):
        threshold = threshold_int / 100.0
        accepted = [
            item
            for item in labeled
            if float(item["vector_score"]) >= threshold and not item.get("constraint_conflict")
        ]
        correct_accepted = [item for item in accepted if item["human_top1_correct"]]
        correct_candidates = [item for item in labeled if item["human_top1_correct"]]
        correct_decisions = sum(
            (float(item["vector_score"]) >= threshold and not item.get("constraint_conflict"))
            == bool(item["human_top1_correct"])
            for item in labeled
        )
        precision = _ratio(len(correct_accepted), len(accepted))
        recall = _ratio(len(correct_accepted), len(correct_candidates))
        rows.append(
            {
                "threshold": threshold,
                "labeled_count": len(labeled),
                "accepted_count": len(accepted),
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "decision_accuracy": _ratio(correct_decisions, len(labeled)),
            }
        )
    return rows


def _write_review_template(path: Path, records: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> None:
    review_records: dict[str, dict[str, Any]] = {}
    for item in records:
        if len(item.get("exact_candidate_ids", [])) == 1 and item.get("legacy_agreement") is not False:
            continue
        existing = labels.get(item["record_id"], {})
        review_records[item["record_id"]] = {
            "top1_correct": existing.get("top1_correct"),
            "expected_entity": existing.get("expected_entity", ""),
            "note": existing.get("note", ""),
        }
    path.write_text(json.dumps({"records": review_records}, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Cached entity-link evaluation",
        "",
        "This run reused cached `atomic_question_analyses.json` files and made zero chat-LLM calls.",
        "All current non-exact mentions and exact links that disagreed with the cached linker were manually reviewed. Other unique exact-name links are presumed correct.",
        "Cached-evidence reachability is reported separately and is not identity ground truth.",
        "",
        "## Summary",
        "",
        "| dataset | mentions | exact | vector accepted | unlinked | coverage | decision accuracy | cached evidence reachable@2hop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in [*summary["datasets"].items(), ("overall", summary["overall"])]:
        lines.append(
            f"| {dataset} | {item['mention_count']} | {item['exact_count']} | {item['vector_accepted_count']} | "
            f"{item['unlinked_count']} | {_pct(item['link_coverage'])} | {_pct(item['decision_accuracy'])} | "
            f"{_pct(item['cached_evidence_reachable_2hop_rate'])} |"
        )

    lines.extend(["", "## Non-exact mentions", ""])
    for item in records:
        if len(item.get("exact_candidate_ids", [])) == 1:
            continue
        lines.extend(
            [
                f"### `{item['record_id']}`",
                "",
                f"- Atomic question: {item['atomic_question']}",
                f"- Mention: `{item['mention']}`",
                f"- Current: `{item['match_type']}` → `{item['linked_entity'] or 'NONE'}`; score={item['vector_score']}",
                f"- Constraint conflict: `{item['constraint_conflict'] or 'NONE'}`",
                f"- Legacy reference: {', '.join(normalize_label(value) for value in item['legacy_entity_ids']) or 'NONE'}",
                f"- Atomic answer: `{item['atomic_answer']}`",
                f"- Used evidence reachable: 1-hop={bool(item['used_hyperedge_hit_1hop'])}, 2-hop={bool(item['used_hyperedge_hit_2hop'])}",
                f"- Human label: `{item['human_top1_correct']}` {item['human_note']}",
                "",
                "| rank | candidate | score | type | description |",
                "|---:|---|---:|---|---|",
            ]
        )
        for candidate in item["vector_candidates"]:
            description = str(candidate["description"]).replace("|", "\\|")
            lines.append(
                f"| {candidate['rank']} | {candidate['entity']} | {candidate['score']:.4f} | "
                f"{candidate['entity_type']} | {description} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = _read_json(path)
    records = payload.get("records", payload) if isinstance(payload, dict) else {}
    return {str(key): dict(value) for key, value in records.items() if isinstance(value, dict)}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
