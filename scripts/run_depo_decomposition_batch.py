from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
for path in (DEPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--questions-file")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--questions-root", default="questions")
    parser.add_argument("--output-root", default="runs/depo_decomposition")
    parser.add_argument("--run-id")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url

    from entity_masking_preprocessor import EntityMaskingPreprocessor
    from hanlp_sdp_parser import HanLPSDPParser
    from llm_client import LLMClient
    from main import run_hanlp_sdp_pipeline
    from models import QuestionRecord

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
    preprocessor = EntityMaskingPreprocessor(llm_client)
    parser = HanLPSDPParser()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _repo_path(args.output_root)

    for questions_file in _resolve_question_files(args):
        dataset = _dataset_name(questions_file)
        output_dir = output_root / dataset / run_id
        items = _slice_items(
            _read_question_items(questions_file),
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
        print(f"Running {dataset}: {len(items)} question(s), output={output_dir}")

        for offset, item in enumerate(items, start=1):
            record = QuestionRecord(question=item["question"], qid=item.get("qid"))
            question_dir = output_dir / _question_dir_name(
                item["index"], record.qid, record.question
            )
            result_path = question_dir / "result.json"
            if args.resume and result_path.exists():
                print(f"[skip] {dataset} #{item['index']} {record.question}")
                continue

            print(f"[run {offset}/{len(items)}] {dataset} #{item['index']} {record.question}")
            decomposition = run_hanlp_sdp_pipeline(
                record=record,
                preprocessor=preprocessor,
                parser=parser,
                llm_client=llm_client,
            )
            _write_json(
                result_path,
                {
                    "question": record.question,
                    "gold_answer": item.get("answer"),
                    "dag": decomposition["atomic_question_dag"].to_dict(),
                },
            )
            print(f"[ok]  {dataset} #{item['index']}")

    return 0


def _resolve_question_files(args: argparse.Namespace) -> list[Path]:
    if args.questions_file:
        return [_repo_path(args.questions_file)]
    questions_root = _repo_path(args.questions_root)
    if args.dataset:
        return [questions_root / args.dataset / "questions.json"]
    return sorted(questions_root.glob("*/questions.json"))


def _read_question_items(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        raw_items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw_items = json.loads(path.read_text(encoding="utf-8"))

    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if isinstance(item, str):
            question = item.strip()
            qid = None
            answer = None
        else:
            question = str(item.get("question", item.get("query", ""))).strip()
            qid_value = item.get("id", item.get("qid"))
            qid = str(qid_value) if qid_value is not None else None
            answer = item.get("answer")
        items.append(
            {
                "index": index,
                "qid": qid,
                "question": question,
                "answer": answer,
            }
        )
    return items


def _slice_items(
    items: list[dict[str, Any]],
    *,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = items[start - 1 : end]
    return selected[:limit] if limit is not None else selected


def _question_dir_name(index: int, qid: str | None, question: str) -> str:
    prefix = f"{index:05d}"
    if qid:
        prefix += f"_{_slug(qid, max_len=48)}"
    return f"{prefix}_{_slug(question, max_len=80)}"


def _slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug[:max_len].strip("-") or "question"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return value


def _dataset_name(questions_file: Path) -> str:
    if questions_file.name == "questions.json":
        return questions_file.parent.name
    return questions_file.parent.name if questions_file.parent.name != "questions" else questions_file.stem


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


if __name__ == "__main__":
    raise SystemExit(main())
