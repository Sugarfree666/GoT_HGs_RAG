import json
from pathlib import Path

from models import QuestionRecord


def read_questions(path: str | Path) -> list[QuestionRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        QuestionRecord(
            question=str(item["question"]).strip(),
            qid=str(qid) if (qid := item.get("id", item.get("qid"))) is not None else None,
        )
        for item in payload
    ]
