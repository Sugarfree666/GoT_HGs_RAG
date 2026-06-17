from __future__ import annotations

from typing import Any

from .config import Config


def generate_atomic_dag_with_depo(
    question: str,
    config: Config,
    *,
    corenlp_url: str = "http://localhost:9000",
    corenlp_memory: str = "4G",
    corenlp_home: str | None = None,
    corenlp_timeout_ms: int = 60000,
) -> dict[str, Any]:
    """DEPO no longer exposes a CoreNLP atomic-DAG generator."""

    del question, config, corenlp_url, corenlp_memory, corenlp_home, corenlp_timeout_ms
    raise RuntimeError(
        "DEPO's CoreNLP line now stops after Step 3 dependency parsing and no longer generates an atomic DAG. "
        "Pass a precomputed DAG with --dag or use --mock-llm."
    )
