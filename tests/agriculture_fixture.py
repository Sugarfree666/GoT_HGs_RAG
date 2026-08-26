from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_agriculture_fixture(project_root: Path) -> None:
    """Create the minimal dataset used by pipeline smoke tests."""

    config_path = project_root / "configs" / "agriculture.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(
            """dataset_root: datasets/agriculture
top_k: 3
model: gpt-4o-mini
embedding_model: text-embedding-3-small
timeout_seconds: 120
temperature: 0.1
""",
            encoding="utf-8",
        )

    root = project_root / "datasets" / "agriculture"
    root.mkdir(parents=True, exist_ok=True)
    _write_text_if_missing(root / "graph_chunk_entity_relation.graphml", _graphml_payload())
    _write_json_if_missing(
        root / "kv_store_full_docs.json",
        {
            "doc-1": {
                "content": "Urban farms can build community support through outreach, education, partnerships, and safe soil practices."
            }
        },
    )
    _write_json_if_missing(
        root / "kv_store_text_chunks.json",
        {
            "chunk-1": {
                "content": "Urban farms build community support through outreach, education, partnerships, and safe soil practices."
            }
        },
    )
    _write_json_if_missing(
        root / "vdb_entity_names.json",
        _vector_store(
            [
                {"__id__": "urban farms", "entity_name": "urban farms"},
                {"__id__": "community support", "entity_name": "community support"},
            ]
        ),
    )
    _write_json_if_missing(
        root / "vdb_hyperedges.json",
        _vector_store(
            [
                {
                    "__id__": "urban farms build community support",
                    "hyperedge_name": "urban farms build community support",
                }
            ]
        ),
    )
    _write_json_if_missing(
        root / "vdb_chunks.json",
        _vector_store([{"__id__": "chunk-1"}]),
    )


def _write_text_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _write_json_if_missing(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _vector_store(rows: list[dict[str, Any]], dimension: int = 1536) -> dict[str, Any]:
    matrix: list[list[float]] = []
    for index, _row in enumerate(rows):
        vector = [0.0] * dimension
        vector[index % dimension] = 1.0
        matrix.append(vector)
    return {"embedding_dim": dimension, "matrix": matrix, "data": rows}


def _graphml_payload() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="role" attr.type="string"/>
  <key id="d1" for="node" attr.name="weight" attr.type="double"/>
  <key id="d2" for="node" attr.name="source_id" attr.type="string"/>
  <key id="d3" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d4" for="node" attr.name="description" attr.type="string"/>
  <key id="d5" for="edge" attr.name="role" attr.type="string"/>
  <key id="d6" for="edge" attr.name="weight" attr.type="double"/>
  <key id="d7" for="edge" attr.name="source_id" attr.type="string"/>
  <graph id="G" edgedefault="undirected">
    <node id="urban farms">
      <data key="d0">entity</data><data key="d1">1.0</data><data key="d2">chunk-1</data><data key="d3">concept</data><data key="d4">Urban farms</data>
    </node>
    <node id="community support">
      <data key="d0">entity</data><data key="d1">1.0</data><data key="d2">chunk-1</data><data key="d3">concept</data><data key="d4">Community support</data>
    </node>
    <node id="urban farms build community support">
      <data key="d0">hyperedge</data><data key="d1">1.0</data><data key="d2">chunk-1</data><data key="d4">Urban farms build community support through outreach and soil safety practices.</data>
    </node>
    <edge source="urban farms" target="urban farms build community support"><data key="d5">link</data><data key="d6">1.0</data><data key="d7">chunk-1</data></edge>
    <edge source="community support" target="urban farms build community support"><data key="d5">link</data><data key="d6">1.0</data><data key="d7">chunk-1</data></edge>
  </graph>
</graphml>
"""
