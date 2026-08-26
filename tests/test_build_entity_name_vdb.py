from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from construct.build_entity_name_vdb import (
    EntityNameVDBConfig,
    build_entity_name_vdb,
    iter_graphml_entity_names,
)


GRAPHML_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="role" attr.type="string" />
  <key id="d1" for="node" attr.name="description" attr.type="string" />
  <graph id="G" edgedefault="undirected">
    <node id="&quot;ALPHA ENTITY&quot;">
      <data key="d0">entity</data>
      <data key="d1">Description that must not be embedded.</data>
    </node>
    <node id="&lt;hyperedge&gt;&quot;Alpha is linked to beta.&quot;">
      <data key="d0">hyperedge</data>
    </node>
    <node id="&quot;BÉTA ENTITY&quot;">
      <data key="d0">entity</data>
      <data key="d1">Another description that must not be embedded.</data>
    </node>
    <edge source="&quot;ALPHA ENTITY&quot;" target="&lt;hyperedge&gt;&quot;Alpha is linked to beta.&quot;" />
  </graph>
</graphml>
"""


class FakeEmbedding:
    embedding_dim = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray(
            [[float(len(text)), float(sum(ord(char) for char in text) % 97 + 1), 1.0] for text in texts],
            dtype=np.float32,
        )


class FailOnSecondEmbedding(FakeEmbedding):
    async def __call__(self, texts: list[str]) -> np.ndarray:
        if self.calls:
            raise RuntimeError("simulated embedding interruption")
        return await super().__call__(texts)


class BuildEntityNameVDBTest(unittest.TestCase):
    def test_iter_graphml_entity_names_returns_only_entity_node_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            graphml_path = Path(temp_dir) / "graph.graphml"
            graphml_path.write_text(GRAPHML_FIXTURE, encoding="utf-8")

            names = list(iter_graphml_entity_names(graphml_path))

        self.assertEqual(names, ['"ALPHA ENTITY"', '"BÉTA ENTITY"'])

    def test_build_writes_name_only_vectors_and_resumes_existing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            graphml_path = dataset_dir / "graph_chunk_entity_relation.graphml"
            graphml_path.write_text(GRAPHML_FIXTURE, encoding="utf-8")
            embedding = FakeEmbedding()
            config = EntityNameVDBConfig(
                dataset_dir=dataset_dir,
                embedding_batch_size=2,
                upsert_batch_size=2,
            )

            first = build_entity_name_vdb(config, embedding_func=embedding)
            output_path = dataset_dir / "vdb_entity_names.json"
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse((dataset_dir / ".entity_name_vdb_build").exists())
            calls_after_first_build = list(embedding.calls)
            second = build_entity_name_vdb(config, embedding_func=embedding)

        self.assertEqual(first["entity_count"], 2)
        self.assertEqual(first["embedded_count"], 2)
        self.assertEqual(first["existing_count"], 0)
        self.assertEqual(first["stored_count"], 2)
        self.assertEqual(payload["embedding_dim"], 3)
        self.assertEqual(
            [row["entity_name"] for row in payload["data"]],
            ['"ALPHA ENTITY"', '"BÉTA ENTITY"'],
        )
        self.assertNotIn("content", payload["data"][0])
        self.assertNotIn("description", payload["data"][0])
        self.assertEqual(calls_after_first_build, [['"ALPHA ENTITY"', '"BÉTA ENTITY"']])
        self.assertEqual(len(payload["data"]), 2)
        self.assertEqual(second["embedded_count"], 0)
        self.assertEqual(second["existing_count"], 2)
        self.assertEqual(embedding.calls, calls_after_first_build)

    def test_interrupted_build_leaves_resumable_stage_not_partial_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            graphml_path = dataset_dir / "graph_chunk_entity_relation.graphml"
            graphml_path.write_text(GRAPHML_FIXTURE, encoding="utf-8")
            config = EntityNameVDBConfig(
                dataset_dir=dataset_dir,
                embedding_batch_size=1,
                upsert_batch_size=1,
            )

            with self.assertRaisesRegex(RuntimeError, "simulated embedding interruption"):
                build_entity_name_vdb(config, embedding_func=FailOnSecondEmbedding())

            final_path = dataset_dir / "vdb_entity_names.json"
            staging_path = dataset_dir / ".entity_name_vdb_build" / "vdb_entity_names.json"
            self.assertFalse(final_path.exists())
            self.assertTrue(staging_path.exists())

            resumed = build_entity_name_vdb(config, embedding_func=FakeEmbedding())

            self.assertTrue(final_path.exists())
            self.assertFalse(staging_path.exists())
            self.assertEqual(resumed["existing_count"], 1)
            self.assertEqual(resumed["embedded_count"], 1)


if __name__ == "__main__":
    unittest.main()
