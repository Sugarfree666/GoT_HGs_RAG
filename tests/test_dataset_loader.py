from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from hyper_branch.config import DatasetConfig
from hyper_branch.data.loaders import HypergraphDatasetLoader


class HypergraphDatasetLoaderTest(unittest.TestCase):
    def test_missing_entity_name_vdb_does_not_fall_back_to_entity_description_vdb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "graph_chunk_entity_relation.graphml").write_text("<graphml />", encoding="utf-8")
            (root / "kv_store_text_chunks.json").write_text(json.dumps({}), encoding="utf-8")
            (root / "kv_store_full_docs.json").write_text(json.dumps({}), encoding="utf-8")
            (root / "vdb_entities.json").write_text(json.dumps({}), encoding="utf-8")

            loader = HypergraphDatasetLoader(DatasetConfig(root=root), logging.getLogger(__name__))

            with self.assertRaisesRegex(FileNotFoundError, "vdb_entity_names.json"):
                loader.load()


if __name__ == "__main__":
    unittest.main()
