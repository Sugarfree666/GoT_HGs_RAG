from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from ..models import ExtractedSubgraph, GraphEdge, GraphNode, VectorMatch
from ..utils import normalize_label, split_source_ids


class KnowledgeHypergraph:
    def __init__(self, nodes: dict[str, GraphNode], edges: dict[str, GraphEdge]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        self.source_to_nodes: dict[str, list[str]] = defaultdict(list)
        self.source_to_edges: dict[str, list[str]] = defaultdict(list)

        for edge_id, edge in edges.items():
            self.adjacency[edge.source].append(edge_id)
            self.adjacency[edge.target].append(edge_id)
            for source_id in edge.source_ids:
                self.source_to_edges[source_id].append(edge_id)

        for node_id, node in nodes.items():
            for source_id in node.source_ids:
                self.source_to_nodes[source_id].append(node_id)

    @classmethod
    def from_graphml(cls, path: Path) -> "KnowledgeHypergraph":
        namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
        tree = ET.parse(path)
        root = tree.getroot()
        key_names = {
            key.attrib["id"]: key.attrib.get("attr.name", key.attrib["id"])
            for key in root.findall("g:key", namespace)
        }
        graph = root.find("g:graph", namespace)
        if graph is None:
            raise ValueError(f"Could not find GraphML graph element in {path}.")

        nodes: dict[str, GraphNode] = {}
        for node_elem in graph.findall("g:node", namespace):
            node_id = node_elem.attrib["id"]
            data = _collect_graphml_data(node_elem, key_names, namespace)
            nodes[node_id] = GraphNode(
                node_id=node_id,
                role=data.get("role", ""),
                weight=float(data.get("weight", 0.0) or 0.0),
                source_ids=split_source_ids(data.get("source_id", "")),
                entity_type=data.get("entity_type"),
                description=data.get("description"),
            )

        edges: dict[str, GraphEdge] = {}
        for index, edge_elem in enumerate(graph.findall("g:edge", namespace)):
            data = _collect_graphml_data(edge_elem, key_names, namespace)
            edge_id = f"edge-{index}"
            weight = data.get("weight")
            if weight is None:
                weight = data.get("weight_float")
            edges[edge_id] = GraphEdge(
                edge_id=edge_id,
                source=edge_elem.attrib["source"],
                target=edge_elem.attrib["target"],
                role=data.get("role", ""),
                weight=float(weight or 0.0),
                source_ids=split_source_ids(data.get("source_id", "")),
            )
        return cls(nodes=nodes, edges=edges)

    def summarize(self) -> dict[str, Any]:
        role_counts = Counter(node.role for node in self.nodes.values())
        edge_role_counts = Counter(edge.role for edge in self.edges.values())
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_roles": dict(role_counts),
            "edge_roles": dict(edge_role_counts),
        }

    def extract_subgraph(
        self,
        seed_node_ids: list[str],
        hops: int,
        seed_entities: list[VectorMatch],
        seed_hyperedges: list[VectorMatch],
    ) -> ExtractedSubgraph:
        valid_seeds = [node_id for node_id in seed_node_ids if node_id in self.nodes]
        if not valid_seeds:
            return ExtractedSubgraph(
                node_ids=[],
                edge_ids=[],
                seed_entities=seed_entities,
                seed_hyperedges=seed_hyperedges,
                source_chunk_ids=[],
                hop_count=hops,
                node_distances={},
                summary={"node_count": 0, "edge_count": 0},
            )

        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in valid_seeds)
        node_distances: dict[str, int] = {seed: 0 for seed in valid_seeds}

        while queue:
            node_id, depth = queue.popleft()
            if depth >= hops:
                continue
            for edge_id in self.adjacency.get(node_id, []):
                edge = self.edges[edge_id]
                neighbor = edge.target if edge.source == node_id else edge.source
                if neighbor not in node_distances:
                    node_distances[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        node_ids = sorted(node_distances.keys(), key=lambda node_id: (node_distances[node_id], normalize_label(node_id)))
        node_set = set(node_ids)
        edge_ids = sorted(
            [
                edge_id
                for edge_id, edge in self.edges.items()
                if edge.source in node_set and edge.target in node_set
            ]
        )

        source_chunk_ids: set[str] = set()
        for node_id in node_ids:
            source_chunk_ids.update(self.nodes[node_id].source_ids)
        for edge_id in edge_ids:
            source_chunk_ids.update(self.edges[edge_id].source_ids)

        summary = self._build_subgraph_summary(node_ids, edge_ids, node_distances, seed_entities, seed_hyperedges)
        return ExtractedSubgraph(
            node_ids=node_ids,
            edge_ids=edge_ids,
            seed_entities=seed_entities,
            seed_hyperedges=seed_hyperedges,
            source_chunk_ids=sorted(source_chunk_ids),
            hop_count=hops,
            node_distances=node_distances,
            summary=summary,
        )

    def _build_subgraph_summary(
        self,
        node_ids: list[str],
        edge_ids: list[str],
        node_distances: dict[str, int],
        seed_entities: list[VectorMatch],
        seed_hyperedges: list[VectorMatch],
    ) -> dict[str, Any]:
        def sort_key(node_id: str) -> tuple[int, float, str]:
            node = self.nodes[node_id]
            return (node_distances.get(node_id, 999), -node.weight, node.display_label)

        top_entities = [
            self.nodes[node_id].display_label
            for node_id in sorted(
                [node_id for node_id in node_ids if self.nodes[node_id].role == "entity"],
                key=sort_key,
            )[:20]
        ]
        top_hyperedges = [
            self.nodes[node_id].display_label
            for node_id in sorted(
                [node_id for node_id in node_ids if self.nodes[node_id].role == "hyperedge"],
                key=sort_key,
            )[:20]
        ]
        return {
            "node_count": len(node_ids),
            "edge_count": len(edge_ids),
            "top_entities": top_entities,
            "top_hyperedges": top_hyperedges,
            "seed_entities": [match.to_dict() for match in seed_entities],
            "seed_hyperedges": [match.to_dict() for match in seed_hyperedges],
        }


def _collect_graphml_data(element: ET.Element, key_names: dict[str, str], namespace: dict[str, str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for data_elem in element.findall("g:data", namespace):
        key_id = data_elem.attrib["key"]
        attr_name = key_names.get(key_id, key_id)
        value = data_elem.text or ""
        if attr_name == "weight" and "weight" in payload:
            payload["weight_float"] = value
        else:
            payload[attr_name] = value
    return payload
