"""In-memory knowledge hypergraph and its vector indexes."""

from __future__ import annotations

import base64
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

CandidatePool = dict[str, set[str]]
EntityLinks = dict[str, list[str]]

_APOSTROPHE_VARIANTS = str.maketrans("’‘`´", "''''")
_CHUNK_ENTITY_EXCLUDED_TYPES = {
    "CATEGORY", "CONCEPT", "CONDITION", "RELATION", "ROLE", "TITLE", "TYPE"
}


class Embedder(Protocol):
    def embed_text(self, text: str) -> np.ndarray: ...


class HypergraphDatabase:
    """Load one dataset snapshot and expose the operations retrieval actually needs."""

    def __init__(self, root: Path) -> None:
        #加载文本块库，读取文本chunk数据,将json文本转成字典
        self.chunks: dict[str, dict[str, Any]] = json.loads(
            (root / "kv_store_text_chunks.json").read_text(encoding="utf-8")
        )
        #加载实体名称向量库
        self.entity_vectors = _VectorIndex.load(
            root / "vdb_entity_names.json",
            label_field="entity_name",
        )
        #加载超边向量库
        self.hyperedge_vectors = _VectorIndex.load(
            root / "vdb_hyperedges.json",
            label_field="hyperedge_name",
        )
        self.chunk_vectors = _VectorIndex.load(
            root / "vdb_chunks.json",
            label_field="__id__",
        )
        #用来保存节点类型
        self.roles: dict[str, str] = {}
        #保存超边来源chunk
        self.sources: dict[str, list[str]] = {}
        #保存实体到关联超边
        self.entity_to_hyperedges: dict[str, list[str]] = defaultdict(list)
        #保存超边到关联的实体
        self.hyperedge_to_entities: dict[str, list[str]] = defaultdict(list)
        #保存chunk包含哪些实体
        self.chunk_to_entities: dict[str, list[str]] = defaultdict(list)
        self.chunk_to_hyperedges: dict[str, list[str]] = defaultdict(list)
        #加载 GraphML 超图
        self._load_graph(root / "graph_chunk_entity_relation.graphml")
        #以规范化后的实体名为键，记录对应的图节点 ID，方便精确匹配实体
        self._entity_ids_by_name: dict[str, list[str]] = defaultdict(list)
        for node_id, role in self.roles.items():
            if role == "entity":
                self._entity_ids_by_name[_lookup_key(node_id)].append(node_id)

    def link_entity_ids(self, mentions: list[str], embedder: Embedder) -> EntityLinks:
        return {
            mention: self._link_entities(mention, embedder)
            for mention in dict.fromkeys(mention.strip() for mention in mentions if mention.strip())
        }

    def candidate_pool(
        self,
        mentions: list[str],
        embedder: Embedder,
        *,
        entity_ids: EntityLinks | None = None,
    ) -> CandidatePool:
        """Link anchors and collect their one-hop and source-enriched two-hop hyperedges."""
        linked = entity_ids if entity_ids is not None else self.link_entity_ids(mentions, embedder)
        anchor_ids = list(
            dict.fromkeys(entity_id for ids in linked.values() for entity_id in ids)
        )
        candidates: CandidatePool = {}

        for anchor_id in anchor_ids:
            first_hops = list(dict.fromkeys(self.entity_to_hyperedges.get(anchor_id, [])))
            for hyperedge_id in first_hops:
                candidates.setdefault(hyperedge_id, set())

            for first_hop_id in first_hops:
                bridge_entities = list(self.hyperedge_to_entities.get(first_hop_id, []))
                for chunk_id in self.sources.get(first_hop_id, []):
                    bridge_entities.extend(self.chunk_to_entities.get(chunk_id, []))

                for entity_id in dict.fromkeys(bridge_entities):
                    if entity_id == anchor_id:
                        continue
                    for second_hop_id in self.entity_to_hyperedges.get(entity_id, []):
                        if second_hop_id not in first_hops:
                            candidates.setdefault(second_hop_id, set()).add(first_hop_id)
        return candidates

    def original_question_candidate_pool(
        self,
        question: str,
        embedder: Embedder,
        *,
        chunk_top_k: int,
    ) -> CandidatePool:
        """Collect every hyperedge sourced from the question's top chunks."""
        if chunk_top_k < 1:
            raise ValueError("chunk_top_k must be at least 1")

        top_chunks = self.chunk_vectors.query(embedder.embed_text(question), chunk_top_k)
        candidates: CandidatePool = {}
        for chunk_id, _score in top_chunks:
            for hyperedge_id in self.chunk_to_hyperedges.get(chunk_id, []):
                candidates.setdefault(hyperedge_id, set())
        return candidates

    def rank(
        self,
        question: str,
        candidates: CandidatePool,
        embedder: Embedder,
    ) -> list[dict[str, Any]]:
        """Rank only the supplied candidate hyperedges and attach their source chunks."""
        #将当前问题编码为向量
        question_vector = embedder.embed_text(question)

        scores = self.hyperedge_vectors.similarities(question_vector, list(candidates))
        ranked_ids = sorted(candidates, key=lambda item: (-scores.get(item, 0.0), item))

        return [
            {
                "id": hyperedge_id,
                "text": _display_text(hyperedge_id),
                "chunks": [
                    (chunk_id, str(self.chunks.get(chunk_id, {}).get("content", "")))
                    for chunk_id in dict.fromkeys(self.sources.get(hyperedge_id, []))
                ],
                "first_hop_texts": [
                    _display_text(first_hop_id)
                    for first_hop_id in sorted(candidates[hyperedge_id])
                ],
            }
            for hyperedge_id in ranked_ids
        ]

    def _link_entities(self, mention: str, embedder: Embedder) -> list[str]:
        exact_ids = self._entity_ids_by_name.get(_lookup_key(mention), [])
        if exact_ids:
            return exact_ids
        label = _display_text(mention)
        base_label = re.sub(r"(?:\s*\([^()]*\)\s*)+$", "", label).strip()
        base_ids = self._entity_ids_by_name.get(_lookup_key(base_label), [])
        if base_label != label and len(base_ids) == 1:
            return base_ids
        label, score = self.entity_vectors.query(embedder.embed_text(mention), 1)[0]
        return [label] if score >= 0.5 else []

    #用 XML 解析器读取 GraphML，然后把 XML 描述的超图结构转换成 Python 字典索引。
    def _load_graph(self, path: Path) -> None:
        #GraphML 是一种基于 XML 的图数据格式。XML结构化数据存储格式，namespace是命名空间，区分不同含义的标签
        namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
        #ET是xml解析器，读取XML文件，获取ElementTree对象，获取根节点也就是graphml
        root = ET.parse(path).getroot()
        #id对应的role字典
        key_names = {
            #key的id属性：优先获取attr.name
            key.attrib["id"]: key.attrib.get("attr.name", key.attrib["id"])
            #找到所有key标签
            for key in root.findall("g:key", namespace)
        }
        #在 XML 根节点 root 下找到 GraphML 中的 <graph> 节点，并保存为 Python 的 Element 对象。
        graph = root.find("g:graph", namespace)
        #在<graph></graph>中寻找所有的<node>
        for element in graph.findall("g:node", namespace):
            node_id = element.attrib["id"]
            #获取当前<node>下的attr.name对应的值
            data = _graphml_data(element, key_names, namespace)
            role = data.get("role", "")
            #获取当前<node>的来源chunk id
            sources = [
                part.strip()
                for part in data.get("source_id", "").split("<SEP>")
                if part.strip()
            ]
            #保存节点类型和chunk来源
            self.roles[node_id] = role
            self.sources[node_id] = sources
            if role == "hyperedge":
                for chunk_id in sources:
                    self.chunk_to_hyperedges[chunk_id].append(node_id)
            #如果是实体节点，记录实体和来源chunk的映射
            if role == "entity" and (
                _display_text(data.get("entity_type", "")).upper()
                not in _CHUNK_ENTITY_EXCLUDED_TYPES
            ):
                for chunk_id in sources:
                    self.chunk_to_entities[chunk_id].append(node_id)
        ##在<graph></graph>中寻找所有的<edge>
        for element in graph.findall("g:edge", namespace):
            #source和target表示当前边连接的两个端点id
            source, target = element.attrib["source"], element.attrib["target"]
            #确定source和target的role
            if self.roles[source] == "entity":
                entity_id, hyperedge_id = source, target
            else:
                entity_id, hyperedge_id = target, source
            #存入实体超边连接
            self.entity_to_hyperedges[entity_id].append(hyperedge_id)
            #存入超边实体连接
            self.hyperedge_to_entities[hyperedge_id].append(entity_id)


class _VectorIndex:
    #rows表示向量库中的元数据，matrix：所有 embedding 组成的二维 numpy 矩阵。
    def __init__(self, rows: list[dict[str, Any]], matrix: np.ndarray, label_field: str) -> None:
        #计算每一个embedding的L2长度，axis=1表示按行计算,结果保持二维形状，便于后续计算。
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        #将零向量的长度从 0 改成 1.0
        norms[norms == 0] = 1.0
        #将矩阵转成float32，并除以自己的长度，进行归一化
        self.matrix = matrix.astype(np.float32) / norms
        #从向量库的每条元数据中取出名称字段，按原有顺序保存为列表。
        self.labels = [str(row[label_field]) for row in rows]
        #将每个名称规范化后，建立“名称 → 向量行号”的字典。
        self.indices = {
            _lookup_key(label): index
            for index, label in enumerate(self.labels)
        }

    @classmethod
    def load(cls, path: Path, *, label_field: str) -> "_VectorIndex":
        #读取向量库文件
        payload = json.loads(path.read_text(encoding="utf-8"))
        #取出列表[{"__id__": "en-581405..."："entity_name": "\"SUCCESS OF 'BILLY ELLIOT THE MUSICAL'\""}]
        rows = list(payload.get("data", []))
        #dimension=1536，向量维数
        dimension = int(payload.get("embedding_dim", 0))
        #取出 JSON 中保存的全部向量数据。
        encoded = payload.get("matrix")
        #把 "matrix" 解码、转为 float32，(N, 1536)的向量矩阵
        #当前向量库中的 "matrix" 不是数字列表而，而是将所有 float32 向量的二进制字节编码成的 Base64 字符串。
        matrix = (
            np.frombuffer(base64.b64decode(encoded), dtype="<f4").reshape(len(rows), dimension)
        )
        return cls(rows, matrix, label_field)
    #在整个向量库中搜索最相近的 Top-K 项
    def query(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        #归一化输入向量
        query = _unit_vector(vector)
        #计算向量库中每一行向量与查询向量的点积。
        scores = self.matrix @ query
        #
        return [
            (self.labels[int(index)], float(scores[int(index)]))
            #返回按分数从小到大排列的行号，反转取top-k
            for index in np.argsort(scores)[::-1][:top_k]
        ]
    #计算
    def similarities(self, vector: np.ndarray, ids: list[str]) -> dict[str, float]:
        query = _unit_vector(vector)
        #初始化结果字典
        result: dict[str, float] = {}
        #遍历候选超边id
        for item_id in ids:
            #找到对应超边行号
            index = self.indices.get(_lookup_key(item_id))
            if index is not None:
                #取出该超边向量，与问题向量做点积，得到余弦相似度。
                result[item_id] = float(np.dot(self.matrix[index], query))
        return result

#计算当前向量的单位向量
def _unit_vector(vector: np.ndarray) -> np.ndarray:
    #将输入转成float32向量
    value = np.asarray(vector, dtype=np.float32)
    #计算该向量的L2长度
    norm = np.linalg.norm(value)
    return value if norm == 0 else value / norm


def _display_text(text: str) -> str:
    #删除text的首尾空格
    value = text.strip()
    #判断文本是否以超边标记 <hyperedge> 开头。如果是则切掉这个前缀
    if value.startswith("<hyperedge>"):
        value = value[len("<hyperedge>"):]
    #处理这种情况"Paris"
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    #将连续空格统一为一个空格
    return " ".join(value.split())


def _lookup_key(text: str) -> str:
    value = unicodedata.normalize(
        "NFKD", _display_text(text).translate(_APOSTROPHE_VARIANTS)
    )
    return "".join(
        char
        for char in value
        if not unicodedata.combining(char)
        and not unicodedata.category(char).startswith("P")
    ).casefold()


#获取attr.name：value
def _graphml_data(
    element: ET.Element,
    key_names: dict[str, str],
    namespace: dict[str, str],
) -> dict[str, str]:
    return {
        #记录{atrrib.name:text}
        key_names.get(child.attrib["key"], child.attrib["key"]): child.text or ""
        #遍历每一个<data>
        for child in element.findall("g:data", namespace)
    }
