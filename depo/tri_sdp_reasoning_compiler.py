from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Iterable

from models import HanLPSDPResult

# ============================================================
# 1. 节点分类与语义图过滤
# ============================================================

# 显式实体占位符；实体是每条推理路径的起点。
ENTITY_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
# 数字、日期和百分数被标为 constraint，例如 1998、3/5、50%。
NUMERIC_RE = re.compile(r"^[+-]?(?:\d[\d,]*(?:\.\d+)?|\d{1,4}(?:[-/]\d{1,2}){1,2})%?$")

# 下列词在 classify_node() 中归为 function。
DETERMINERS = {"a", "an", "the"}
WH_WORDS = {"what", "which", "who", "whom", "whose", "where", "when"}
# 疑问词虽然是 function，但允许作为语义路径的终点。
WH_ANCHORS = WH_WORDS | {"why", "how"}
RELATIVE_PRONOUNS = {"that"}
LIGHT_VERBS = {
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
}
# 介词既是 function，也是介词收缩增强的中心节点。
PREPOSITIONS = {
    "of", "at", "in", "on", "for", "by", "to", "from", "with", "about",
    "as", "into", "onto", "over", "under", "after", "before", "during",
    "through",
}
# 范围/并列词不进入语义图；and、or 也用于识别并列实体组。
SCOPE_WORDS = {"and", "or", "among", "between", "than"}
SEMANTIC_SCOPE_WORDS = SCOPE_WORDS | {"both", "either", "neither"}
FUNCTION_WORDS = (
    DETERMINERS
    | WH_WORDS
    | RELATIVE_PRONOUNS
    | LIGHT_VERBS
    | PREPOSITIONS
    | SCOPE_WORDS
)
# 下列词与数值一起归为 constraint。
ORDER_CUES = {
    "first", "earliest", "latest", "last", "older", "oldest", "younger", "youngest",
}
APPROX_CUES = {"approximately", "about", "around", "roughly"}

# ============================================================
# 2. PAS 结构增强
# ============================================================

# 固定 HanLP 将英文所有格拆为 ``'`` 和 ``s`` 两个 token。
POSSESSIVE_MARKERS = {"'", "s"}
POSSESSIVE_OWNER_RELATIONS = {"poss_ARG2"}
POSSESSIVE_POSSESSED_RELATIONS = {"poss_ARG1", "adj_ARG1", "noun_ARG1", "modifier"}

# ============================================================
# 3. 边代价（论文公式 2）
#    Core / Augmented Core = 1，Structural = 2，General = 3，Invalid = +∞
# ============================================================

# 原始 PAS 核心论元边：cost = 1。
CORE_RELATION_PREFIXES = (
    "verb_ARG", "noun_ARG", "adj_ARG", "act_ARG", "pat_ARG", "eff_ARG",
)
CORE_RELATIONS = {"loc", "twhen"}
# 原始 PAS 结构边：cost = 2。
STRUCTURAL_RELATION_PREFIXES = ("conj_ARG", "relative_ARG", "comp_")
STRUCTURAL_RELATIONS = {"comparison", "modifier"}
# 无语义功能连接：不加入语义图，等价于 cost = +∞。
INVALID_RELATION_MARKERS = (
    "prep_ARG", "bv", "det", "aux", "root", "punct", "case", "cop",
    "quote", "paren", "bracket",
)
# 介词、所有格收缩后得到的增强核心边：cost = 1。
AUGMENTED_CORE_RULES = {
    "pas_preposition_contraction", "pas_possessive_contraction",
}
# 并列实体共享句法连接的结构边：cost = 2。
STRUCTURAL_RULES = {"pas_coordination_candidate_attachment"}


@dataclass
class TokenReasoningNode:
    id: str
    text: str
    kind: str


@dataclass
class TokenReasoningEdge:
    relations: set[str] = field(default_factory=set)
    #记录何种方式增强
    rules: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RawPasEdge:
    head_id: str
    relation: str
    dep_id: str


@dataclass
#一条实体推理路径
class TokenReasoningPath:
    nodes: list[str]


@dataclass
#记录多个path
class TokenReasoningStructureResult:
    paths: list[TokenReasoningPath]


@dataclass
#记录四个状态
class _WorkingState:
    nodes: dict[str, TokenReasoningNode]
    #原始PAS边
    raw_edges: list[RawPasEdge]
    #三类增强后的边
    edges: dict[tuple[str, str], TokenReasoningEdge]
    syntax_heads: dict[str, int]


def compile_token_reasoning_structure(
    hanlp_sdp_result: HanLPSDPResult,
    explicit_entities: list[str],
) -> TokenReasoningStructureResult:
    """Run Step 4 and return the selected token paths for Step 5."""
    #建立初始PAS图，将句法语义关系转化成图结构
    state = build_evidence_graph(hanlp_sdp_result)
    #获得显示实体id
    entity_ids = _resolve_explicit_entity_ids(state.nodes, explicit_entities)
    #介词收缩增强
    add_pas_preposition_contraction_edges(state)
    #所有格搜索增强
    add_pas_possessive_contraction_edges(state)
    #并列关系增强
    add_pas_coordination_candidate_attachment_edges(state, entity_ids)
    #生成最终的带权无向图
    graph = _semantic_graph(state.nodes, state.edges)
    return TokenReasoningStructureResult(
        paths=_select_entity_best_paths(state.nodes, graph, entity_ids)
    )


def classify_node(text: str, index: int) -> str:
    """Classify one PAS token as entity, content, constraint, or function."""

    token = text.strip()
    lower = token.lower()
    if index == 0 or lower == "root" or not token or _is_punctuation(token):
        return "function"
    if ENTITY_RE.fullmatch(token):
        return "entity"
    if NUMERIC_RE.fullmatch(token) or lower in ORDER_CUES | APPROX_CUES:
        return "constraint"
    return "function" if lower in FUNCTION_WORDS else "content"


def build_evidence_graph(hanlp_sdp_result: HanLPSDPResult) -> _WorkingState:
    """Construct the undirected PAS evidence graph before augmentation."""
    #HanLP token列表中的节点封装成TokenReasoningNode形式
    nodes = _build_token_nodes(hanlp_sdp_result)
    #构造原始PAS边
    raw_edges = [
        RawPasEdge(
            head_id=str(edge.head_idx),
            relation=edge.relation,
            dep_id=str(edge.dep_idx),
        )
        for edge in hanlp_sdp_result.edges
    ]
    edges: dict[tuple[str, str], TokenReasoningEdge] = {}
    #遍历所有PAS边加入图
    for edge in raw_edges:
        _add_edge(edges, edge.head_id, edge.dep_id, relation=edge.relation)
    #标记所有格节点，将"'s"标记成function
    _mark_possessive_markers(nodes, raw_edges)
    #返回状态
    return _WorkingState(
        nodes=nodes,
        raw_edges=raw_edges,
        edges=edges,
        #把key为0的root节点删去
        syntax_heads={
            key: value
            for key, value in hanlp_sdp_result.syntax_heads.items()
            if key != "0"
        },
    )

#
def add_pas_preposition_contraction_edges(state: _WorkingState) -> None:
    """Contract preposition ARG1--preposition--ARG2 into an augmented core edge."""
    #为了保持运行结果一致，先排序再遍历，否则同一个问题运行结果不一致
    for preposition in sorted(state.nodes.values(), key=lambda node: int(node.id)):
        #判断当前节点是不是介词
        if preposition.text.lower() not in PREPOSITIONS:
            continue
        #获取介词的两侧参数
        arg1_ids, arg2_ids = _preposition_arguments(preposition.id, state)
        for arg1_id in arg1_ids:
            if not _is_salient(state.nodes[arg1_id]):
                continue
            for arg2_id in arg2_ids:
                if arg1_id == arg2_id or not _is_salient(state.nodes[arg2_id]):
                    continue
                #只允许entity、content参与收缩
                _add_edge(
                    state.edges,
                    arg1_id,
                    arg2_id,
                    rule="pas_preposition_contraction",
                )


def add_pas_possessive_contraction_edges(state: _WorkingState) -> None:
    """Contract HanLP's split possessive path: owner -- ' s -- possessed."""
    #记录已处理的 ' 和 s 节点 id
    contracted_markers: set[str] = set()
    #按照原始token顺序排序
    tokens = sorted(state.nodes.values(), key=lambda node: int(node.id))

    for marker, suffix in zip(tokens, tokens[1:]):
        if marker.text != "'" or suffix.text.lower() != "s":
            continue
        #分别从原始边中取出’和s链接的两端端点
        owners_a, possessed_a = _possessive_arguments(marker.id, state)
        owners_b, possessed_b = _possessive_arguments(suffix.id, state)
        #合并去重
        for owner_id in set(owners_a) | set(owners_b):
            for possessed_id in set(possessed_a) | set(possessed_b):
                #合并去重后直接添加语义边
                _add_edge(
                    state.edges,
                    owner_id,
                    possessed_id,
                    rule="pas_possessive_contraction",
                )
                #记录 ' 和 s，表示它们已经被直接边替代。
                contracted_markers.update({marker.id, suffix.id})

    for key in list(state.edges):
        if contracted_markers.intersection(key):
            del state.edges[key]


def add_pas_coordination_candidate_attachment_edges(
    state: _WorkingState,
    entity_ids: list[str],
) -> None:
    """Attach coordinated explicit entities to their shared syntactic head."""
    #表示至少两个实体才能存在并列
    if len(entity_ids) < 2:
        return
    #寻找并列实体组合：实体 + and/or + 实体
    for connector_id, member_ids in _coordination_groups(state, set(entity_ids)):
        #寻找共同链接的中心词
        attachment_id = _shared_syntactic_attachment(state, connector_id, member_ids)
        if attachment_id is None:
            continue
        #给每个并列实体添加连接
        for member_id in member_ids:
            if _edge_key(member_id, attachment_id) not in state.edges:
                _add_edge(
                    state.edges,
                    member_id,
                    attachment_id,
                    rule="pas_coordination_candidate_attachment",
                )


def _semantic_graph(
    nodes: dict[str, TokenReasoningNode],
    edges: dict[tuple[str, str], TokenReasoningEdge],
) -> dict[str, list[tuple[str, int]]]:
    """Build the paper's weighted undirected semantic graph G_q^u."""

    graph: dict[str, list[tuple[str, int]]] = {}
    for (source_id, target_id), edge in edges.items():
        if not _semantic_node_allowed(nodes[source_id]) or not _semantic_node_allowed(
            nodes[target_id]
        ):
            continue
        if _is_pure_coordination(edge):
            continue
        cost = _edge_cost(edge)
        if cost is None:
            continue
        graph.setdefault(source_id, []).append((target_id, cost))
        graph.setdefault(target_id, []).append((source_id, cost))

    for neighbors in graph.values():
        neighbors.sort(key=lambda item: (item[1], int(item[0])))
    return graph


def _select_entity_best_paths(
    nodes: dict[str, TokenReasoningNode],
    graph: dict[str, list[tuple[str, int]]],
    entity_ids: list[str],
) -> list[TokenReasoningPath]:
    #找到语义节点
    semantic_ids = {
        node_id
        for node_id, neighbors in graph.items()
        #两个条件，至少有一个相邻节点且是语义节点
        if neighbors and _is_semantic_node(nodes[node_id])
    }
    #找边界节点
    boundary_ids = _ordered_ids(
        (node_id for node_id in semantic_ids if len(graph[node_id]) == 1), nodes
    )
    selected: list[TokenReasoningPath] = []

    for entity_id in _ordered_ids(entity_ids, nodes):
        candidates = [
            path
            for boundary_id in boundary_ids
            if boundary_id != entity_id
            for path in [
                _shortest_path(
                    graph,
                    nodes,
                    entity_id,
                    boundary_id,
                    blocked_ids=set(entity_ids) - {entity_id},
                )
            ]
            if path
        ]
        if not candidates:
            continue
        #统计所有候选路径中出现的语义节点
        branch_ids = {
            node_id
            for path in candidates
            for node_id in path
            if node_id != entity_id and node_id in semantic_ids
        }
        #选择sp分数最高的路径
        best_path = max(
            candidates,
            key=lambda path: _sp_score(entity_id, path, branch_ids),
        )
        selected.append(_path_from_ids(nodes, best_path))
    return selected

#Djistra代码实现
def _shortest_path(
    graph: dict[str, list[tuple[str, int]]],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
    *,
    blocked_ids: set[str],
) -> list[str]:
    """Dijkstra with deterministic ties: cost, edge count, then token order."""

    if source_id not in graph or target_id not in graph:
        return []

    start_key = _path_key([source_id])
    heap: list[tuple[int, int, tuple[int, ...], str, list[str]]] = [
        (0, 0, start_key, source_id, [source_id])
    ]
    best: dict[str, tuple[int, int, tuple[int, ...]]] = {
        source_id: (0, 0, start_key)
    }
    while heap:
        cost, edge_count, path_key, node_id, path = heapq.heappop(heap)
        if best[node_id] != (cost, edge_count, path_key):
            continue
        if node_id == target_id:
            return path
        for neighbor_id, edge_cost in graph[node_id]:
            if neighbor_id in path or neighbor_id in blocked_ids:
                continue
            next_path = [*path, neighbor_id]
            candidate = (
                cost + edge_cost,
                edge_count + 1,
                _path_key(next_path),
            )
            if neighbor_id not in best or candidate < best[neighbor_id]:
                best[neighbor_id] = candidate
                heapq.heappush(heap, (*candidate, neighbor_id, next_path))
    return []


def _sp_score(entity_id: str, path: list[str], branch_ids: set[str]) -> float:
    """Compute 2|C_m(P)| / (|S_m| + |V(P) minus {m}|)."""
    #去除实体节点后的路径节点集合
    path_without_entity = [node_id for node_id in path if node_id != entity_id]
    #branch_ids所有路径涉及的语义节点数量
    denominator = len(branch_ids) + len(path_without_entity)
    return (
        2 * sum(node_id in branch_ids for node_id in path_without_entity) / denominator
        if denominator
        else 0.0
    )


def _preposition_arguments(
    preposition_id: str,
    state: _WorkingState,
) -> tuple[list[str], list[str]]:
    arg1 = {
        edge.dep_id
        for edge in state.raw_edges
        if edge.head_id == preposition_id
        and edge.dep_id in state.nodes
        and edge.relation == "prep_ARG1"
    }
    arg2 = {
        edge.dep_id
        for edge in state.raw_edges
        if edge.head_id == preposition_id
        and edge.dep_id in state.nodes
        and edge.relation == "prep_ARG2"
    }
    return _ordered_ids(arg1, state.nodes), _ordered_ids(arg2, state.nodes)


def _possessive_arguments(
    marker_id: str,
    state: _WorkingState,
) -> tuple[list[str], list[str]]:
    owners: set[str] = set()
    possessed: set[str] = set()
    for edge in state.raw_edges:
        if marker_id not in {edge.head_id, edge.dep_id}:
            continue
        related_id = edge.dep_id if edge.head_id == marker_id else edge.head_id
        if related_id not in state.nodes:
            continue
        if edge.relation in POSSESSIVE_OWNER_RELATIONS:
            owners.add(related_id)
        elif edge.relation in POSSESSIVE_POSSESSED_RELATIONS:
            possessed.add(related_id)
    return _ordered_ids(owners, state.nodes), _ordered_ids(possessed, state.nodes)


def _coordination_groups(
    state: _WorkingState,
    entity_ids: set[str],
) -> list[tuple[str, list[str]]]:
    groups: dict[str, set[str]] = {}
    for edge in state.raw_edges:
        if not _is_coordination_group_relation(edge.relation):
            continue
        if (
            edge.head_id in state.nodes
            and _is_connector(state.nodes[edge.head_id])
            and edge.dep_id in entity_ids
        ):
            groups.setdefault(edge.head_id, set()).add(edge.dep_id)
        elif (
            edge.dep_id in state.nodes
            and _is_connector(state.nodes[edge.dep_id])
            and edge.head_id in entity_ids
        ):
            groups.setdefault(edge.dep_id, set()).add(edge.head_id)

    return sorted(
        (
            (connector_id, _ordered_ids(member_ids, state.nodes))
            for connector_id, member_ids in groups.items()
            if len(member_ids) >= 2
        ),
        key=lambda group: int(group[0]),
    )

#判断一组并列实体是否共享同一个外部语法中心，如果共享就返回那个语法中心节点
def _shared_syntactic_attachment(
    state: _WorkingState,
    connector_id: str,
    member_ids: list[str],
) -> str | None:
    
    group_ids = {*member_ids, connector_id}
    #对于每一个并列实体分别找它们共享的外部head
    attachments = {
        _external_syntax_head(member_id, group_ids, state.syntax_heads)
        for member_id in member_ids
    }
    if len(attachments) != 1:
        return None
    #获取唯一head，next取下一个元素，iter迭代器
    attachment_id = next(iter(attachments))
    return attachment_id 

#从一个并列实体出发，沿着依存句法 head 链向上寻找第一个不属于并列结构内部的外部语义中心节点。
def _external_syntax_head(
    member_id: str,
    group_ids: set[str],
    syntax_heads: dict[str, int],
) -> str | None:
    current = member_id
    seen: set[str] = set()
    while current in syntax_heads and current not in seen:
        seen.add(current)
        #获取语法头节点
        head_id = str(syntax_heads[current])
        #如果没在组合中
        if head_id not in group_ids:
            return head_id
        current = head_id
    return None


def _is_coordination_group_relation(relation: str) -> bool:
    return "coord" in relation or relation in {"conj_member", "disj_member"}


def _is_pure_coordination(edge: TokenReasoningEdge) -> bool:
    markers = ("coord", "conj_member", "disj_member", "_and_c", "_or_c")
    return bool(edge.relations) and all(
        any(marker in relation for marker in markers) for relation in edge.relations
    )


def _is_connector(node: TokenReasoningNode) -> bool:
    return node.text.lower() in {"and", "or"}


def _semantic_node_allowed(node: TokenReasoningNode) -> bool:
    lower = node.text.lower()
    if node.id == "0" or lower == "root" or _is_punctuation(node.text):
        return False
    if lower in SEMANTIC_SCOPE_WORDS | DETERMINERS | PREPOSITIONS | LIGHT_VERBS | RELATIVE_PRONOUNS:
        return False
    return node.kind != "function" or lower in WH_ANCHORS


def _is_semantic_node(node: TokenReasoningNode) -> bool:
    if node.kind == "entity" or ENTITY_RE.fullmatch(node.text):
        return False
    return node.text.lower() in WH_ANCHORS or (
        _semantic_node_allowed(node) and node.kind in {"content", "constraint"}
    )


def _edge_cost(edge: TokenReasoningEdge) -> int | None:
    """Return the Equation (2) cost; ``None`` represents an invalid edge."""

    if edge.rules & AUGMENTED_CORE_RULES:
        return 1
    if edge.rules & STRUCTURAL_RULES:
        return 2
    costs = [_pas_relation_cost(relation) for relation in edge.relations]
    return min((cost for cost in costs if cost is not None), default=None)


def _pas_relation_cost(relation: str) -> int | None:
    if any(marker in relation for marker in INVALID_RELATION_MARKERS):
        return None
    if relation in CORE_RELATIONS or relation.startswith(CORE_RELATION_PREFIXES):
        return 1
    if (
        relation in STRUCTURAL_RELATIONS
        or relation.startswith(STRUCTURAL_RELATION_PREFIXES)
        or relation.endswith("_mod")
    ):
        return 2
    return 3

#判断节点是否是重要节点
def _is_salient(node: TokenReasoningNode) -> bool:
    return node.kind in {"entity", "content"}


def _build_token_nodes(result: HanLPSDPResult) -> dict[str, TokenReasoningNode]:
    nodes = {"0": TokenReasoningNode("0", "ROOT", "function")}
    for index, token in enumerate(result.tokens, start=1):
        text = str(token)
        nodes[str(index)] = TokenReasoningNode(
            id=str(index), text=text, kind=classify_node(text, index)
        )
    return nodes


def _mark_possessive_markers(
    nodes: dict[str, TokenReasoningNode],
    raw_edges: list[RawPasEdge],
) -> None:
    #先创建一个空state
    state = _WorkingState(nodes, raw_edges, {}, {})
    for node in nodes.values():
        #遍历每个节点，找是否存在's
        if node.text.lower() not in POSSESSIVE_MARKERS:
            continue
        #找到这个所有格的对象
        owners, possessed = _possessive_arguments(node.id, state)
        if owners or possessed:
            node.kind = "function"


def _resolve_explicit_entity_ids(
    nodes: dict[str, TokenReasoningNode],
    explicit_entities: list[str],
) -> list[str]:
    order = {entity: index for index, entity in enumerate(explicit_entities)}
    return sorted(
        {
            node.id
            for node in nodes.values()
            if node.text in order and ENTITY_RE.fullmatch(node.text)
        },
        key=lambda node_id: (order[nodes[node_id].text], int(node_id)),
    )

#向当前的推理图中加入一条无向边；如果这条边已经存在，就把新的 relation 或 rule 信息合并进去，而不是重复创建边。
def _add_edge(
    edges: dict[tuple[str, str], TokenReasoningEdge],
    source_id: str,
    target_id: str,
    *,
    relation: str | None = None,
    rule: str | None = None,
) -> None:
    #统一边的表示，对于A-B,B-A两条边视为一条边
    key = _edge_key(source_id, target_id)
    #如果这个 key 已经存在，就返回它对应的 value；如果 key 不存在，就先把默认值放进去，再返回这个默认值。
    edge = edges.setdefault(key, TokenReasoningEdge())
    #添加关系
    if relation:
        edge.relations.add(relation)
    #添加增强规则pas_preposition_contraction等等
    if rule:
        edge.rules.add(rule)

#判断是否存在重复边
def _edge_key(source_id: str, target_id: str) -> tuple[str, str]:
    source, target = str(source_id), str(target_id)
    return (source, target) if int(source) <= int(target) else (target, source)

#按 token 在句子中的位置，对节点 id 排序。
def _ordered_ids(
    node_ids: Iterable[str], nodes: dict[str, TokenReasoningNode]
) -> list[str]:
    return sorted(
        {str(node_id) for node_id in node_ids if str(node_id) in nodes},
        key=int,
    )


def _path_key(path: list[str]) -> tuple[int, ...]:
    return tuple(int(node_id) for node_id in path)


def _path_from_ids(
    nodes: dict[str, TokenReasoningNode], node_ids: list[str]
) -> TokenReasoningPath:
    return TokenReasoningPath([nodes[node_id].text for node_id in node_ids])


def _is_punctuation(text: str) -> bool:
    return bool(text) and all(not character.isalnum() for character in text)
