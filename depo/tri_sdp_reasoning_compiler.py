from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from math import inf, isinf

from models import HanLPSDPEdge, HanLPSDPResult

# ============================================================
# 1. 节点分类与语义图过滤
# ============================================================

# 显式实体占位符；实体是每条推理路径的起点。
ENTITY_RE = re.compile(r"^ENTITY[A-Z0-9]*$")
# 数字、日期和百分数被标为 constraint，例如 1998、3/5、50%。
NUMERIC_RE = re.compile(r"^[+-]?(?:\d[\d,]*(?:\.\d+)?|\d{1,4}(?:[-/]\d{1,2}){1,2})%?$")

# 下列词在 classify_node() 中归为 function。
DETERMINERS = {"a", "an", "the"}
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


@dataclass
#Step4 构图和增强共享的内部状态
class _WorkingState:
    nodes: dict[str, TokenReasoningNode]
    pas_edges: list[HanLPSDPEdge]
    graph: dict[str, dict[str, TokenReasoningEdge]]
    syntax_heads: dict[str, int]


def compile_token_reasoning_structure(
    hanlp_sdp_result: HanLPSDPResult,
    explicit_entities: list[str],
) -> list[list[str]]:
    """执行 Step4，返回供 Step5 使用的实体语义路径。"""
    #建立初始PAS图，将句法语义关系转化成图结构
    state = build_evidence_graph(hanlp_sdp_result)
    #按实体占位符顺序取得其在句中的全部 token 节点
    #记录实体id是哪些
    entity_ids = [
        node.id
        for entity in explicit_entities
        for node in state.nodes.values()
        if node.text == entity
    ]
    #介词收缩增强
    add_pas_preposition_contraction_edges(state)
    #所有格搜索增强
    add_pas_possessive_contraction_edges(state)
    #并列关系增强
    #根据关系找并列组合
    add_pas_coordination_candidate_attachment_edges(state, entity_ids)
    #在增强后的唯一图上，按节点类型和边代价直接进行最短路径搜索。
    return _select_entity_best_paths(state.nodes, state.graph, entity_ids)


def classify_node(text: str) -> str:
    """Classify one PAS token as entity, content, constraint, or function."""
    #对token进行一个处理
    token = text.strip()
    lower = token.lower()
    #对节点进行分类
    if not token or _is_punctuation(token):
        return "function"
    if ENTITY_RE.fullmatch(token):
        return "entity"
    if NUMERIC_RE.fullmatch(token) or lower in ORDER_CUES | APPROX_CUES:
        return "constraint"
    return "function" if lower in FUNCTION_WORDS else "content"


def build_evidence_graph(hanlp_sdp_result: HanLPSDPResult) -> _WorkingState:
    """Construct the single undirected graph from HanLP PAS edges."""
    #把每一个token转成一个图节点
    nodes = _build_token_nodes(hanlp_sdp_result)
    #初始化唯一的无向邻接图
    graph: dict[str, dict[str, TokenReasoningEdge]] = {}
    for edge in hanlp_sdp_result.edges:
        _add_edge(graph, str(edge.head_idx), str(edge.dep_idx), relation=edge.relation)

    state = _WorkingState(
        nodes=nodes,
        pas_edges=hanlp_sdp_result.edges,
        graph=graph,
        syntax_heads={
            key: value
            for key, value in hanlp_sdp_result.syntax_heads.items()
            if key != "0"
        },
    )
    #把所有格标记符号 ' 和 s 标为 function 节点，避免它们被当作有语义内容的路径节点。
    _mark_possessive_markers(state)
    return state


def add_pas_preposition_contraction_edges(state: _WorkingState) -> None:
    """Contract preposition ARG1--preposition--ARG2 into an augmented core edge."""
    contracted_prepositions: set[str] = set()
    #为了保持运行结果一致，先排序再遍历，否则同一个问题运行结果不一致
    for preposition in state.nodes.values():
        #判断当前节点是不是介词
        if preposition.text.lower() not in PREPOSITIONS:
            continue
        #获取介词的两侧参数
        arg1_ids, arg2_ids = _preposition_arguments(preposition.id, state)
        for arg1_id in arg1_ids:
            if state.nodes[arg1_id].kind not in {"entity", "content"}:
                continue
            for arg2_id in arg2_ids:
                if (
                    arg1_id == arg2_id
                    or state.nodes[arg2_id].kind not in {"entity", "content"}
                ):
                    continue
                #只允许entity、content参与收缩
                _add_edge(
                    state.graph,
                    arg1_id,
                    arg2_id,
                    rule="pas_preposition_contraction",
                )
                contracted_prepositions.add(preposition.id)
    #删除介词原来连接节点的边
    for preposition_id in contracted_prepositions:
        for neighbor_id in list(state.graph[preposition_id]):
            del state.graph[neighbor_id][preposition_id]
        del state.graph[preposition_id]


def add_pas_possessive_contraction_edges(state: _WorkingState) -> None:
    """Contract HanLP's split possessive path: owner -- ' s -- possessed."""
    #记录已处理的 ' 和 s 节点 id
    contracted_markers: set[str] = set()
   #获取当前token节点id
    tokens = list(state.nodes.values())
    #将相邻元素配对
    for marker, suffix in zip(tokens, tokens[1:]):
        if marker.text != "'" or suffix.text.lower() != "s":
            continue
        #分别从原始边中取出’和s链接的两端端点
        owners_a, possessed_a = _possessive_arguments(marker.id, state)
        owners_b, possessed_b = _possessive_arguments(suffix.id, state)
        #合并去重
        for owner_id in owners_a | owners_b:
            for possessed_id in possessed_a | possessed_b:
                #合并去重后直接添加语义边
                _add_edge(
                    state.graph,
                    owner_id,
                    possessed_id,
                    rule="pas_possessive_contraction",
                )
                #记录 ' 和 s，表示它们已经被直接边替代。
                contracted_markers.update({marker.id, suffix.id})

    for marker_id in contracted_markers:
        if marker_id not in state.graph:
            continue
        for neighbor_id in list(state.graph[marker_id]):
            del state.graph[neighbor_id][marker_id]
        del state.graph[marker_id]


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
        #给每个并列实体添加连接
        for member_id in member_ids:
            #若实体已经与中心词相连，就不重复加边，保留原始 PAS 边及其代价；否则补一条结构边
            if attachment_id not in state.graph.get(member_id, {}):
                _add_edge(
                    state.graph,
                    member_id,
                    attachment_id,
                    rule="pas_coordination_candidate_attachment",
                )


def _select_entity_best_paths(
    nodes: dict[str, TokenReasoningNode],
    graph: dict[str, dict[str, TokenReasoningEdge]],
    entity_ids: list[str],
) -> list[list[str]]:
    #找到所有非实体语义节点。后面算SP分数
    semantic_ids = {
        node_id
        for node_id in graph
        if _is_semantic_node(nodes[node_id])
    }
    #语义节点中度数为 1 的节点是边界节点。
    boundary_ids = [
        node_id for node_id in semantic_ids if len(graph[node_id]) == 1
    ]
    selected: list[list[str]] = []
    #遍历处理每个实体
    for entity_id in entity_ids:
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
        selected.append([nodes[node_id].text for node_id in best_path])
    return selected

# 使用 Dijkstra 在语义图中寻找从实体到边界语义节点的最短路径。
def _shortest_path(
    graph: dict[str, dict[str, TokenReasoningEdge]],
    nodes: dict[str, TokenReasoningNode],
    source_id: str,
    target_id: str,
    *,
    blocked_ids: set[str],
) -> list[str]:
    """按边代价搜索最短路径；代价相同时依次比较边数和 token 位置。"""

    # 起点或终点不在图中时，没有可搜索的路径。
    if source_id not in graph or target_id not in graph:
        return []

    # 堆元素依次为：总代价、边数、路径 token 序列、当前节点、完整路径。
    # path_key 使相同代价的结果始终按 token 位置稳定选择。
    start_key = _path_key([source_id])
    heap: list[tuple[int, int, tuple[int, ...], str, list[str]]] = [
        (0, 0, start_key, source_id, [source_id])
    ]
    best: dict[str, tuple[int, int, tuple[int, ...]]] = {
        source_id: (0, 0, start_key)
    }
    while heap:
        # 每次取出当前代价最小的候选路径。
        cost, edge_count, path_key, node_id, path = heapq.heappop(heap)
        # 若该候选已被更优路径替代，则跳过。
        if best[node_id] != (cost, edge_count, path_key):
            continue
        # Dijkstra 首次到达终点时，该路径就是最优路径。
        if node_id == target_id:
            return path
        # 只保留允许进入语义路径、且边代价不是无穷大的相邻节点。
        traversable_neighbors = [
            (neighbor_id, _edge_cost(edge))
            for neighbor_id, edge in graph[node_id].items()
            if _semantic_node_allowed(nodes[neighbor_id])
            and not isinf(_edge_cost(edge))
        ]
        for neighbor_id, edge_cost in sorted(
            traversable_neighbors,
            key=lambda item: (item[1], int(item[0])),
        ):
            # 不重复经过节点，也不经过其他实体节点。
            if neighbor_id in path or neighbor_id in blocked_ids:
                continue
            # 扩展一条边，并构造用于比较优劣的新路径信息。
            next_path = [*path, neighbor_id]
            candidate = (
                cost + edge_cost,
                edge_count + 1,
                _path_key(next_path),
            )
            # 仅在首次到达或找到更优路径时，才加入待搜索堆。
            if neighbor_id not in best or candidate < best[neighbor_id]:
                best[neighbor_id] = candidate
                heapq.heappush(heap, (*candidate, neighbor_id, next_path))
    # 所有候选路径都无法到达终点。
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

#根据PAS关系识别介词节点
def _preposition_arguments(
    preposition_id: str,
    state: _WorkingState,
) -> tuple[set[str], set[str]]:
    arg1 = {
        str(edge.dep_idx)
        for edge in state.pas_edges
        if str(edge.head_idx) == preposition_id
        and str(edge.dep_idx) in state.nodes
        and edge.relation == "prep_ARG1"
    }
    arg2 = {
        str(edge.dep_idx)
        for edge in state.pas_edges
        if str(edge.head_idx) == preposition_id
        and str(edge.dep_idx) in state.nodes
        and edge.relation == "prep_ARG2"
    }
    return arg1, arg2


def _possessive_arguments(
    marker_id: str,
    state: _WorkingState,
) -> tuple[set[str], set[str]]:
    owners: set[str] = set()
    possessed: set[str] = set()
    for edge in state.pas_edges:
        head_id, dep_id = str(edge.head_idx), str(edge.dep_idx)
        if marker_id not in {head_id, dep_id}:
            continue
        related_id = dep_id if head_id == marker_id else head_id
        if related_id not in state.nodes:
            continue
        if edge.relation in POSSESSIVE_OWNER_RELATIONS:
            owners.add(related_id)
        elif edge.relation in POSSESSIVE_POSSESSED_RELATIONS:
            possessed.add(related_id)
    return owners, possessed


def _coordination_groups(
    state: _WorkingState,
    entity_ids: set[str],
) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for edge in state.pas_edges:
        #判断是否是并列关系
        if not (
            "coord" in edge.relation
            or edge.relation in {"conj_member", "disj_member"}
        ):
            continue
        #找到该边的首尾节点
        head_id, dep_id = str(edge.head_idx), str(edge.dep_idx)
        if (
            head_id in state.nodes
            #判断是不是连接词
            and state.nodes[head_id].text.lower() in {"and", "or"}
            #依赖节点是不是实体
            and dep_id in entity_ids
        ):
            #给连接词增加一个实体
            """{
                "5":{
                     "3"
                     }  
                }"""
            groups.setdefault(head_id, []).append(dep_id)
            #同理，判断尾节点是不是实体或
        elif (
            dep_id in state.nodes
            and state.nodes[dep_id].text.lower() in {"and", "or"}
            and head_id in entity_ids
        ):
            groups.setdefault(dep_id, []).append(head_id)

    return [
        (connector_id, member_ids)
        for connector_id, member_ids in groups.items()
        if len(member_ids) >= 2
    ]

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
    #attachments去重，如果句法头节点不是一个
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
    #实体节点
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


def _is_pure_coordination(edge: TokenReasoningEdge) -> bool:
    markers = ("coord", "conj_member", "disj_member", "_and_c", "_or_c")
    return bool(edge.relations) and all(
        any(marker in relation for marker in markers) for relation in edge.relations
    )


def _semantic_node_allowed(node: TokenReasoningNode) -> bool:
    lower = node.text.lower()
    if node.id == "0" or _is_punctuation(node.text):
        return False
    if lower in SEMANTIC_SCOPE_WORDS | DETERMINERS | PREPOSITIONS | LIGHT_VERBS | RELATIVE_PRONOUNS:
        return False
    return node.kind != "function"


def _is_semantic_node(node: TokenReasoningNode) -> bool:
    if node.kind == "entity" or ENTITY_RE.fullmatch(node.text):
        return False
    return _semantic_node_allowed(node) and node.kind in {"content", "constraint"}


def _edge_cost(edge: TokenReasoningEdge) -> int | float:
    """Return the Equation (2) cost; ``inf`` represents an invalid edge."""

    if _is_pure_coordination(edge):
        return inf
    if edge.rules & AUGMENTED_CORE_RULES:
        return 1
    if edge.rules & STRUCTURAL_RULES:
        return 2
    costs = [_pas_relation_cost(relation) for relation in edge.relations]
    return min(costs, default=inf)


def _pas_relation_cost(relation: str) -> int | float:
    if any(marker in relation for marker in INVALID_RELATION_MARKERS):
        return inf
    #核心语义边
    if relation in CORE_RELATIONS or relation.startswith(CORE_RELATION_PREFIXES):
        return 1
    if (
        relation in STRUCTURAL_RELATIONS
        or relation.startswith(STRUCTURAL_RELATION_PREFIXES)
        or relation.endswith("_mod")
    ):
        return 2
    return 3

#构建图节点
def _build_token_nodes(result: HanLPSDPResult) -> dict[str, TokenReasoningNode]:
    nodes = {"0": TokenReasoningNode("0", "ROOT", "function")}
    for index, token in enumerate(result.tokens, start=1):
        text = str(token)
        nodes[str(index)] = TokenReasoningNode(
            #给节点分类
            id=str(index), text=text, kind=classify_node(text)
        )
    return nodes

#将‘ s节点设置类型
def _mark_possessive_markers(
    state: _WorkingState,
) -> None:
    for node in state.nodes.values():
        if node.text.lower() not in POSSESSIVE_MARKERS:
            continue
        owners, possessed = _possessive_arguments(node.id, state)
        if owners or possessed:
            node.kind = "function"


#向当前的推理图中加入一条无向边；如果这条边已经存在，就把新的 relation 或 rule 信息合并进去，而不是重复创建边。
def _add_edge(
    graph: dict[str, dict[str, TokenReasoningEdge]],
    source_id: str,
    target_id: str,
    *,
    relation: str | None = None,
    rule: str | None = None,
) -> None:
    #同一条边对象同时挂在两个端点的邻接表中，表示无向连接。
    edge = graph.setdefault(source_id, {}).get(target_id)
    if edge is None:
        edge = TokenReasoningEdge()
        graph[source_id][target_id] = edge
        graph.setdefault(target_id, {})[source_id] = edge
    #添加关系
    if relation:
        edge.relations.add(relation)
    #添加增强规则pas_preposition_contraction等等
    if rule:
        edge.rules.add(rule)

def _path_key(path: list[str]) -> tuple[int, ...]:
    return tuple(int(node_id) for node_id in path)


def _is_punctuation(text: str) -> bool:
    return bool(text) and all(not character.isalnum() for character in text)
