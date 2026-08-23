"""Load a HanLP PAS parser and convert its output to DEPO records."""

from __future__ import annotations

from typing import Any
#导入两个数据结构
from models import HanLPSDPEdge, HanLPSDPResult

class HanLPSDPParser:
    def __init__(self) -> None:
        #初始化保存HanLP模型，第一次加载，以后复用
        self._pipeline: Any | None = None

    def parse(self, text: str) -> HanLPSDPResult:
        #加载模型，然后将结果转成字典形式
        payload = _as_mapping(self._load_pipeline()(text))
        tokens = payload["tok"]
        return HanLPSDPResult(
            tokens=tokens,
            edges=_edges(payload["sdp/pas"]),
            syntax_heads=_syntax_heads(payload["dep"]),
        )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            import hanlp
            #从HanLP预训练模型列表中导入模型
            from hanlp.pretrained.mtl import (
                EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE,
            )
            self._pipeline = hanlp.load(
                EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
            )
        return self._pipeline


def _as_mapping(document: Any) -> dict[str, Any]:
    if isinstance(document, dict):
        return document
    #判断HanLP返回的是不是Document对象，它可能提供转字典方法
    if hasattr(document, "to_dict"):
        return document.to_dict()
    return dict(document)

def _edges(graph: list[list[tuple[int, str]]]) -> list[HanLPSDPEdge]:
    return [
        #创建一个边对象
        HanLPSDPEdge(head_idx=head_idx, relation=relation, dep_idx=dep_idx)
        for dep_idx, dependencies in enumerate(graph, start=1)
        for head_idx, relation in dependencies
    ]

#把普通依存图转换成
def _syntax_heads(graph: list[tuple[int, str]]) -> dict[str, int]:
    return {
        str(dep_idx): head_idx
        for dep_idx, (head_idx, _) in enumerate(graph, start=1)
    }
