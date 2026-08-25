"""加载并复用固定的 HanLP PAS 模型。"""

from __future__ import annotations

from typing import Any

from models import HanLPSDPEdge, HanLPSDPResult


class HanLPSDPParser:
    def __init__(self) -> None:
        self._pipeline: Any | None = None

    def parse(self, text: str) -> HanLPSDPResult:
        #调用模型
        payload = self._load_pipeline()(text)
        return HanLPSDPResult(
            tokens=payload["tok"],
            #转成三元组
            edges=[
                HanLPSDPEdge(head_idx, relation, dep_idx)
                for dep_idx, dependencies in enumerate(payload["sdp/pas"], start=1)
                for head_idx, relation in dependencies
            ],
            #保存输出的依存句法树，用于后面处理并列结构
            syntax_heads={
                str(dep_idx): head_idx
                for dep_idx, (head_idx, _) in enumerate(payload["dep"], start=1)
            },
        )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            import hanlp
            from hanlp.pretrained.mtl import (
                EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE,
            )
            self._pipeline = hanlp.load(
                EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
            )
        return self._pipeline
