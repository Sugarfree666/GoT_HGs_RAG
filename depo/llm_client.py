"""供 DEPO 实体识别和 Step5 共用的轻量 OpenAI 兼容 JSON 客户端。"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


class LLMClient:
    """请求符合结构的 JSON，并对暂时性 API 或解析失败进行有限重试。"""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """发起一次 JSON 对话请求。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
