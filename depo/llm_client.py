"""供 DEPO 实体识别和 Step5 共用的轻量 OpenAI 兼容 JSON 客户端。"""

from __future__ import annotations

import json
import re
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

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """发送一次确定性对话请求；JSON 无效时带纠错反馈重试。"""

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return self._parse_json(content)
            except Exception as exc:  # OpenAI errors and JSON errors both retry.
                last_error = exc
                if attempt == max_retries - 1:
                    break
                # 向下一次尝试提供具体失败原因，而不是静默重复请求。
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid parseable JSON "
                            f"or the request failed with: {exc}. Return only valid JSON."
                        ),
                    }
                )

        raise RuntimeError(f"LLM did not return valid JSON after {max_retries} attempts: {last_error}")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """接受模型返回的 Markdown JSON 围栏，但顶层必须是对象。"""

        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                raise
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object at the top level.")
        return value
