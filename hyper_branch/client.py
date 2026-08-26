
from __future__ import annotations

import json
from typing import Any
from urllib import request

import numpy as np

class OpenAIClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        embedding_model: str,
        timeout_seconds: int,
        temperature: float,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        #创建嵌入缓存

    #调用 embedding 接口，把文本转换成向量；
    def embed_text(self, text: str) -> np.ndarray:
        response = self._post(
            "/embeddings",
            {"model": self.embedding_model, "input": text},
        )
        #转成folat32浮点数向量
        return np.asarray(response["data"][0]["embedding"], dtype=np.float32)
    
    #调用 chat completion 接口，让 LLM 回答原子问题。
    def answer_atomic_question(
        self,
        *,
        system_prompt: str,
        original_question: str,
        atomic_question: str,
        dependency_answers: list[dict[str, str]],
        evidence_blocks: list[dict[str, Any]],
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_question": original_question,
                            "atomic_question": atomic_question,
                            "dependency_answers": dependency_answers,
                            "evidence_blocks": evidence_blocks,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": 900,
            #要求模型返回JSON对象
            "response_format": {"type": "json_object"},
        }
        response = self._post("/chat/completions", payload)
        parsed = json.loads(response["choices"][0]["message"]["content"])
        return str(parsed["answer"]).strip()

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        #创建 HTTP 请求对象
        req = request.Request(
            #请求接口地址
            f"{self.base_url}{endpoint}",
            #发送的数据
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            #请求头
            headers={
                #告诉服务端请求正文是 JSON
                "Content-Type": "application/json",
                #发送api-key
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        #发送请求，等待timeout_seconds，获取response
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            #读取返回内容
            return json.loads(response.read().decode("utf-8"))
