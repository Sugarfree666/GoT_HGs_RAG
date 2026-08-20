"""从配置的提示词目录加载具名模板，并维护小型内存缓存。"""

from __future__ import annotations

from pathlib import Path


class PromptManager:
    """按名称读取提示词文件，避免同一运行中重复访问磁盘。"""

    def __init__(self, prompt_dir: Path) -> None:
        self.prompt_dir = prompt_dir
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        """读取并缓存指定名称的 Markdown 提示词。"""

        if name not in self._cache:
            path = self.prompt_dir / f"{name}.md"
            self._cache[name] = path.read_text(encoding="utf-8")
        return self._cache[name]
