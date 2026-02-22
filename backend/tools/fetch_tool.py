from __future__ import annotations

from typing import Any, Dict, List, Optional

from provider.fetch.crawl4ai import Crawl4AIProvider
from provider.fetch.firecrawl import FirecrawlProvider
from provider.fetch.mineru import MinerUProvider

from utils.logger import get_logger

logger = get_logger(__name__)

DESCRIPTION = """
用途：拉取指定 link_id 的网页正文与元信息

何时调用：
- 需要核验搜索摘要中的关键断言/数字/定义/预测方法时
- 需要提取政策法规条款、统计口径、市场规模数据表、份额信息、技术路线描述等可引用证据时
- 需要从页面内外进一步访问链接时

输入：
- link_id：目标页面标识（仅接受 link_id，不接受真实URL）

输出建议字段：
- link_id：回显
- content：正文文本
- status/error：错误原因（如失败）

使用规则：
- 只能传入 link_id
- 避免重复拉取同一 link_id，已拉取内容应复用
"""


class FetchTool:
    """Fetch tool with provider fallback and normalized content."""

    def __init__(
        self,
        *,
        firecrawl_provider: Optional[FirecrawlProvider] = None,
        crawl4ai_provider: Optional[Crawl4AIProvider] = None,
        mineru_provider: Optional[MinerUProvider] = None,
    ) -> None:
        self.firecrawl_provider = firecrawl_provider or FirecrawlProvider()
        self.crawl4ai_provider = crawl4ai_provider or Crawl4AIProvider()
        self.mineru_provider = mineru_provider or MinerUProvider()

    async def fetch(self, url: str, extraction_focus: Optional[List[str]] = None) -> Dict[str, Any]:
        extraction_focus = extraction_focus or []
        if not url:
            return {"content": "", "summary": "", "raw": {}, "focus": extraction_focus}

        last_error: Optional[Exception] = None
        for provider_name in self._provider_order(url):
            provider = self._provider(provider_name)
            try:
                raw = await provider.crawl(url=url)
                content = self._extract_text(raw)
                return {
                    "content": content,
                    "summary": content[:500],
                    "raw": raw,
                    "provider": provider_name,
                    "focus": extraction_focus,
                }
            except Exception as exc:
                last_error = exc
                logger.warning("Fetch provider failed: %s", provider_name)
                continue

        logger.warning("All fetch providers failed, using offline fallback for %s", url)
        return {
            "content": f"Offline fallback content for {url}. Extracted focus: {', '.join(extraction_focus)}",
            "summary": f"Offline fallback for {url}",
            "raw": {"fallback": True, "last_error": str(last_error) if last_error else ""},
            "provider": "fallback",
            "focus": extraction_focus,
        }

    def _provider_order(self, url: str) -> List[str]:
        lowered = url.lower()
        if lowered.endswith(".pdf"):
            return ["mineru", "firecrawl", "crawl4ai"]
        return ["firecrawl", "crawl4ai", "mineru"]

    def _provider(self, name: str):
        if name == "firecrawl":
            return self.firecrawl_provider
        if name == "crawl4ai":
            return self.crawl4ai_provider
        return self.mineru_provider

    def _extract_text(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            if payload.get("markdown"):
                return str(payload.get("markdown"))
            if payload.get("content"):
                return str(payload.get("content"))
            if payload.get("summary"):
                return str(payload.get("summary"))
            return str(payload)
        if isinstance(payload, list):
            parts: List[str] = []
            for item in payload[:3]:
                if isinstance(item, dict):
                    parts.append(str(item.get("markdown") or item.get("content") or item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(payload)
