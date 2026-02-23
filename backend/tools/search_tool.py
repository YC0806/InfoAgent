from __future__ import annotations

from typing import Dict, List, Optional

from provider.search.bravesearch import BraveSearchProvider
from utils.logger import get_logger

logger = get_logger(__name__)

DESCRIPTION = """
用途：输入query在互联网上检索信息 

何时调用：
- 需要为某个模板子项（如 B1 政策法规、C4 竞争格局）寻找权威来源与候选页面时
- 需要补齐缺口、寻找双源验证、或解决来源冲突时
- 需要收集预测材料、统计口径说明、标准与专利线索时

输入建议（按实现字段调整）：
- query：检索词（建议包含：行业/赛道 + 子项标签 + 地域 + 年份/时间窗 + 指标口径）

输出：
对每条结果返回至少：
- link_id：链接标识
- title：标题
- description：摘要
"""


class SearchTool:
    """Search tool wrapper with normalized output."""

    def __init__(self, provider: Optional[BraveSearchProvider] = None) -> None:
        self.provider = provider or BraveSearchProvider()

    async def search(
        self, query: str, count: int = 10, freshness: Optional[str] = None
    ) -> List[Dict[str, str]]:
        if not query.strip():
            return []
        try:
            results = await self.provider.search(
                query=query, count=count, safesearch="moderate", freshness=freshness
            )
            normalized: List[Dict[str, str]] = []
            for item in results:
                url = item.get("url")
                if not url:
                    continue
                normalized.append(
                    {
                        "title": item.get("title") or "",
                        "url": url,
                        "description": item.get("description") or "",
                    }
                )
            if normalized:
                return normalized
        except Exception as exc:
            logger.warning("Brave search failed, fallback to local candidates: %s", exc)
        return self._fallback_candidates(query=query, count=count)

    async def search_news(
        self, query: str, count: int = 20, freshness: str = "pw"
    ) -> List[Dict[str, str]]:
        """Search news articles via Brave News API.

        Args:
            query: Search query text.
            count: Number of results (default 20).
            freshness: Time filter — pd (past day), pw (past week), pm (past month).
        """
        if not query.strip():
            return []
        try:
            results = await self.provider.search_news(
                query=query, count=count, safesearch="moderate", freshness=freshness
            )
            normalized: List[Dict[str, str]] = []
            for item in results:
                url = item.get("url")
                if not url:
                    continue
                normalized.append(
                    {
                        "title": item.get("title") or "",
                        "url": url,
                        "description": item.get("description") or "",
                        "source": item.get("source") or "",
                        "page_age": item.get("page_age") or "",
                    }
                )
            if normalized:
                return normalized
        except Exception as exc:
            logger.warning("Brave news search failed, fallback to web search: %s", exc)
        return await self.search(query=query, count=count, freshness=freshness)

    def _fallback_candidates(self, query: str, count: int) -> List[Dict[str, str]]:
        tokens = [item for item in query.replace("，", " ").replace(",", " ").split(" ") if item]
        seed = "-".join(tokens[:4]) or "topic"
        base = [
            ("Wikipedia", f"https://en.wikipedia.org/wiki/{seed}"),
            ("ArXiv", f"https://arxiv.org/search/?query={seed}"),
            ("World Bank", f"https://www.worldbank.org/en/search?q={seed}"),
            ("OECD", f"https://www.oecd.org/search/?q={seed}"),
            ("Gov Report", f"https://www.gov.cn/search/{seed}"),
        ]
        items: List[Dict[str, str]] = []
        for title, url in base[: max(1, min(count, len(base)))]:
            items.append(
                {
                    "title": f"{title} | {query}",
                    "url": url,
                    "description": f"Fallback candidate for query: {query}",
                }
            )
        return items
