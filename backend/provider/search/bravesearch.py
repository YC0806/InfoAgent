from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from provider.search.base import BaseSearchProvider
from utils.config import settings


class BraveSearchProvider(BaseSearchProvider):
    """Brave Search API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.search.brave.com/res/v1/web/search",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or settings.brave_api_key
        self.base_url = base_url
        self.timeout = timeout

    async def search(
        self,
        query: str,
        count: int = 10,
        offset: int = 0,
        country: Optional[str] = None,
        safesearch: Optional[str] = None,
        freshness: Optional[str] = None,
        summary: bool = False,
    ) -> List[dict]:
        params: Dict[str, Any] = {
            "q": query,
            "count": count,
            "offset": offset,
        }
        if country:
            params["country"] = country
        if safesearch:
            params["safesearch"] = safesearch
        if freshness:
            params["freshness"] = freshness
        if summary:
            params["summary"] = 1

        response = await self._get_json(params)
        results = response.get("web", {}).get("results", [])
        normalized: List[dict] = []
        for item in results:
            normalized.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "description": item.get("description") or item.get("snippet"),
                    "raw": item,
                }
            )
        return normalized

    async def _get_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY is not set")
        url = f"{self.base_url}?{urlencode(params)}"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }

        def _request() -> Dict[str, Any]:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())

        return await asyncio.to_thread(_request)
