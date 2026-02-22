from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

from provider.fetch.base import BaseCrawlProvider
from utils.config import settings


class FirecrawlProvider(BaseCrawlProvider):
    """Firecrawl crawl provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.firecrawl.dev",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or settings.firecrawl_api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def crawl(
        self,
        url: str,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_discovery_depth: int = 2,
        limit: int = 1000,
    ) -> List[dict]:
        payload: Dict[str, Any] = {
            "url": url,
            "maxDiscoveryDepth": max_discovery_depth,
            "limit": limit,
        }
        if include_paths:
            payload["includePaths"] = include_paths
        if exclude_paths:
            payload["excludePaths"] = exclude_paths

        response = await self._post_json("/v2/crawl", payload)
        data = response.get("data")
        if isinstance(data, list):
            return data
        if isinstance(response, list):
            return response
        return [response]

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY is not set")
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = json.dumps(payload).encode("utf-8")

        def _request() -> Dict[str, Any]:
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        return await asyncio.to_thread(_request)
