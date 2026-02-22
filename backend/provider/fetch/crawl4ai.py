from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

from provider.fetch.base import BaseCrawlProvider
from utils.config import settings


class Crawl4AIProvider(BaseCrawlProvider):
    """Crawl4AI REST API provider."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        self.base_url = (base_url or settings.crawl4ai_base_url or "http://localhost:11235").rstrip("/")
        self.timeout = timeout

    async def crawl(
        self,
        url: str,
        browser_config: Optional[Dict[str, Any]] = None,
        crawler_config: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        payload: Dict[str, Any] = {
            "urls": [url],
        }
        if browser_config:
            payload["browser_config"] = browser_config
        if crawler_config:
            payload["crawler_config"] = crawler_config

        response = await self._post_json("/crawl", payload)
        data = response.get("data")
        if isinstance(data, list):
            return data
        if isinstance(response, list):
            return response
        return [response]

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = json.dumps(payload).encode("utf-8")

        def _request() -> Dict[str, Any]:
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        return await asyncio.to_thread(_request)
