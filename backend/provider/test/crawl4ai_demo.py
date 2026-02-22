from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from repo root or backend/ without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from provider.fetch.crawl4ai import Crawl4AIProvider
from utils.config import settings


async def main() -> None:
    base_url = settings.crawl4ai_base_url or "http://localhost:11235"
    provider = Crawl4AIProvider(base_url=base_url)
    results = await provider.crawl(
        url="https://www.stcn.com/article/detail/1491990.html",
        browser_config={"type": "BrowserConfig", "params": {"headless": True}},
        crawler_config={"type": "CrawlerRunConfig", "params": {"cache_mode": "bypass"}},
    )
    print(f"Crawl4AI results: {len(results)}")
    for item in results[:3]:
        print(item)


if __name__ == "__main__":
    asyncio.run(main())
