from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from repo root or backend/ without installing the package.
CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from provider.fetch.firecrawl import FirecrawlProvider
from utils.config import settings


async def main() -> None:
    if not settings.firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is not set")

    provider = FirecrawlProvider()
    results = await provider.crawl(
        url="https://docs.firecrawl.dev",
        include_paths=["^/docs/.*$"],
        max_discovery_depth=1,
        limit=5,
    )
    print(f"Firecrawl results: {len(results)}")
    for item in results[:3]:
        print(item)


if __name__ == "__main__":
    asyncio.run(main())
