from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from repo root or backend/ without installing the package.
CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from provider.search.bravesearch import BraveSearchProvider
from utils.config import settings


async def main() -> None:
    if not settings.brave_api_key:
        raise RuntimeError("BRAVE_API_KEY is not set")

    provider = BraveSearchProvider()
    results = await provider.search(
        query="firecrawl docs",
        count=5,
        safesearch="moderate",
    )
    print(f"Brave results: {len(results)}")
    for item in results[:3]:
        print(item)


if __name__ == "__main__":
    asyncio.run(main())
