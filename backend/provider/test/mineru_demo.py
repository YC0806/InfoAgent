from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from repo root or backend/ without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from provider.fetch.mineru import MinerUProvider
from utils.config import settings


async def main() -> None:
    if not settings.mineru_api_key:
        raise RuntimeError("MINERU_API_KEY is not set")

    provider = MinerUProvider()
    results = await provider.crawl(
        url="http://www.unbank.info/static/pages/2063/313360.html",
        model_version="MinerU-HTML",
    )
    print(f"MinerU results: {len(results)}")
    for item in results[:1]:
        print(
            {
                "task_id": item.get("task_id"),
                "full_zip_url": item.get("full_zip_url"),
                "markdown_len": len(item.get("markdown") or ""),
                "json_keys": list((item.get("json") or {}).keys())
                if isinstance(item.get("json"), dict)
                else None,
            }
        )


if __name__ == "__main__":
    asyncio.run(main())
