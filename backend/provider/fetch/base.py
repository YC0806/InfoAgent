from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

class BaseCrawlProvider(ABC):
    @abstractmethod
    async def crawl(self, url: str, **kwargs) -> List[dict]:
        """Crawl url and return results."""
