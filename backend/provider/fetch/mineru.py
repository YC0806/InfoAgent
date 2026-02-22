from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import Request, urlopen

from provider.fetch.base import BaseCrawlProvider
from utils.config import settings


class MinerUProvider(BaseCrawlProvider):
    """MinerU parsing provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 300,
        poll_interval: int = 5,
        max_wait: int = 600,
        default_model_version: str = "MinerU-HTML",
    ) -> None:
        self.api_key = api_key or settings.mineru_api_key
        self.base_url = (
            (base_url or settings.mineru_base_url or "https://mineru.net").rstrip("/")
        )
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.default_model_version = default_model_version

    async def crawl(
        self,
        url: str,
        model_version: Optional[str] = None,
        is_ocr: Optional[bool] = None,
        enable_formula: Optional[bool] = None,
        enable_table: Optional[bool] = None,
        language: Optional[str] = None,
        data_id: Optional[str] = None,
        callback: Optional[str] = None,
        seed: Optional[str] = None,
        extra_formats: Optional[List[str]] = None,
        page_ranges: Optional[str] = None,
        **kwargs: Any,
    ) -> List[dict]:
        payload: Dict[str, Any] = {"url": url}
        payload["model_version"] = model_version or self.default_model_version

        if is_ocr is not None:
            payload["is_ocr"] = is_ocr
        if enable_formula is not None:
            payload["enable_formula"] = enable_formula
        if enable_table is not None:
            payload["enable_table"] = enable_table
        if language:
            payload["language"] = language
        if data_id:
            payload["data_id"] = data_id
        if callback:
            payload["callback"] = callback
        if seed:
            payload["seed"] = seed
        if extra_formats:
            payload["extra_formats"] = extra_formats
        if page_ranges:
            payload["page_ranges"] = page_ranges

        response = await self._post_json("/api/v4/extract/task", payload)
        print(f"MinerU task creation response: {response}")
        code = response.get("code")
        if code != 0:
            raise RuntimeError(f"MinerU task creation failed: {response}")

        task_id = response.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"MinerU task_id missing in response: {response}")

        status = await self._poll_task(task_id)
        data = status.get("data", {})
        if data.get("state") != "done":
            raise RuntimeError(f"MinerU task failed: {status}")

        full_zip_url = data.get("full_zip_url")
        markdown_text = ""
        json_payload: Any = None
        files: Dict[str, List[str]] = {"markdown": [], "json": []}

        if full_zip_url:
            zip_bytes = await self._download_bytes(full_zip_url)
            markdown_text, json_payload, files = self._extract_zip(zip_bytes)

        return {
            "task_id": task_id,
            "source_url": url,
            "full_zip_url": full_zip_url,
            "markdown": markdown_text,
            "json": json_payload,
            "files": files,
            "raw": status,
            "meta": {"data_id": data.get("data_id")},
        }

    async def _poll_task(self, task_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.max_wait
        while True:
            status = await self._get_json(f"/api/v4/extract/task/{task_id}")
            code = status.get("code")
            if code != 0:
                raise RuntimeError(f"MinerU status error: {status}")

            data = status.get("data", {})
            state = data.get("state")
            if state in {"done", "failed"}:
                if state == "failed":
                    raise RuntimeError(
                        f"MinerU task failed: {data.get('err_msg') or status}"
                    )
                return status

            if time.monotonic() >= deadline:
                raise TimeoutError(f"MinerU task timeout after {self.max_wait}s: {task_id}")

            await asyncio.sleep(self.poll_interval)

    def _extract_zip(
        self, zip_bytes: bytes
    ) -> tuple[str, Optional[Any], Dict[str, List[str]]]:
        markdown_text = ""
        json_payload: Optional[Any] = None
        files: Dict[str, List[str]] = {"markdown": [], "json": []}

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                lower = name.lower()
                if lower.endswith(".md"):
                    files["markdown"].append(name)
                elif lower.endswith(".json"):
                    files["json"].append(name)

            markdown_text = self._read_first_text(zf, files["markdown"])
            json_payload = self._read_first_json(zf, files["json"])

        return markdown_text, json_payload, files

    def _read_first_text(self, zf: zipfile.ZipFile, names: Iterable[str]) -> str:
        for name in names:
            data = zf.read(name)
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("utf-8", errors="replace")
        return ""

    def _read_first_json(
        self, zf: zipfile.ZipFile, names: Iterable[str]
    ) -> Optional[Any]:
        for name in names:
            data = zf.read(name)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
        return None

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("MINERU_API_KEY is not set")
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

        return await self._retry_to_thread(_request, "MinerU POST")

    async def _get_json(self, path: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("MINERU_API_KEY is not set")
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        def _request() -> Dict[str, Any]:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        return await self._retry_to_thread(_request, "MinerU GET")

    async def _download_bytes(self, url: str) -> bytes:
        def _request() -> bytes:
            req = Request(url, method="GET")
            with urlopen(req, timeout=self.timeout) as resp:
                return resp.read()

        return await self._retry_to_thread(_request, "MinerU download")

    async def _retry_to_thread(self, func, label: str, retries: int = 3, delay: int = 3):
        last_exc: Optional[BaseException] = None
        for attempt in range(1, retries + 1):
            try:
                return await asyncio.to_thread(func)
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                await asyncio.sleep(delay)
        raise RuntimeError(f"{label} failed after {retries} retries") from last_exc
