import re
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlparse, urlunparse

from utils.utils import generate_id

@dataclass
class RegistryEntry:
    id: str
    canonical_url: str
    domain: str
    created_at: datetime
    last_seen_at: datetime
    final_url: Optional[str] = None
    content_type: Optional[str] = None
    title: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class URLRegistry:
    _TRACKING_KEYS = {"gclid", "fbclid"}
    _TRACKING_PREFIXES = ("utm_",)
    _DEFAULT_PORTS = {"http": 80, "https": 443}
    _URL_REGEX = re.compile(r"(?P<url>(?:https?://|www\.)[^\s<>\"]+)")
    _TRAILING_PUNCTUATION = ".,);:!?]}>"

    def __init__(
        self,
        *,
        allow_schemes: Iterable[str] = ("http", "https"),
        drop_fragments: bool = True,
        remove_default_ports: bool = True,
        remove_tracking: bool = True,
        max_url_length: int = 2048,
    ) -> None:
        self.allow_schemes = {scheme.lower() for scheme in allow_schemes}
        self.drop_fragments = drop_fragments
        self.remove_default_ports = remove_default_ports
        self.remove_tracking = remove_tracking
        self.max_url_length = max_url_length
        self._by_id: Dict[str, RegistryEntry] = {}
        self._by_key: Dict[str, str] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "allow_schemes": sorted(self.allow_schemes),
                "drop_fragments": self.drop_fragments,
                "remove_default_ports": self.remove_default_ports,
                "remove_tracking": self.remove_tracking,
                "max_url_length": self.max_url_length,
            },
            "entries": {
                entry_id: {
                    "id": entry.id,
                    "canonical_url": entry.canonical_url,
                    "domain": entry.domain,
                    "created_at": entry.created_at.isoformat(),
                    "last_seen_at": entry.last_seen_at.isoformat(),
                    "final_url": entry.final_url,
                    "content_type": entry.content_type,
                    "title": entry.title,
                    "meta": entry.meta,
                }
                for entry_id, entry in self._by_id.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "URLRegistry":
        config = payload.get("config", {})
        registry = cls(
            allow_schemes=config.get("allow_schemes", ("http", "https")),
            drop_fragments=config.get("drop_fragments", True),
            remove_default_ports=config.get("remove_default_ports", True),
            remove_tracking=config.get("remove_tracking", True),
            max_url_length=config.get("max_url_length", 2048),
        )
        entries = payload.get("entries", {})
        for entry_id, raw in entries.items():
            created_at = _parse_datetime(raw.get("created_at"))
            last_seen_at = _parse_datetime(raw.get("last_seen_at"))
            entry = RegistryEntry(
                id=raw.get("id", entry_id),
                canonical_url=raw.get("canonical_url", ""),
                domain=raw.get("domain", ""),
                created_at=created_at,
                last_seen_at=last_seen_at,
                final_url=raw.get("final_url"),
                content_type=raw.get("content_type"),
                title=raw.get("title"),
                meta=raw.get("meta") or {},
            )
            registry._by_id[entry.id] = entry
            if entry.canonical_url:
                key = sha256(entry.canonical_url.encode("utf-8")).hexdigest()
                registry._by_key[key] = entry.id
        return registry

    def register(
        self,
        url: str,
        *,
        source: Optional[str] = None,
        base_url: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not url:
            return None
        url = url.strip()
        if base_url:
            url = urljoin(base_url, url)
        if len(url) > self.max_url_length:
            return None

        canonical_url = self._canonicalize(url)
        if not canonical_url:
            return None

        canonical_key = sha256(canonical_url.encode("utf-8")).hexdigest()
        now = datetime.utcnow()
        existing_id = self._by_key.get(canonical_key)
        if existing_id:
            entry = self._by_id[existing_id]
            entry.last_seen_at = now
            if meta:
                entry.meta.update(meta)
            if source:
                entry.meta.setdefault("source", source)
            return existing_id

        new_id = generate_id()
        entry = RegistryEntry(
            id=new_id,
            canonical_url=canonical_url,
            domain=urlparse(canonical_url).netloc,
            created_at=now,
            last_seen_at=now,
            meta=meta or {},
        )
        if source:
            entry.meta.setdefault("source", source)
        self._by_id[new_id] = entry
        self._by_key[canonical_key] = new_id
        return new_id

    def resolve(self, link_id: str) -> Optional[str]:
        entry = self._by_id.get(link_id)
        if not entry:
            return None
        return entry.final_url or entry.canonical_url

    def get_entry(self, link_id: str) -> Optional[RegistryEntry]:
        return self._by_id.get(link_id)

    def _canonicalize(self, url: str) -> Optional[str]:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        if not parsed.scheme or parsed.scheme.lower() not in self.allow_schemes:
            return None
        if not parsed.netloc:
            return None

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        if self.remove_default_ports:
            host, port = self._split_host_port(netloc)
            if port and self._DEFAULT_PORTS.get(scheme) == port:
                netloc = host

        query = parsed.query
        if query:
            query = self._canonicalize_query(query)

        fragment = "" if self.drop_fragments else parsed.fragment
        canonical = urlunparse((scheme, netloc, path, "", query, fragment))
        return canonical

    def _canonicalize_query(self, query: str) -> str:
        pairs = parse_qsl(query, keep_blank_values=True)
        if self.remove_tracking:
            pairs = [
                (k, v)
                for k, v in pairs
                if k not in self._TRACKING_KEYS and not k.startswith(self._TRACKING_PREFIXES)
            ]
        pairs.sort(key=lambda item: (item[0], item[1]))
        return "&".join(f"{quote(k)}={quote(v)}" for k, v in pairs)

    @staticmethod
    def _split_host_port(netloc: str) -> Tuple[str, Optional[int]]:
        if ":" not in netloc:
            return netloc, None
        host, port_text = netloc.rsplit(":", 1)
        try:
            return host, int(port_text)
        except ValueError:
            return netloc, None


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.utcnow()
    return datetime.utcnow()
