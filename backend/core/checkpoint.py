from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.domains import Brief, FacetPlan
from core.url_registry import URLRegistry
from utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages checkpoint files for a single run under a run directory.

    All writes use atomic rename (write to temp file, then rename) to prevent
    corruption from mid-write crashes.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.facets_dir = run_dir / "facets"

    @classmethod
    def create(cls, base_dir: Path, run_id: str) -> CheckpointManager:
        """Create a new run directory and return a CheckpointManager for it."""
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "facets").mkdir(exist_ok=True)
        logger.info("Created checkpoint dir: %s", run_dir)
        return cls(run_dir)

    @classmethod
    def load(cls, run_dir: Path) -> CheckpointManager:
        """Load an existing run directory."""
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return cls(run_dir)

    # --- atomic write helper ---

    def _atomic_write(self, path: Path, data: Any) -> None:
        """Write JSON data atomically: write to temp file then rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            tmp_path.rename(path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read a JSON file, returning None if it doesn't exist."""
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --- run_meta ---

    def save_run_meta(self, meta: Dict[str, Any]) -> None:
        self._atomic_write(self.run_dir / "run_meta.json", meta)

    def load_run_meta(self) -> Optional[Dict[str, Any]]:
        return self._read_json(self.run_dir / "run_meta.json")

    # --- brief ---

    def save_brief(self, brief: Brief) -> None:
        self._atomic_write(self.run_dir / "brief.json", brief.model_dump())
        logger.info("Saved brief checkpoint")

    def load_brief(self) -> Optional[Brief]:
        data = self._read_json(self.run_dir / "brief.json")
        if data is None:
            return None
        return Brief.model_validate(data)

    # --- facet_plan ---

    def save_facet_plan(self, facet_plan: FacetPlan) -> None:
        self._atomic_write(
            self.run_dir / "facet_plan.json", facet_plan.model_dump()
        )
        logger.info("Saved facet_plan checkpoint")

    def load_facet_plan(self) -> Optional[FacetPlan]:
        data = self._read_json(self.run_dir / "facet_plan.json")
        if data is None:
            return None
        return FacetPlan.model_validate(data)

    # --- url_registry ---

    def save_url_registry(self, registry: URLRegistry) -> None:
        self._atomic_write(
            self.run_dir / "url_registry.json", registry.to_dict()
        )

    def load_url_registry(self) -> Optional[URLRegistry]:
        data = self._read_json(self.run_dir / "url_registry.json")
        if data is None:
            return None
        return URLRegistry.from_dict(data)

    # --- per-facet planner state ---

    def save_facet_planner_state(
        self, facet_task_id: str, state_dump: Dict[str, Any]
    ) -> None:
        facet_dir = self.facets_dir / facet_task_id
        facet_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(facet_dir / "planner_state.json", state_dump)
        logger.debug("Saved planner_state for facet %s", facet_task_id)

    def load_facet_planner_state(
        self, facet_task_id: str
    ) -> Optional[Dict[str, Any]]:
        return self._read_json(
            self.facets_dir / facet_task_id / "planner_state.json"
        )

    # --- per-facet result ---

    def save_facet_result(
        self, facet_task_id: str, result: Dict[str, Any]
    ) -> None:
        facet_dir = self.facets_dir / facet_task_id
        facet_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(facet_dir / "result.json", result)
        logger.info("Saved facet result for %s", facet_task_id)

    def load_facet_result(
        self, facet_task_id: str
    ) -> Optional[Dict[str, Any]]:
        return self._read_json(
            self.facets_dir / facet_task_id / "result.json"
        )

    # --- final result ---

    def save_final_result(self, result: Dict[str, Any]) -> None:
        self._atomic_write(self.run_dir / "result.json", result)
        logger.info("Saved final result checkpoint")

    def load_final_result(self) -> Optional[Dict[str, Any]]:
        return self._read_json(self.run_dir / "result.json")
