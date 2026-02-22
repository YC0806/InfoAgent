"""
Tooling-layer dependency helpers.

Re-exports ToolBudget for backward compatibility and provides
a convenience builder for standalone tool testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.tooling.governance import ToolBudget


@dataclass
class Deps(ToolBudget):
    """Minimal dependency container for standalone tool testing."""

    project_root: Path = Path(".")


def build_deps(project_root: str | Path, tool_call_limit: int | None) -> Deps:
    if tool_call_limit is None or tool_call_limit <= 0:
        tool_call_limit = 0  # unlimited
    return Deps(
        project_root=Path(project_root).resolve(),
        total=tool_call_limit,
    )
