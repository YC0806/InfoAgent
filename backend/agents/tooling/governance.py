"""
Tool governance: budget enforcement, response annotation, and prepare callbacks.

### Features
1. **ToolBudget**: Tracks per-agent tool call budget with remaining/exhausted checks.
2. **budget_tag**: Appends budget status to tool responses so LLM stays aware.
3. **prepare_tool_with_budget**: PydanticAI prepare callback that disables tools when budget=0.
4. **Budget constants**: Default tool budgets for each agent type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from utils.logger import get_logger

if TYPE_CHECKING:
    from agents.deps import AgentDeps

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default budgets per agent role
# ---------------------------------------------------------------------------
PLANNER_TOOL_BUDGET = 20
SEARCH_TOOL_BUDGET = 5
FETCH_TOOL_BUDGET = 10


# ---------------------------------------------------------------------------
# Budget state
# ---------------------------------------------------------------------------
@dataclass
class ToolBudget:
    """Tracks tool call budget for a single agent run."""

    total: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def consume(self, cost: int = 1) -> int:
        """Consume budget and return remaining count."""
        self.used += cost
        remaining = self.remaining
        logger.info("Tool budget consumed: used=%d, remaining=%d/%d", self.used, remaining, self.total)
        if remaining == 0:
            logger.warning("Tool budget exhausted: %d/%d used", self.used, self.total)
        return remaining


# ---------------------------------------------------------------------------
# Budget tag for tool responses
# ---------------------------------------------------------------------------
def budget_tag(deps: AgentDeps) -> str:
    """Generate budget status string to append to every tool response.

    This keeps the LLM continuously aware of its remaining budget.
    """
    budget = deps.tool_budget
    if not budget:
        return ""
    r, t = budget.remaining, budget.total
    if r == 0:
        return (
            f"\n\n🚫 [工具预算已耗尽 ({budget.used}/{t})。"
            f"不可再调用任何工具，请立即输出最终结果文本。]"
        )
    if r <= 3:
        return (
            f"\n\n⚠️ [工具预算警告: 剩余 {r}/{t} 次。"
            f"请合理规划剩余调用，预算耗尽后必须直接输出最终结果。]"
        )
    return f"\n\n[工具预算: 剩余 {r}/{t} 次调用]"


def consume_budget(deps: AgentDeps) -> str:
    """Consume 1 budget unit and return the budget tag string.

    Call this at the END of every tool function, then append the result to the
    tool's return value.
    """
    if deps.tool_budget:
        deps.tool_budget.consume()
    return budget_tag(deps)


# ---------------------------------------------------------------------------
# PydanticAI prepare callback
# ---------------------------------------------------------------------------
async def prepare_tool_with_budget(
    ctx: RunContext[AgentDeps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """PydanticAI per-tool prepare callback.

    Returns ``None`` (disabling the tool) when the tool budget is exhausted.
    This forces the LLM to produce a text response instead of calling tools.
    """
    budget = ctx.deps.tool_budget
    if budget and budget.exhausted:
        logger.warning(
            "Tool '%s' disabled — budget exhausted (%d/%d)",
            tool_def.name, budget.used, budget.total,
        )
        return None
    return tool_def


# ---------------------------------------------------------------------------
# System prompt fragment
# ---------------------------------------------------------------------------
def budget_system_prompt(budget_total: int) -> str:
    """Return the budget rules section to embed in an agent's system prompt."""
    return f"""
## 工具调用预算规则
你拥有 **{budget_total} 次** 工具调用预算。每次调用工具消耗 1 次。
- 每次工具响应末尾会显示「剩余 N/{budget_total} 次调用」。
- 当剩余 ≤ 3 时你会看到 ⚠️ 警告，请优先完成最关键的操作。
- 当预算为 0 时所有工具将自动禁用，你 **必须立即输出最终结果文本**，不可再尝试调用工具。
- 请合理规划工具使用：避免重复查询、合并可一次完成的操作。"""
