from __future__ import annotations

import json

from pydantic_ai import Agent, RunContext

from agents.deps import AgentDeps
from agents.model import get_model
from agents.schemas import FetchWorkResult
from agents.tooling.governance import (
    FETCH_TOOL_BUDGET,
    budget_system_prompt,
    consume_budget,
    prepare_tool_with_budget,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_fetch_result_schema = json.dumps(FetchWorkResult.model_json_schema(), ensure_ascii=False, indent=2)

fetch_agent = Agent(
    get_model(),
    output_type=FetchWorkResult,
    deps_type=AgentDeps,
    system_prompt=f"""你是半自主抓取代理（Fetch Agent）。你的职责是抓取网页内容并提取结构化证据。

## 工作流程
1. 调用 get_candidates_for_gap 获取当前 gap 的可用候选列表
2. 根据标题/描述/来源评估哪些候选最值得抓取
3. 调用 reserve_candidates 领取选中的候选（防止重复抓取）
4. 对每个选中的候选调用 fetch_url 抓取内容
5. 从抓取的内容中提取结构化证据
6. 输出 FetchWorkResult

## 候选选择标准
- 优先选择与 gap 描述高度相关的候选
- 注意来源多样性（不同域名优先）
- 优先权威来源（政府、学术、知名机构）
- 跳过明显低质量的候选（标题模糊、描述无关）
- 每次选 2-4 个候选

## 证据提取标准
- 提取具体的声明（claim）：数据点、定义、结论
- 附上原文引用（citation_text）：直接引用原文段落
- 评估每条证据的置信度（confidence）
- 说明每条证据与 gap 的关联（relevance_to_gap）

## 输出 JSON Schema
```json
{_fetch_result_schema}
```"""
    + budget_system_prompt(FETCH_TOOL_BUDGET),
)


@fetch_agent.tool(prepare=prepare_tool_with_budget)
async def get_candidates_for_gap(
    ctx: RunContext[AgentDeps], gap_id: str, limit: int = 8
) -> str:
    """从 Planner 的候选池获取指定 gap 的可用候选。

    Args:
        gap_id: 目标 gap ID
        limit: 返回候选数量上限
    """
    logger.info("Getting candidates for gap_id=%s, limit=%d", gap_id, limit)
    state = ctx.deps.planner_state
    if not state:
        logger.error("PlannerState not initialized when getting candidates for gap_id=%s", gap_id)
        return "错误: PlannerState 未初始化"
    version, candidates = state.list_candidates_for_gap(gap_id, limit)
    logger.info("Found %d candidates for gap_id=%s (view version=%d)", len(candidates), gap_id, version)
    lines = []
    for c in candidates:
        lines.append(
            f"[{c.candidate_id}] {c.title}\n"
            f"    URL: {c.canonical_url}\n"
            f"    域名: {c.domain}\n"
            f"    描述: {c.description}\n"
            f"    评分: {c.score}"
        )
    if lines:
        result = f"视图版本: {version}\n候选列表:\n" + "\n".join(lines)
    else:
        logger.warning("No available candidates for gap_id=%s", gap_id)
        result = "无可用候选。"
    return result + consume_budget(ctx.deps)


@fetch_agent.tool(prepare=prepare_tool_with_budget)
async def reserve_candidates(
    ctx: RunContext[AgentDeps], candidate_ids: list[str]
) -> str:
    """领取候选以防止并发重复抓取。

    Args:
        candidate_ids: 要领取的候选 ID 列表
    """
    logger.info("Reserving %d candidates: %s", len(candidate_ids), candidate_ids)
    state = ctx.deps.planner_state
    if not state:
        logger.error("PlannerState not initialized when reserving candidates")
        return "错误: PlannerState 未初始化"
    reserved = state.reserve_candidates("fetch_task", candidate_ids)
    logger.info("Successfully reserved %d/%d candidates", len(reserved), len(candidate_ids))
    result = f"成功领取 {len(reserved)} 个候选: {reserved}"
    return result + consume_budget(ctx.deps)


@fetch_agent.tool(prepare=prepare_tool_with_budget)
async def fetch_url(
    ctx: RunContext[AgentDeps], url: str, extraction_focus: str
) -> str:
    """抓取 URL 内容。内容会自动截断以适应上下文窗口。

    Args:
        url: 要抓取的 URL
        extraction_focus: 抽取重点（如 "definition", "data", "mechanism"）
    """
    logger.info("Fetching URL: %s, focus=%s", url, extraction_focus)
    try:
        payload = await ctx.deps.fetch_tool.fetch(
            url, extraction_focus=[extraction_focus]
        )
        content = (
            payload.get("content", "")
            or payload.get("markdown", "")
            or str(payload.get("raw", ""))
        )
        if len(content) > 8000:
            logger.info("Content truncated from %d to 8000 chars for URL: %s", len(content), url)
            content = (
                content[:8000]
                + f"\n\n... [内容已截断，共 {len(content)} 字符]"
            )
        if content:
            logger.info("Successfully fetched URL: %s, content length=%d", url, len(content))
            result = f"=== 抓取内容 ({url}) ===\n{content}"
        else:
            logger.warning("Fetched URL but content is empty: %s", url)
            result = f"抓取成功但内容为空: {url}"
    except Exception as e:
        logger.error("Failed to fetch URL: %s, error=%s", url, e)
        result = f"抓取失败 ({url}): {str(e)}"
    return result + consume_budget(ctx.deps)
