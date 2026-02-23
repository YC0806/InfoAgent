from __future__ import annotations

from pydantic_ai import Agent, RunContext, UsageLimits

from agents.deps import AgentDeps
from agents.model import get_model
from agents.schemas import FetchWorkResult
from agents.search_agent import search_agent
from agents.fetch_agent import fetch_agent
from agents.tooling.governance import (
    FETCH_TOOL_BUDGET,
    PLANNER_TOOL_BUDGET,
    SEARCH_TOOL_BUDGET,
    ToolBudget,
    budget_system_prompt,
    consume_budget,
    prepare_tool_with_budget,
)
from core.domains import EvidenceItem
from utils.logger import get_logger
from utils.utils import generate_id
from utils.config import settings

if settings.logfire_enabled and settings.logfire_token:
    import logfire
    logfire.configure(token=settings.logfire_token, scrubbing=False)
    logfire.instrument_pydantic_ai()

logger = get_logger(__name__)

planner_agent = Agent(
    get_model(),
    output_type=str,
    deps_type=AgentDeps,
    system_prompt="""你是 Gap 驱动的研究规划者（Planner）。你的核心职责是执行 OODA 循环，
驱动 gap 从 open → shrinking → closed，直到所有研究需求被满足。

## 工作流程
1. 首先调用 get_gap_status 了解当前所有 gap 的状态
2. 选择最高优先级的 open gap（考虑：哪个 gap 最关键、哪个最容易推进）
3. 对于没有候选的 gap → 调用 delegate_search 搜索候选
4. 对于有候选的 gap → 调用 delegate_fetch 抓取和提取证据
5. 抓取完成后，调用 get_gap_evidence 查看已收集的证据
6. 调用 assess_and_close_gap 判断 gap 是否可以关闭
7. 再次调用 get_gap_status 检查整体进展
8. 重复直到所有 gap 关闭或无法继续进展

## 搜索策略指导
delegate_search 的 query_intent 参数支持以下值：
- **broad**: 宽泛搜索，获取概览
- **focused**: 精确搜索，针对具体主题
- **verification**: 验证已有信息
- **counterpoint**: 寻找反方观点
- **news**: 新闻搜索 — 适用于时效性强的 gap（含"最新"、"动态"、"趋势"、"进展"等需求）

### 搜索类型选择规则
- 当 gap 需求描述中包含时效性关键词（最新、动态、趋势、进展、发布、news、latest、recent）时，**优先使用 query_intent="news"**
- 当 gap 需求是深度分析、背景研究、技术原理时，使用 query_intent="broad" 或 "focused"
- 可以对同一 gap 进行多次搜索，使用不同的 query_intent 和查询角度

### 查询多样化策略
- 用不同语言搜索：中文查询+英文查询，英文源通常更丰富
- 用不同粒度搜索：泛化查询（"AI latest news"）+ 具体查询（"Claude Code security update"）
- 用不同时间范围：近期搜索 + 较长时间范围搜索
- 搜索结果不好时换关键词角度，而不是重复相同查询

## 决策规则
- gap 关闭需要证据确实满足需求描述，不是"有东西就关"
- 搜索结果质量不好时，用不同的查询策略重新搜索（换关键词、换语言、加限定）
- 注意来源多样性，不要所有证据来自同一网站
- 如果某个 gap 反复搜索都无法满足，可以在最终输出中说明
- 当所有 gap 都关闭后，输出最终状态摘要并结束

## 输出
当你认为研究完成（或无法继续进展），直接输出最终状态摘要文本，不再调用工具。"""
    + budget_system_prompt(PLANNER_TOOL_BUDGET),
    retries=2,
    instrument=True
)


@planner_agent.tool(prepare=prepare_tool_with_budget)
async def get_gap_status(ctx: RunContext[AgentDeps]) -> str:
    """获取所有 gap、requirement、candidate 的当前状态概览。
    每次决策前应先调用此工具了解全局状态。"""
    logger.info("Planner requesting gap status overview")
    state = ctx.deps.planner_state
    if not state:
        logger.error("PlannerState not initialized when requesting gap status")
        return "错误: PlannerState 未初始化"
    result = state.get_status_text()
    return result + consume_budget(ctx.deps)


@planner_agent.tool(prepare=prepare_tool_with_budget)
async def get_gap_evidence(ctx: RunContext[AgentDeps], gap_id: str) -> str:
    """获取某个 gap 已收集的所有证据摘要。
    在判断 gap 是否可以关闭前调用此工具。

    Args:
        gap_id: 要查看证据的 gap ID
    """
    logger.info("Planner requesting evidence for gap_id=%s", gap_id)
    state = ctx.deps.planner_state
    if not state:
        logger.error("PlannerState not initialized when requesting evidence for gap_id=%s", gap_id)
        return "错误: PlannerState 未初始化"
    result = state.get_gap_evidence_text(gap_id)
    return result + consume_budget(ctx.deps)


@planner_agent.tool(prepare=prepare_tool_with_budget)
async def delegate_search(
    ctx: RunContext[AgentDeps],
    gap_id: str,
    search_query: str,
    query_intent: str,
) -> str:
    """委派搜索任务给 SearchAgent。搜索结果会自动入候选池。

    Args:
        gap_id: 本次搜索服务的 gap ID
        search_query: 搜索查询文本（你来决定最优查询）
        query_intent: 搜索意图 - broad/focused/verification/counterpoint/news（news 表示优先新闻搜索）
    """
    logger.info("Delegating search: gap_id=%s, query='%s', intent=%s", gap_id, search_query, query_intent)
    state = ctx.deps.planner_state
    if not state:
        logger.error("PlannerState not initialized when delegating search for gap_id=%s", gap_id)
        return "错误: PlannerState 未初始化"

    prompt = (
        f"搜索查询: {search_query}\n"
        f"搜索意图: {query_intent}\n"
        f"Gap 上下文: {state.get_gap_context(gap_id)}"
    )

    # Swap to sub-agent budget for the search_agent run
    parent_budget = ctx.deps.tool_budget
    ctx.deps.tool_budget = ToolBudget(total=SEARCH_TOOL_BUDGET)
    try:
        result = await search_agent.run(prompt, deps=ctx.deps, usage=ctx.usage, usage_limits=UsageLimits(request_limit=None))
    except Exception as e:
        logger.error("SearchAgent failed for gap_id=%s, query='%s': %s", gap_id, search_query, e)
        raise
    finally:
        ctx.deps.tool_budget = parent_budget  # restore planner budget

    state.ingest_candidates(gap_id, result.output.candidates)
    logger.info(
        "Search completed: gap_id=%s, query='%s', candidates=%d",
        gap_id, result.output.query_used, len(result.output.candidates),
    )
    response = (
        f"搜索完成。查询: '{result.output.query_used}'\n"
        f"发现 {len(result.output.candidates)} 个候选\n"
        f"评估: {result.output.relevance_notes}"
    )
    return response + consume_budget(ctx.deps)


@planner_agent.tool(prepare=prepare_tool_with_budget)
async def delegate_fetch(
    ctx: RunContext[AgentDeps],
    gap_id: str,
    instruction: str,
) -> str:
    """委派抓取任务给 FetchAgent。FetchAgent 会自主选择候选、抓取、提取证据。

    Args:
        gap_id: 本次抓取服务的 gap ID
        instruction: 给 FetchAgent 的指令（要提取什么样的证据）
    """
    logger.info("Delegating fetch: gap_id=%s, instruction='%s'", gap_id, instruction[:100])
    state = ctx.deps.planner_state
    if not state:
        logger.error("PlannerState not initialized when delegating fetch for gap_id=%s", gap_id)
        return "错误: PlannerState 未初始化"

    prompt = (
        f"Gap ID: {gap_id}\n"
        f"抓取指令: {instruction}\n"
        f"Gap 上下文: {state.get_gap_context(gap_id)}"
    )

    # Swap to sub-agent budget for the fetch_agent run
    parent_budget = ctx.deps.tool_budget
    ctx.deps.tool_budget = ToolBudget(total=FETCH_TOOL_BUDGET)
    try:
        result = await fetch_agent.run(prompt, deps=ctx.deps, usage=ctx.usage, usage_limits=UsageLimits(request_limit=None))
    except Exception as e:
        logger.error("FetchAgent failed for gap_id=%s: %s", gap_id, e)
        raise
    finally:
        ctx.deps.tool_budget = parent_budget  # restore planner budget

    fetch_result = result.output

    # Convert ExtractedEvidence to EvidenceItem and ingest
    evidence_items = []
    for ev in fetch_result.evidence:
        evidence_items.append(
            EvidenceItem(
                evidence_id=generate_id(),
                from_candidate_id="",
                url=ev.source_url,
                title=ev.claim[:80],
                citations=[{"url": ev.source_url, "text": ev.citation_text}],
                extracted_payload={
                    "claim": ev.claim,
                    "citation_text": ev.citation_text,
                    "relevance_to_gap": ev.relevance_to_gap,
                    "confidence": ev.confidence,
                },
            )
        )
    state.ingest_evidence(gap_id, evidence_items)

    # Mark candidates as fetched/dropped
    state.mark_candidates_fetched(fetch_result.selected_candidate_ids)
    state.mark_candidates_dropped(fetch_result.dropped)

    logger.info(
        "Fetch completed: gap_id=%s, selected=%d, evidence=%d",
        gap_id, len(fetch_result.selected_candidate_ids), len(fetch_result.evidence),
    )
    response = (
        f"抓取完成。选中 {len(fetch_result.selected_candidate_ids)} 个候选\n"
        f"提取 {len(fetch_result.evidence)} 条证据\n"
        f"覆盖评估: {fetch_result.coverage_assessment}"
    )
    return response + consume_budget(ctx.deps)


@planner_agent.tool(prepare=prepare_tool_with_budget)
async def assess_and_close_gap(
    ctx: RunContext[AgentDeps],
    gap_id: str,
    should_close: bool,
    reasoning: str,
) -> str:
    """评估 gap 是否应该关闭。这是你的判断——基于已收集证据是否满足需求。

    Args:
        gap_id: 要评估的 gap ID
        should_close: 你的判断——是否关闭此 gap
        reasoning: 你的推理过程
    """
    logger.info("Assessing gap: gap_id=%s, should_close=%s", gap_id, should_close)
    state = ctx.deps.planner_state
    if not state:
        logger.error("PlannerState not initialized when assessing gap_id=%s", gap_id)
        return "错误: PlannerState 未初始化"

    if should_close:
        state.close_gap(gap_id)
        logger.info("Gap closed: gap_id=%s, reasoning=%s", gap_id, reasoning[:100])
        response = f"Gap {gap_id} 已关闭。理由: {reasoning}"
    else:
        logger.info("Gap kept open: gap_id=%s, reasoning=%s", gap_id, reasoning[:100])
        response = f"Gap {gap_id} 保持 open。理由: {reasoning}"
    return response + consume_budget(ctx.deps)
