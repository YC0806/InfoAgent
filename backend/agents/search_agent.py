from __future__ import annotations

import json
from urllib.parse import urlparse

from pydantic_ai import Agent, RunContext

from agents.deps import AgentDeps
from agents.model import get_model
from agents.schemas import SearchWorkResult
from agents.tooling.governance import (
    SEARCH_TOOL_BUDGET,
    budget_system_prompt,
    consume_budget,
    prepare_tool_with_budget,
)
from core.domains import CandidateItem
from utils.logger import get_logger
from utils.utils import generate_id

logger = get_logger(__name__)

_search_result_schema = json.dumps(SearchWorkResult.model_json_schema(), ensure_ascii=False, indent=2)

search_agent = Agent(
    get_model(),
    output_type=SearchWorkResult,
    deps_type=AgentDeps,
    system_prompt=f"""你是搜索代理（Search Agent）。你的职责是执行 Web 搜索和新闻搜索并评估结果。

## 工作流程
1. 分析给定的搜索查询和 gap 上下文
2. 判断查询类型：时效性查询（含"最新"、"动态"、"趋势"、"news"、"latest"等）优先使用 execute_news_search
3. 制定多策略搜索计划：
   - 对时效性查询：先新闻搜索获取最新动态，再 web 搜索补充深度内容
   - 对深度研究查询：先 web 搜索获取全面信息，可选新闻搜索补充最新进展
4. 执行搜索，获取结果
5. 评估每个结果对目标 gap 的相关性
6. 输出结构化的候选列表（SearchWorkResult）

## 搜索策略
- **新闻搜索**（execute_news_search）：适用于时效性强的查询，如最新动态、行业趋势、产品发布、人物动态
- **Web 搜索**（execute_search）：适用于深度研究、背景信息、技术文档、学术内容
- 对于"最新动态"类查询，务必同时使用新闻搜索和 web 搜索以获得全面覆盖
- freshness 参数：pd=过去一天、pw=过去一周、pm=过去一月，根据时效需求选择

## 查询优化策略
- 如果原查询太宽泛，添加限定词
- 如果原查询是中文但目标可能有英文优质来源，**务必用英文关键词搜索**（如 "AI news 2026"、"LLM latest"）
- 使用具体人名/产品名/公司名搜索（如 "Andrej Karpathy"、"Claude Code"、"GPT-5"）
- 考虑添加时间限定提升时效性
- 如果需要反方观点，在查询中加入"争议"、"风险"、"批评"等词
- 多角度搜索：中文+英文、具体名词+泛化查询、不同时间范围

## 候选评估
- 评估每个结果的相关性、权威性、时效性
- 为每个候选分配 0-1 的 score（综合评分）
- 新闻结果的时效性权重更高
- 过滤明显不相关的结果，不要全部入池

## 输出 JSON Schema
```json
{_search_result_schema}
```"""
    + budget_system_prompt(SEARCH_TOOL_BUDGET),
)


@search_agent.tool(prepare=prepare_tool_with_budget)
async def execute_search(
    ctx: RunContext[AgentDeps], query: str, count: int = 10
) -> str:
    """调用 Brave Search API 执行搜索，返回原始结果列表。

    Args:
        query: 搜索查询文本
        count: 返回结果数量（默认 10）
    """
    logger.info("Executing search: query='%s', count=%d", query, count)
    try:
        results = await ctx.deps.search_tool.search(query=query, count=count)
    except Exception as e:
        logger.error("Search execution failed: query='%s', error=%s", query, e)
        raise
    logger.info("Search returned %d results for query='%s'", len(results), query)
    lines = []
    for i, item in enumerate(results):
        lines.append(
            f"[{i}] {item.get('title', 'N/A')}\n"
            f"    URL: {item.get('url', '')}\n"
            f"    摘要: {item.get('description', '')}"
        )
    result = "\n".join(lines) if lines else "无搜索结果。"
    return result + consume_budget(ctx.deps)


@search_agent.tool(prepare=prepare_tool_with_budget)
async def execute_news_search(
    ctx: RunContext[AgentDeps], query: str, count: int = 20, freshness: str = "pw"
) -> str:
    """搜索最新新闻和动态。适用于时效性强的查询（行业动态、产品发布、人物新闻等）。

    Args:
        query: 搜索查询文本
        count: 返回结果数量（默认 20）
        freshness: 时间过滤 - pd(过去一天), pw(过去一周), pm(过去一月)
    """
    logger.info("Executing news search: query='%s', count=%d, freshness=%s", query, count, freshness)
    try:
        results = await ctx.deps.search_tool.search_news(query=query, count=count, freshness=freshness)
    except Exception as e:
        logger.error("News search execution failed: query='%s', error=%s", query, e)
        raise
    logger.info("News search returned %d results for query='%s'", len(results), query)
    lines = []
    for i, item in enumerate(results):
        source_info = f" [{item.get('source', '')}]" if item.get("source") else ""
        age_info = f" ({item.get('page_age', '')})" if item.get("page_age") else ""
        lines.append(
            f"[{i}] {item.get('title', 'N/A')}{source_info}{age_info}\n"
            f"    URL: {item.get('url', '')}\n"
            f"    摘要: {item.get('description', '')}"
        )
    result = "\n".join(lines) if lines else "无新闻搜索结果。"
    return result + consume_budget(ctx.deps)
