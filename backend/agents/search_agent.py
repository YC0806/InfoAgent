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
    system_prompt=f"""你是搜索代理（Search Agent）。你的职责是执行 Web 搜索并评估结果。

## 工作流程
1. 分析给定的搜索查询和 gap 上下文
2. 如果查询可以优化，先用 execute_search 尝试改进后的查询
3. 执行搜索，获取结果
4. 评估每个结果对目标 gap 的相关性
5. 输出结构化的候选列表（SearchWorkResult）

## 查询优化策略
- 如果原查询太宽泛，添加限定词
- 如果原查询是中文但目标可能有英文优质来源，可以同时用两种语言搜索
- 考虑添加时间限定（如"2024"、"最新"）提升时效性
- 如果需要反方观点，在查询中加入"争议"、"风险"、"批评"等词

## 候选评估
- 评估每个结果的相关性、权威性、时效性
- 为每个候选分配 0-1 的 score（综合评分）
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
