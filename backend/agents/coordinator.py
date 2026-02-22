from __future__ import annotations

import json

from pydantic_ai import Agent, RunContext

from agents.deps import AgentDeps
from agents.model import get_model
from core.domains import Brief, FacetPlan
from utils.logger import get_logger

logger = get_logger(__name__)

_brief_schema = json.dumps(Brief.model_json_schema(), ensure_ascii=False, indent=2)
_facet_plan_schema = json.dumps(FacetPlan.model_json_schema(), ensure_ascii=False, indent=2)

brief_agent = Agent(
    get_model(),
    output_type=Brief,
    deps_type=AgentDeps,
    system_prompt=f"""你是信息研究系统的协调者。分析用户的研究查询，输出结构化的 Brief。

你需要理解：
- 主题实体与边界（topic）
- 范围——地理/时间/领域（scope）
- 目标受众（audience）
- 质量偏好——权威性/时效性/多源/反方（quality_preferences）
- 输出要求（output_requirements）

根据查询语言自动适配输出语言，不要使用固定模板。
对于中文查询使用中文输出，对于英文查询使用英文输出。

## 输出 JSON Schema
```json
{_brief_schema}
```""",
)


facet_plan_agent = Agent(
    get_model(),
    output_type=FacetPlan,
    deps_type=AgentDeps,
    system_prompt=f"""基于 Brief 将研究任务分解为多个 FacetTask。

## 分解原则
- 根据查询内容动态决定需要哪些 facet 和多少个，不要固定使用 3 个
- 每个 facet 应该是独立可并行的研究单元
- 为每个 requirement 设定合理的 completion_rule
- seed_terms 应该是语义相关的搜索词，不是简单的分词

## 输出 JSON Schema
```json
{_facet_plan_schema}
```""",
)
