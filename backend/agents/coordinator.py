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

## 查询类型识别
请识别查询的类型并在 quality_preferences 中标注：
- **动态追踪型**（含"最新"、"动态"、"趋势"、"进展"、"news"等）：在 quality_preferences 中注明"query_type: trending"，scope 中明确时间范围（如"过去一周"、"近一个月"），强调时效性和新闻源
- **深度研究型**（含"分析"、"比较"、"原理"、"综述"等）：标注"query_type: research"，强调权威性和多源验证
- **数据分析型**（含"数据"、"统计"、"排名"、"市场规模"等）：标注"query_type: data"，强调数据源权威性和口径一致性

根据查询语言自动适配输出语言，不要使用固定模板。
对于中文查询使用中文输出，对于英文查询使用英文输出。

## 请按照以下 JSON 格式返回分析结果：
{_brief_schema}
""",
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

## 动态追踪类查询的特殊分解策略
当 Brief 的 quality_preferences 中包含 "query_type: trending" 时，应从多角度分解以确保全面覆盖：
- **产品发布与技术更新**：新产品发布、版本更新、功能升级（seed_terms 示例：具体产品名如 "Claude Code"、"GPT-5"、"Gemini"）
- **研究突破与论文**：重要研究成果、新论文、新方法（seed_terms 示例："AI research breakthrough 2026"、"LLM paper"）
- **行业人物动态**：关键人物的言论、博客、演讲（seed_terms 示例：具体人名如 "Andrej Karpathy"、"Sam Altman"）
- **投融资与商业**：融资、收购、合作、商业策略（seed_terms 示例："AI startup funding"、"AI公司融资"）
- **政策与监管**：政策变化、法规更新、行业标准（seed_terms 示例："AI regulation"、"AI政策"）

根据具体主题选择 3-5 个最相关的角度，不必覆盖所有角度。

## seed_terms 最佳实践
- 同时包含中文和英文关键词，因为英文源通常更丰富
- 包含具体的人名、公司名、产品名（不要只用泛化词）
- 包含适合新闻搜索的关键词（如 "latest"、"news"、"announcement"、"release"）
- 为每个 facet 标注搜索优先级提示：在 requirement 描述中注明"优先新闻搜索"或"优先深度搜索"

## 输出 JSON Schema
```json
{_facet_plan_schema}
```""",
)
