from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic_ai import UsageLimits

from agents.coordinator import brief_agent, facet_plan_agent
from agents.deps import AgentDeps, PlannerState
from agents.planner import planner_agent
from agents.tooling.governance import PLANNER_TOOL_BUDGET, ToolBudget
from core.checkpoint import CheckpointManager
from core.domains import UserQuery
from core.url_registry import URLRegistry
from tools.fetch_tool import FetchTool
from tools.search_tool import SearchTool
from utils.config import settings
from utils.logger import get_logger
from utils.utils import generate_id

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "runs"


class InfoAgentOrchestrator:
    """End-to-end V3 orchestration loop with PydanticAI Deep Agents."""

    def __init__(
        self,
        *,
        search_tool: Optional[SearchTool] = None,
        fetch_tool: Optional[FetchTool] = None,
        timeout_seconds: int = 300,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.search_tool = search_tool or SearchTool()
        self.fetch_tool = fetch_tool or FetchTool()
        self.timeout_seconds = timeout_seconds
        self.output_dir = output_dir or _DEFAULT_OUTPUT_DIR

    async def run(
        self,
        query_text: str,
        language: str = "zh-CN",
        *,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # --- Checkpoint setup ---
        resuming = run_id is not None
        if resuming:
            ckpt = CheckpointManager.load(self.output_dir / run_id)
            logger.info("Resuming run %s", run_id)
        else:
            run_id = generate_id()
            ckpt = CheckpointManager.create(self.output_dir, run_id)
            logger.info("Starting new run %s", run_id)

        user_query = UserQuery(raw_query=query_text, language=language)

        # Save / update run meta
        ckpt.save_run_meta({
            "run_id": run_id,
            "query": query_text,
            "language": language,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
        })

        # --- Restore or create URL registry ---
        url_registry = ckpt.load_url_registry() if resuming else None
        if url_registry is None:
            url_registry = URLRegistry()

        # Shared deps (planner_state will be set per facet)
        deps = AgentDeps(
            settings=settings,
            search_tool=self.search_tool,
            fetch_tool=self.fetch_tool,
            url_registry=url_registry,
        )

        # Step 1: Build Brief via LLM (or restore from checkpoint)
        brief = ckpt.load_brief() if resuming else None
        if brief is not None:
            logger.info("Restored brief from checkpoint: topic=%s", brief.topic)
        else:
            logger.info("Building Brief for query: %s", query_text)
            brief_result = await brief_agent.run(
                f"分析以下研究查询并生成 Brief 并按照JSON Shema格式输出:\n\n查询: {query_text}\n语言: {language}，请求时间: {datetime.strftime(user_query.timestamp, '%Y-%m-%d %H:%M:%S')}",
                deps=deps,
                usage_limits=UsageLimits(request_limit=None)
            )
            brief = brief_result.output
            logger.info("Brief built: topic=%s", brief.topic)
            ckpt.save_brief(brief)

        # Step 2: Build FacetPlan via LLM (or restore from checkpoint)
        facet_plan = ckpt.load_facet_plan() if resuming else None
        if facet_plan is not None:
            logger.info(
                "Restored facet_plan from checkpoint: %d facets",
                len(facet_plan.facet_specs),
            )
        else:
            logger.info("Building FacetPlan from Brief")
            facet_plan_result = await facet_plan_agent.run(
                f"基于以下 Brief 生成 FacetPlan 并按照JSON Shema格式输出:\n\n{brief.model_dump_json(indent=2)}",
                deps=deps,
                usage_limits=UsageLimits(request_limit=None)
            )
            facet_plan = facet_plan_result.output
            logger.info(
                "FacetPlan built: %d facets", len(facet_plan.facet_specs)
            )
            ckpt.save_facet_plan(facet_plan)

        if not facet_plan.facet_specs:
            result = {
                "run_id": run_id,
                "brief": brief.model_dump(),
                "facet_plan": facet_plan.model_dump(),
                "result": "no_facet",
            }
            ckpt.save_final_result(result)
            ckpt.save_run_meta({
                "run_id": run_id,
                "query": query_text,
                "language": language,
                "status": "completed",
                "started_at": datetime.utcnow().isoformat(),
            })
            return result

        # Step 3: For each FacetTask, run Planner Agent
        facet_results: List[Dict[str, Any]] = []
        for facet_task in facet_plan.facet_specs:
            fid = facet_task.facet_task_id

            # Check if this facet is already completed
            existing_result = ckpt.load_facet_result(fid)
            if existing_result and existing_result.get("all_gaps_closed"):
                logger.info(
                    "Skipping completed facet: %s", facet_task.objective
                )
                facet_results.append(existing_result)
                continue

            logger.info(
                "Running Planner for facet: %s", facet_task.objective
            )

            # Restore or initialize PlannerState
            saved_state = ckpt.load_facet_planner_state(fid) if resuming else None
            if saved_state is not None:
                planner_state = PlannerState.from_dump(facet_task, saved_state)
                logger.info("Restored PlannerState from checkpoint for facet '%s'", facet_task.objective)
            else:
                planner_state = PlannerState(facet_task)

            # Wire incremental save callback
            def _make_save_cb(ps: PlannerState, task_id: str) -> None:
                def _save() -> None:
                    ckpt.save_facet_planner_state(task_id, ps.dump())
                    ckpt.save_url_registry(url_registry)
                return _save
            planner_state.on_state_change = _make_save_cb(planner_state, fid)

            deps.planner_state = planner_state
            deps.tool_budget = ToolBudget(total=PLANNER_TOOL_BUDGET)
            logger.info(
                "Planner tool budget set: %d calls for facet '%s'",
                PLANNER_TOOL_BUDGET, facet_task.objective,
            )

            facet_context = (
                f"开始研究任务。\n"
                f"目标: {facet_task.objective}\n"
                f"种子搜索词: {', '.join(facet_task.seed_terms)}\n"
                f"验收标准:\n"
            )
            for req in facet_task.acceptance_criteria:
                facet_context += (
                    f"  - [{req.requirement_type}] {req.description} "
                    f"(完成规则: {req.completion_rule})\n"
                    "按照JSON Shema格式输出\n"
                )

            try:
                planner_result = await planner_agent.run(
                    facet_context, deps=deps, usage_limits=UsageLimits(request_limit=None)
                )
                summary = planner_result.output
            except TimeoutError:
                logger.warning(
                    "Planner timed out for facet: %s",
                    facet_task.objective,
                )
                summary = "Planner 超时终止"

            facet_result = {
                "facet_task_id": fid,
                "objective": facet_task.objective,
                "summary": summary,
                "planner_dump": planner_state.dump(),
                "all_gaps_closed": planner_state.all_gaps_closed(),
            }
            facet_results.append(facet_result)

            # Save facet checkpoint
            ckpt.save_facet_planner_state(fid, planner_state.dump())
            ckpt.save_facet_result(fid, facet_result)
            ckpt.save_url_registry(url_registry)

        # Aggregate results
        all_closed = all(r["all_gaps_closed"] for r in facet_results)
        result = {
            "run_id": run_id,
            "brief": brief.model_dump(),
            "facet_plan": facet_plan.model_dump(),
            "facet_results": facet_results,
            "status_summary": {
                "all_gaps_closed": all_closed,
                "facets_completed": sum(
                    1 for r in facet_results if r["all_gaps_closed"]
                ),
                "facets_total": len(facet_results),
            },
        }
        ckpt.save_final_result(result)
        ckpt.save_run_meta({
            "run_id": run_id,
            "query": query_text,
            "language": language,
            "status": "completed" if all_closed else "partial",
            "finished_at": datetime.utcnow().isoformat(),
        })

        logger.info("Run %s finished. Output dir: %s", run_id, ckpt.run_dir)
        return result
