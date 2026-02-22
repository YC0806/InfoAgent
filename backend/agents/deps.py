from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from agents.tooling.governance import ToolBudget
from core.domains import (
    CandidateItem,
    EvidenceItem,
    FacetTask,
    GapState,
    RequirementState,
    WorkPlan,
)
from core.url_registry import URLRegistry
from tools.fetch_tool import FetchTool
from tools.search_tool import SearchTool
from utils.config import Settings
from utils.logger import get_logger
from utils.utils import generate_id

logger = get_logger(__name__)


class PlannerState:
    """Stateful wrapper around WorkPlan. Handles all state reads/writes for Planner Agent.

    Key design: PlannerState only does state management, never decision-making.
    All decisions (which gap to pick, what query to generate, whether to close a gap)
    are made by the Planner Agent's LLM through tool calls.
    """

    def __init__(self, facet_task: FacetTask) -> None:
        self.facet_task = facet_task
        self.on_state_change: Optional[Callable[[], None]] = None
        self.work_plan = WorkPlan(
            plan_id=generate_id(),
            facet_task_id=facet_task.facet_task_id,
        )
        logger.info("PlannerState initialized for facet: %s", facet_task.objective)
        self.compile_facet_task()

    @classmethod
    def from_dump(cls, facet_task: FacetTask, dump: dict) -> PlannerState:
        """Restore PlannerState from a checkpoint dump (skips compile_facet_task)."""
        instance = object.__new__(cls)
        instance.facet_task = facet_task
        instance.on_state_change = None
        wp_data = dump.get("work_plan", {})
        instance.work_plan = WorkPlan(
            plan_id=wp_data.get("plan_id", generate_id()),
            facet_task_id=facet_task.facet_task_id,
            requirements={
                k: RequirementState.model_validate(v)
                for k, v in wp_data.get("requirements", {}).items()
            },
            gaps={
                k: GapState.model_validate(v)
                for k, v in wp_data.get("gaps", {}).items()
            },
            candidates={
                k: CandidateItem.model_validate(v)
                for k, v in wp_data.get("candidates", {}).items()
            },
            evidence={
                k: EvidenceItem.model_validate(v)
                for k, v in wp_data.get("evidence", {}).items()
            },
            tasks={},
            events=[],
        )
        logger.info(
            "PlannerState restored from dump for facet: %s "
            "(gaps=%d, candidates=%d, evidence=%d)",
            facet_task.objective,
            len(instance.work_plan.gaps),
            len(instance.work_plan.candidates),
            len(instance.work_plan.evidence),
        )
        return instance

    def compile_facet_task(self) -> None:
        """Initialize requirements and gaps from FacetTask acceptance criteria."""
        for requirement in self.facet_task.acceptance_criteria:
            self.work_plan.requirements[requirement.requirement_id] = RequirementState(
                requirement=requirement,
            )
            gap = GapState(
                gap_id=generate_id(),
                linked_requirement_id=requirement.requirement_id,
                gap_statement=requirement.description,
                phase="exploration",
            )
            self.work_plan.gaps[gap.gap_id] = gap
        self.work_plan.log_event(
            action="compile_facet_task",
            rationale="FacetTask compiled to requirements and initial gaps",
            outputs={
                "requirements": len(self.work_plan.requirements),
                "gaps": len(self.work_plan.gaps),
            },
        )
        logger.info(
            "Compiled facet task: %d requirements, %d gaps",
            len(self.work_plan.requirements),
            len(self.work_plan.gaps),
        )

    def list_candidates_for_gap(
        self, gap_id: str, limit: int = 8
    ) -> tuple[int, List[CandidateItem]]:
        """Return versioned candidate view for a gap. Prefers gap-linked candidates."""
        candidates = [
            c
            for c in self.work_plan.candidates.values()
            if gap_id in c.signals.get("gap_ids", [])
            and c.status in {"new", "queued_for_fetch"}
            and c.reserved_by is None
        ]
        if not candidates:
            candidates = [
                c
                for c in self.work_plan.candidates.values()
                if c.status in {"new", "queued_for_fetch"} and c.reserved_by is None
            ]
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]
        candidate_ids = [item.candidate_id for item in ranked]
        view = self.work_plan.create_view(gap_id, candidate_ids)
        return view.version, ranked

    def reserve_candidates(
        self, task_id: str, candidate_ids: Sequence[str]
    ) -> List[str]:
        """Mark candidates as reserved to prevent duplicate fetching."""
        reserved: List[str] = []
        for candidate_id in candidate_ids:
            candidate = self.work_plan.candidates.get(candidate_id)
            if not candidate or candidate.reserved_by is not None:
                continue
            candidate.reserved_by = task_id
            candidate.status = "queued_for_fetch"
            reserved.append(candidate_id)
        return reserved

    def ingest_candidates(
        self, gap_id: str, candidates: List[CandidateItem]
    ) -> None:
        """Add search result candidates to the pool, linked to a gap."""
        added = 0
        for item in candidates:
            existing = self._find_candidate_by_url(item.canonical_url)
            if existing:
                existing.score = max(existing.score, item.score)
                existing.description = existing.description or item.description
                existing.seen_in_queries.extend(item.seen_in_queries)
                gap_ids = set(existing.signals.get("gap_ids", []))
                gap_ids.add(gap_id)
                existing.signals["gap_ids"] = list(gap_ids)
                continue
            item.signals.setdefault("gap_ids", []).append(gap_id)
            self.work_plan.candidates[item.candidate_id] = item
            added += 1

        gap = self.work_plan.gaps.get(gap_id)
        if gap and added:
            gap.state = "shrinking"
            gap.phase = "focus"
        logger.info("Ingested candidates for gap_id=%s: %d new added", gap_id, added)
        if added:
            self._notify_state_change()

    def ingest_evidence(self, gap_id: str, evidence_items: List[EvidenceItem]) -> None:
        """Add extracted evidence to the pool, linked to a gap."""
        gap = self.work_plan.gaps.get(gap_id)
        if not gap:
            return
        requirement_state = self.work_plan.requirements.get(gap.linked_requirement_id)

        for item in evidence_items:
            item.linked_gap_ids = [gap_id]
            if requirement_state:
                item.linked_requirement_ids = [gap.linked_requirement_id]
            self.work_plan.evidence[item.evidence_id] = item
            if requirement_state:
                requirement_state.evidence_links.append(item.evidence_id)
            gap.close_evidence_ids.append(item.evidence_id)
        logger.info("Ingested %d evidence items for gap_id=%s", len(evidence_items), gap_id)
        if evidence_items:
            self._notify_state_change()

    def close_gap(self, gap_id: str) -> None:
        """Close a gap and mark its linked requirement as met."""
        gap = self.work_plan.gaps.get(gap_id)
        if not gap:
            return
        gap.state = "closed"
        gap.phase = "done"
        requirement_state = self.work_plan.requirements.get(gap.linked_requirement_id)
        if requirement_state and requirement_state.evidence_links:
            requirement_state.state = "met"
        logger.info("Gap closed: gap_id=%s, statement='%s'", gap_id, gap.gap_statement[:80])
        self._notify_state_change()

    def get_status_text(self) -> str:
        """Format all gap/requirement/candidate status as LLM-readable text."""
        lines: List[str] = []
        lines.append(f"=== 研究计划状态 ===")
        lines.append(f"目标: {self.facet_task.objective}")
        lines.append("")

        for gap_id, gap in self.work_plan.gaps.items():
            req = self.work_plan.requirements.get(gap.linked_requirement_id)
            req_type = req.requirement.requirement_type if req else "unknown"
            req_state = req.state if req else "unknown"

            candidate_count = sum(
                1
                for c in self.work_plan.candidates.values()
                if gap_id in c.signals.get("gap_ids", [])
            )
            evidence_count = len(gap.close_evidence_ids)

            lines.append(f"Gap [{gap_id}]: {gap.gap_statement}")
            lines.append(
                f"  状态: {gap.state} | 阶段: {gap.phase} | "
                f"需求类型: {req_type} | 需求满足: {req_state}"
            )
            lines.append(f"  候选数: {candidate_count} | 证据数: {evidence_count}")
            lines.append("")

        total = len(self.work_plan.gaps)
        closed = sum(1 for g in self.work_plan.gaps.values() if g.state == "closed")
        lines.append(f"总计: {closed}/{total} gaps 已关闭")
        return "\n".join(lines)

    def get_gap_context(self, gap_id: str) -> str:
        """Get context about a specific gap for sub-agent prompts."""
        gap = self.work_plan.gaps.get(gap_id)
        if not gap:
            return f"Gap {gap_id} 不存在"
        req = self.work_plan.requirements.get(gap.linked_requirement_id)
        req_desc = req.requirement.description if req else "无"
        req_type = req.requirement.requirement_type if req else "unknown"
        return (
            f"Gap: {gap.gap_statement}\n"
            f"需求类型: {req_type}\n"
            f"需求描述: {req_desc}\n"
            f"当前状态: {gap.state} | 阶段: {gap.phase}"
        )

    def get_gap_evidence_text(self, gap_id: str) -> str:
        """Format all evidence for a gap as LLM-readable text."""
        gap = self.work_plan.gaps.get(gap_id)
        if not gap:
            return f"Gap {gap_id} 不存在"

        if not gap.close_evidence_ids:
            return f"Gap [{gap_id}] 暂无证据。\n描述: {gap.gap_statement}"

        lines: List[str] = []
        lines.append(f"=== Gap [{gap_id}] 的证据 ===")
        lines.append(f"描述: {gap.gap_statement}")
        lines.append("")

        for eid in gap.close_evidence_ids:
            ev = self.work_plan.evidence.get(eid)
            if not ev:
                continue
            lines.append(f"证据 [{eid}]:")
            lines.append(f"  来源: {ev.url}")
            lines.append(f"  标题: {ev.title}")
            payload = ev.extracted_payload
            if payload.get("claim"):
                lines.append(f"  声明: {payload['claim']}")
            if payload.get("citation_text"):
                citation = payload["citation_text"]
                if len(citation) > 500:
                    citation = citation[:500] + "..."
                lines.append(f"  引用: {citation}")
            if payload.get("confidence"):
                lines.append(f"  置信度: {payload['confidence']}")
            if payload.get("relevance_to_gap"):
                lines.append(f"  相关性: {payload['relevance_to_gap']}")
            lines.append("")

        return "\n".join(lines)

    def all_gaps_closed(self) -> bool:
        """Check if all gaps are closed."""
        return all(g.state == "closed" for g in self.work_plan.gaps.values())

    def mark_candidates_fetched(self, candidate_ids: List[str]) -> None:
        """Mark candidates as fetched after successful fetch."""
        for cid in candidate_ids:
            candidate = self.work_plan.candidates.get(cid)
            if candidate:
                candidate.status = "fetched"
                candidate.reserved_by = None

    def mark_candidates_dropped(
        self, dropped: List[Dict[str, str]]
    ) -> None:
        """Mark candidates as dropped with reasons."""
        for item in dropped:
            candidate = self.work_plan.candidates.get(item.get("candidate_id", ""))
            if candidate:
                candidate.status = "dropped"
                candidate.reserved_by = None
                candidate.signals["drop_reason"] = item.get("reason", "unknown")

    def dump(self) -> Dict[str, Any]:
        """Serialize the full state for inspection."""
        return {
            "work_plan": {
                "requirements": {
                    k: v.model_dump() for k, v in self.work_plan.requirements.items()
                },
                "gaps": {
                    k: v.model_dump() for k, v in self.work_plan.gaps.items()
                },
                "candidates": {
                    k: v.model_dump() for k, v in self.work_plan.candidates.items()
                },
                "evidence": {
                    k: v.model_dump() for k, v in self.work_plan.evidence.items()
                },
                "tasks": {
                    k: v.model_dump() for k, v in self.work_plan.tasks.items()
                },
                "events": [e.model_dump() for e in self.work_plan.events],
            }
        }

    def _notify_state_change(self) -> None:
        """Call the on_state_change callback if registered."""
        if self.on_state_change is not None:
            try:
                self.on_state_change()
            except Exception:
                logger.warning("on_state_change callback failed", exc_info=True)

    def _find_candidate_by_url(self, canonical_url: str) -> Optional[CandidateItem]:
        for candidate in self.work_plan.candidates.values():
            if candidate.canonical_url == canonical_url:
                return candidate
        return None


@dataclass
class AgentDeps:
    """Shared dependency container for all agents."""

    settings: Settings
    search_tool: SearchTool
    fetch_tool: FetchTool
    url_registry: URLRegistry
    planner_state: Optional[PlannerState] = None
    tool_budget: Optional[ToolBudget] = None
