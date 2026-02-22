from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from utils.utils import generate_id


GAP_PHASES = ("exploration", "focus", "verification", "done")
REQUIREMENT_STATES = ("unmet", "partial", "met")
GAP_STATES = ("open", "shrinking", "closed")
TASK_STATES = ("pending", "running", "done", "failed")


class UserQuery(BaseModel):
    raw_query: str
    language: str = "zh-CN"
    constraints: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Brief(BaseModel):
    topic: str = Field(description="研究主题，简洁清晰")
    scope: str = Field(description="研究范围描述，包括地理/时间/领域等限定")
    audience: str = Field(description="目标受众")
    quality_preferences: List[str] = Field(description="质量偏好列表，如 authority-first, multi-source, include-counterpoint, recent-data")
    output_requirements: List[str] = Field(description="输出要求列表，如 structured summary, citations, data tables")
    language: str = Field(default="zh-CN", description="查询语言代码，如 zh-CN 或 en")


class EvidenceRequirement(BaseModel):
    requirement_id: str = Field(description="唯一标识，8位随机字符串")
    requirement_type: str = Field(description="证据需求类型: definition/data/mechanism/case/counterpoint/policy/trend/comparison")
    description: str = Field(description="需求描述")
    quality_constraints: Dict[str, Any] = Field(default_factory=dict, description="质量约束，如 {\"min_sources\": 2}")
    completion_rule: Dict[str, Any] = Field(default_factory=dict, description="完成规则，如 {\"min_evidence\": 2}")


class FacetTask(BaseModel):
    facet_task_id: str = Field(description="唯一标识，8位随机字符串")
    objective: str = Field(description="子目标描述")
    seed_terms: List[str] = Field(description="语义相关的种子搜索词列表")
    acceptance_criteria: List[EvidenceRequirement] = Field(description="验收标准，EvidenceRequirement 列表")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="约束条件")


class FacetPlan(BaseModel):
    plan_id: str = Field(description="唯一标识，8位随机字符串")
    facet_specs: List[FacetTask] = Field(description="FacetTask 列表")
    global_acceptance: Dict[str, Any] = Field(default_factory=dict, description="全局验收条件，如 {\"min_sources\": 3, \"with_citations\": true}")
    plan_version: str = Field(default="v3", description="固定为 v3")
    trace_meta: Dict[str, Any] = Field(default_factory=dict, description="追踪元数据，可留空")


class RequirementState(BaseModel):
    requirement: EvidenceRequirement
    state: str = "unmet"
    evidence_links: List[str] = Field(default_factory=list)


class GapState(BaseModel):
    gap_id: str
    linked_requirement_id: str
    gap_statement: str
    state: str = "open"
    phase: str = "exploration"
    close_evidence_ids: List[str] = Field(default_factory=list)


class CandidateItem(BaseModel):
    candidate_id: str = Field(description="唯一标识，8位随机字符串")
    canonical_url: str = Field(description="候选 URL")
    title: str = Field(description="结果标题")
    description: str = Field(description="结果描述/摘要")
    domain: str = Field(description="域名")
    source_type: str = Field(default="web", description="来源类型，默认 web")
    published_at: Optional[str] = Field(default=None, description="发布日期（可选）")
    signals: Dict[str, Any] = Field(default_factory=dict, description="附加元数据信号")
    score: float = Field(default=0.0, description="综合相关性评分 0-1")
    status: str = Field(default="new", description="候选状态，默认 new")
    seen_in_queries: List[str] = Field(default_factory=list, description="出现在哪些查询中")
    reserved_by: Optional[str] = Field(default=None, description="被哪个任务领取")


class EvidenceItem(BaseModel):
    evidence_id: str
    from_candidate_id: str
    url: str
    title: str
    citations: List[Dict[str, Any]]
    extracted_payload: Dict[str, Any] = Field(default_factory=dict)
    linked_requirement_ids: List[str] = Field(default_factory=list)
    linked_gap_ids: List[str] = Field(default_factory=list)
    quality_flags: List[str] = Field(default_factory=list)


class PlanEvent(BaseModel):
    timestamp: str
    action: str
    rationale: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)


class WorkTask(BaseModel):
    task_id: str
    task_type: str
    facet_task_id: str
    linked_gap_id: str
    linked_requirement_id: str
    strategy_tag: str
    expected_yield: str
    status: str = "pending"
    dependencies: List[str] = Field(default_factory=list)
    query_id: Optional[str] = None
    query_text: Optional[str] = None
    query_intent: Optional[str] = None
    instruction: str = ""
    view_version: Optional[int] = None
    policy: Dict[str, Any] = Field(default_factory=dict)
    extraction_focus: List[str] = Field(default_factory=list)


class WorkResult(BaseModel):
    task_id: str
    task_type: str
    outcome: str
    used_gap_id: str
    summary: str
    produced_candidates: List[CandidateItem] = Field(default_factory=list)
    selected_candidate_ids: List[str] = Field(default_factory=list)
    dropped_candidate_ids: List[Dict[str, str]] = Field(default_factory=list)
    evidence_added: List[EvidenceItem] = Field(default_factory=list)
    coverage_claim: Dict[str, Any] = Field(default_factory=dict)
    signals: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    fail_reason: Optional[str] = None


class CandidateView(BaseModel):
    gap_id: str
    version: int
    candidate_ids: List[str]
    created_at: str


class WorkPlan(BaseModel):
    plan_id: str
    facet_task_id: str
    requirements: Dict[str, RequirementState] = Field(default_factory=dict)
    gaps: Dict[str, GapState] = Field(default_factory=dict)
    candidates: Dict[str, CandidateItem] = Field(default_factory=dict)
    evidence: Dict[str, EvidenceItem] = Field(default_factory=dict)
    tasks: Dict[str, WorkTask] = Field(default_factory=dict)
    views: Dict[str, List[CandidateView]] = Field(default_factory=dict)
    events: List[PlanEvent] = Field(default_factory=list)

    def log_event(
        self,
        action: str,
        rationale: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.events.append(
            PlanEvent(
                timestamp=datetime.utcnow().isoformat(),
                action=action,
                rationale=rationale,
                inputs=inputs or {},
                outputs=outputs or {},
            )
        )

    def create_view(self, gap_id: str, candidate_ids: List[str]) -> CandidateView:
        current = self.views.setdefault(gap_id, [])
        version = len(current) + 1
        view = CandidateView(
            gap_id=gap_id,
            version=version,
            candidate_ids=candidate_ids,
            created_at=datetime.utcnow().isoformat(),
        )
        current.append(view)
        return view

    def get_view(self, gap_id: str, version: int) -> Optional[CandidateView]:
        for view in self.views.get(gap_id, []):
            if view.version == version:
                return view
        return None


def new_requirement(requirement_type: str, description: str) -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=generate_id(),
        requirement_type=requirement_type,
        description=description,
        completion_rule={"min_evidence": 1},
    )


def new_work_task(
    *,
    task_type: str,
    facet_task_id: str,
    linked_gap_id: str,
    linked_requirement_id: str,
    strategy_tag: str,
    expected_yield: str,
) -> WorkTask:
    return WorkTask(
        task_id=generate_id(),
        task_type=task_type,
        facet_task_id=facet_task_id,
        linked_gap_id=linked_gap_id,
        linked_requirement_id=linked_requirement_id,
        strategy_tag=strategy_tag,
        expected_yield=expected_yield,
    )
