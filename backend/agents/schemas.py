from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from core.domains import CandidateItem


class SearchWorkResult(BaseModel):
    """Structured output from SearchAgent."""

    candidates: List[CandidateItem] = Field(description="筛选后的候选列表")
    query_used: str = Field(description="实际使用的搜索查询")
    relevance_notes: str = Field(description="对搜索结果质量的整体评价")


class ExtractedEvidence(BaseModel):
    """A single piece of evidence extracted from fetched content."""

    claim: str = Field(description="具体的声明/数据点/结论")
    citation_text: str = Field(description="直接引用的原文段落")
    source_url: str = Field(description="证据来源 URL")
    relevance_to_gap: str = Field(description="与 gap 的关联说明")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度评分 0.0-1.0")


class FetchWorkResult(BaseModel):
    """Structured output from FetchAgent."""

    selected_candidate_ids: List[str] = Field(description="实际选中并抓取的候选 ID 列表")
    dropped: List[dict] = Field(default_factory=list, description="被放弃的候选及原因，如 [{\"id\": \"xxx\", \"reason\": \"...\"}]")
    evidence: List[ExtractedEvidence] = Field(description="提取的证据列表")
    coverage_assessment: str = Field(description="对 gap 覆盖程度的评估")
