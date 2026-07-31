"""Pydantic request/response models for the AI service HTTP API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SummarizeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    max_chars: int = Field(1200, ge=200, le=4000)


class SummarizeResponse(BaseModel):
    provider: str
    model: str
    summary: str


class MemoryUpsertRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    workspace_path: str = Field("", max_length=4096)
    source_phase: str = Field("", max_length=128)
    source_provider: str = Field("", max_length=128)
    source_model: str = Field("", max_length=256)
    outcome_label: str = Field("unknown", max_length=32)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        if len(v) > 20:
            raise ValueError("tags must contain at most 20 items")
        for tag in v:
            if len(tag) > 64:
                raise ValueError("each tag must be at most 64 characters")
        return v


class MemoryUpsertResponse(BaseModel):
    id: int


class MemorySearchRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    workspace_path: str = Field("", max_length=4096)
    include_cross_session: bool = True
    current_phase: str = Field("", max_length=128)


class MemoryItem(BaseModel):
    id: int
    session_id: str
    score: float
    content: str
    tags: List[str]
    source_phase: str
    source_provider: str
    source_model: str
    outcome_label: str
    created_at: str
    score_breakdown: Dict[str, float]


class MemorySearchResponse(BaseModel):
    items: List[MemoryItem]


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    task_type: str = Field("default", max_length=32)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    provider: str
    model: str
    output: str


class SelfConsistencyRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    n: int = Field(3, ge=1, le=7)
    temperature: float = Field(0.7, ge=0.0, le=1.0)


class SelfConsistencyResponse(BaseModel):
    provider: str
    model: str
    output: str
    samples: int


class ReflectRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    workspace_path: str = Field("", max_length=4096)
    task: str = Field(..., min_length=1)
    transcript: str = Field(..., min_length=1)
    outcome: str = Field("unknown", max_length=32)


class ReflectResponse(BaseModel):
    provider: str
    model: str
    lesson: str


class InsightsRequest(BaseModel):
    workspace_path: str = Field(..., min_length=1, max_length=4096)
    min_lessons: int = Field(10, ge=1, le=100)


class InsightItem(BaseModel):
    text: str
    id: int


class InsightsResponse(BaseModel):
    provider: str
    model: str
    insights: List[InsightItem]
    lessons_consolidated: int
