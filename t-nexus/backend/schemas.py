from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: Optional[str] = ""

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    
    class Config:
        from_attributes = True

class KPI(BaseModel):
    label: str
    value: str
    delta: str

class ConversationStats(BaseModel):
    total: int
    avgDuration: str

class TrafficData(BaseModel):
    labels: List[str]
    values: List[int]

class RatingDistribution(BaseModel):
    positive: int
    neutral: int
    negative: int

class IncidentResponse(BaseModel):
    id: str
    impact: str
    status: str
    timestamp: str

class OverviewResponse(BaseModel):
    period: str
    kpis: List[KPI]
    conversations: ConversationStats
    successRate: float
    traffic: TrafficData
    ratings: RatingDistribution
    incidents: List[IncidentResponse]

class LLMMetrics(BaseModel):
    requests: int
    avgTime: float
    avgTokens: int
    avgResponseLength: int
    labels: List[str]
    positive: List[int]
    negative: List[int]

class SlowQuery(BaseModel):
    label: str
    duration: str

class HippoRAGMetrics(BaseModel):
    relevance: float
    topQueries: List[str]
    slowQueries: List[SlowQuery]

class TagWeight(BaseModel):
    label: str
    weight: int

class UserMetrics(BaseModel):
    topQueries: List[str]
    completionRate: float
    tagCloud: List[TagWeight]

class MetricsResponse(BaseModel):
    llm: LLMMetrics
    hipporag: HippoRAGMetrics
    user: UserMetrics
    popularQueries: List[str]

class ManualReviewItemResponse(BaseModel):
    id: int
    title: str
    question: str
    modelResponse: str
    adminResponse: str
    
    class Config:
        from_attributes = True

class ManualReviewUpdate(BaseModel):
    adminResponse: str

class NotificationResponse(BaseModel):
    id: int
    title: str
    message: str
    ts: str
    
    class Config:
        from_attributes = True

class SettingsUpdate(BaseModel):
    reportingWindow: Optional[str] = "Last 7 days"
    telegramAlerts: Optional[bool] = True
    rabbitmqHealth: Optional[bool] = True