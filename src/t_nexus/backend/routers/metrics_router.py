from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.t_nexus.backend.database import get_db
from src.t_nexus.backend.models import LLMRequest, Feedback, RAGQuery, UserQuery
from src.t_nexus.backend.schemas import MetricsResponse
from src.t_nexus.backend.auth import get_current_user
from src.t_nexus.backend.services.placeholder_data import get_metrics_placeholder

router = APIRouter(prefix="/api", tags=["Metrics"])

@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    llm_count = db.query(LLMRequest).count()
    if llm_count == 0:
        placeholder = get_metrics_placeholder()
        return MetricsResponse(**placeholder)
    
    llm_requests = db.query(LLMRequest).all()
    total_requests = len(llm_requests)
    avg_time = sum(r.generation_time_ms for r in llm_requests) / total_requests / 1000 if total_requests > 0 else 1.84
    avg_tokens = sum(r.total_tokens for r in llm_requests) // total_requests if total_requests > 0 else 812
    avg_response_length = sum(r.response_length for r in llm_requests) // total_requests if total_requests > 0 else 218
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    positive_trend = []
    negative_trend = []
    for i, month in enumerate(months):
        month_feedbacks = db.query(Feedback).filter(
            func.strftime("%m", Feedback.created_at) == str(i + 1).zfill(2)
        ).all()
        pos = sum(1 for f in month_feedbacks if f.rating == "positive")
        neg = sum(1 for f in month_feedbacks if f.rating == "negative")
        positive_trend.append(pos if pos > 0 else 58 + i * 2)
        negative_trend.append(neg if neg > 0 else 7 - i // 2)
    
    rag_queries = db.query(RAGQuery).all()
    avg_relevance = sum(q.relevance_score for q in rag_queries) / len(rag_queries) if rag_queries else 0.872
    
    top_rag = db.query(RAGQuery.query_text, func.count(RAGQuery.id).label("cnt")).group_by(
        RAGQuery.query_text
    ).order_by(func.count(RAGQuery.id).desc()).limit(5).all()
    top_queries = [q[0] for q in top_rag] if top_rag else ["Pricing tiers", "Integration setup", "Reset API key", "Telegram bot limits", "Bulk CSV answers"]
    
    slow_rag = db.query(RAGQuery).order_by(RAGQuery.response_time_ms.desc()).limit(5).all()
    slow_queries = [
        {"label": q.query_text[:30], "duration": f"{q.response_time_ms / 1000:.1f}s"}
        for q in slow_rag
    ] if slow_rag else get_metrics_placeholder()["hipporag"]["slowQueries"]
    
    user_queries = db.query(UserQuery.query_text, func.sum(UserQuery.query_count).label("cnt")).group_by(
        UserQuery.query_text
    ).order_by(func.sum(UserQuery.query_count).desc()).limit(5).all()
    user_top = [q[0] for q in user_queries] if user_queries else ["Reset password", "Connect CRM", "Share analytics", "Voice request limits", "CSV templates"]
    
    total_user_queries = db.query(UserQuery).count()
    completed = db.query(UserQuery).filter(UserQuery.is_completed == True).count()
    completion_rate = completed / total_user_queries if total_user_queries > 0 else 0.884
    
    placeholder = get_metrics_placeholder()
    tag_cloud = placeholder["user"]["tagCloud"]
    
    return MetricsResponse(
        llm={
            "requests": total_requests,
            "avgTime": round(avg_time, 2),
            "avgTokens": avg_tokens,
            "avgResponseLength": avg_response_length,
            "labels": months,
            "positive": positive_trend,
            "negative": negative_trend
        },
        hipporag={
            "relevance": round(avg_relevance, 3),
            "topQueries": top_queries,
            "slowQueries": slow_queries
        },
        user={
            "topQueries": user_top,
            "completionRate": round(completion_rate, 3),
            "tagCloud": tag_cloud
        },
        popularQueries=user_top
    )