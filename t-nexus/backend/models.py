from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from backend.database import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    SUPPORT = "support"
    VIEWER = "viewer"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), default="")
    role = Column(String(50), default=UserRole.SUPPORT)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    telegram_user_id = Column(String(100), index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, default=0)
    request_type = Column(String(50))
    is_successful = Column(Boolean, default=True)
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    content = Column(Text)
    is_from_user = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")

class LLMRequest(Base):
    __tablename__ = "llm_requests"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    generation_time_ms = Column(Float, default=0)
    response_length = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    rating = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class RAGQuery(Base):
    __tablename__ = "rag_queries"
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)
    relevance_score = Column(Float, default=0)
    response_time_ms = Column(Float, default=0)
    sources_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class ManualReviewItem(Base):
    __tablename__ = "manual_review_items"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    question = Column(Text)
    model_response = Column(Text)
    admin_response = Column(Text, default="")
    status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Incident(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(50), unique=True)
    impact = Column(Text)
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    message = Column(Text)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserQuery(Base):
    __tablename__ = "user_queries"
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)
    query_count = Column(Integer, default=1)
    is_completed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)