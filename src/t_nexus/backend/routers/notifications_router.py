from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta

from src.t_nexus.backend.database import get_db
from src.t_nexus.backend.models import Notification as NotificationModel
from src.t_nexus.backend.schemas import NotificationResponse
from src.t_nexus.backend.auth import get_current_user
from src.t_nexus.backend.services.placeholder_data import get_notifications_placeholder

router = APIRouter(prefix="/api", tags=["Notifications"])

def ensure_placeholder_notifications(db: Session):
    count = db.query(NotificationModel).count()
    if count == 0:
        placeholder = get_notifications_placeholder()
        for item in placeholder:
            db_item = NotificationModel(
                id=item["id"],
                title=item["title"],
                message=item["message"]
            )
            db.add(db_item)
        db.commit()

def format_timestamp(created_at: datetime) -> str:
    now = datetime.utcnow()
    diff = now - created_at
    if diff < timedelta(minutes=1):
        return "just now"
    elif diff < timedelta(hours=1):
        return f"{int(diff.total_seconds() // 60)}m ago"
    elif diff < timedelta(days=1):
        return f"{int(diff.total_seconds() // 3600)}h ago"
    else:
        return created_at.strftime("%b %d")

@router.get("/notifications", response_model=List[NotificationResponse])
def list_notifications(db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    ensure_placeholder_notifications(db)
    notifications = db.query(NotificationModel).order_by(NotificationModel.created_at.desc()).limit(20).all()
    placeholder = get_notifications_placeholder()
    ts_map = {item["id"]: item["ts"] for item in placeholder}
    return [
        NotificationResponse(
            id=n.id,
            title=n.title,
            message=n.message,
            ts=ts_map.get(n.id, format_timestamp(n.created_at))
        )
        for n in notifications
    ]

@router.post("/notifications/{notification_id}/read")
def mark_notification_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    notification = db.query(NotificationModel).filter(NotificationModel.id == notification_id).first()
    if notification:
        notification.is_read = True
        db.commit()
    return {"status": "ok"}