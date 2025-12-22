from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from src.t_nexus.backend.database import get_db
from src.t_nexus.backend.models import ManualReviewItem as ManualReviewModel
from src.t_nexus.backend.schemas import ManualReviewItemResponse, ManualReviewUpdate
from src.t_nexus.backend.auth import get_current_user
from src.t_nexus.backend.services.placeholder_data import get_manual_review_placeholder

router = APIRouter(prefix="/api", tags=["Manual Review"])

def ensure_placeholder_data(db: Session):
    count = db.query(ManualReviewModel).count()
    if count == 0:
        placeholder_items = get_manual_review_placeholder()
        for item in placeholder_items:
            db_item = ManualReviewModel(
                id=item["id"],
                title=item["title"],
                question=item["question"],
                model_response=item["modelResponse"],
                admin_response=item["adminResponse"]
            )
            db.add(db_item)
        db.commit()

@router.get("/manual-review", response_model=List[ManualReviewItemResponse])
def list_manual_review_items(db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    ensure_placeholder_data(db)
    items = db.query(ManualReviewModel).order_by(ManualReviewModel.id.desc()).all()
    return [
        ManualReviewItemResponse(
            id=item.id,
            title=item.title,
            question=item.question,
            modelResponse=item.model_response,
            adminResponse=item.admin_response or ""
        )
        for item in items
    ]

@router.get("/manual-review/{item_id}", response_model=ManualReviewItemResponse)
def get_manual_review_item(item_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    item = db.query(ManualReviewModel).filter(ManualReviewModel.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return ManualReviewItemResponse(
        id=item.id,
        title=item.title,
        question=item.question,
        modelResponse=item.model_response,
        adminResponse=item.admin_response or ""
    )

@router.put("/manual-review/{item_id}", response_model=ManualReviewItemResponse)
def update_manual_review_item(
    item_id: int,
    payload: ManualReviewUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    item = db.query(ManualReviewModel).filter(ManualReviewModel.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    item.admin_response = payload.adminResponse
    item.status = "reviewed"
    db.commit()
    db.refresh(item)
    return ManualReviewItemResponse(
        id=item.id,
        title=item.title,
        question=item.question,
        modelResponse=item.model_response,
        adminResponse=item.admin_response or ""
    )