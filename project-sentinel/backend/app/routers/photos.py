import os
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.config import settings
from app.database import get_db
from app.models.photo import PhotoVerification
from app.models.project import Project
from app.schemas.photo import PhotoClassifyResponse, BoundingBox
from app.ml.photo_verifier import PhotoVerifier

router = APIRouter(prefix="/photos", tags=["photos"])


@router.post("/classify", response_model=PhotoClassifyResponse)
async def classify_photo(
    file: UploadFile = File(..., description="Site photograph image file (JPEG, PNG)"),
    project_id: str = Form(..., description="Project ID, e.g. PS-RD-1042"),
    reported_progress_pct: Optional[float] = Form(None, description="Optional reported progress percentage"),
    update_project: bool = Form(False, description="Whether to update verified progress with estimate"),
    db: AsyncSession = Depends(get_db),
):
    """
    Real YOLOv8 Field-Photo Classifier (Static Dataset / Fine-Tuned).
    
    1. Runs real YOLOv8 classification (completed vs incomplete).
    2. Maps model confidence to verified completion estimate (75-100% if completed, 20-45% if incomplete).
    3. Computes photo_gap = abs(reported_progress_pct - photo_verified_estimate).
    4. Persists record and returns response with data_source: 'real:data/photos/'.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename provided")

    # Get reported progress from project if not provided
    if reported_progress_pct is None:
        stmt = select(Project).where(Project.id == project_id)
        proj = (await db.execute(stmt)).scalar_one_or_none()
        reported_progress_pct = proj.reported_progress if proj else 65.0

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{project_id}_{timestamp_str}_{Path(file.filename).name}"
    save_path = settings.UPLOADS_DIR / safe_filename

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        result = PhotoVerifier.classify_image(
            image_path=str(save_path),
            project_id=project_id,
            reported_progress_pct=reported_progress_pct,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLOv8 vision inference failed: {e}")

    # Store verification in database
    photo_record = PhotoVerification(
        project_id=project_id,
        image_path=str(save_path),
        filename=safe_filename,
        model_version="yolov8_construction_cls",
        classifier_mode=result["classifier_mode"],
        label=result["label"],
        confidence=result["confidence"],
        completion_proxy_score=result["photo_verified_estimate"],
        boxes=[],
        detected_classes={"label": result["label"]},
        heuristic_notes=result["heuristic_notes"],
        created_at=datetime.utcnow(),
    )
    db.add(photo_record)

    if update_project:
        stmt = select(Project).where(Project.id == project_id)
        proj = (await db.execute(stmt)).scalar_one_or_none()
        if proj:
            proj.verified_progress = result["photo_verified_estimate"]
            proj.last_verified = datetime.utcnow().strftime("%Y-%m-%d")

    await db.commit()

    return PhotoClassifyResponse(
        project_id=project_id,
        filename=safe_filename,
        label=result["label"],
        confidence=result["confidence"],
        photo_verified_estimate=result["photo_verified_estimate"],
        photo_gap=result["photo_gap"],
        reported_progress_pct=result["reported_progress_pct"],
        classifier_mode=result["classifier_mode"],
        heuristic_notes=result["heuristic_notes"],
        data_source="real:data/photos/",
        timestamp=photo_record.created_at,
    )


@router.get("/{project_id}", response_model=List[PhotoClassifyResponse])
async def get_project_photos(project_id: str, db: AsyncSession = Depends(get_db)):
    """Returns history of all verified field photos for a project."""
    stmt = (
        select(PhotoVerification)
        .where(PhotoVerification.project_id == project_id)
        .order_by(desc(PhotoVerification.created_at))
        .limit(20)
    )
    records = (await db.execute(stmt)).scalars().all()

    return [
        PhotoClassifyResponse(
            project_id=r.project_id,
            filename=r.filename,
            label=r.label,
            confidence=r.confidence,
            photo_verified_estimate=r.completion_proxy_score,
            photo_gap=round(abs(65.0 - r.completion_proxy_score), 1),
            reported_progress_pct=65.0,
            classifier_mode=r.classifier_mode,
            heuristic_notes=r.heuristic_notes,
            data_source="real:data/photos/",
            timestamp=r.created_at,
        )
        for r in records
    ]
