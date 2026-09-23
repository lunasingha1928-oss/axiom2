from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates")

class PhotoClassifyResponse(BaseModel):
    project_id: str
    filename: str
    label: str = Field(..., description="Classification: completed vs incomplete")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Real YOLOv8 model confidence score")
    photo_verified_estimate: float = Field(..., ge=0.0, le=100.0, description="Estimated completion percentage")
    photo_gap: float = Field(..., ge=0.0, description="Absolute gap between reported and verified")
    reported_progress_pct: float = Field(..., description="Officially reported progress")
    classifier_mode: str = "trained_yolov8_cls"
    heuristic_notes: Optional[str] = None
    data_source: str = "real:data/photos/"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
