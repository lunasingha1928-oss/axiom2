from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text
from app.database import Base

class PhotoVerification(Base):
    __tablename__ = "photo_verifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(64), index=True, nullable=False)
    image_path = Column(String(512), nullable=False)
    filename = Column(String(255), nullable=False)
    
    # Model inference results
    model_version = Column(String(64), default="yolov8n")
    classifier_mode = Column(String(64), default="pretrained_yolov8_heuristic")  # clearly labeled
    label = Column(String(64), nullable=False)  # "completed_stage", "active_construction", "early_groundwork"
    confidence = Column(Float, nullable=False)  # overall model confidence float (0.0 to 1.0)
    completion_proxy_score = Column(Float, nullable=False)  # 0.0 to 100.0
    
    # Detailed vision outputs
    boxes = Column(JSON, nullable=False)  # List of {class, confidence, bbox: [x1, y1, x2, y2]}
    detected_classes = Column(JSON, nullable=False)  # Summary count of detected classes
    heuristic_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
