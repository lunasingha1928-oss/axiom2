from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON
from app.database import Base

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(64), index=True, nullable=False)
    risk_score = Column(Integer, nullable=False)  # 0-100
    risk_level = Column(String(32), nullable=False)  # low, medium, high
    status = Column(String(32), nullable=False)  # on-track, delayed, critical
    
    # Detailed breakdown components (raw value, normalized score, contribution)
    factors_json = Column(JSON, nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow, index=True)
