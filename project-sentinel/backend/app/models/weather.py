from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Index
from app.database import Base

class WeatherReading(Base):
    __tablename__ = "weather_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(64), index=True, nullable=False)
    date = Column(String(32), nullable=False, index=True)  # YYYY-MM-DD
    rainfall_mm = Column(Float, default=0.0)
    temp_max_c = Column(Float, nullable=True)
    temp_min_c = Column(Float, nullable=True)
    temp_mean_c = Column(Float, nullable=True)
    source = Column(String(32), default="open-meteo")
    fetched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_weather_proj_date", "project_id", "date", unique=True),
    )
