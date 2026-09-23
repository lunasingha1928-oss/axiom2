from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey
from app.database import Base

class SatelliteImageCache(Base):
    """
    Persistent cache for Copernicus Sentinel-2 Level-2A before/after true-color
    satellite imagery, actual scene dates, cloud cover metrics, and fetch status.
    """
    __tablename__ = "satellite_image_cache"

    project_id = Column(String(64), ForeignKey("projects.id"), primary_key=True, index=True)
    before_image_path = Column(String(255), nullable=True)
    after_image_path = Column(String(255), nullable=True)
    before_scene_date = Column(String(32), nullable=True)
    after_scene_date = Column(String(32), nullable=True)
    cloud_cover_before = Column(Float, nullable=True)
    cloud_cover_after = Column(Float, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    fetch_status = Column(String(32), nullable=False, default="ok")  # "ok", "stale", "failed"
    last_error = Column(Text, nullable=True)
