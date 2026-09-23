from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Index
from app.database import Base

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), index=True, nullable=False)  # e.g. "ASSET-RL-2217-TRK1"
    project_id = Column(String(64), index=True, nullable=True)
    sensor_type = Column(String(64), index=True, nullable=False)  # "vibration", "temperature", "strain", "tilt"
    value = Column(Float, nullable=False)
    unit = Column(String(32), nullable=False)  # "mm/s", "deg_C", "microstrain", "deg"
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_sensor_asset_type_time", "asset_id", "sensor_type", "timestamp"),
    )

class AssetHealthLog(Base):
    __tablename__ = "asset_health_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), index=True, nullable=False)
    project_id = Column(String(64), index=True, nullable=True)
    health_score = Column(Float, nullable=False)  # 0.0 to 100.0
    status = Column(String(32), nullable=False)  # "healthy", "warning", "critical"
    vibration_rms = Column(Float, nullable=True)
    temperature_c = Column(Float, nullable=True)
    strain_microstrain = Column(Float, nullable=True)
    tilt_deg = Column(Float, nullable=True)
    anomaly_z_score = Column(Float, default=0.0)
    details = Column(JSON, nullable=True)
    computed_at = Column(DateTime, default=datetime.utcnow, index=True)
