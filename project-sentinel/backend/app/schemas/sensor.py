from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class SensorReadingInput(BaseModel):
    asset_id: str
    project_id: Optional[str] = None
    sensor_type: str
    value: float
    unit: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class SensorBatchIngest(BaseModel):
    readings: List[SensorReadingInput]

class SensorIngestResponse(BaseModel):
    status: str = "success"
    ingested_count: int
    asset_id: str
    latest_health_score: float
    asset_status: str
    data_source: str = "simulated:data/sensors.csv"

class AssetHealthResponse(BaseModel):
    asset_id: str
    project_id: Optional[str] = None
    current_health_score: float = Field(..., ge=0.0, le=100.0)
    status: str = Field(..., description="healthy, warning, critical")
    vibration_rms: Optional[float] = None
    temperature_c: Optional[float] = None
    strain_microstrain: Optional[float] = None
    tilt_deg: Optional[float] = None
    anomaly_z_score: float = 0.0
    data_source: str = "simulated:data/sensors.csv"
    last_updated: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
