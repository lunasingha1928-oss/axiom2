from typing import List, Optional
from pydantic import BaseModel, Field

class WeatherPoint(BaseModel):
    date: str
    rainfall_mm: float
    temp_max_c: Optional[float] = None
    temp_min_c: Optional[float] = None
    temp_mean_c: Optional[float] = None

class WeatherResponse(BaseModel):
    project_id: str
    latitude: float
    longitude: float
    rainfall_30d_total_mm: float
    rainfall_7d_forecast_mm: float
    rainfall_risk_score: float = Field(..., ge=0.0, le=100.0)
    readings: List[WeatherPoint] = []
