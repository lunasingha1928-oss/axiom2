from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.project import Project
from app.schemas.weather import WeatherResponse, WeatherPoint
from app.services.weather_service import WeatherService

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/{project_id}", response_model=WeatherResponse)
async def get_project_weather(project_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns precipitation and meteorological history for a project's location
    joined strictly from static data/rainfall.csv (no live API calls).
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalar_one_or_none()

    p_code = project.project_code if project else None
    lat = project.latitude if project and project.latitude else 20.5937
    lng = project.longitude if project and project.longitude else 78.9629

    result = WeatherService.get_project_rainfall(project_id, p_code)

    return WeatherResponse(
        project_id=project_id,
        latitude=lat,
        longitude=lng,
        rainfall_30d_total_mm=result["rainfall_30d_total_mm"],
        rainfall_7d_forecast_mm=0.0,
        rainfall_risk_score=result["rainfall_exposure"],
        readings=[WeatherPoint(**p) for p in result["readings"]],
    )
