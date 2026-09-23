from app.routers.projects import router as projects_router
from app.routers.photos import router as photos_router
from app.routers.sensors import router as sensors_router
from app.routers.weather import router as weather_router
from app.routers.risk import router as risk_router
from app.routers.pipeline import router as pipeline_router
from app.routers.satellite import router as satellite_router

__all__ = [
    "projects_router",
    "photos_router",
    "sensors_router",
    "weather_router",
    "risk_router",
    "pipeline_router",
    "satellite_router",
]

