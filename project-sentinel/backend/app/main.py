import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db, AsyncSessionLocal
from app.routers import (
    projects_router,
    photos_router,
    sensors_router,
    weather_router,
    risk_router,
    pipeline_router,
    satellite_router,
)
from app.routers.projects import seed_default_projects

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentinel-backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Project Sentinel Database...")
    await init_db()
    
    # Auto-seed production registers if table is empty
    async with AsyncSessionLocal() as session:
        try:
            seeded = await seed_default_projects(session)
            if seeded > 0:
                logger.info(f"Database initialized with {seeded} monitored project records.")
        except Exception as e:
            logger.warning(f"Project auto-seed notice: {e}")

    # Optional Daily Satellite Cache Refresh Scheduler
    scheduler = None
    if settings.ENABLE_SATELLITE_SCHEDULER:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from scripts.refresh_satellite_cache import refresh_all_satellite_imagery
            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                refresh_all_satellite_imagery,
                "interval",
                hours=settings.SATELLITE_CACHE_REFRESH_HOURS,
                id="satellite_cache_refresh",
                name="Refresh Sentinel-2 satellite cache",
            )
            scheduler.start()
            logger.info(f"Satellite background scheduler active (refresh interval: {settings.SATELLITE_CACHE_REFRESH_HOURS}h).")
        except Exception as ex:
            logger.warning(f"Could not start satellite scheduler: {ex}")
    else:
        logger.info("Satellite background scheduler is disabled (ENABLE_SATELLITE_SCHEDULER=false).")

    yield

    if scheduler and scheduler.running:
        scheduler.shutdown()
        logger.info("Satellite background scheduler stopped.")
    logger.info("Project Sentinel Backend shutting down.")


app = FastAPI(
    title="Project Sentinel Backend & ML Pipeline",
    description="Real backend API for infrastructure monitoring, field-photo vision verification (YOLOv8), Sentinel-2 NDBI change detection, XGBoost + SHAP predictive risk scoring, linear asset anomaly detection, and MoSPI PAIMANA ingestion.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for Uploads and Satellite Cache
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")
app.mount("/satellite-cache", StaticFiles(directory=str(settings.SATELLITE_CACHE_DIR)), name="satellite-cache")

# Include Routers under /api
app.include_router(projects_router, prefix=settings.API_V1_STR)
app.include_router(photos_router, prefix=settings.API_V1_STR)
app.include_router(sensors_router, prefix=settings.API_V1_STR)
app.include_router(weather_router, prefix=settings.API_V1_STR)
app.include_router(risk_router, prefix=settings.API_V1_STR)
app.include_router(pipeline_router, prefix=settings.API_V1_STR)
app.include_router(satellite_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "service": "Project Sentinel Backend API",
        "version": settings.VERSION,
        "status": "operational",
        "endpoints": {
            "projects": "/api/projects",
            "risk_breakdown": "/api/projects/{id}/risk-breakdown",
            "photo_classify": "/api/photos/classify",
            "sensor_ingest": "/api/sensors/ingest",
            "asset_health": "/api/assets/{id}/health",
            "weather": "/api/weather/{project_id}",
            "paimana_ingest": "/api/pipeline/ingest-paimana",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def healthcheck():
    return {"status": "healthy"}
