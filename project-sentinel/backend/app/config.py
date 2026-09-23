import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = DATA_DIR / "models"
SATELLITE_CACHE_DIR = DATA_DIR / "satellite_cache"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SATELLITE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    PROJECT_NAME: str = "Project Sentinel Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    DATA_DIR: Path = DATA_DIR
    UPLOADS_DIR: Path = UPLOADS_DIR
    MODELS_DIR: Path = MODELS_DIR
    SATELLITE_CACHE_DIR: Path = SATELLITE_CACHE_DIR
    SATELLITE_CACHE_REFRESH_HOURS: int = 24
    
    # Copernicus Data Space Ecosystem (CDSE) Sentinel Hub credentials
    CDSE_CLIENT_ID: str | None = os.getenv("CDSE_CLIENT_ID")
    CDSE_CLIENT_SECRET: str | None = os.getenv("CDSE_CLIENT_SECRET")
    ENABLE_SATELLITE_SCHEDULER: bool = os.getenv("ENABLE_SATELLITE_SCHEDULER", "false").lower() in ("true", "1", "yes")
    
    # Database configuration (PostgreSQL target, SQLite zero-config default)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{DATA_DIR.as_posix()}/sentinel_v2.db"
    )
    SYNC_DATABASE_URL: str = os.getenv(
        "SYNC_DATABASE_URL",
        f"sqlite:///{DATA_DIR.as_posix()}/sentinel_v2.db"
    )
    
    # YOLOv8 Vision Model
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    
    # Risk weights (documented defaults totaling 1.0)
    WEIGHT_PROGRESS_GAP: float = 0.30
    WEIGHT_BUDGET_VARIANCE: float = 0.25
    WEIGHT_SENSOR_HEALTH: float = 0.15
    WEIGHT_RAINFALL_RISK: float = 0.15
    WEIGHT_DELAY_MONTHS: float = 0.15
    
    # Weather configuration
    OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    HEAVY_RAINFALL_THRESHOLD_MM: float = 150.0  # 30-day baseline threshold for high flood/monsoon risk
    
    # Sensor telemetry thresholds
    VIBRATION_ALERT_RMS_MM_S: float = 4.5  # ISO 10816 Class II vibration threshold
    VIBRATION_CRITICAL_RMS_MM_S: float = 7.1
    STRAIN_ALERT_MICROSTRAIN: float = 800.0
    STRAIN_CRITICAL_MICROSTRAIN: float = 1400.0

settings = Settings()
