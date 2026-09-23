"""
Project Sentinel — Real Sentinel-2 Satellite Imagery Service (CDSE / Sentinel Hub)

Fetches real Level-2A before/after true-color satellite imagery from the
Copernicus Data Space Ecosystem (CDSE) using the official sentinelhub Python SDK.
"""

import os
import math
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

from app.config import settings

logger = logging.getLogger("sentinel-satellite-service")

class SatelliteImageryUnavailable(Exception):
    """Raised when Sentinel-2 imagery cannot be fetched or processed from CDSE."""
    pass


EVALSCRIPT_TRUE_COLOR = """//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: {
      bands: 3,
      sampleType: "AUTO"
    }
  };
}

function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
"""


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Converts WGS84 lat/lon to web mercator tile coordinate at given zoom."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Converts web mercator tile coordinate to top-left WGS84 lat/lon."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def calculate_aoi_bbox(lat: float, lng: float, zoom: int = 15):
    """
    Computes a WGS84 BBox for the project site matching the 3x2 tile grid extent
    used in fetch_real_satellite_images.py.
    """
    from sentinelhub import BBox, CRS

    cx, cy = deg2num(lat, lng, zoom)
    cols, rows = 3, 2
    min_tx = cx - (cols // 2)
    max_tx = min_tx + cols
    min_ty = cy - (rows // 2)
    max_ty = min_ty + rows

    nw_lat, nw_lng = num2deg(min_tx, min_ty, zoom)
    se_lat, se_lng = num2deg(max_tx, max_ty, zoom)

    min_lon = min(nw_lng, se_lng)
    max_lon = max(nw_lng, se_lng)
    min_lat = min(nw_lat, se_lat)
    max_lat = max(nw_lat, se_lat)

    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)


def get_cdse_sh_config():
    """
    Returns an SHConfig initialized with Copernicus Data Space Ecosystem endpoints
    and credentials from environment variables.
    """
    from sentinelhub import SHConfig

    client_id = settings.CDSE_CLIENT_ID
    client_secret = settings.CDSE_CLIENT_SECRET

    if not client_id or not client_secret:
        raise SatelliteImageryUnavailable(
            "CDSE credentials missing. Please set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in environment or backend/.env."
        )

    config = SHConfig()
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_client_id = client_id.strip()
    config.sh_client_secret = client_secret.strip()
    return config


def _extract_cloud_cover(feature: Dict[str, Any]) -> float:
    """Extracts cloud cover percentage safely from STAC feature properties."""
    props = feature.get("properties", {})
    for key in ("eo:cloud_cover", "cloudCover", "cloud_cover", "cloudCoverage"):
        val = props.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 100.0


def _extract_datetime(feature: Dict[str, Any]) -> str:
    """Extracts acquisition datetime safely from STAC feature properties."""
    props = feature.get("properties", {})
    dt = props.get("datetime")
    if dt:
        return str(dt)
    return datetime.utcnow().isoformat() + "Z"


def _search_lowest_cloud_scene(
    catalog,
    bbox,
    center_dt: datetime,
    is_interval: bool = False,
    window_days: int = 14,
    max_cloud_threshold: float = 30.0,
) -> Tuple[str, float]:
    """
    Searches Catalog for Sentinel-2 L2A scenes with cloud cover <= max_cloud_threshold.
    If no scene is found within window_days, widens search once to 30 days.
    Returns (scene_datetime_iso, cloud_cover_pct).
    """
    from sentinelhub import DataCollection

    collection_id = "sentinel-2-l2a"

    def run_query(days: int):
        if is_interval:
            start_t = center_dt - timedelta(days=days)
            end_t = center_dt
        else:
            start_t = center_dt - timedelta(days=days)
            end_t = center_dt + timedelta(days=days)

        start_str = start_t.strftime("%Y-%m-%dT00:00:00Z")
        end_str = end_t.strftime("%Y-%m-%dT23:59:59Z")

        try:
            iterator = catalog.search(
                collection=collection_id,
                bbox=bbox,
                time=(start_str, end_str),
            )
            features = list(iterator)
            return features, start_str, end_str
        except Exception as err:
            logger.warning(f"Catalog search error for range ({start_str} - {end_str}): {err}")
            raise

    # Phase 1: ±14 day window
    features, s_str, e_str = run_query(window_days)
    valid_scenes = []
    for f in features:
        cc = _extract_cloud_cover(f)
        if cc <= max_cloud_threshold:
            valid_scenes.append((f, cc))

    # Phase 2: Widen to ±30 days if no scene under threshold
    if not valid_scenes and window_days < 30:
        logger.info(f"No scene <= {max_cloud_threshold}% in ±{window_days}d window, widening to ±30d...")
        features, s_str, e_str = run_query(30)
        for f in features:
            cc = _extract_cloud_cover(f)
            if cc <= max_cloud_threshold:
                valid_scenes.append((f, cc))

    if not valid_scenes:
        if features:
            # Fall back to the lowest cloud scene available in the 30-day window
            features.sort(key=lambda x: _extract_cloud_cover(x))
            best = features[0]
            cc = _extract_cloud_cover(best)
            dt = _extract_datetime(best)
            logger.info(f"Using lowest available scene in window ({cc:.1f}% cloud cover).")
            return dt, cc
        raise SatelliteImageryUnavailable(
            f"No Sentinel-2 L2A scenes found near {center_dt.strftime('%Y-%m-%d')} within 30-day window."
        )

    # Sort valid scenes by lowest cloud cover
    valid_scenes.sort(key=lambda x: x[1])
    best_feature, best_cc = valid_scenes[0]
    return _extract_datetime(best_feature), best_cc


def _download_scene_image(
    config,
    bbox,
    scene_datetime_str: str,
    output_path: Path,
    width: int = 768,
    height: int = 512,
):
    """
    Renders and saves true-color JPEG via SentinelHub Process API for the selected scene.
    """
    from sentinelhub import SentinelHubRequest, DataCollection, MimeType

    try:
        # Parse timestamp to create a 24-hour bracket around scene
        scene_dt = datetime.fromisoformat(scene_datetime_str.replace("Z", "+00:00"))
        time_from = (scene_dt - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
        time_to = (scene_dt + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        time_from = scene_datetime_str[:10] + "T00:00:00Z"
        time_to = scene_datetime_str[:10] + "T23:59:59Z"

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_TRUE_COLOR,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
                time_interval=(time_from, time_to),
                mosaicking_order="leastCC",
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.JPG)
        ],
        bbox=bbox,
        size=(width, height),
        config=config,
    )

    try:
        data = request.get_data()
        if not data:
            raise SatelliteImageryUnavailable("Process API returned empty response data.")

        img_data = data[0]
        if isinstance(img_data, np.ndarray):
            im = Image.fromarray(img_data)
            im.save(output_path, "JPEG", quality=95)
        elif isinstance(img_data, bytes):
            output_path.write_bytes(img_data)
        elif isinstance(img_data, Image.Image):
            img_data.save(output_path, "JPEG", quality=95)
        else:
            raise SatelliteImageryUnavailable(f"Unsupported image data format: {type(img_data)}")
    except Exception as e:
        raise SatelliteImageryUnavailable(f"Failed to download Sentinel-2 image: {e}") from e


def fetch_before_after_images(
    project_id: str,
    lat: float,
    lng: float,
    project_start_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main service function:
    1. Authenticates to CDSE Sentinel Hub.
    2. Searches catalog for lowest-cloud before scene (around project_start_date)
       and lowest-cloud after scene (recent 14 days).
    3. Renders true-color JPEG for project AOI bounding box.
    4. Saves to SATELLITE_CACHE_DIR/{project_id}_before.jpg and _after.jpg.
    5. Returns real metadata (paths, dates, cloud cover percentages).
    """
    try:
        config = get_cdse_sh_config()
    except SatelliteImageryUnavailable:
        raise
    except Exception as err:
        raise SatelliteImageryUnavailable(f"Authentication setup failed: {err}") from err

    try:
        from sentinelhub import SentinelHubCatalog
        catalog = SentinelHubCatalog(config=config)
        bbox = calculate_aoi_bbox(lat, lng)

        # 1. Determine baseline date
        if project_start_date:
            try:
                before_center = datetime.strptime(project_start_date[:10], "%Y-%m-%d")
            except Exception:
                before_center = datetime.utcnow() - timedelta(days=365)
        else:
            before_center = datetime.utcnow() - timedelta(days=365)

        now = datetime.utcnow()

        # 2. Search catalog for before scene
        logger.info(f"[{project_id}] Searching before scene around {before_center.strftime('%Y-%m-%d')}...")
        before_scene_dt, cloud_before = _search_lowest_cloud_scene(
            catalog=catalog,
            bbox=bbox,
            center_dt=before_center,
            is_interval=False,
            window_days=14,
        )

        # 3. Search catalog for after scene (last 14 days from today)
        logger.info(f"[{project_id}] Searching after scene within last 14 days from {now.strftime('%Y-%m-%d')}...")
        after_scene_dt, cloud_after = _search_lowest_cloud_scene(
            catalog=catalog,
            bbox=bbox,
            center_dt=now,
            is_interval=True,
            window_days=14,
        )

        # 4. Save images to cache directory
        cache_dir = settings.SATELLITE_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        before_path = cache_dir / f"{project_id}_before.jpg"
        after_path = cache_dir / f"{project_id}_after.jpg"

        logger.info(f"[{project_id}] Downloading before scene ({before_scene_dt[:10]}, {cloud_before:.1f}% cloud)...")
        _download_scene_image(config, bbox, before_scene_dt, before_path)

        logger.info(f"[{project_id}] Downloading after scene ({after_scene_dt[:10]}, {cloud_after:.1f}% cloud)...")
        _download_scene_image(config, bbox, after_scene_dt, after_path)

        return {
            "project_id": project_id,
            "before_image_path": str(before_path.resolve()),
            "after_image_path": str(after_path.resolve()),
            "before_scene_date": before_scene_dt[:10],
            "after_scene_date": after_scene_dt[:10],
            "cloud_cover_before": round(float(cloud_before), 1),
            "cloud_cover_after": round(float(cloud_after), 1),
        }

    except SatelliteImageryUnavailable:
        raise
    except Exception as exc:
        raise SatelliteImageryUnavailable(
            f"Failed to fetch Sentinel-2 imagery for project '{project_id}': {exc}"
        ) from exc
