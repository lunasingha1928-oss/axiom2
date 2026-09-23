"""
Project Sentinel — Satellite Verification Service (Placeholder / Demo Layer)

NOTE: This module does NOT perform real satellite data acquisition or processing.
All outputs are heuristic simulations seeded from project_id for demo purposes.

Simulates (does not execute):
1. Before/after Sentinel-2 tile acquisition.
2. Atmospheric correction and pan-sharpening.
3. Change detection via Normalized Difference Built-up Index (NDBI).
   Formula: NDBI = (SWIR - NIR) / (SWIR + NIR)
4. Outputs `satellite_estimated_progress_pct` and `discrepancy_satellite`.

Real Sentinel-2 API integration is planned for a future release.
"""

import numpy as np
from typing import Dict, Any, Tuple

def compute_ndbi_index(swir_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Computes Normalized Difference Built-up Index (NDBI).
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    Values > 0 typically correspond to built-up, concrete, and paved construction surfaces.
    """
    denominator = swir_band + nir_band
    denominator[denominator == 0] = 1e-6
    ndbi = (swir_band - nir_band) / denominator
    return np.clip(ndbi, -1.0, 1.0)

def simulate_sentinel2_change_detection(
    project_id: str,
    reported_progress: float,
    seed_offset: int = 42
) -> Dict[str, Any]:
    """
    Runs atmospheric correction, pan-sharpening simulation, and NDBI differencing
    on before/after Sentinel-2 multispectral observation tiles for a given project.
    """
    # Deterministic simulation based on project_id
    hash_val = sum(ord(c) for c in project_id) + seed_offset
    np.random.seed(hash_val % 10000)

    # 1. Simulate atmospheric correction & pan-sharpened bands (SWIR & NIR)
    # Grid size representing 100x100m observation window
    grid_size = (32, 32)
    
    # Base terrain (NIR high for vegetation, SWIR lower)
    nir_before = np.random.uniform(0.35, 0.65, grid_size)
    swir_before = np.random.uniform(0.15, 0.35, grid_size)
    ndbi_before = compute_ndbi_index(swir_before, nir_before)

    # After image (Construction built-up area increases SWIR and reduces vegetation NIR)
    # Estimate built-up expansion proportional to actual physical construction
    target_progress = reported_progress
    # Realistic visual variance: some projects lag significantly on physical ground
    if "PS-RD-1042" in project_id: # NH-44 Sangareddy known lag
        target_progress = 52.0
    elif "PS-RL-2015" in project_id:
        target_progress = 44.0
    elif "PS-PW-3308" in project_id:
        target_progress = 48.0
    elif "PS-WT-4401" in project_id:
        target_progress = 88.0

    built_up_ratio = np.clip(target_progress / 100.0, 0.05, 0.95)
    built_up_mask = np.random.uniform(0, 1, grid_size) < built_up_ratio

    nir_after = nir_before.copy()
    swir_after = swir_before.copy()
    nir_after[built_up_mask] = np.random.uniform(0.12, 0.28, np.sum(built_up_mask))
    swir_after[built_up_mask] = np.random.uniform(0.40, 0.70, np.sum(built_up_mask))
    
    ndbi_after = compute_ndbi_index(swir_after, nir_after)

    # 2. Pixel differencing & Built-up thresholding
    ndbi_diff = ndbi_after - ndbi_before
    new_construction_pixels = np.sum(ndbi_diff > 0.15)
    total_pixels = grid_size[0] * grid_size[1]
    
    # 3. Calculate satellite estimated progress %
    raw_sat_progress = (new_construction_pixels / total_pixels) * 100.0
    # Scale to calibrated progress range
    satellite_estimated_progress_pct = float(np.clip(raw_sat_progress * 1.15, 5.0, 98.0))

    # 4. Discrepancy computation
    discrepancy_sat = float(abs(reported_progress - satellite_estimated_progress_pct))

    return {
        "project_id": project_id,
        "data_source": "simulated:heuristic_placeholder",
        "satellite_source": "Placeholder — Sentinel-2 API integration planned",
        "preprocessing": "Placeholder — no real atmospheric correction applied",
        "methodology": "Heuristic simulation calibrated per-project for demo purposes (not live change detection)",
        "mean_ndbi_before": float(np.mean(ndbi_before)),
        "mean_ndbi_after": float(np.mean(ndbi_after)),
        "mean_ndbi_delta": float(np.mean(ndbi_diff)),
        "new_built_up_extent_pct": float(round(raw_sat_progress, 1)),
        "satellite_estimated_progress_pct": float(round(satellite_estimated_progress_pct, 1)),
        "reported_progress_pct": float(round(reported_progress, 1)),
        "discrepancy_satellite": float(round(discrepancy_sat, 1)),
        "resolution_meters": 10,
        "cloud_cover_pct": 2.1,
    }
