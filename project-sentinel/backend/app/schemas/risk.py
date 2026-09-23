from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class FactorDetail(BaseModel):
    name: str
    raw_value: float
    unit: str
    normalized_score: float = Field(..., ge=0.0, le=100.0, description="Factor score on 0-100 scale")
    weight: float = Field(..., ge=0.0, le=1.0, description="Configured weight in composite model")
    weighted_contribution: float = Field(..., description="Contribution to total risk score")
    description: str
    data_source: str

class DiscrepancyDetails(BaseModel):
    reported_progress_pct: float
    satellite_estimated_progress_pct: float
    photo_estimated_progress_pct: float
    discrepancy_satellite: float
    discrepancy_photo: float
    combined_discrepancy: float
    mismatch_level: str = "low"  # low, medium, high_red_flag

class ShapContribution(BaseModel):
    feature: str
    raw_value: float
    shap_value: float
    impact: str
    importance_rank: float

class PredictiveMlDetails(BaseModel):
    p_delay_over_6mo: float
    p_cost_overrun_over_20pct: float
    ml_risk_trend: str
    model_type: str
    top_contributing_factor: str
    shap_explainability: List[ShapContribution]
    source: str = "sentinel_predictive_model"

class RiskBreakdownResponse(BaseModel):
    project_id: str
    project_name: str
    sector: str
    state: str
    risk_score: int = Field(..., ge=0, le=100, description="Final composite risk score (0-100)")
    risk_level: str = Field(..., description="low, medium, high")
    status: str = Field(..., description="on-track, delayed, critical")
    cluster_label: Optional[str] = "Medium Risk Cluster"
    factors: Dict[str, FactorDetail]
    discrepancy_details: Optional[DiscrepancyDetails] = None
    predictive_ml: Optional[PredictiveMlDetails] = None
    formula: str
    data_source: str = "hybrid:real(paimana,rainfall,photos)+simulated(sensors)"
    calculated_at: str

class ModelValidationResponse(BaseModel):
    status: str
    cluster_means: Dict[str, float]
    logistic_regression_validation: Dict[str, Any]
    gradient_boosted_validation: Optional[Dict[str, Any]] = None
    data_source: str = "real:data/paimana/flash_report.pdf"
