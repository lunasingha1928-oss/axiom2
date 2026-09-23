from typing import Optional, List, Tuple, Literal, Any, Dict
from pydantic import BaseModel, Field

SectorType = Literal["Roads", "Railways", "Power", "Water"]
RiskLevelType = Literal["low", "medium", "high"]
ProjectStatusType = Literal["on-track", "delayed", "critical"]

class SiteImagerySchema(BaseModel):
    lat: float
    lng: float
    zoom: int = 14
    satVerified: bool = False
    beforeDate: str = ""
    afterDate: str = ""
    changeDetected: float = 0.0
    ndbiDelta: float = 0.0
    aoi: Optional[List[float]] = None

class ProjectBase(BaseModel):
    id: str
    name: str
    sector: SectorType
    state: str
    reportMonth: str = "2026-08"
    sanctionedCost: float = Field(..., description="Sanctioned cost, INR crore")
    originalCost: float = 0.0
    revisedCost: float = 0.0
    expenditure: float = Field(..., description="Cumulative expenditure, INR crore")
    reportedProgress: float = Field(..., description="Officially reported progress, %")
    verifiedProgress: float = Field(..., description="Independently verified progress, %")
    satelliteEstimatedProgress: float = 0.0
    photoEstimatedProgress: float = 0.0
    
    # Discrepancy signals
    discrepancySatellite: float = 0.0
    discrepancyPhoto: float = 0.0
    combinedDiscrepancy: float = 0.0
    mismatchRedFlag: bool = False
    
    # Predictive ML (XGBoost & SHAP)
    xgboostPDelay: float = 0.0
    xgboostPCostOverrun: float = 0.0
    shapTopFactors: Optional[List[Dict[str, Any]]] = None
    
    riskScore: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    riskLevel: RiskLevelType
    status: ProjectStatusType
    delayMonths: int = Field(0, description="Months behind schedule")
    startDate: Optional[str] = None
    expectedEndDate: Optional[str] = None
    revisedEndDate: Optional[str] = None
    lastVerified: str
    
    # Engineered features
    budgetVariancePct: float = 0.0
    scheduleSlippage: float = 0.0
    sectorBaselineDeviation: float = 0.0
    clusterLabel: Optional[str] = "Medium Risk"
    dataSource: str = "real:data/paimana/flash_report.pdf"

class ProjectRecordResponse(ProjectBase):
    pass

class ProjectDetailResponse(ProjectBase):
    site: Optional[SiteImagerySchema] = None

class ProjectCreate(BaseModel):
    id: str
    project_code: Optional[str] = None
    name: str
    sector: SectorType
    state: str
    sanctioned_cost: float
    expenditure: float = 0.0
    reported_progress: float = 0.0
    verified_progress: float = 0.0
    start_date: Optional[str] = None
    expected_end_date: Optional[str] = None
    delay_months: int = 0
    last_verified: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
