from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, Text, ForeignKey, UniqueConstraint
from app.database import Base

class Project(Base):
    __tablename__ = "projects"

    id = Column(String(64), primary_key=True, index=True)  # e.g. "PS-RD-1042" or code "705728"
    project_code = Column(String(64), index=True, nullable=True)
    name = Column(String(255), nullable=False)
    sector = Column(String(64), nullable=False, index=True)  # Roads, Railways, Power, Water
    state = Column(String(128), nullable=False, index=True)
    report_month = Column(String(16), default="2026-08", index=True)  # Keyed by report_month
    
    # Financials (INR Crore)
    original_cost = Column(Float, nullable=False, default=0.0)
    revised_cost = Column(Float, nullable=False, default=0.0)
    sanctioned_cost = Column(Float, nullable=False, default=0.0)
    expenditure = Column(Float, nullable=False, default=0.0)
    
    # Timeline & Milestones
    start_date = Column(String(32), nullable=True)
    expected_end_date = Column(String(32), nullable=True)
    delay_months = Column(Integer, nullable=False, default=0)
    contractor_delay_rate = Column(Float, default=0.15)  # Historical contractor delay probability
    milestone_dates = Column(JSON, nullable=True)        # Milestone schedule vs actuals
    last_verified = Column(String(32), nullable=True)
    
    # Progress Metrics
    reported_progress = Column(Float, nullable=False, default=0.0)  # %
    verified_progress = Column(Float, nullable=False, default=0.0)  # %
    satellite_estimated_progress = Column(Float, default=0.0)        # % (NDBI change detection)
    photo_estimated_progress = Column(Float, default=0.0)            # % (YOLOv8 vision classification)
    
    # Discrepancy Signals
    discrepancy_satellite = Column(Float, default=0.0)  # |reported - satellite|
    discrepancy_photo = Column(Float, default=0.0)      # |reported - photo|
    combined_discrepancy = Column(Float, default=0.0)   # max(sat, photo) or weighted mismatch
    
    # Engineered PAIMANA Features
    budget_variance_pct = Column(Float, default=0.0)        # expenditure vs expected spend given elapsed time
    schedule_slippage = Column(Float, default=0.0)          # elapsed_time_pct vs reported_progress_pct
    sector_baseline_deviation = Column(Float, default=0.0)  # pace vs sector/state baseline pace
    
    # ML Predictive Model & SHAP Explainability
    xgboost_p_delay = Column(Float, default=0.0)            # P(delay > 6 months)
    xgboost_p_cost_overrun = Column(Float, default=0.0)     # P(cost overrun > 20%)
    shap_top_factors = Column(JSON, nullable=True)          # Ranked SHAP feature contributions
    
    # Composite Risk Scoring & Sanity Check
    risk_score = Column(Integer, nullable=False, default=0)         # 0-100 (Transparent Formula)
    risk_level = Column(String(32), nullable=False, default="low")  # low, medium, high
    status = Column(String(32), nullable=False, default="on-track") # on-track, delayed, critical
    cluster_label = Column(String(64), default="Low Risk")          # K-Means Cluster sanity check
    
    # Metadata & Data Source Tracking
    data_source = Column(String(128), default="real:data/paimana/flash_report.pdf")
    
    # Geographic Location & Satellite geometry
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    zoom = Column(Integer, default=14)
    sat_verified = Column(Boolean, default=False)
    before_date = Column(String(32), nullable=True)
    after_date = Column(String(32), nullable=True)
    change_detected = Column(Float, default=0.0)
    ndbi_diff = Column(Float, default=0.0)
    aoi_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProjectMonthlySnapshot(Base):
    """
    Time-series record keyed by (project_id, report_month) to track monthly slippage,
    spending trends, and discrepancy progression over time.
    """
    __tablename__ = "project_monthly_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(64), ForeignKey("projects.id"), index=True, nullable=False)
    report_month = Column(String(16), index=True, nullable=False)  # e.g. "2026-08"
    
    reported_progress = Column(Float, nullable=False)
    verified_progress = Column(Float, nullable=False)
    satellite_estimated_progress = Column(Float, default=0.0)
    photo_estimated_progress = Column(Float, default=0.0)
    discrepancy = Column(Float, default=0.0)
    
    expenditure = Column(Float, nullable=False)
    sanctioned_cost = Column(Float, nullable=False)
    delay_months = Column(Integer, default=0)
    risk_score = Column(Integer, default=0)
    
    data_source = Column(String(128), default="real:data/paimana/flash_report.pdf")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("project_id", "report_month", name="uq_project_month"),
    )
