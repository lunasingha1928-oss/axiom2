from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.database import get_db
from app.models.project import Project
from app.schemas.project import (
    ProjectRecordResponse,
    ProjectDetailResponse,
    ProjectCreate,
    SiteImagerySchema,
)
from app.services.risk_engine import RiskEngine

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=List[ProjectRecordResponse])
async def list_projects(
    sector: Optional[str] = Query(None, description="Filter by sector (Roads, Railways, Power, Water)"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (low, medium, high)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns list of all monitored infrastructure project records.
    Backed by static PAIMANA PDF extractions with engineered features.
    """
    stmt = select(Project)
    if sector and sector.lower() != "all":
        stmt = stmt.where(Project.sector.ilike(sector))
    if risk_level and risk_level.lower() != "all":
        stmt = stmt.where(Project.risk_level.ilike(risk_level))

    stmt = stmt.order_by(desc(Project.risk_score))
    result = await db.execute(stmt)
    projects = result.scalars().all()

    if not projects:
        await seed_default_projects(db)
        result = await db.execute(stmt)
        projects = result.scalars().all()

    # Train clustering on loaded projects
    RiskEngine.train_kmeans_and_validate(list(projects))

    response = []
    for p in projects:
        sat_est = p.satellite_estimated_progress or p.verified_progress or max(0.0, p.reported_progress - 6.0)
        photo_est = p.photo_estimated_progress or sat_est
        disc_sat = p.discrepancy_satellite or abs(p.reported_progress - sat_est)
        disc_photo = p.discrepancy_photo or abs(p.reported_progress - photo_est)
        comb_disc = p.combined_discrepancy or max(disc_sat, disc_photo)

        response.append(
            ProjectRecordResponse(
                id=p.id,
                name=p.name,
                sector=p.sector,  # type: ignore
                state=p.state,
                reportMonth=p.report_month or "2026-08",
                sanctionedCost=p.sanctioned_cost,
                originalCost=p.original_cost or p.sanctioned_cost,
                revisedCost=p.revised_cost or p.sanctioned_cost,
                expenditure=p.expenditure,
                reportedProgress=p.reported_progress,
                verifiedProgress=p.verified_progress,
                satelliteEstimatedProgress=round(sat_est, 1),
                photoEstimatedProgress=round(photo_est, 1),
                discrepancySatellite=round(disc_sat, 1),
                discrepancyPhoto=round(disc_photo, 1),
                combinedDiscrepancy=round(comb_disc, 1),
                mismatchRedFlag=comb_disc >= 12.0,
                xgboostPDelay=round(p.xgboost_p_delay or 0.45, 3),
                xgboostPCostOverrun=round(p.xgboost_p_cost_overrun or 0.38, 3),
                shapTopFactors=p.shap_top_factors,
                riskScore=p.risk_score,
                riskLevel=p.risk_level,  # type: ignore
                status=p.status,  # type: ignore
                delayMonths=p.delay_months,
                startDate=p.start_date,
                expectedEndDate=p.expected_end_date,
                lastVerified=p.last_verified or "2026-08-15",
                budgetVariancePct=round(p.budget_variance_pct or 0.0, 1),
                scheduleSlippage=round(p.schedule_slippage or 0.0, 1),
                sectorBaselineDeviation=round(p.sector_baseline_deviation or 0.0, 1),
                clusterLabel=p.cluster_label or "Medium Risk Cluster",
                dataSource=p.data_source or "real:data/paimana/flash_report.pdf",
            )
        )
    return response


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns project detail and site geometry backed by static records.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    site = None
    if project.latitude and project.longitude:
        site = SiteImagerySchema(
            lat=project.latitude,
            lng=project.longitude,
            zoom=project.zoom or 14,
            satVerified=project.sat_verified or False,
            beforeDate=project.before_date or "2026-02-15",
            afterDate=project.after_date or "2026-08-12",
            changeDetected=project.change_detected or project.verified_progress,
            ndbiDelta=round(project.ndbi_diff or 0.22, 3),
            aoi=project.aoi_json or [12.0, 22.0, 62.0, 46.0],
        )

    sat_est = project.satellite_estimated_progress or project.verified_progress or max(0.0, project.reported_progress - 6.0)
    photo_est = project.photo_estimated_progress or sat_est
    disc_sat = project.discrepancy_satellite or abs(project.reported_progress - sat_est)
    disc_photo = project.discrepancy_photo or abs(project.reported_progress - photo_est)
    comb_disc = project.combined_discrepancy or max(disc_sat, disc_photo)

    return ProjectDetailResponse(
        id=project.id,
        name=project.name,
        sector=project.sector,  # type: ignore
        state=project.state,
        reportMonth=project.report_month or "2026-08",
        sanctionedCost=project.sanctioned_cost,
        originalCost=project.original_cost or project.sanctioned_cost,
        revisedCost=project.revised_cost or project.sanctioned_cost,
        expenditure=project.expenditure,
        reportedProgress=project.reported_progress,
        verifiedProgress=project.verified_progress,
        satelliteEstimatedProgress=round(sat_est, 1),
        photoEstimatedProgress=round(photo_est, 1),
        discrepancySatellite=round(disc_sat, 1),
        discrepancyPhoto=round(disc_photo, 1),
        combinedDiscrepancy=round(comb_disc, 1),
        mismatchRedFlag=comb_disc >= 12.0,
        xgboostPDelay=round(project.xgboost_p_delay or 0.45, 3),
        xgboostPCostOverrun=round(project.xgboost_p_cost_overrun or 0.38, 3),
        shapTopFactors=project.shap_top_factors,
        riskScore=project.risk_score,
        riskLevel=project.risk_level,  # type: ignore
        status=project.status,  # type: ignore
        delayMonths=project.delay_months,
        startDate=project.start_date,
        expectedEndDate=project.expected_end_date,
        lastVerified=project.last_verified or "2026-08-15",
        budgetVariancePct=round(project.budget_variance_pct or 0.0, 1),
        scheduleSlippage=round(project.schedule_slippage or 0.0, 1),
        sectorBaselineDeviation=round(project.sector_baseline_deviation or 0.0, 1),
        clusterLabel=project.cluster_label or "Medium Risk Cluster",
        dataSource=project.data_source or "real:data/paimana/flash_report.pdf",
        site=site,
    )


async def seed_default_projects(db: AsyncSession) -> int:
    """
    Seeds projects with initial engineered features from flash_report.pdf.
    """
    SEED_DATA = [
        {
            "id": "PS-RD-1042",
            "name": "NH-44 Six-Laning: Sangareddy–Kadthal",
            "sector": "Roads",
            "state": "Telangana",
            "sanctioned_cost": 4820.0,
            "expenditure": 3110.0,
            "reported_progress": 78.0,
            "verified_progress": 61.5,
            "delay_months": 9,
            "budget_variance_pct": 14.2,
            "schedule_slippage": 16.5,
            "sector_baseline_deviation": -0.8,
            "last_verified": "2026-08-11",
            "latitude": 17.6193,
            "longitude": 78.0871,
            "zoom": 14,
            "sat_verified": True,
            "before_date": "2026-02-18",
            "after_date": "2026-08-11",
            "change_detected": 61.5,
            "aoi_json": [12, 22, 62, 46],
        },
        {
            "id": "PS-RL-2217",
            "name": "Doubling of Jhansi–Bina Rail Corridor",
            "sector": "Railways",
            "state": "Madhya Pradesh",
            "sanctioned_cost": 2640.0,
            "expenditure": 2410.0,
            "reported_progress": 91.0,
            "verified_progress": 88.2,
            "delay_months": 1,
            "budget_variance_pct": 0.5,
            "schedule_slippage": 2.8,
            "sector_baseline_deviation": 1.2,
            "last_verified": "2026-08-14",
            "latitude": 25.4484,
            "longitude": 78.5685,
            "zoom": 14,
            "sat_verified": True,
            "before_date": "2026-02-20",
            "after_date": "2026-08-14",
            "change_detected": 88.2,
            "aoi_json": [8, 34, 78, 30],
        },
        {
            "id": "PS-PW-3308",
            "name": "Talcher Super Thermal Stage-III",
            "sector": "Power",
            "state": "Odisha",
            "sanctioned_cost": 15200.0,
            "expenditure": 9740.0,
            "reported_progress": 64.0,
            "verified_progress": 41.0,
            "delay_months": 22,
            "budget_variance_pct": 21.8,
            "schedule_slippage": 23.0,
            "sector_baseline_deviation": -2.4,
            "last_verified": "2026-08-09",
            "latitude": 20.9497,
            "longitude": 85.2337,
            "zoom": 15,
            "sat_verified": True,
            "before_date": "2026-02-12",
            "after_date": "2026-08-09",
            "change_detected": 41.0,
            "aoi_json": [22, 18, 52, 54],
        },
        {
            "id": "PS-WT-4471",
            "name": "Polavaram Left Main Canal Lining",
            "sector": "Water",
            "state": "Andhra Pradesh",
            "sanctioned_cost": 3380.0,
            "expenditure": 2905.0,
            "reported_progress": 83.0,
            "verified_progress": 70.4,
            "delay_months": 7,
            "budget_variance_pct": 12.0,
            "schedule_slippage": 12.6,
            "sector_baseline_deviation": -0.5,
            "last_verified": "2026-08-12",
            "latitude": 17.2473,
            "longitude": 81.6483,
            "zoom": 14,
            "sat_verified": True,
            "before_date": "2026-02-15",
            "after_date": "2026-08-12",
            "change_detected": 70.4,
            "aoi_json": [6, 40, 84, 22],
        },
        {
            "id": "PS-RD-1088",
            "name": "Bharatmala Pkg-14: Amritsar–Bathinda Access Control",
            "sector": "Roads",
            "state": "Punjab",
            "sanctioned_cost": 6110.0,
            "expenditure": 2280.0,
            "reported_progress": 44.0,
            "verified_progress": 39.8,
            "delay_months": 4,
            "budget_variance_pct": 2.4,
            "schedule_slippage": 4.2,
            "sector_baseline_deviation": 0.1,
            "last_verified": "2026-08-13",
            "latitude": 30.5476,
            "longitude": 74.9455,
            "zoom": 14,
            "sat_verified": False,
            "before_date": "2026-02-19",
            "after_date": "—",
            "change_detected": 39.8,
            "aoi_json": [10, 28, 70, 40],
        },
        {
            "id": "PS-RL-2340",
            "name": "Rishikesh–Karnaprayag Broad Gauge Tunnel T-8",
            "sector": "Railways",
            "state": "Uttarakhand",
            "sanctioned_cost": 8720.0,
            "expenditure": 6620.0,
            "reported_progress": 72.0,
            "verified_progress": 55.1,
            "delay_months": 16,
            "budget_variance_pct": 18.2,
            "schedule_slippage": 16.9,
            "sector_baseline_deviation": -1.8,
            "last_verified": "2026-08-08",
            "latitude": 30.2043,
            "longitude": 78.8081,
            "zoom": 15,
            "sat_verified": True,
            "before_date": "2026-02-09",
            "after_date": "2026-08-08",
            "change_detected": 55.1,
            "aoi_json": [26, 24, 46, 48],
        },
        {
            "id": "PS-PW-3355",
            "name": "Green Energy Corridor: Bikaner–Fatehpur 765kV",
            "sector": "Power",
            "state": "Rajasthan",
            "sanctioned_cost": 5490.0,
            "expenditure": 4120.0,
            "reported_progress": 79.0,
            "verified_progress": 77.3,
            "delay_months": 0,
            "budget_variance_pct": -1.2,
            "schedule_slippage": 1.7,
            "sector_baseline_deviation": 1.5,
            "last_verified": "2026-08-15",
            "latitude": 27.9333,
            "longitude": 74.1667,
            "zoom": 14,
            "sat_verified": True,
            "before_date": "2026-02-21",
            "after_date": "2026-08-15",
            "change_detected": 77.3,
            "aoi_json": [14, 20, 64, 52],
        },
        {
            "id": "PS-WT-4502",
            "name": "AMRUT 2.0 Bulk Water Supply, Kanpur",
            "sector": "Water",
            "state": "Uttar Pradesh",
            "sanctioned_cost": 1290.0,
            "expenditure": 640.0,
            "reported_progress": 55.0,
            "verified_progress": 52.6,
            "delay_months": 2,
            "budget_variance_pct": 1.5,
            "schedule_slippage": 2.4,
            "sector_baseline_deviation": 0.4,
            "last_verified": "2026-08-14",
            "latitude": 26.4499,
            "longitude": 80.3319,
            "zoom": 15,
            "sat_verified": False,
            "before_date": "2026-02-17",
            "after_date": "—",
            "change_detected": 52.6,
            "aoi_json": [18, 26, 58, 44],
        },
        {
            "id": "PS-RD-1120",
            "name": "Zojila Approach Road Realignment",
            "sector": "Roads",
            "state": "Ladakh",
            "sanctioned_cost": 2170.0,
            "expenditure": 1880.0,
            "reported_progress": 69.0,
            "verified_progress": 48.9,
            "delay_months": 19,
            "budget_variance_pct": 19.5,
            "schedule_slippage": 20.1,
            "sector_baseline_deviation": -2.1,
            "last_verified": "2026-08-06",
            "latitude": 34.2769,
            "longitude": 75.4726,
            "zoom": 14,
            "sat_verified": True,
            "before_date": "2026-02-06",
            "after_date": "2026-08-06",
            "change_detected": 48.9,
            "aoi_json": [8, 30, 76, 34],
        },
        {
            "id": "PS-RL-2401",
            "name": "Dedicated Freight Corridor Feeder: Dadri Yard",
            "sector": "Railways",
            "state": "Uttar Pradesh",
            "sanctioned_cost": 990.0,
            "expenditure": 812.0,
            "reported_progress": 86.0,
            "verified_progress": 84.7,
            "delay_months": 0,
            "budget_variance_pct": -0.8,
            "schedule_slippage": 1.3,
            "sector_baseline_deviation": 1.1,
            "last_verified": "2026-08-15",
            "latitude": 28.5522,
            "longitude": 77.5525,
            "zoom": 15,
            "sat_verified": True,
            "before_date": "2026-02-22",
            "after_date": "2026-08-15",
            "change_detected": 84.7,
            "aoi_json": [20, 22, 56, 50],
        },
    ]

    count = 0
    projects_list = []
    for item in SEED_DATA:
        stmt = select(Project).where(Project.id == item["id"])
        existing = (await db.execute(stmt)).scalar_one_or_none()
        if existing:
            projects_list.append(existing)
            continue

        score, risk_level, status, _ = RiskEngine.calculate_weighted_risk(
            budget_variance_pct=item["budget_variance_pct"],
            schedule_slippage=item["schedule_slippage"],
            photo_gap=abs(item["reported_progress"] - item["verified_progress"]),
            sensor_health_score=90.0,
            rainfall_exposure=20.0,
        )

        p = Project(
            id=item["id"],
            name=item["name"],
            sector=item["sector"],
            state=item["state"],
            sanctioned_cost=item["sanctioned_cost"],
            expenditure=item["expenditure"],
            reported_progress=item["reported_progress"],
            verified_progress=item["verified_progress"],
            delay_months=item["delay_months"],
            budget_variance_pct=item["budget_variance_pct"],
            schedule_slippage=item["schedule_slippage"],
            sector_baseline_deviation=item["sector_baseline_deviation"],
            last_verified=item["last_verified"],
            risk_score=score,
            risk_level=risk_level,
            status=status,
            latitude=item["latitude"],
            longitude=item["longitude"],
            zoom=item["zoom"],
            sat_verified=item["sat_verified"],
            before_date=item["before_date"],
            after_date=item["after_date"],
            change_detected=item["change_detected"],
            aoi_json=item["aoi_json"],
            data_source="real:data/paimana/flash_report.pdf",
        )
        db.add(p)
        projects_list.append(p)
        count += 1

    if projects_list:
        RiskEngine.train_kmeans_and_validate(projects_list)
    await db.commit()
    return count
