import os
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.pipeline.paimana_parser import PaimanaParser

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

LOCAL_PDF_PATH = settings.DATA_DIR / "paimana" / "flash_report.pdf"


@router.post("/ingest-paimana")
async def ingest_paimana_pdf(
    file: Optional[UploadFile] = File(None, description="MoSPI PAIMANA Flash Report PDF file"),
    use_local_sample: bool = Form(True, description="Use static data/paimana/flash_report.pdf file"),
    max_pages: Optional[int] = Form(None, description="Max pages to parse"),
    db: AsyncSession = Depends(get_db),
):
    """
    MoSPI PAIMANA Flash Report Ingestion Pipeline (Static PDF -> Structured Data).
    
    1. Extracts Table 6 ('All Ongoing Projects') from local /data/paimana/flash_report.pdf or upload.
    2. Extracts: project_id, name, sector, state, sanctioned_cost, expenditure_to_date, reported_progress_pct, start_date, expected_end_date.
    3. Engineers: budget_variance_pct, schedule_slippage, sector_baseline_deviation.
    4. Persists into database schema with data_source: 'real:data/paimana/flash_report.pdf'.
    """
    if file and file.filename:
        pdf_bytes = await file.read()
        records = PaimanaParser.extract_from_pdf(pdf_bytes, max_pages=max_pages)
    elif use_local_sample or LOCAL_PDF_PATH.exists():
        if not LOCAL_PDF_PATH.exists():
            raise HTTPException(status_code=404, detail="Local data/paimana/flash_report.pdf not found")
        records = PaimanaParser.extract_from_pdf(str(LOCAL_PDF_PATH), max_pages=max_pages)
    else:
        raise HTTPException(
            status_code=400,
            detail="Please provide a PDF file upload or set use_local_sample=True"
        )

    if not records:
        raise HTTPException(status_code=422, detail="No ongoing project records could be extracted from the PDF")

    saved_count = await PaimanaParser.ingest_and_save_records(db, records)

    return {
        "status": "success",
        "extracted_count": len(records),
        "saved_to_db": saved_count,
        "data_source": "real:data/paimana/flash_report.pdf",
        "sample_records": records[:5],
    }
