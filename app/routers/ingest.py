import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.schemas import CrawlRequest
from app.auth.security import require_api_key
from app.services.retriever import delete_by_label

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge"


async def _run_crawl(url: str, label: str) -> None:
    from app.services.crawler import crawl_and_ingest
    try:
        count = await crawl_and_ingest(url, label)
        logger.info(f"Background crawl complete: {count} chunks for label='{label}'.")
    except Exception as e:
        logger.error(f"Background crawl failed for {url}: {e}")


@router.post("/crawl")
async def crawl(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
):
    background_tasks.add_task(_run_crawl, request.url, request.label)
    return {"detail": f"Crawl started for {request.url} with label='{request.label}'."}


@router.post("/knowledge")
async def ingest_knowledge(_: str = Depends(require_api_key)):
    from app.services.ingestor import ingest_file

    if not KNOWLEDGE_DIR.exists():
        raise HTTPException(status_code=404, detail="Knowledge directory not found.")

    md_files = list(KNOWLEDGE_DIR.glob("*.md"))
    if not md_files:
        return {"detail": "No .md files found in knowledge directory.", "files": []}

    results = {}
    for md_file in md_files:
        count = await ingest_file(md_file)
        results[md_file.name] = count

    return {"detail": "Knowledge ingestion complete.", "chunks_by_file": results}


@router.delete("/content/{label}")
async def delete_content(label: str, _: str = Depends(require_api_key)):
    count = await delete_by_label(label)
    return {"detail": f"Deleted {count} chunks with label='{label}'."}
