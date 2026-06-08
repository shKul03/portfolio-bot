import logging
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.database import get_pool

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


async def _check_db() -> bool:
    try:
        pool = await get_pool()
        if pool is None:
            return False
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False


async def _check_embedder() -> bool:
    try:
        from app.services.embedder import check_embedder_health
        return await check_embedder_health()
    except Exception:
        return False


async def _check_llm() -> bool:
    from app.config import settings
    try:
        if settings.LLM_PROVIDER == "groq":
            if not settings.GROQ_API_KEY:
                return False
            from groq import AsyncGroq
            client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            await client.models.list()
            return True
        else:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                return r.status_code == 200
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health():
    db_ok = await _check_db()
    embedder_ok = await _check_embedder()
    llm_ok = await _check_llm()

    return HealthResponse(
        status="ok",
        db=db_ok,
        embedder=embedder_ok,
        llm=llm_ok,
    )


@router.get("/ping")
async def ping():
    db_ok = False
    try:
        pool = await get_pool()
        if pool is not None:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
            logger.debug("ping ok")
    except Exception:
        pass
    return {"pong": True, "db": db_ok}
