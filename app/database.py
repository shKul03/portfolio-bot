import asyncio
import logging
import asyncpg
from pgvector.asyncpg import register_vector
from app.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool | None:
    return _pool


async def execute_with_retry(pool: asyncpg.Pool, query: str, *args):
    for attempt in range(2):
        try:
            async with pool.acquire() as conn:
                return await conn.fetch(query, *args)
        except (asyncpg.ConnectionDoesNotExistError, asyncpg.TooManyConnectionsError) as e:
            if attempt == 0:
                await asyncio.sleep(1)
                continue
            raise


async def init_db() -> None:
    global _pool
    pool_kwargs = dict(
        dsn=settings.asyncpg_url,
        min_size=1,
        max_size=5,
        max_inactive_connection_lifetime=270,
        max_cached_statement_lifetime=0,
        command_timeout=30,
        server_settings={"application_name": "portfolio-bot"},
        init=register_vector,
    )
    if settings.asyncpg_ssl:
        pool_kwargs["ssl"] = "require"

    try:
        _pool = await asyncpg.create_pool(**pool_kwargs)
        logger.info("Database connection pool established.")
    except Exception as e:
        logger.warning(f"Database unavailable on startup — continuing without DB: {e}")
        _pool = None
        return

    # Schema setup is best-effort: a failure here must not destroy the pool.
    try:
        await _setup_schema()
    except Exception as e:
        logger.warning(f"Schema setup failed (pool still available): {e}")


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def _setup_schema() -> None:
    if _pool is None:
        return
    for attempt in range(2):
        try:
            async with _pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(768),
                        metadata JSONB DEFAULT '{}',
                        label TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
                    ON documents USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_label_idx ON documents (label);
                """)
            logger.info("Database schema and HNSW index ready.")
            return
        except (asyncpg.ConnectionDoesNotExistError, asyncpg.TooManyConnectionsError) as e:
            if attempt == 0:
                await asyncio.sleep(1)
                continue
            raise
