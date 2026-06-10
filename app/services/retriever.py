import logging
import json
from app.database import get_pool

logger = logging.getLogger(__name__)


async def similarity_search(
    embedding: list[float],
    top_k: int = 5,
    min_score: float = 0.42,
) -> list[tuple[str, float, dict]]:
    pool = await get_pool()
    if pool is None:
        logger.warning("Database pool not available — returning empty results.")
        return []

    # Pass embedding as a Python list — pgvector asyncpg codec handles encoding.
    query = """
        SELECT
            content,
            1 - (embedding <=> $1) AS score,
            metadata
        FROM documents
        WHERE 1 - (embedding <=> $1) >= $2
        ORDER BY score DESC
        LIMIT $3;
    """

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, embedding, min_score, top_k)
            results = []
            for row in rows:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                results.append((row["content"], float(row["score"]), metadata))
            logger.info(
                "Retrieved %s chunks",
                len(results)
            )

            for i, (content, score, metadata) in enumerate(results):
                logger.info(
                    "[%s] score=%.4f label=%s preview=%s",
                    i,
                    score,
                    metadata.get("label"),
                    content[:200].replace("\n", " ")
                )
            return results
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []


async def delete_by_label(label: str) -> int:
    pool = await get_pool()
    if pool is None:
        return 0
    try:
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM documents WHERE label = $1;", label)
            count = int(result.split()[-1])
            logger.info(f"Deleted {count} chunks with label='{label}'.")
            return count
    except Exception as e:
        logger.error(f"Delete by label failed: {e}")
        return 0
