import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


async def embed_text(text: str) -> list[float]:
    url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": settings.EMBEDDING_MODEL, "prompt": text}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if not embedding or len(embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"Unexpected embedding dimension: got {len(embedding) if embedding else 0}, "
                    f"expected {EMBEDDING_DIM}"
                )
            return embedding
    except Exception as e:
        logger.error(f"Embedding failed for text snippet '{text[:60]}...': {e}")
        raise


async def embed_batch(texts: list[str]) -> list[list[float]]:
    results = []
    for text in texts:
        embedding = await embed_text(text)
        results.append(embedding)
    return results


async def check_embedder_health() -> bool:
    try:
        await embed_text("health check")
        return True
    except Exception:
        return False
