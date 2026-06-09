import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

NOMIC_EMBED_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
EMBED_MODEL = "nomic-embed-text-v1.5"
EMBEDDING_DIM = 768


async def embed(text: str, task_type: str = "search_document") -> list[float]:
    """Embed text using Nomic AI nomic-embed-text-v1.5 (768 dimensions).

    task_type: "search_document" for indexed content, "search_query" for queries.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            NOMIC_EMBED_URL,
            headers={
                "Authorization": f"Bearer {settings.NOMIC_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "texts": [text.replace("\n", " ")],
                "task_type": task_type,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


async def embed_batch(
    texts: list[str], task_type: str = "search_document"
) -> list[list[float]]:
    """Embed a batch of texts in a single API call."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            NOMIC_EMBED_URL,
            headers={
                "Authorization": f"Bearer {settings.NOMIC_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "texts": [t.replace("\n", " ") for t in texts],
                "task_type": task_type,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
