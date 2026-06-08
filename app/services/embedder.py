import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

GROQ_EMBED_URL = "https://api.groq.com/openai/v1/embeddings"
EMBED_MODEL = "nomic-embed-text-v1_5"
EMBEDDING_DIM = 768


async def embed(text: str) -> list[float]:
    """Embed text using Groq nomic-embed-text-v1_5 (768 dimensions)."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GROQ_EMBED_URL,
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "input": text.replace("\n", " "),
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in a single API call."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GROQ_EMBED_URL,
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "input": [t.replace("\n", " ") for t in texts],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
