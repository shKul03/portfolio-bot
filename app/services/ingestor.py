import logging
import json
from pathlib import Path
from app.services.embedder import embed
from app.database import get_pool
from app.config import settings

logger = logging.getLogger(__name__)

PLACEHOLDER = "(To be filled in)"


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def _is_skippable(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if stripped == PLACEHOLDER:
        return True
    if PLACEHOLDER in stripped and len(stripped) < len(PLACEHOLDER) + 100:
        return True
    return False


async def _upsert_chunk(
    conn,
    doc_id: str,
    content: str,
    embedding: list[float],
    metadata: dict,
    label: str,
) -> None:
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
    await conn.execute(
        """
        INSERT INTO documents (id, content, embedding, metadata, label)
        VALUES ($1, $2, $3::vector, $4::jsonb, $5)
        ON CONFLICT (id) DO UPDATE
            SET content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                label = EXCLUDED.label;
        """,
        doc_id,
        content,
        embedding_str,
        json.dumps(metadata),
        label,
    )


async def ingest_file(file_path: str | Path) -> int:
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return 0

    content = path.read_text(encoding="utf-8")

    if _is_skippable(content):
        logger.info(f"Skipping {path.name} — empty or placeholder.")
        return 0

    pool = await get_pool()
    if pool is None:
        logger.error("No database pool available for ingestion.")
        return 0

    chunks = chunk_text(content)
    filename = path.name
    count = 0

    async with pool.acquire() as conn:
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            doc_id = f"{filename}::chunk_{i}"
            try:
                embedding = await embed(chunk)
                metadata = {"source": "knowledge_file", "filename": filename, "chunk_index": i}
                await _upsert_chunk(conn, doc_id, chunk, embedding, metadata, label=filename)
                count += 1
            except Exception as e:
                logger.error(f"Failed to ingest chunk {i} of {filename}: {e}")

    logger.info(f"Ingested {count} chunks from {filename}.")
    return count


async def ingest_chunks_with_label(
    chunks: list[str],
    label: str,
    base_id: str,
) -> int:
    pool = await get_pool()
    if pool is None:
        logger.error("No database pool available for ingestion.")
        return 0

    count = 0
    async with pool.acquire() as conn:
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            doc_id = f"{base_id}::chunk_{i}"
            try:
                embedding = await embed(chunk)
                metadata = {"source": "web_crawl", "label": label, "chunk_index": i}
                await _upsert_chunk(conn, doc_id, chunk, embedding, metadata, label=label)
                count += 1
            except Exception as e:
                logger.error(f"Failed to ingest crawled chunk {i} (label={label}): {e}")

    return count
