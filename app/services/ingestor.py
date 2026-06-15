import logging
import json
from pathlib import Path
from app.services.embedder import embed
from app.database import get_pool
from app.config import settings

logger = logging.getLogger(__name__)

PLACEHOLDER = "(To be filled in)"
SKIP_DIRECTIVE = "<!-- skip -->"


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Split text into chunks respecting markdown boundaries.
    Priority: split on ## headings, then # headings, then double
    newlines, then single newlines, then sentences, then words.
    chunk_size is in CHARACTERS not words.
    """
    chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP

    separators = ["\n## ", "\n# ", "\n\n", "\n", ". ", " "]

    def split_recursive(text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep = separators[0]
        splits = text.split(sep)

        chunks = []
        current = ""

        for split in splits:
            candidate = (current + sep + split).strip() if current else split.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(split.strip()) > chunk_size:
                    sub_chunks = split_recursive(split.strip(), separators[1:])
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = split.strip()

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if len(c.strip()) > 80]

    raw_chunks = split_recursive(text, separators)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i-1][-overlap:] if len(raw_chunks[i-1]) > overlap else raw_chunks[i-1]
        overlapped.append(prev_tail + " " + raw_chunks[i])

    return overlapped


def _is_skippable(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if stripped.startswith(SKIP_DIRECTIVE):
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
    # Pass embedding as a Python list — pgvector asyncpg codec encodes it correctly.
    # Pass metadata as a JSON string with explicit ::jsonb cast.
    await conn.execute(
        """
        INSERT INTO documents (id, content, embedding, metadata, label)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        ON CONFLICT (id) DO UPDATE
            SET content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                label = EXCLUDED.label;
        """,
        doc_id,
        content,
        embedding,
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
                metadata = {"source": "knowledge_file", "label": filename, "chunk_index": i}
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
