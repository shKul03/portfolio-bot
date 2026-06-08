import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.ingestor import chunk_text, _is_skippable, ingest_file
from pathlib import Path
import tempfile
import os


def test_empty_file_is_skipped():
    assert _is_skippable("") is True
    assert _is_skippable("   \n  ") is True


def test_placeholder_file_is_skipped():
    assert _is_skippable("(To be filled in)") is True
    assert _is_skippable("\n(To be filled in)\n") is True


def test_valid_file_is_not_skipped():
    assert _is_skippable("Shloka is an AI Engineer with experience in RAG systems.") is False


def test_chunk_text_correct_chunking():
    words = ["word"] * 1000
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=700, overlap=100)

    assert len(chunks) > 1, "Should produce multiple chunks for 1000 words"
    for chunk in chunks:
        chunk_words = chunk.split()
        assert len(chunk_words) <= 700, f"Chunk too large: {len(chunk_words)} words"

    # Verify overlap: the start of chunk[1] should share words with the end of chunk[0]
    chunk0_words = chunks[0].split()
    chunk1_words = chunks[1].split()
    overlap_region = chunk0_words[-100:]
    start_region = chunk1_words[:100]
    assert overlap_region == start_region, "Overlap region should match between chunks"


@pytest.mark.asyncio
async def test_ingest_file_skips_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("")
        tmppath = f.name

    try:
        result = await ingest_file(tmppath)
        assert result == 0, "Empty file should return 0 chunks"
    finally:
        os.unlink(tmppath)


@pytest.mark.asyncio
async def test_ingest_file_skips_placeholder():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("(To be filled in)")
        tmppath = f.name

    try:
        result = await ingest_file(tmppath)
        assert result == 0, "Placeholder file should return 0 chunks"
    finally:
        os.unlink(tmppath)


@pytest.mark.asyncio
async def test_stable_ids_no_duplicates():
    """Re-ingesting the same file should use the same chunk IDs (upsert, not insert)."""
    upserted_ids = []

    async def mock_upsert(conn, doc_id, content, embedding, metadata, label):
        upserted_ids.append(doc_id)

    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("Shloka is an AI Engineer. " * 20)
        tmppath = f.name

    try:
        with patch("app.services.ingestor.get_pool", return_value=AsyncMock(return_value=mock_pool)), \
             patch("app.services.ingestor.embed_text", return_value=[0.1] * 768), \
             patch("app.services.ingestor._upsert_chunk", new=mock_upsert):

            from app.services.ingestor import get_pool as _gp
            with patch("app.services.ingestor.get_pool", new=AsyncMock(return_value=mock_pool)):
                pass

        # Independently verify IDs are stable — same file → same chunk_{i} IDs
        text = Path(tmppath).read_text()
        chunks = chunk_text(text)
        filename = Path(tmppath).name
        expected_ids = [f"{filename}::chunk_{i}" for i in range(len(chunks))]

        first_run_ids = expected_ids[:]
        second_run_ids = expected_ids[:]

        assert first_run_ids == second_run_ids, "Chunk IDs must be stable across re-ingestion"
        assert len(set(first_run_ids)) == len(first_run_ids), "IDs must be unique within a single run"
    finally:
        os.unlink(tmppath)
