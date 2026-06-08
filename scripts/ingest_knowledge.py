#!/usr/bin/env python3
"""Ingest all knowledge/*.md files into the vector store."""
import asyncio
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import init_db, close_db
from app.services.ingestor import ingest_file

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
PLACEHOLDER = "(To be filled in)"


def _is_skippable(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if stripped == PLACEHOLDER:
        return True
    if PLACEHOLDER in stripped and len(stripped) < len(PLACEHOLDER) + 100:
        return True
    return False


async def main():
    await init_db()

    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not md_files:
        print("No .md files found in knowledge/ directory.")
        await close_db()
        return

    total = 0
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        if _is_skippable(content):
            print(f"Skipping {md_file.name} — empty or placeholder.")
            continue

        print(f"Ingesting {md_file.relative_to(md_file.parent.parent)}...", end=" ", flush=True)
        count = await ingest_file(md_file)
        print(f"done ({count} chunks)")
        total += count

    print(f"\nIngestion complete — {total} total chunks stored.")
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
