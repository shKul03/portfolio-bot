import logging
import re
from urllib.parse import urljoin, urlparse
from app.services.ingestor import chunk_text, ingest_chunks_with_label
from app.config import settings

logger = logging.getLogger(__name__)

COMMON_PATHS = [
    "/", "/about", "/projects", "/experience", "/skills",
    "/contact", "/blog", "/work", "/portfolio",
]


def _clean_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def _fetch_page(page, url: str) -> str | None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        html = await page.content()
        return _clean_html(html)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


async def _discover_urls_from_sitemap(page, base_url: str) -> list[str]:
    sitemap_urls = [
        urljoin(base_url, "/sitemap.xml"),
        urljoin(base_url, "/sitemap_index.xml"),
    ]
    found_urls = []

    for sitemap_url in sitemap_urls:
        try:
            await page.goto(sitemap_url, wait_until="domcontentloaded", timeout=10000)
            content = await page.content()
            urls = re.findall(r"<loc>(https?://[^<]+)</loc>", content)
            base_domain = urlparse(base_url).netloc
            for url in urls:
                if urlparse(url).netloc == base_domain:
                    found_urls.append(url)
            if found_urls:
                logger.info(f"Found {len(found_urls)} URLs from sitemap at {sitemap_url}.")
                return found_urls
        except Exception:
            continue

    return []


async def crawl_and_ingest(url: str, label: str) -> int:
    from playwright.async_api import async_playwright

    base_url = url.rstrip("/")
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    total_chunks = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (compatible; PortfolioBotCrawler/1.0)"
        )
        page = await context.new_page()

        discovered_urls = await _discover_urls_from_sitemap(page, origin)

        if not discovered_urls:
            logger.info("No sitemap found — falling back to common path crawling.")
            discovered_urls = [urljoin(origin, path) for path in COMMON_PATHS]

        visited = set()
        for crawl_url in discovered_urls:
            if crawl_url in visited:
                continue
            visited.add(crawl_url)

            text = await _fetch_page(page, crawl_url)
            if not text or len(text) < 100:
                continue

            chunks = chunk_text(
                text,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP,
            )
            base_id = f"{label}::{crawl_url}"
            count = await ingest_chunks_with_label(chunks, label=label, base_id=base_id)
            total_chunks += count
            logger.info(f"Crawled {crawl_url} → {count} chunks ingested.")

        await browser.close()

    logger.info(f"Crawl complete for label='{label}': {total_chunks} total chunks.")
    return total_chunks
