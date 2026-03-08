"""
pipeline/scraper.py - Sitemap-based 2 Satoshis blog scraper
Fetches all post URLs from ar.ca sitemap, then scrapes each individually via Firecrawl.
"""

import asyncio
import httpx
import os
import xml.etree.ElementTree as ET
from datetime import datetime

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
BLOG_BASE = "https://www.ar.ca/blog"
SITEMAP_URLS = [
    "https://www.ar.ca/sitemap.xml",
    "https://www.ar.ca/blog-sitemap.xml",
    "https://www.ar.ca/sitemap_index.xml",
]


async def fetch_sitemap_urls(client: httpx.AsyncClient) -> list[str]:
    """Try multiple sitemap locations and extract all blog post URLs."""
    blog_urls = []

    for sitemap_url in SITEMAP_URLS:
        try:
            resp = await client.get(sitemap_url, timeout=15)
            if resp.status_code != 200:
                continue

            root = ET.fromstring(resp.text)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            # Check if this is a sitemap index (points to other sitemaps)
            sub_sitemaps = root.findall(".//sm:sitemap/sm:loc", ns)
            if sub_sitemaps:
                for sub in sub_sitemaps:
                    sub_url = sub.text.strip()
                    if "blog" in sub_url or "post" in sub_url:
                        try:
                            sub_resp = await client.get(sub_url, timeout=15)
                            if sub_resp.status_code == 200:
                                sub_root = ET.fromstring(sub_resp.text)
                                for loc in sub_root.findall(".//sm:url/sm:loc", ns):
                                    url = loc.text.strip()
                                    if "/blog/" in url:
                                        blog_urls.append(url)
                        except Exception:
                            pass

            # Direct sitemap entries
            for loc in root.findall(".//sm:url/sm:loc", ns):
                url = loc.text.strip()
                if "/blog/" in url:
                    blog_urls.append(url)

            if blog_urls:
                print(f"[Scraper] Found {len(blog_urls)} blog URLs from {sitemap_url}")
                break

        except Exception as e:
            print(f"[Scraper] Sitemap {sitemap_url} failed: {e}")
            continue

    return list(set(blog_urls))  # deduplicate


async def scrape_url_firecrawl(client: httpx.AsyncClient, url: str) -> dict | None:
    """Scrape a single URL via Firecrawl scrape endpoint."""
    try:
        resp = await client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
                "timeout": 30000,
            },
            timeout=45,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data.get("success"):
            return None

        content = data.get("data", {})
        markdown = content.get("markdown", "")
        metadata = content.get("metadata", {})

        if len(markdown) < 200:  # skip near-empty pages
            return None

        return {
            "url": url,
            "title": metadata.get("title", ""),
            "content": markdown,
            "published_at": metadata.get("publishedTime") or metadata.get("og:article:published_time"),
        }

    except Exception as e:
        print(f"[Scraper] Failed to scrape {url}: {e}")
        return None


async def fallback_direct_scrape(client: httpx.AsyncClient, url: str) -> dict | None:
    """Direct HTTP fetch + basic HTML stripping as fallback."""
    try:
        resp = await client.get(url, timeout=20, follow_redirects=True)
        if resp.status_code != 200:
            return None

        html = resp.text

        # Extract title
        title = ""
        if "<title>" in html:
            title = html.split("<title>")[1].split("</title>")[0].strip()

        # Strip scripts, styles, nav
        import re
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
        html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL)
        html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL)
        html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL)

        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        # Try to find the main content block (after "Market Recap" or post title)
        if "And That's Our Two Satoshis" in text or "And That's Our 2 Satoshis" in text:
            # Has full post content
            if len(text) < 300:
                return None
            return {"url": url, "title": title, "content": text[:50000], "published_at": None}

        return None

    except Exception as e:
        print(f"[Scraper] Direct fetch failed for {url}: {e}")
        return None


async def crawl_blog(max_pages: int = 500) -> list[dict]:
    """
    Main entry point. 
    Strategy:
    1. Try sitemap to get all post URLs
    2. If sitemap fails, fall back to known URL patterns + Firecrawl crawl
    3. Scrape each URL individually
    """
    posts = []

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (compatible; ArcaCollective/1.0)"},
        timeout=60,
    ) as client:

        # Step 1: Get URLs from sitemap
        print("[Scraper] Attempting sitemap discovery...")
        urls = await fetch_sitemap_urls(client)

        # Step 2: If sitemap failed, try Firecrawl map endpoint
        if not urls and FIRECRAWL_API_KEY:
            print("[Scraper] Sitemap empty, trying Firecrawl /map...")
            try:
                resp = await client.post(
                    "https://api.firecrawl.dev/v1/map",
                    headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                    json={"url": BLOG_BASE, "limit": max_pages},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    urls = [u for u in data.get("links", []) if "/blog/" in u and u != BLOG_BASE]
                    print(f"[Scraper] Firecrawl map returned {len(urls)} URLs")
            except Exception as e:
                print(f"[Scraper] Firecrawl map failed: {e}")

        # Step 3: Fallback to crawl if still nothing
        if not urls and FIRECRAWL_API_KEY:
            print("[Scraper] Falling back to Firecrawl crawl...")
            try:
                resp = await client.post(
                    "https://api.firecrawl.dev/v1/crawl",
                    headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                    json={
                        "url": BLOG_BASE,
                        "limit": max_pages,
                        "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Poll for completion
                    job_id = data.get("id")
                    if job_id:
                        for _ in range(60):
                            await asyncio.sleep(5)
                            status_resp = await client.get(
                                f"https://api.firecrawl.dev/v1/crawl/{job_id}",
                                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                                timeout=30,
                            )
                            if status_resp.status_code == 200:
                                status_data = status_resp.json()
                                if status_data.get("status") == "completed":
                                    for page in status_data.get("data", []):
                                        meta = page.get("metadata", {})
                                        url = meta.get("url") or meta.get("sourceURL", "")
                                        if "/blog/" in url:
                                            posts.append({
                                                "url": url,
                                                "title": meta.get("title", ""),
                                                "content": page.get("markdown", ""),
                                                "published_at": meta.get("publishedTime"),
                                            })
                                    print(f"[Scraper] Crawl completed: {len(posts)} posts")
                                    return [p for p in posts if len(p.get("content", "")) > 200]
                                elif status_data.get("status") == "failed":
                                    break
            except Exception as e:
                print(f"[Scraper] Crawl failed: {e}")

        if not urls:
            print("[Scraper] Could not discover URLs. Returning empty list.")
            return []

        print(f"[Scraper] Scraping {len(urls)} individual URLs...")

        # Step 4: Scrape each URL with concurrency limit
        semaphore = asyncio.Semaphore(5)  # 5 concurrent requests

        async def scrape_with_semaphore(url: str) -> dict | None:
            async with semaphore:
                result = None
                if FIRECRAWL_API_KEY:
                    result = await scrape_url_firecrawl(client, url)
                if not result:
                    result = await fallback_direct_scrape(client, url)
                return result

        tasks = [scrape_with_semaphore(url) for url in urls[:max_pages]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, dict) and r:
                posts.append(r)

        print(f"[Scraper] Successfully scraped {len(posts)} posts")
        return posts


if __name__ == "__main__":
    # Test run
    async def main():
        posts = await crawl_blog(max_pages=500)
        total_words = sum(len(p["content"].split()) for p in posts)
        print(f"\nResults: {len(posts)} posts, ~{total_words:,} words")
        for p in posts[:5]:
            print(f"  - {p['title'][:60]} ({len(p['content'])} chars)")

    asyncio.run(main())
