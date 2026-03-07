# The Collective - 2 Satoshis Blog Scraper
# Crawls all Arca "That's Our 2 Satoshis" posts via Firecrawl
# Stores clean text in arca_posts table for RAG retrieval

import asyncio
import hashlib
import json
import os
import sys
import time

import asyncpg
import httpx

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://")

BLOG_URL = "https://www.ar.ca/blog/tag/market-recap"
FIRECRAWL_BASE = "https://api.firecrawl.dev/v1"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

async def get_conn():
    return await asyncpg.connect(DATABASE_URL)


async def init_posts_table():
    conn = await get_conn()
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS arca_posts (
            id SERIAL PRIMARY KEY,
            post_id TEXT UNIQUE NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            published_date TEXT,
            content TEXT NOT NULL,
            word_count INTEGER,
            scraped_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    await conn.close()
    print("[Scraper] arca_posts table ready")


# ---------------------------------------------------------------------------
# Firecrawl
# ---------------------------------------------------------------------------

async def crawl_blog(max_pages: int = 200) -> list:
    """
    Use Firecrawl to crawl all 2 Satoshis blog posts.
    Returns list of {url, title, content, publishedDate} dicts.
    """
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Start crawl job
    print(f"[Scraper] Starting Firecrawl crawl of {BLOG_URL}...")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{FIRECRAWL_BASE}/crawl",
            headers=headers,
            json={
                "url": BLOG_URL,
                "limit": max_pages,
                "scrapeOptions": {
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                    "excludeTags": ["nav", "footer", "header", ".hs-nav", ".blog-listing", ".related-posts"]
                },
                "includePaths": ["/blog/"],
                "excludePaths": ["/blog/tag/", "/blog/author/"],
            }
        )
        if resp.status_code != 200:
            raise Exception(f"Firecrawl crawl start failed: {resp.status_code} {resp.text}")

        crawl_data = resp.json()
        crawl_id = crawl_data.get("id")
        if not crawl_id:
            raise Exception(f"No crawl ID returned: {crawl_data}")
        print(f"[Scraper] Crawl started — ID: {crawl_id}")

    # Poll for completion
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            await asyncio.sleep(5)
            status_resp = await client.get(
                f"{FIRECRAWL_BASE}/crawl/{crawl_id}",
                headers=headers
            )
            status_data = status_resp.json()
            status = status_data.get("status")
            completed = status_data.get("completed", 0)
            total = status_data.get("total", 0)
            print(f"[Scraper] Status: {status} — {completed}/{total} pages")

            if status == "completed":
                return status_data.get("data", [])
            elif status == "failed":
                raise Exception(f"Crawl failed: {status_data}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_post(raw: dict) -> dict | None:
    """Extract and clean a post from Firecrawl result."""
    url = raw.get("metadata", {}).get("url", "") or raw.get("url", "")
    title = raw.get("metadata", {}).get("title", "") or raw.get("metadata", {}).get("og:title", "")
    published = raw.get("metadata", {}).get("article:published_time", "") or \
                raw.get("metadata", {}).get("datePublished", "")
    content = raw.get("markdown", "") or raw.get("content", "")

    # Skip index/tag pages and very short pages
    if not content or len(content) < 500:
        return None
    if "/blog/tag/" in url or "/blog/author/" in url:
        return None
    if "market-recap" in url and url.endswith("/blog/tag/market-recap"):
        return None

    # Clean up common HubSpot boilerplate
    lines = content.split("\n")
    cleaned_lines = []
    skip_patterns = [
        "subscribe", "newsletter", "follow us", "share this",
        "related posts", "you might also like", "cookie", "privacy policy",
        "©", "all rights reserved", "arca.com", "ar.ca/blog"
    ]
    for line in lines:
        line_lower = line.lower()
        if any(p in line_lower for p in skip_patterns):
            continue
        cleaned_lines.append(line)

    clean_content = "\n".join(cleaned_lines).strip()
    word_count = len(clean_content.split())

    if word_count < 200:
        return None

    post_id = hashlib.md5(url.encode()).hexdigest()

    return {
        "post_id": post_id,
        "url": url,
        "title": title,
        "published_date": published,
        "content": clean_content,
        "word_count": word_count
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

async def save_posts(posts: list) -> int:
    conn = await get_conn()
    saved = 0
    for post in posts:
        try:
            result = await conn.execute("""
                INSERT INTO arca_posts (post_id, url, title, published_date, content, word_count)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (post_id) DO NOTHING
            """, post["post_id"], post["url"], post["title"],
                post["published_date"], post["content"], post["word_count"])
            if result == "INSERT 0 1":
                saved += 1
        except Exception as e:
            print(f"[Scraper] WARNING: Failed to save {post['url']}: {e}")
    await conn.close()
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("[Scraper] Starting 2 Satoshis ingestion...")

    if not FIRECRAWL_API_KEY:
        print("[Scraper] ERROR: FIRECRAWL_API_KEY not set")
        sys.exit(1)
    if not DATABASE_URL:
        print("[Scraper] ERROR: DATABASE_URL not set")
        sys.exit(1)

    await init_posts_table()

    # Crawl
    raw_pages = await crawl_blog(max_pages=300)
    print(f"[Scraper] Crawl returned {len(raw_pages)} pages")

    # Clean
    posts = []
    for raw in raw_pages:
        cleaned = clean_post(raw)
        if cleaned:
            posts.append(cleaned)

    print(f"[Scraper] {len(posts)} valid posts after cleaning")

    # Save
    saved = await save_posts(posts)
    print(f"[Scraper] Saved {saved} new posts to arca_posts table")
    print(f"[Scraper] Total unique posts: {len(posts)}")

    # Summary
    conn = await get_conn()
    total = await conn.fetchval("SELECT COUNT(*) FROM arca_posts")
    total_words = await conn.fetchval("SELECT SUM(word_count) FROM arca_posts")
    await conn.close()
    print(f"[Scraper] Database now has {total} posts, ~{total_words:,} total words")
    print("[Scraper] Done. Ready for RAG indexing.")


if __name__ == "__main__":
    asyncio.run(main())
