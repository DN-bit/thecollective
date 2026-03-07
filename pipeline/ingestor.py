# The Collective - Automated News Ingestion Pipeline
# Pulls from RSS feeds + CryptoPanic, scores impact, submits to /generate
# Runs as a Render Cron Job every 2 hours

import os
import sys
import json
import hashlib
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

import httpx
import feedparser
import asyncpg

logging.basicConfig(level=logging.INFO, format='[Ingestor] %(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "https://collective-api-3plq.onrender.com")

# RSS feeds — free, no API key needed
RSS_FEEDS = [
    # Macro / Traditional Finance
    {"url": "https://feeds.bloomberg.com/markets/news.rss",         "domain": "macro",     "source": "Bloomberg"},
    {"url": "https://feeds.reuters.com/reuters/businessNews",        "domain": "macro",     "source": "Reuters"},
    {"url": "https://www.ft.com/rss/home",                          "domain": "macro",     "source": "FT"},
    # Crypto
    {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/",      "domain": "macro",     "source": "CoinDesk"},
    {"url": "https://cointelegraph.com/rss",                        "domain": "macro",     "source": "CoinTelegraph"},
    {"url": "https://theblock.co/rss.xml",                          "domain": "defi",      "source": "TheBlock"},
    {"url": "https://decrypt.co/feed",                              "domain": "sentiment", "source": "Decrypt"},
    # Policy / Regulatory
    {"url": "https://www.sec.gov/rss/news/press.xml",               "domain": "policy",    "source": "SEC"},
    {"url": "https://www.cftc.gov/rss/pressreleases.xml",           "domain": "policy",    "source": "CFTC"},
]

# CryptoPanic API — free tier available at cryptopanic.com
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/?auth_token={key}&filter=important&public=true"

# ---------------------------------------------------------------------------
# Impact scoring — keyword-based heuristic
# ---------------------------------------------------------------------------

IMPACT_RULES = [
    # High impact (0.75 - 0.95)
    (0.90, ["war", "invasion", "nuclear", "sanctions", "fed rate", "federal reserve", "rate hike", "rate cut",
            "emergency", "collapse", "bankruptcy", "contagion", "systemic"]),
    (0.85, ["iran", "israel", "russia", "ukraine", "china tariff", "sec enforcement", "cftc action",
            "executive order", "ban", "seized", "arrested", "indicted"]),
    (0.80, ["etf approved", "etf rejected", "spot bitcoin", "spot ethereum", "strategic reserve",
            "treasury", "sovereign", "ipo", "acquisition", "merger"]),
    (0.75, ["inflation", "cpi", "pce", "gdp", "recession", "fomc", "powell", "yellen",
            "clarity act", "fit21", "regulation", "legislation", "congress"]),
    # Medium impact (0.50 - 0.70)
    (0.65, ["partnership", "integration", "launch", "upgrade", "hack", "exploit", "vulnerability",
            "defi protocol", "tvl", "yield", "airdrop", "token"]),
    (0.55, ["analyst", "report", "forecast", "prediction", "survey", "institutional",
            "fund", "investment", "allocation"]),
    (0.50, ["price", "market", "trading", "volume", "liquidation", "long", "short"]),
    # Low impact (0.30)
    (0.30, []),  # default
]

def score_impact(title: str, summary: str) -> float:
    text = (title + " " + summary).lower()
    for score, keywords in IMPACT_RULES:
        if any(kw in text for kw in keywords):
            return score
    return 0.30


CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "defi", "nft",
    "token", "stablecoin", "web3", "solana", "sol", "ripple", "xrp", "binance",
    "coinbase", "exchange", "wallet", "mining", "validator", "staking", "yield",
    "protocol", "dao", "dex", "amm", "tvl", "airdrop", "altcoin", "memecoin",
    "digital asset", "digital currency", "central bank digital", "cbdc",
    "sec crypto", "cftc crypto", "fed crypto", "treasury crypto",
    "spot etf", "bitcoin etf", "ethereum etf", "crypto etf",
    "tether", "usdc", "usdt", "stablecoin", "circle", "paxos",
    "blackrock bitcoin", "fidelity bitcoin", "grayscale",
    "on-chain", "layer 2", "layer2", "rollup", "zk proof",
    "arca", "dragonfly", "a16z crypto", "paradigm", "multicoin"
]

def is_crypto_relevant(title: str, summary: str) -> bool:
    """Returns True only if the story has a genuine crypto angle."""
    text = (title + " " + summary).lower()
    return any(kw in text for kw in CRYPTO_KEYWORDS)


def infer_domain(title: str, summary: str, default_domain: str) -> str:
    text = (title + " " + summary).lower()
    if any(k in text for k in ["sec", "cftc", "congress", "senate", "regulation", "legislation", "law", "clarity", "fit21"]):
        return "policy"
    if any(k in text for k in ["defi", "protocol", "tvl", "yield", "liquidity", "amm", "dex", "lending"]):
        return "defi"
    if any(k in text for k in ["twitter", "sentiment", "community", "social", "reddit", "meme", "culture"]):
        return "sentiment"
    return default_domain


def make_event_id(title: str) -> str:
    """Deterministic ID based on title — prevents duplicate ingestion."""
    h = hashlib.md5(title.encode()).hexdigest()[:12]
    return f"auto_{h}"


# ---------------------------------------------------------------------------
# Deduplication — check if event_id already in corpus
# ---------------------------------------------------------------------------

async def already_ingested(conn, event_id: str) -> bool:
    row = await conn.fetchrow("SELECT id FROM corpus WHERE event_id = $1", event_id)
    return row is not None


# ---------------------------------------------------------------------------
# RSS ingestion
# ---------------------------------------------------------------------------

async def fetch_rss_items(client: httpx.AsyncClient, feed: dict) -> list:
    items = []
    try:
        resp = await client.get(feed["url"], timeout=10)
        parsed = feedparser.parse(resp.text)
        for entry in parsed.entries[:10]:  # top 10 per feed
            title = entry.get("title", "").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()
            # Strip HTML tags from summary
            import re
            summary = re.sub(r'<[^>]+>', '', summary)[:500]
            if not title or len(title) < 20:
                continue
            items.append({
                "title": title,
                "summary": summary,
                "source": feed["source"],
                "default_domain": feed["domain"],
            })
    except Exception as e:
        log.warning(f"RSS fetch failed for {feed['source']}: {e}")
    return items


# ---------------------------------------------------------------------------
# CryptoPanic ingestion
# ---------------------------------------------------------------------------

async def fetch_cryptopanic(client: httpx.AsyncClient) -> list:
    if not CRYPTOPANIC_API_KEY:
        return []
    items = []
    try:
        url = CRYPTOPANIC_URL.format(key=CRYPTOPANIC_API_KEY)
        resp = await client.get(url, timeout=10)
        data = resp.json()
        for post in data.get("results", [])[:15]:
            title = post.get("title", "").strip()
            if not title:
                continue
            # CryptoPanic gives votes — use as impact signal
            votes = post.get("votes", {})
            positive = votes.get("positive", 0)
            negative = votes.get("negative", 0)
            important = votes.get("important", 0)
            items.append({
                "title": title,
                "summary": f"CryptoPanic importance signals — positive:{positive} negative:{negative} important:{important}",
                "source": "CryptoPanic",
                "default_domain": "macro",
                "vote_boost": min(important * 0.02, 0.2),  # boost impact for highly voted items
            })
    except Exception as e:
        log.warning(f"CryptoPanic fetch failed: {e}")
    return items


# ---------------------------------------------------------------------------
# Submit to /generate endpoint
# ---------------------------------------------------------------------------

async def submit_to_collective(client: httpx.AsyncClient, item: dict) -> Optional[dict]:
    description = item["title"]
    if item.get("summary") and len(item["summary"]) > 30:
        description += f". {item['summary'][:300]}"

    impact = score_impact(item["title"], item.get("summary", ""))
    impact = min(impact + item.get("vote_boost", 0.0), 0.95)

    # Skip low-impact items to conserve OpenAI budget
    if impact < 0.45:
        log.info(f"Skipping low-impact item (score {impact:.2f}): {item['title'][:60]}")
        return None

    domain = infer_domain(item["title"], item.get("summary", ""), item["default_domain"])

    try:
        resp = await client.post(
            f"{API_BASE}/generate",
            json={
                "description": description,
                "impact": impact,
                "domain": domain,
                "source": item["source"],
            },
            timeout=60  # generation can take a while
        )
        if resp.status_code == 200:
            data = resp.json()
            log.info(f"✓ Generated [{domain}] {impact:.2f} — {item['title'][:60]}")
            return data
        else:
            log.warning(f"Generate failed {resp.status_code}: {item['title'][:60]}")
            return None
    except Exception as e:
        log.warning(f"Submit error: {e} — {item['title'][:60]}")
        return None


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

async def run():
    log.info("=== Ingestion cycle starting ===")

    # DB connection for deduplication
    database_url = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://")
    if not database_url:
        log.error("DATABASE_URL not set — cannot deduplicate")
        sys.exit(1)

    db_conn = await asyncpg.connect(database_url)

    async with httpx.AsyncClient(
        headers={"User-Agent": "TheCollective/1.0 (Arca Intelligence Ingestor)"},
        follow_redirects=True
    ) as client:

        # Collect all items
        all_items = []

        # RSS
        for feed in RSS_FEEDS:
            items = await fetch_rss_items(client, feed)
            all_items.extend(items)
            log.info(f"RSS {feed['source']}: {len(items)} items")

        # CryptoPanic
        cp_items = await fetch_cryptopanic(client)
        all_items.extend(cp_items)
        log.info(f"CryptoPanic: {len(cp_items)} items")

        log.info(f"Total raw items: {len(all_items)}")

        # Deduplicate and submit
        submitted = 0
        skipped_dup = 0
        skipped_low = 0

        skipped_irrelevant = 0

        for item in all_items:
            # Crypto relevance gate — drop non-crypto stories entirely
            if not is_crypto_relevant(item["title"], item.get("summary", "")):
                skipped_irrelevant += 1
                log.info(f"SKIP (not crypto): {item['title'][:60]}")
                continue

            event_id = make_event_id(item["title"])

            # Check deduplication
            if await already_ingested(db_conn, event_id):
                skipped_dup += 1
                continue

            # Score and submit
            impact = score_impact(item["title"], item.get("summary", ""))
            if impact < 0.45:
                skipped_low += 1
                continue

            result = await submit_to_collective(client, item)
            if result and result.get("status") == "generated":
                submitted += 1
                # Pace requests — don't hammer OpenAI
                await asyncio.sleep(3)

        await db_conn.close()

        log.info(f"=== Cycle complete: {submitted} generated, {skipped_dup} duplicates, {skipped_low} low-impact, {skipped_irrelevant} not crypto-relevant ===")


if __name__ == "__main__":
    asyncio.run(run())
