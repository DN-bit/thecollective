# The Collective - Automated News Ingestion Pipeline
# Pulls from RSS feeds + CryptoPanic, scores impact, submits to /generate
# Runs as a Render Cron Job every 2 hours

import os
import re
import sys
import json
import hashlib
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import feedparser
import asyncpg

logging.basicConfig(level=logging.INFO, format='[Ingestor] %(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "https://collective-api-3plq.onrender.com")

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

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/?auth_token={key}&filter=important&public=true"

# ---------------------------------------------------------------------------
# Relevance — two-tier system
# ---------------------------------------------------------------------------

# Tier 1: Strong crypto signals — sufficient on their own
CRYPTO_STRONG = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "ripple", "xrp",
    "binance", "coinbase", "tether", "usdc", "usdt", "circle", "paxos",
    "grayscale", "blackrock bitcoin", "fidelity bitcoin",
    "crypto", "blockchain", "defi", "nft", "stablecoin", "web3",
    "digital asset", "digital currency", "cbdc", "central bank digital",
    "bitcoin etf", "ethereum etf", "spot etf",
    "on-chain", "layer 2", "layer2", "rollup", "zk proof", "zk-proof",
    "arca", "dragonfly capital", "a16z crypto", "paradigm fund", "multicoin capital",
    "airdrop", "memecoin", "altcoin", "dao governance",
]

# Tier 2: Ambiguous — only accepted from known crypto sources
CRYPTO_AMBIGUOUS = [
    "token", "protocol", "exchange", "wallet", "mining", "validator",
    "staking", "yield", "tvl", "dex", "amm", "liquidity pool",
]

# Sources where ambiguous terms are fine
CRYPTO_SOURCES = {"CoinDesk", "CoinTelegraph", "TheBlock", "Decrypt", "CryptoPanic"}

# Hard exclusion patterns — drop even if crypto keywords appear
NOISE_PATTERNS = [
    r"\bpokemon\b", r"\bminecraft\b", r"\bfortnite\b", r"\bgaming token\b",
    r"\bnft art\b", r"\bnft drop\b", r"\bcelebrity nft\b", r"\bmeme coin launch\b",
    r"\binfluencer\b", r"\byoutuber\b", r"\btwitch\b",
    r"\bexchange student\b",
    r"\bprotocol (meeting|talks|agreement|accord)\b",
    r"\bmining (company|stock|copper|gold|coal|iron)\b(?!.*crypto|.*bitcoin)",
    r"\btoken (gesture|of appreciation|ring)\b",
    r"\bwallet (size|theft|lost|found)\b(?!.*crypto|.*bitcoin)",
]

def is_crypto_relevant(title: str, summary: str, source: str = "") -> bool:
    text = (title + " " + summary).lower()

    # Hard exclusions first
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    # Strong signal — always relevant
    if any(kw in text for kw in CRYPTO_STRONG):
        return True

    # Ambiguous terms — only from crypto-focused sources
    if source in CRYPTO_SOURCES:
        if any(kw in text for kw in CRYPTO_AMBIGUOUS):
            return True

    return False


# ---------------------------------------------------------------------------
# Impact scoring
# ---------------------------------------------------------------------------

IMPACT_RULES = [
    (0.90, ["war", "invasion", "nuclear", "sanctions", "fed rate", "federal reserve", "rate hike", "rate cut",
            "emergency", "collapse", "bankruptcy", "contagion", "systemic"]),
    (0.85, ["iran", "israel", "russia", "ukraine", "china tariff", "sec enforcement", "cftc action",
            "executive order", "ban", "seized", "arrested", "indicted"]),
    (0.80, ["etf approved", "etf rejected", "spot bitcoin", "spot ethereum", "strategic reserve",
            "treasury", "sovereign", "ipo", "acquisition", "merger"]),
    (0.75, ["inflation", "cpi", "pce", "gdp", "recession", "fomc", "powell", "yellen",
            "clarity act", "fit21", "regulation", "legislation", "congress"]),
    (0.65, ["partnership", "integration", "launch", "upgrade", "hack", "exploit", "vulnerability",
            "defi protocol", "tvl", "yield", "airdrop"]),
    (0.55, ["analyst", "report", "forecast", "prediction", "survey", "institutional",
            "fund", "investment", "allocation"]),
    (0.50, ["price", "market", "trading", "volume", "liquidation"]),
    (0.30, []),
]

def score_impact(title: str, summary: str) -> float:
    text = (title + " " + summary).lower()
    for score, keywords in IMPACT_RULES:
        if any(kw in text for kw in keywords):
            return score
    return 0.30


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

def infer_domain(title: str, summary: str, default_domain: str) -> str:
    text = (title + " " + summary).lower()
    if any(k in text for k in ["sec", "cftc", "congress", "senate", "regulation", "legislation", "law", "clarity", "fit21"]):
        return "policy"
    if any(k in text for k in ["defi", "protocol", "tvl", "yield", "liquidity", "amm", "dex", "lending"]):
        return "defi"
    if any(k in text for k in ["twitter", "sentiment", "community", "social", "reddit", "meme", "culture"]):
        return "sentiment"
    return default_domain


# ---------------------------------------------------------------------------
# Deduplication — semantic fingerprint catches near-duplicates
# ---------------------------------------------------------------------------

def make_event_id(title: str) -> str:
    """Normalized title hash — catches same story with minor wording differences."""
    normalized = re.sub(r'[^\w\s]', '', title.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    stop = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
            "in", "on", "at", "to", "for", "of", "and", "or", "but", "as"}
    tokens = [w for w in normalized.split() if w not in stop]
    fingerprint = " ".join(tokens[:8])
    h = hashlib.md5(fingerprint.encode()).hexdigest()[:12]
    return f"auto_{h}"


def make_content_hash(title: str, summary: str) -> Optional[str]:
    """Key entity hash — catches rephrased versions of same story."""
    tokens = re.findall(
        r'\b([A-Z]{2,6}|\d+\.?\d*[bBmMkK]?|bitcoin|ethereum|solana|binance|coinbase)\b',
        title + " " + summary[:200]
    )
    if len(tokens) >= 3:
        key = " ".join(sorted(set(t.lower() for t in tokens[:6])))
        return hashlib.md5(key.encode()).hexdigest()[:12]
    return None


async def already_ingested(conn, event_id: str, content_hash: Optional[str] = None) -> bool:
    row = await conn.fetchrow("SELECT id FROM corpus WHERE event_id = $1", event_id)
    if row:
        return True
    if content_hash:
        row = await conn.fetchrow(
            "SELECT id FROM corpus WHERE source LIKE $1",
            f"%hash:{content_hash}%"
        )
        if row:
            return True
    return False


# ---------------------------------------------------------------------------
# RSS ingestion
# ---------------------------------------------------------------------------

async def fetch_rss_items(client: httpx.AsyncClient, feed: dict) -> list:
    items = []
    try:
        resp = await client.get(feed["url"], timeout=10)
        parsed = feedparser.parse(resp.text)
        for entry in parsed.entries[:10]:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()
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
            votes = post.get("votes", {})
            important = votes.get("important", 0)
            items.append({
                "title": title,
                "summary": f"CryptoPanic signals — positive:{votes.get('positive',0)} negative:{votes.get('negative',0)} important:{important}",
                "source": "CryptoPanic",
                "default_domain": "macro",
                "vote_boost": min(important * 0.02, 0.2),
            })
    except Exception as e:
        log.warning(f"CryptoPanic fetch failed: {e}")
    return items


# ---------------------------------------------------------------------------
# Submit to /generate
# ---------------------------------------------------------------------------

async def submit_to_collective(
    client: httpx.AsyncClient, item: dict, content_hash: Optional[str] = None
) -> Optional[dict]:
    description = item["title"]
    if item.get("summary") and len(item["summary"]) > 30:
        description += f". {item['summary'][:300]}"

    impact = score_impact(item["title"], item.get("summary", ""))
    impact = min(impact + item.get("vote_boost", 0.0), 0.95)

    if impact < 0.55:
        return None

    domain = infer_domain(item["title"], item.get("summary", ""), item["default_domain"])
    source = item["source"]
    if content_hash:
        source = f"{source} hash:{content_hash}"

    try:
        resp = await client.post(
            f"{API_BASE}/generate",
            json={"description": description, "impact": impact, "domain": domain, "source": source},
            timeout=60
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

    database_url = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://")
    if not database_url:
        log.error("DATABASE_URL not set")
        sys.exit(1)

    db_conn = await asyncpg.connect(database_url)

    async with httpx.AsyncClient(
        headers={"User-Agent": "TheCollective/1.0 (Arca Intelligence Ingestor)"},
        follow_redirects=True
    ) as client:

        all_items = []

        for feed in RSS_FEEDS:
            items = await fetch_rss_items(client, feed)
            all_items.extend(items)
            log.info(f"RSS {feed['source']}: {len(items)} items")

        cp_items = await fetch_cryptopanic(client)
        all_items.extend(cp_items)
        log.info(f"CryptoPanic: {len(cp_items)} items")
        log.info(f"Total raw items: {len(all_items)}")

        submitted = 0
        skipped_dup = 0
        skipped_low = 0
        skipped_irrelevant = 0

        for item in all_items:
            title = item["title"]
            summary = item.get("summary", "")
            source = item["source"]

            # Relevance gate
            if not is_crypto_relevant(title, summary, source):
                skipped_irrelevant += 1
                log.debug(f"SKIP (irrelevant): {title[:60]}")
                continue

            # Impact pre-check before DB query
            impact = score_impact(title, summary)
            if impact < 0.55:
                skipped_low += 1
                continue

            event_id = make_event_id(title)
            content_hash = make_content_hash(title, summary)

            if await already_ingested(db_conn, event_id, content_hash):
                skipped_dup += 1
                continue

            result = await submit_to_collective(client, item, content_hash)
            if result and result.get("status") == "generated":
                submitted += 1
                await asyncio.sleep(3)

        await db_conn.close()

        log.info(
            f"=== Cycle complete: {submitted} generated | "
            f"{skipped_dup} dupes | {skipped_low} low-impact | "
            f"{skipped_irrelevant} irrelevant ==="
        )


if __name__ == "__main__":
    asyncio.run(run())
