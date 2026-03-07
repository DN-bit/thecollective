# The Collective - Central API
# FastAPI application - Central Mind HTTP interface

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

import anthropic
import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repo root is on the path so base.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import SpecialistNode, IntelligenceEvent

# RAG - import with fallback so API still works if rag.py has issues
try:
    from pipeline.rag import retrieve_relevant_posts, format_rag_context, index_posts
    RAG_ENABLED = True
except ImportError:
    RAG_ENABLED = False
    print("[Collective] WARNING: RAG module not found, running without 2 Satoshis context")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="The Collective API",
    description="Agent-to-agent intelligence generation system",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

async def get_db_conn():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    database_url = database_url.replace("postgres://", "postgresql://")
    return await asyncpg.connect(database_url)


async def init_db():
    conn = await get_db_conn()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS corpus (
                id SERIAL PRIMARY KEY,
                event_id TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                impact FLOAT NOT NULL,
                domain TEXT NOT NULL,
                intelligence JSONB NOT NULL,
                confidence FLOAT,
                judged BOOLEAN DEFAULT FALSE,
                judge_scores JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                output JSONB NOT NULL,
                confidence FLOAT,
                compute_cost_usd FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                judge_name TEXT NOT NULL,
                logic_score FLOAT NOT NULL,
                truth_score FLOAT NOT NULL,
                source_score FLOAT NOT NULL,
                average_score FLOAT NOT NULL,
                notes TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(event_id, judge_name)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id SERIAL PRIMARY KEY,
                coin_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                sector TEXT,
                market_cap_rank INTEGER,
                market_cap_usd FLOAT,
                current_price_usd FLOAT,
                price_change_24h FLOAT,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_impacts (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                coin_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                impact_direction TEXT,
                impact_severity TEXT,
                rationale TEXT,
                confidence FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(event_id, coin_id)
            )
        """)
        # Safe column additions for existing tables
        await conn.execute("ALTER TABLE portfolio_impacts ADD COLUMN IF NOT EXISTS mechanism TEXT")
    finally:
        await conn.close()


async def sync_assets_from_coingecko():
    """Fetch top 100 tokens from CoinGecko and upsert into assets table."""
    import httpx as _httpx

    api_key = os.getenv("COINGECKO_API_KEY", "")
    headers = {"Accept": "application/json"}
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1}

    if api_key:
        # Demo API key — pass as both header and query param for compatibility
        headers["x-cg-demo-api-key"] = api_key
        params["x_cg_demo_api_key"] = api_key

    async with _httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params=params,
            timeout=30,
            headers=headers
        )
        coins = resp.json()

    # CoinGecko rate limit returns a dict, not a list
    if not isinstance(coins, list):
        raise ValueError(f"CoinGecko returned unexpected response: {coins}")

    conn = await get_db_conn()
    try:
        count = 0
        for coin in coins:
            if not isinstance(coin, dict):
                continue
            await conn.execute("""
                INSERT INTO assets (coin_id, symbol, name, sector, market_cap_rank, market_cap_usd, current_price_usd, price_change_24h, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (coin_id) DO UPDATE SET
                    market_cap_rank = EXCLUDED.market_cap_rank,
                    market_cap_usd = EXCLUDED.market_cap_usd,
                    current_price_usd = EXCLUDED.current_price_usd,
                    price_change_24h = EXCLUDED.price_change_24h,
                    updated_at = NOW()
            """,
                coin["id"],
                coin["symbol"].upper(),
                coin["name"],
                "other",
                coin.get("market_cap_rank"),
                float(coin.get("market_cap") or 0),
                float(coin.get("current_price") or 0),
                float(coin.get("price_change_percentage_24h") or 0)
            )
            count += 1
    finally:
        await conn.close()

    print(f"[Collective] Synced {count} assets from CoinGecko")
    return count


@app.on_event("startup")
async def startup():
    # Init all DB tables
    try:
        await init_db()
        print("[Collective] Database initialized successfully")
    except Exception as e:
        print(f"[Collective] WARNING: DB init failed: {e}")

    # Sync top 100 tokens — await directly so it completes before first request
    try:
        await sync_assets_from_coingecko()
    except Exception as e:
        print(f"[Collective] WARNING: CoinGecko sync failed: {e}")

    # Index any unembedded 2 Satoshis posts
    if RAG_ENABLED:
        try:
            indexed = await index_posts()
            if indexed > 0:
                print(f"[Collective] RAG: indexed {indexed} new posts")
            else:
                print("[Collective] RAG: all posts already indexed")
        except Exception as e:
            print(f"[Collective] WARNING: RAG indexing failed: {e}")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EventRequest(BaseModel):
    description: str = Field(..., min_length=10)
    impact: float = Field(..., ge=0.0, le=1.0)
    domain: str = Field(default="macro")
    source: Optional[str] = Field(default="manual")


class JudgeRequest(BaseModel):
    event_id: str
    judge_name: str
    scores: Dict[str, float]
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# MacroNode
# ---------------------------------------------------------------------------

class MacroNode(SpecialistNode):
    def __init__(self):
        super().__init__(
            domain="macro",
            system_prompt="""You are MacroNode, a specialist in crypto macro intelligence.
Analyze geopolitical events, Fed policy, institutional flows, and market structure.
Always return structured JSON with chain-of-thought reasoning, scenarios, and citations.""",
            model="claude-sonnet-4-5"
        )

    def _format_prompt(self, event: IntelligenceEvent) -> str:
        rag_section = getattr(self, "rag_context", "")
        rag_block = f"\n\n{rag_section}\n" if rag_section else ""
        return f"""Analyze this crypto macro event and return a JSON object.

Event: {event.description}
Impact Score: {event.impact_score}/1.0
Source: {event.source}
Timestamp: {event.timestamp.isoformat()}{rag_block}

Return only valid JSON, no markdown fences, no extra text:
{{
    "market_regime": "bull",
    "chain_of_thought": ["first reasoning step", "second reasoning step", "third reasoning step"],
    "scenarios": [
        {{"outcome": "base case description", "probability": 0.55, "rationale": "why this is most likely"}},
        {{"outcome": "bull case description", "probability": 0.25, "rationale": "why this could happen"}},
        {{"outcome": "bear case description", "probability": 0.20, "rationale": "why this could happen"}}
    ],
    "recommendation": "specific actionable advice for a digital assets fund",
    "citations": ["source or reference 1", "source or reference 2"],
    "confidence": 0.75,
    "key_metrics": {{"metric_name": "metric_value"}}
}}

Replace all example values with your actual analysis. All probability values must be numbers between 0 and 1. confidence must be a number between 0 and 1."""""



# ---------------------------------------------------------------------------
# SentimentNode — xAI Grok live X/Twitter sentiment
# ---------------------------------------------------------------------------

class SentimentNode:
    """
    Uses xAI Grok to fetch live X/Twitter sentiment on a crypto asset.
    Grok has real-time X data access — no scraping needed.
    Returns crowd sentiment score + narrative + delta vs team judgment.
    """
    def __init__(self):
        self.node_id = "sentiment_node_v1"
        self.xai_api_key = os.getenv("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1"

    async def get_sentiment(self, asset: str, context: str = "") -> dict:
        """
        Query Grok for live X sentiment on an asset.
        asset: token symbol or name e.g. "SOL", "Bitcoin", "Ethereum"
        context: optional event context e.g. "price down 8% today"
        """
        if not self.xai_api_key:
            raise Exception("XAI_API_KEY not set")

        context_line = f"Context: {context}" if context else ""

        prompt = f"""You have access to real-time X (Twitter) data. Analyze current sentiment for {asset} in crypto markets.

{context_line}

Search recent X posts about ${asset} #{asset} and crypto discussions mentioning {asset}.

Return a JSON object with this exact structure:
{{
    "asset": "{asset}",
    "crowd_sentiment_score": 7.2,
    "sentiment_label": "bullish",
    "sentiment_summary": "2-3 sentence summary of what X is saying right now",
    "dominant_narratives": ["narrative 1", "narrative 2", "narrative 3"],
    "key_accounts_sentiment": "what influential crypto accounts are saying",
    "price_driver_theory": "the crowd's leading theory for why price is moving",
    "contrarian_signals": "any notable dissenting views or warnings",
    "post_volume": "high|medium|low",
    "confidence": 0.80,
    "data_freshness": "estimated recency of X data used"
}}

crowd_sentiment_score is 1-10 (1=extremely bearish, 5=neutral, 10=extremely bullish).
Return only valid JSON, no markdown fences."""

        headers = {
            "Authorization": f"Bearer {self.xai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "grok-4-1-fast-reasoning",
            "input": [
                {
                    "role": "system",
                    "content": "You are a crypto market sentiment analyst with access to real-time X data. Always return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "tools": [
                {"type": "x_search"},
                {"type": "web_search"}
            ],
            "max_output_tokens": 2000,
            "temperature": 0.3
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/responses",
                headers=headers,
                json=payload
            )
            if resp.status_code != 200:
                raise Exception(f"xAI API error: {resp.status_code} {resp.text}")

            data = resp.json()
            # Responses API returns output array - find the message content
            raw = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            raw = part.get("text", "").strip()
                            break
            if not raw:
                raise Exception(f"No text output in response: {data}")

            # Clean JSON
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw = part
                        break

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            return json.loads(raw)

    async def get_price_driver(self, asset: str, price_change_pct: float) -> dict:
        """
        Specifically ask Grok why an asset's price is moving right now.
        price_change_pct: e.g. -8.5 for down 8.5%
        """
        direction = "up" if price_change_pct > 0 else "down"
        magnitude = abs(price_change_pct)

        prompt = f"""You have real-time X data access. {asset} is {direction} {magnitude:.1f}% right now.

Search X for the leading explanations crypto traders and analysts are giving for this move.

Return JSON:
{{
    "asset": "{asset}",
    "price_change_pct": {price_change_pct},
    "leading_explanation": "the dominant theory on X for why this is happening",
    "supporting_factors": ["factor 1", "factor 2", "factor 3"],
    "disputed_narratives": ["narrative being debated or rejected"],
    "key_posts_summary": "summary of the most cited posts/threads",
    "crowd_sentiment_score": 5.0,
    "is_fundamental": true,
    "is_technical": false,
    "is_sentiment_driven": false,
    "confidence": 0.75
}}

is_fundamental: true if the move is attributed to news/fundamentals
is_technical: true if attributed to chart levels/liquidations/flows
is_sentiment_driven: true if attributed to hype/fear/social momentum

Return only valid JSON, no markdown."""

        headers = {
            "Authorization": f"Bearer {self.xai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "grok-4-1-fast-reasoning",
            "input": [
                {"role": "system", "content": "You are a crypto market analyst with real-time X data access. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "tools": [
                {"type": "x_search"},
                {"type": "web_search"}
            ],
            "max_output_tokens": 2000,
            "temperature": 0.3
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/responses",
                headers=headers,
                json=payload
            )
            if resp.status_code != 200:
                raise Exception(f"xAI API error: {resp.status_code} {resp.text}")

            data = resp.json()
            raw = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            raw = part.get("text", "").strip()
                            break
            if not raw:
                raise Exception(f"No text output in response: {data}")

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            return json.loads(raw)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "system": "The Collective",
        "status": "online",
        "version": "0.1.0",
        "nodes": ["macro", "defi", "policy", "sentiment"],
        "message": "Ready to synchronize."
    }


@app.get("/health")
async def health():
    db_ok = False
    try:
        conn = await get_db_conn()
        await conn.close()
        db_ok = True
    except Exception:
        pass
    return {"status": "ok", "db_connected": db_ok, "timestamp": datetime.now().isoformat()}


@app.post("/generate")
async def generate(request: EventRequest):
    event = IntelligenceEvent(
        id=f"evt_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        timestamp=datetime.now(),
        source=request.source or "manual",
        description=request.description,
        impact_score=request.impact,
        relevant_domains=[request.domain],
        raw_data=None
    )

    domain_map = {"macro": MacroNode}
    node = domain_map.get(request.domain, MacroNode)()

    # Inject RAG context from relevant 2 Satoshis posts
    if RAG_ENABLED:
        try:
            relevant_posts = await retrieve_relevant_posts(request.description, top_k=3)
            node.rag_context = format_rag_context(relevant_posts)
            print(f"[Collective] RAG: injected {len(relevant_posts)} relevant posts")
        except Exception as e:
            print(f"[Collective] WARNING: RAG retrieval failed: {e}")
            node.rag_context = ""
    else:
        node.rag_context = ""

    try:
        outputs = await node.generate(event, n_variants=1)
        if not outputs:
            raise HTTPException(status_code=422, detail="No outputs passed quality threshold")

        best = outputs[0]

        try:
            conn = await get_db_conn()
            await conn.execute("""
                INSERT INTO corpus (event_id, description, impact, domain, intelligence, confidence, judged)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                ON CONFLICT (event_id) DO NOTHING
            """, event.id, request.description, request.impact, request.domain,
                json.dumps(best.output), best.confidence, False)
            await conn.close()
        except Exception as db_err:
            print(f"[Collective] WARNING: DB write failed: {db_err}")

        return {
            "status": "generated",
            "event_id": event.id,
            "node_id": best.node_id,
            "domain": request.domain,
            "intelligence": best.output,
            "confidence": best.confidence,
            "compute_cost_usd": best.compute_cost_usd,
            "timestamp": best.timestamp.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corpus")
async def get_corpus(limit: int = 20, offset: int = 0, judged: Optional[bool] = None, domain: Optional[str] = None, sort: Optional[str] = "newest"):
    try:
        conn = await get_db_conn()
        # Count total for pagination
        count_query = "SELECT COUNT(*) FROM corpus WHERE 1=1"
        data_query = "SELECT * FROM corpus WHERE 1=1"
        params = []
        i = 1
        if judged is not None:
            count_query += f" AND judged = ${i}"
            data_query  += f" AND judged = ${i}"
            params.append(judged); i += 1
        if domain:
            count_query += f" AND domain = ${i}"
            data_query  += f" AND domain = ${i}"
            params.append(domain); i += 1
        total = await conn.fetchval(count_query, *params)
        sort_clause = {
            "newest": "ORDER BY created_at DESC",
            "oldest": "ORDER BY created_at ASC",
            "confidence": "ORDER BY confidence DESC",
            "score": "ORDER BY (judge_scores->>'average')::float DESC NULLS LAST",
        }.get(sort, "ORDER BY created_at DESC")
        data_query  += f" {sort_clause} LIMIT ${i} OFFSET ${i+1}"
        params.extend([limit, offset])
        rows = await conn.fetch(data_query, *params)
        # Get judgment counts for all returned entries
        if rows:
            event_ids = [r['event_id'] for r in rows]
            counts = await conn.fetch("""
                SELECT event_id, COUNT(*) as judgment_count, array_agg(judge_name) as judges
                FROM judgments WHERE event_id = ANY($1)
                GROUP BY event_id
            """, event_ids)
            count_map = {r['event_id']: {'judgment_count': r['judgment_count'], 'judges': list(r['judges'])} for r in counts}
        else:
            count_map = {}
        await conn.close()
        entries = []
        for r in rows:
            e = dict(r)
            e.update(count_map.get(e['event_id'], {'judgment_count': 0, 'judges': []}))
            entries.append(e)
        return {
            "status": "ok",
            "total": total,
            "limit": limit,
            "offset": offset,
            "count": len(entries),
            "entries": entries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/judge/{event_id}")
async def judge_entry(event_id: str, request: JudgeRequest):
    try:
        conn = await get_db_conn()

        # Verify entry exists
        exists = await conn.fetchval("SELECT 1 FROM corpus WHERE event_id = $1", event_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

        avg = (request.scores.get("logic", 3) + request.scores.get("truth", 3) + request.scores.get("source", 3)) / 3.0

        # Upsert into judgments table (one row per judge per entry)
        await conn.execute("""
            INSERT INTO judgments (event_id, judge_name, logic_score, truth_score, source_score, average_score, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (event_id, judge_name) DO UPDATE SET
                logic_score = EXCLUDED.logic_score,
                truth_score = EXCLUDED.truth_score,
                source_score = EXCLUDED.source_score,
                average_score = EXCLUDED.average_score,
                notes = EXCLUDED.notes,
                created_at = NOW()
        """, event_id, request.judge_name,
            request.scores.get("logic", 3),
            request.scores.get("truth", 3),
            request.scores.get("source", 3),
            avg, request.notes)

        # Aggregate all judgments for this entry
        agg = await conn.fetchrow("""
            SELECT COUNT(*) as count,
                   AVG(logic_score) as logic,
                   AVG(truth_score) as truth,
                   AVG(source_score) as source,
                   AVG(average_score) as average,
                   array_agg(judge_name) as judges
            FROM judgments WHERE event_id = $1
        """, event_id)

        # Update corpus with aggregated scores and mark judged
        await conn.execute("""
            UPDATE corpus SET
                judged = TRUE,
                judge_scores = $1::jsonb
            WHERE event_id = $2
        """, json.dumps({
            "logic": round(agg["logic"], 2),
            "truth": round(agg["truth"], 2),
            "source": round(agg["source"], 2),
            "average": round(agg["average"], 2),
            "judgment_count": agg["count"],
            "judges": list(agg["judges"])
        }), event_id)

        await conn.close()
        return {
            "status": "judged",
            "event_id": event_id,
            "judge_name": request.judge_name,
            "judgment_count": agg["count"],
            "judges": list(agg["judges"]),
            "aggregate_scores": {
                "logic": round(agg["logic"], 2),
                "truth": round(agg["truth"], 2),
                "source": round(agg["source"], 2),
                "average": round(agg["average"], 2),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    try:
        conn = await get_db_conn()
        row = await conn.fetchrow("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE judged = TRUE) AS judged,
                COUNT(*) FILTER (WHERE judged = FALSE) AS pending_judgment,
                AVG(confidence) AS avg_confidence,
                COUNT(DISTINCT domain) AS active_domains
            FROM corpus
        """)
        await conn.close()
        return {"status": "ok", "corpus": dict(row)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/judges")
async def judge_stats():
    """Analytics on judge behavior - who has judged what, score distributions, activity."""
    try:
        conn = await get_db_conn()

        # Per-judge summary
        judge_rows = await conn.fetch("""
            SELECT
                judge_name,
                COUNT(*) as total_judgments,
                ROUND(AVG(average_score)::numeric, 2) as avg_score,
                ROUND(AVG(logic_score)::numeric, 2) as avg_logic,
                ROUND(AVG(truth_score)::numeric, 2) as avg_truth,
                ROUND(AVG(source_score)::numeric, 2) as avg_source,
                COUNT(*) FILTER (WHERE average_score >= 4.0) as high_scores,
                COUNT(*) FILTER (WHERE average_score <= 2.0) as low_scores,
                MAX(created_at) as last_active
            FROM judgments
            GROUP BY judge_name
            ORDER BY total_judgments DESC
        """)

        # Score distribution across all judges
        dist_rows = await conn.fetch("""
            SELECT
                judge_name,
                CASE
                    WHEN average_score >= 4.5 THEN '5 - Exceptional'
                    WHEN average_score >= 3.5 THEN '4 - Strong'
                    WHEN average_score >= 2.5 THEN '3 - Moderate'
                    WHEN average_score >= 1.5 THEN '2 - Weak'
                    ELSE '1 - Poor'
                END as score_bucket,
                COUNT(*) as count
            FROM judgments
            GROUP BY judge_name, score_bucket
            ORDER BY judge_name, score_bucket
        """)

        # Most judged events (consensus items)
        consensus_rows = await conn.fetch("""
            SELECT
                j.event_id,
                c.description,
                COUNT(*) as judge_count,
                ROUND(AVG(j.average_score)::numeric, 2) as consensus_score,
                array_agg(j.judge_name ORDER BY j.average_score DESC) as judges,
                MAX(j.average_score) - MIN(j.average_score) as score_spread
            FROM judgments j
            JOIN corpus c ON c.event_id = j.event_id
            GROUP BY j.event_id, c.description
            HAVING COUNT(*) >= 2
            ORDER BY judge_count DESC, consensus_score DESC
            LIMIT 10
        """)

        # Total unique events judged
        total_events = await conn.fetchval("SELECT COUNT(DISTINCT event_id) FROM judgments")
        total_judgments = await conn.fetchval("SELECT COUNT(*) FROM judgments")

        await conn.close()

        return {
            "status": "ok",
            "summary": {
                "total_judgments": total_judgments,
                "total_events_judged": total_events,
                "active_judges": len(judge_rows)
            },
            "judges": [dict(r) for r in judge_rows],
            "distributions": [dict(r) for r in dist_rows],
            "consensus_items": [dict(r) for r in consensus_rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/digest")
async def get_digest(hours: int = 24, domain: Optional[str] = None):
    # anthropic already imported at top

    try:
        conn = await get_db_conn()
        query = """
            SELECT event_id, description, domain, intelligence, confidence, created_at
            FROM corpus
            WHERE created_at >= NOW() - ($1 || ' hours')::INTERVAL
            AND judged = TRUE
        """
        params: list = [str(hours)]
        if domain:
            query += " AND domain = $2"
            params.append(domain)
        query += " ORDER BY confidence DESC LIMIT 50"
        rows = await conn.fetch(query, *params)
        await conn.close()
        entries = [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not entries:
        return {"status": "empty", "message": f"No judged entries in the last {hours} hours.", "hours": hours}

    corpus_summary = json.dumps([{
        "domain": e["domain"],
        "event": e["description"],
        "regime": e["intelligence"].get("market_regime", "unknown") if isinstance(e["intelligence"], dict) else "unknown",
        "recommendation": e["intelligence"].get("recommendation", "") if isinstance(e["intelligence"], dict) else "",
        "confidence": e["confidence"],
    } for e in entries], indent=2)

    # Fetch relevant 2 Satoshis context for digest
    rag_block = ""
    if RAG_ENABLED:
        try:
            digest_query = " ".join([e["description"][:50] for e in entries[:5]])
            relevant_posts = await retrieve_relevant_posts(digest_query, top_k=2)
            rag_block = "\n\n" + format_rag_context(relevant_posts) if relevant_posts else ""
        except Exception as e:
            print(f"[Collective] WARNING: RAG failed for digest: {e}")

    synthesis_prompt = f"""Synthesize a daily intelligence brief for Arca, a digital assets investment firm.{rag_block}

{len(entries)} judged entries from the last {hours} hours:
{corpus_summary}

Return JSON:
{{
    "date": "{datetime.now().strftime('%B %d, %Y')}",
    "period_hours": {hours},
    "entry_count": {len(entries)},
    "overall_regime": "bull|bear|neutral|mixed",
    "regime_confidence": 0.X,
    "executive_summary": "2-3 sentences for LP communications",
    "key_themes": [{{"theme": "...", "signal_strength": "high|medium|low", "summary": "..."}}],
    "domain_briefs": {{"macro": "...", "defi": "...", "policy": "...", "sentiment": "..."}},
    "top_recommendations": [{{"priority": 1, "action": "...", "rationale": "...", "domain": "..."}}],
    "risks": [{{"risk": "...", "severity": "high|medium|low", "domain": "..."}}],
    "base_layer_topics": ["topic 1", "topic 2"],
    "lp_talking_points": ["point 1", "point 2", "point 3"]
}}"""

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            system="You are a senior analyst at Arca. Return only valid JSON, no markdown fences.",
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        brief = json.loads(raw.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis error: {e}")

    return {"status": "ok", "brief": brief, "source_entries": len(entries), "generated_at": datetime.now().isoformat()}


@app.get("/sentiment/{asset}")
async def get_sentiment(asset: str, context: Optional[str] = None):
    """
    Get live X/Twitter sentiment for a crypto asset via Grok.
    asset: token symbol e.g. SOL, BTC, ETH
    """
    try:
        node = SentimentNode()
        result = await node.get_sentiment(asset.upper(), context or "")

        # Store in corpus as sentiment domain entry
        try:
            event_id = f"sent_{asset.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            description = f"X/Twitter sentiment analysis for {asset.upper()}: {result.get('sentiment_label', 'neutral')} ({result.get('crowd_sentiment_score', 5)}/10)"
            conn = await get_db_conn()
            await conn.execute("""
                INSERT INTO corpus (event_id, description, impact, domain, intelligence, confidence, judged)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                ON CONFLICT (event_id) DO NOTHING
            """, event_id, description, result.get("confidence", 0.7),
                "sentiment", json.dumps(result), result.get("confidence", 0.7), False)
            await conn.close()
        except Exception as db_err:
            print(f"[Collective] WARNING: Sentiment DB write failed: {db_err}")

        return {"status": "ok", "asset": asset.upper(), "sentiment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/price-driver/{asset}")
async def get_price_driver(asset: str, price_change_pct: float = 0.0):
    """
    Ask Grok why an asset price is moving right now.
    asset: token symbol e.g. SOL, BTC, ETH
    price_change_pct: e.g. -8.5 for down 8.5%
    """
    try:
        node = SentimentNode()
        result = await node.get_price_driver(asset.upper(), price_change_pct)

        # Store in corpus
        try:
            direction = "up" if price_change_pct > 0 else "down"
            event_id = f"price_{asset.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            description = f"Price driver analysis: {asset.upper()} {direction} {abs(price_change_pct):.1f}% — {result.get('leading_explanation', '')[:100]}"
            conn = await get_db_conn()
            await conn.execute("""
                INSERT INTO corpus (event_id, description, impact, domain, intelligence, confidence, judged)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                ON CONFLICT (event_id) DO NOTHING
            """, event_id, description, min(abs(price_change_pct) / 20.0, 0.95),
                "sentiment", json.dumps(result), result.get("confidence", 0.7), False)
            await conn.close()
        except Exception as db_err:
            print(f"[Collective] WARNING: Price driver DB write failed: {db_err}")

        return {"status": "ok", "asset": asset.upper(), "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assets")
async def get_assets(sector: Optional[str] = None):
    """Get top 100 tokens, optionally filtered by sector."""
    try:
        conn = await get_db_conn()
        if sector:
            rows = await conn.fetch(
                "SELECT * FROM assets WHERE sector = $1 ORDER BY market_cap_rank ASC", sector)
        else:
            rows = await conn.fetch("SELECT * FROM assets ORDER BY market_cap_rank ASC")
        await conn.close()
        return {"status": "ok", "count": len(rows), "assets": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assets/sync")
async def manual_asset_sync():
    """Manually trigger a CoinGecko sync."""
    try:
        count = await sync_assets_from_coingecko()
        return {"status": "ok", "synced": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Token risk profiles — fundamental context that enriches impact analysis
TOKEN_RISK_PROFILES = {
    "bitcoin":           "Digital gold narrative, institutional safe-haven, highly correlated to macro risk-off. Regulatory clarity generally positive. Mining energy exposure.",
    "ethereum":          "Smart contract platform, ETF approved, staking yield, highly correlated to DeFi/NFT sectors. SEC scrutiny on staking. Leading L1.",
    "tether":            "Largest stablecoin, systemic liquidity provider, regulatory target for reserve transparency. De-peg risk in stress scenarios.",
    "binancecoin":       "Native token of Binance exchange and BNB Chain. Direct legal exposure to Binance regulatory actions globally. CZ DOJ conviction overhang.",
    "solana":            "High-throughput L1, FTX collapse association still present, strong developer momentum. Centralization concerns, validator concentration.",
    "ripple":            "XRP — long-running SEC lawsuit partially resolved. Institutional cross-border payments focus. Regulatory clarity improving but not complete.",
    "usd-coin":          "Circle-issued stablecoin, regulated, transparent reserves. Benefits from regulatory clarity. Competes with USDT.",
    "staked-ether":      "Lido staked ETH, liquid staking derivative. Smart contract risk, slashing risk, ETH correlation. Dominant LST by TVL.",
    "dogecoin":          "Meme coin, retail sentiment driven, Elon Musk correlation. High beta to risk-on/risk-off. No fundamental utility.",
    "cardano":           "Academic PoS L1, slow development pace, strong community. Limited DeFi ecosystem. Low regulatory risk profile.",
    "tron":              "High-throughput L1, dominant in USDT transfers, Justin Sun legal exposure (SEC fraud charges). Centralized.",
    "avalanche-2":       "EVM-compatible L1, subnet architecture, institutional focus. Competing with ETH L2s. Strong enterprise partnerships.",
    "shiba-inu":         "Meme coin ecosystem, SHIB army retail base, Shibarium L2. Pure sentiment play, no fundamental value drivers.",
    "chainlink":         "Decentralized oracle network, critical DeFi infrastructure. Benefits from DeFi growth. Low regulatory risk, high utility.",
    "polkadot":          "Multi-chain interoperability, parachain auctions, Web3 Foundation. Underperformed vs peers, ecosystem fragmentation risk.",
    "bitcoin-cash":      "BTC fork, P2P payments focus, low institutional interest. Follows BTC directionally with higher volatility.",
    "uniswap":           "Leading DEX, UNI governance token, fee switch debate ongoing. SEC scrutiny on DeFi. High regulatory risk.",
    "litecoin":          "BTC silver, payments focus, MimbleWimble privacy upgrade. Low institutional interest, follows BTC.",
    "near":              "Sharded L1, developer-friendly, AI blockchain narrative. Strong VC backing, chain abstraction focus.",
    "internet-computer": "Dfinity ICP, on-chain computation, controversial launch history. Niche use case, limited DeFi.",
    "dai":               "Decentralized stablecoin (MakerDAO/Sky), crypto-collateralized. RWA exposure growing. Regulatory uncertainty on decentralized stables.",
    "ethereum-classic":  "Original ETH chain post-DAO fork. PoW, low developer activity, speculative.",
    "monero":            "Privacy coin, regulatory target globally. Delistings ongoing. High regulatory risk.",
    "stellar":           "XLM, cross-border payments, Ripple competitor. SDF-controlled supply, institutional partnerships.",
    "aave":              "Leading DeFi lending protocol, multi-chain, GHO stablecoin. Regulatory risk on DeFi lending. High TVL.",
    "maker":             "MKR/DAI system, RWA collateral growing, Sky rebrand. Decentralized governance, regulatory uncertainty.",
    "cosmos":            "ATOM, IBC interoperability hub, modular blockchain ecosystem. Fragmented value accrual.",
    "filecoin":          "Decentralized storage, FIL token. Long development cycles, storage market competition.",
    "hedera-hashgraph":  "Enterprise DLT, HBAR, governed by Hedera council (Google, IBM etc). Centralized but enterprise-grade.",
    "aptos":             "Move-based L1, a16z backed, high TPS. Competing with Sui. Growing DeFi ecosystem.",
    "sui":               "Move-based L1, Mysten Labs, fast finality. Strong gaming and consumer app focus.",
    "hyperliquid":       "On-chain perps DEX, HYPE token, high TVL. Fully on-chain order book. Regulatory risk on derivatives.",
    "pepe":              "Meme coin, pure sentiment, no utility. Extreme volatility. Retail speculation only.",
    "render-token":      "Decentralized GPU rendering, AI/compute narrative. Benefits from AI demand for GPU.",
    "fetch-ai":          "AI agent network, ASI Alliance merger with OCEAN and AGIX. AI x crypto narrative.",
    "bittensor":         "Decentralized ML training network, TAO token. AI x crypto narrative, high valuation.",
    "arbitrum":          "Leading ETH L2, ARB governance token. Benefits from ETH ecosystem growth. Fee compression risk.",
    "optimism":          "ETH L2, OP Stack, Superchain vision. Strong ecosystem, retroactive funding model.",
    "injective-protocol":"DeFi-native L1, INJ, derivatives and RWA focus. High performance, growing ecosystem.",
    "mantle":            "ETH L2, BitDAO treasury backing. Enterprise focus. Competing in crowded L2 market.",
}


# ---------------------------------------------------------------------------
# Evidence Layer — DefiLlama + on-chain data enrichment
# ---------------------------------------------------------------------------

# Maps CoinGecko coin_id to DefiLlama protocol slug
DEFI_LLAMA_SLUGS = {
    "aave":              "aave",
    "uniswap":           "uniswap",
    "maker":             "makerdao",
    "dai":               "makerdao",
    "staked-ether":      "lido",
    "chainlink":         "chainlink",
    "injective-protocol":"injective",
    "hyperliquid":       "hyperliquid",
    "arbitrum":          "arbitrum",
    "optimism":          "optimism",
    "near":              "near",
    "avalanche-2":       "avalanche",
    "solana":            "solana",
    "ethereum":          "ethereum",
    "sui":               "sui",
    "aptos":             "aptos",
    "mantle":            "mantle",
}


async def fetch_defi_llama_tvl(protocol_slug: str) -> Optional[Dict]:
    """Fetch current TVL and 24h/7d change from DefiLlama."""
    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://api.llama.fi/protocol/{protocol_slug}")
            if resp.status_code != 200:
                return None
            data = resp.json()
            tvl_history = data.get("tvl", [])
            if len(tvl_history) < 2:
                return None
            current_tvl = tvl_history[-1].get("totalLiquidityUSD", 0)
            tvl_24h_ago = tvl_history[-2].get("totalLiquidityUSD", current_tvl) if len(tvl_history) > 1 else current_tvl
            tvl_7d_ago  = tvl_history[-8].get("totalLiquidityUSD", current_tvl) if len(tvl_history) > 7 else current_tvl
            change_24h = ((current_tvl - tvl_24h_ago) / tvl_24h_ago * 100) if tvl_24h_ago else 0
            change_7d  = ((current_tvl - tvl_7d_ago)  / tvl_7d_ago  * 100) if tvl_7d_ago  else 0
            return {
                "tvl_usd": current_tvl,
                "tvl_formatted": f"${current_tvl/1e9:.2f}B" if current_tvl > 1e9 else f"${current_tvl/1e6:.1f}M",
                "change_24h": round(change_24h, 2),
                "change_7d":  round(change_7d, 2),
                "chain_tvls": data.get("chainTvls", {}),
            }
    except Exception as e:
        print(f"[Evidence] DefiLlama error for {protocol_slug}: {e}")
        return None


async def fetch_defi_llama_stablecoins() -> Optional[Dict]:
    """Fetch stablecoin peg data from DefiLlama."""
    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://stablecoins.llama.fi/stablecoins?includePrices=true")
            if resp.status_code != 200:
                return None
            data = resp.json()
            coins = data.get("peggedAssets", [])
            result = {}
            for coin in coins:
                name = coin.get("name", "").lower()
                price = coin.get("price", 1.0)
                peg_deviation = abs(price - 1.0) * 100 if price else 0
                if any(s in name for s in ["tether", "usdc", "dai", "usdt"]):
                    result[coin.get("symbol", name)] = {
                        "price": price,
                        "peg_deviation_pct": round(peg_deviation, 4),
                        "circulating_usd": coin.get("circulating", {}).get("peggedUSD", 0),
                        "at_peg": peg_deviation < 0.1
                    }
            return result
    except Exception as e:
        print(f"[Evidence] Stablecoin fetch error: {e}")
        return None


async def fetch_defi_llama_yields(protocol: str) -> Optional[list]:
    """Fetch top yield pools for a protocol."""
    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://yields.llama.fi/pools")
            if resp.status_code != 200:
                return None
            pools = resp.json().get("data", [])
            matching = [p for p in pools if protocol.lower() in p.get("project", "").lower()]
            matching.sort(key=lambda x: x.get("tvlUsd", 0), reverse=True)
            return [{
                "pool": p.get("symbol", ""),
                "chain": p.get("chain", ""),
                "tvl_usd": p.get("tvlUsd", 0),
                "apy": round(p.get("apy", 0), 2),
                "apy_7d_mean": round(p.get("apyMean30d", 0), 2),
            } for p in matching[:5]]
    except Exception as e:
        print(f"[Evidence] Yields fetch error for {protocol}: {e}")
        return None


async def gather_evidence(impacts: list, event_description: str) -> Dict[str, Dict]:
    """
    For HIGH and MEDIUM severity impacts, gather on-chain evidence from DefiLlama.
    Returns a dict keyed by coin_id with evidence data.
    """
    evidence = {}
    tasks = []

    for impact in impacts:
        coin_id = impact.get("coin_id", "")
        severity = impact.get("impact_severity", "low")
        mechanism = impact.get("mechanism", "")

        if severity not in ("high", "medium"):
            continue

        slug = DEFI_LLAMA_SLUGS.get(coin_id)
        coin_evidence = {"coin_id": coin_id, "sources": []}

        # Fetch TVL if it's a DeFi protocol
        if slug:
            tvl_data = await fetch_defi_llama_tvl(slug)
            if tvl_data:
                signal = "⚠️ TVL declining" if (tvl_data["change_24h"] or 0) < -2 else ("✅ TVL stable" if abs(tvl_data["change_24h"] or 0) < 1 else "📈 TVL rising")
                coin_evidence["tvl"] = tvl_data
                coin_evidence["sources"].append({
                    "type": "defi_llama_tvl",
                    "label": f"DefiLlama TVL: {tvl_data['tvl_formatted']} ({(tvl_data['change_24h'] or 0):+.1f}% 24h)",
                    "signal": signal,
                    "supports_thesis": (tvl_data["change_24h"] or 0) < -1,
                    "data": tvl_data
                })

            # Fetch yield data for lending protocols
            if any(x in coin_id for x in ["aave", "maker", "compound"]):
                yields = await fetch_defi_llama_yields(slug)
                if yields:
                    top = yields[0] if yields else None
                    if top:
                        coin_evidence["sources"].append({
                            "type": "defi_llama_yields",
                            "label": f"Top pool: {top['pool']} — APY {top['apy'] or 0}%, TVL ${(top['tvl_usd'] or 0)/1e6:.1f}M",
                            "signal": "⚠️ APY spike (stress)" if (top['apy'] or 0) > 15 else "✅ APY normal",
                            "supports_thesis": (top['apy'] or 0) > 15,
                            "data": yields
                        })

        # Stablecoin peg check
        if any(x in coin_id for x in ["tether", "usd-coin", "dai", "stablecoin"]):
            stables = await fetch_defi_llama_stablecoins()
            if stables:
                for sym, data in stables.items():
                    coin_evidence["sources"].append({
                        "type": "peg_check",
                        "label": f"{sym} peg: ${data['price']:.4f} ({data['peg_deviation_pct']:+.4f}% deviation)",
                        "signal": "⚠️ Depeg risk" if not data["at_peg"] else "✅ At peg",
                        "supports_thesis": not data["at_peg"],
                        "data": data
                    })

        if coin_evidence["sources"]:
            evidence[coin_id] = coin_evidence

    return evidence


@app.post("/portfolio-impact/{event_id}")
async def portfolio_impact(event_id: str, force_refresh: bool = False):
    """Run deep portfolio impact analysis against top 100 tokens with on-chain evidence."""

    try:
        conn = await get_db_conn()
        event = await conn.fetchrow("SELECT * FROM corpus WHERE event_id = $1", event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

        # Return cached unless force_refresh
        if not force_refresh:
            existing = await conn.fetch(
                "SELECT * FROM portfolio_impacts WHERE event_id = $1 ORDER BY confidence DESC", event_id)
            if existing:
                await conn.close()
                # Re-gather evidence on cache hit too (live data)
                impacts = [dict(r) for r in existing]
                try:
                    evidence = await gather_evidence(impacts, dict(event).get("description", ""))
                except Exception as e:
                    print(f"[Collective] WARNING: Evidence gathering failed: {e}")
                    evidence = {}
                return {"status": "ok", "event_id": event_id, "cached": True, "impacts": impacts, "evidence": evidence}

        assets = await conn.fetch(
            "SELECT coin_id, symbol, name, sector, current_price_usd, price_change_24h, market_cap_rank FROM assets ORDER BY market_cap_rank ASC LIMIT 100")
        await conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not assets:
        raise HTTPException(status_code=404, detail="No assets found — run /assets/sync first")

    # Build enriched asset list with risk profiles
    asset_lines = []
    for i, a in enumerate(assets):
        ad = dict(a)
        profile = TOKEN_RISK_PROFILES.get(ad['coin_id'], "")
        profile_str = f" | PROFILE: {profile}" if profile else ""
        price_change = ad['price_change_24h'] or 0.0
        price = ad['current_price_usd'] or 0.0
        asset_lines.append(
            f"{i+1}. {ad['symbol']} ({ad['name']}) — sector: {ad['sector']}, "
            f"rank: #{ad['market_cap_rank']}, 24h: {price_change:+.1f}%, "
            f"price: ${price:,.2f}{profile_str}"
        )
    asset_list = "\n".join(asset_lines)

    event_dict = dict(event)
    intel = event_dict.get("intelligence", {})
    if isinstance(intel, str):
        intel = json.loads(intel)

    # Build full intelligence context
    cot = intel.get("chain_of_thought", [])
    scenarios = intel.get("scenarios", [])
    cot_str = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(cot)]) if cot else "  Not available"
    scenarios_str = "\n".join([
        f"  - {s.get('outcome','')}: {s.get('probability',0)*100:.0f}% probability — {s.get('rationale','')}"
        for s in scenarios
    ]) if scenarios else "  Not available"

    prompt = f"""You are a senior portfolio analyst at Arca, a digital assets investment firm and registered investment advisor.

Your job is to produce a rigorous, specific assessment of how a market event impacts each of the top 100 crypto tokens.
Be precise. Cite specific mechanisms — legal exposure, liquidity contagion, narrative correlation, protocol dependency, regulatory classification risk.
Do NOT give generic market sentiment explanations. Every rationale must be token-specific and mechanistically grounded.

═══ MARKET EVENT ═══
Description: {event_dict['description']}
Domain: {event_dict['domain']}
Impact Score: {event_dict['impact']}
Market Regime: {intel.get('market_regime', 'unknown')}
Confidence: {event_dict['confidence']}

═══ MACRONODE ANALYSIS ═══
Chain of Thought:
{cot_str}

Scenarios:
{scenarios_str}

Recommendation: {intel.get('recommendation', 'N/A')}
Key Metrics: {json.dumps(intel.get('key_metrics', {}), indent=2)}

═══ TOP 100 TOKENS ═══
{asset_list}

═══ INSTRUCTIONS ═══
1. Identify the 12-18 tokens most meaningfully impacted by this specific event
2. For each token provide:
   - impact_direction: "positive" | "negative" | "neutral"
   - impact_severity: "high" | "medium" | "low"  
   - mechanism: the specific channel through which this event affects this token
     (e.g. "direct legal exposure", "liquidity contagion via BNB Chain DeFi", 
     "safe haven rotation", "regulatory classification risk as security", 
     "mining revenue impact", "stablecoin de-peg risk")
   - rationale: 2-3 sentences. Be specific. Reference the token's actual risk profile,
     not just general market sentiment. Mention price levels or on-chain metrics if relevant.
   - confidence: 0.0-1.0

3. Write a 3-4 sentence executive summary suitable for an LP update call.
   Reference specific tokens, mechanisms, and Arca's positioning context.

Return JSON:
{{
    "impacts": [
        {{
            "coin_id": "binancecoin",
            "symbol": "BNB",
            "name": "BNB",
            "impact_direction": "negative",
            "impact_severity": "high",
            "mechanism": "direct legal exposure",
            "rationale": "BNB faces direct downside from any Binance regulatory action given its utility is tightly coupled to the exchange's ecosystem. CZ's DOJ conviction creates ongoing headline risk. BNB Chain DeFi TVL (~$4B) could see outflows if exchange solvency concerns resurface.",
            "confidence": 0.88
        }}
    ],
    "summary": "Executive summary for LP communications..."
}}"""

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4000,
            system="You are a senior portfolio analyst at Arca, a digital assets investment firm. Return only valid JSON, no markdown fences.",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    try:
        conn = await get_db_conn()
        # Add mechanism column if it doesn't exist yet
        await conn.execute("ALTER TABLE portfolio_impacts ADD COLUMN IF NOT EXISTS mechanism TEXT")
        for impact in result.get("impacts", []):
            await conn.execute("""
                INSERT INTO portfolio_impacts
                    (event_id, coin_id, symbol, name, impact_direction, impact_severity, mechanism, rationale, confidence)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (event_id, coin_id) DO UPDATE SET
                    impact_direction = EXCLUDED.impact_direction,
                    impact_severity = EXCLUDED.impact_severity,
                    mechanism = EXCLUDED.mechanism,
                    rationale = EXCLUDED.rationale,
                    confidence = EXCLUDED.confidence
            """,
                event_id,
                impact.get("coin_id", ""),
                impact.get("symbol", ""),
                impact.get("name", ""),
                impact.get("impact_direction", "neutral"),
                impact.get("impact_severity", "low"),
                impact.get("mechanism", ""),
                impact.get("rationale", ""),
                float(impact.get("confidence", 0.5))
            )
        await conn.close()
    except Exception as e:
        print(f"[Collective] WARNING: Impact persist failed: {e}")

    # Gather on-chain evidence for HIGH/MEDIUM impacts
    try:
        evidence = await gather_evidence(result.get("impacts", []), event_dict.get("description", ""))
    except Exception as e:
        print(f"[Collective] WARNING: Evidence gathering failed: {e}")
        evidence = {}

    return {
        "status": "ok",
        "event_id": event_id,
        "cached": False,
        "summary": result.get("summary", ""),
        "impacts": result.get("impacts", []),
        "evidence": evidence
    }


@app.get("/debug")
async def debug():
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=50,
            messages=[{"role": "user", "content": "Return this JSON: {\"message\": \"hello from The Collective\"}"}]
        )
        return {"status": "ok", "response": response.content[0].text}
    except Exception as e:
        return {"status": "error", "error": str(e), "type": type(e).__name__}


@app.post("/migrate")
async def run_migration():
    """Run any pending DB migrations. Safe to call multiple times."""
    try:
        conn = await get_db_conn()
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                judge_name TEXT NOT NULL,
                logic_score FLOAT NOT NULL,
                truth_score FLOAT NOT NULL,
                source_score FLOAT NOT NULL,
                average_score FLOAT NOT NULL,
                notes TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(event_id, judge_name)
            );
            ALTER TABLE portfolio_impacts ADD COLUMN IF NOT EXISTS mechanism TEXT;
        """)
        await conn.close()
        return {"status": "ok", "message": "Migration complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
