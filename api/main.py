# The Collective - Central API
# FastAPI application - Central Mind HTTP interface

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repo root is on the path so base.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import SpecialistNode, IntelligenceEvent

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
            );
            CREATE TABLE IF NOT EXISTS generations (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                output JSONB NOT NULL,
                confidence FLOAT,
                compute_cost_usd FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
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
            );
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
            );
        """)
    finally:
        await conn.close()


async def sync_assets_from_coingecko():
    """Fetch top 100 tokens from CoinGecko and upsert into assets table."""
    import httpx as _httpx

    api_key = os.getenv("COINGECKO_API_KEY", "")
    headers = {"Accept": "application/json"}
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1}

    # Demo key uses x-cg-demo-api-key header; Pro key uses x-cg-pro-api-key
    if api_key:
        headers["x-cg-demo-api-key"] = api_key

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
            model="gpt-4o-mini"
        )

    def _format_prompt(self, event: IntelligenceEvent) -> str:
        return f"""Analyze this crypto macro event:

Event: {event.description}
Impact Score: {event.impact_score}/1.0
Source: {event.source}
Timestamp: {event.timestamp.isoformat()}

Return JSON:
{{
    "market_regime": "bull|bear|neutral",
    "chain_of_thought": ["step 1", "step 2", "step 3"],
    "scenarios": [
        {{"outcome": "...", "probability": 0.X, "rationale": "..."}},
        {{"outcome": "...", "probability": 0.X, "rationale": "..."}},
        {{"outcome": "...", "probability": 0.X, "rationale": "..."}}
    ],
    "recommendation": "specific actionable advice",
    "citations": ["source 1", "source 2"],
    "confidence": 0.X,
    "key_metrics": {{"metric": "value"}}
}}"""


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
async def get_corpus(limit: int = 20, judged: Optional[bool] = None, domain: Optional[str] = None):
    try:
        conn = await get_db_conn()
        query = "SELECT * FROM corpus WHERE 1=1"
        params = []
        i = 1
        if judged is not None:
            query += f" AND judged = ${i}"; params.append(judged); i += 1
        if domain:
            query += f" AND domain = ${i}"; params.append(domain); i += 1
        query += f" ORDER BY created_at DESC LIMIT ${i}"; params.append(limit)
        rows = await conn.fetch(query, *params)
        await conn.close()
        return {"status": "ok", "count": len(rows), "entries": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/judge/{event_id}")
async def judge_entry(event_id: str, request: JudgeRequest):
    try:
        conn = await get_db_conn()
        result = await conn.execute("""
            UPDATE corpus SET judged = TRUE, judge_scores = $1::jsonb WHERE event_id = $2
        """, json.dumps(request.scores), event_id)
        await conn.close()
        if result == "UPDATE 0":
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        return {"status": "judged", "event_id": event_id, "scores": request.scores}
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


@app.get("/digest")
async def get_digest(hours: int = 24, domain: Optional[str] = None):
    import openai

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

    synthesis_prompt = f"""Synthesize a daily intelligence brief for Arca, a digital assets investment firm.

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
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": synthesis_prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        brief = json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis error: {e}")

    return {"status": "ok", "brief": brief, "source_entries": len(entries), "generated_at": datetime.now().isoformat()}


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


@app.post("/portfolio-impact/{event_id}")
async def portfolio_impact(event_id: str, force_refresh: bool = False):
    """Run deep portfolio impact analysis against top 100 tokens."""
    import openai as _openai

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
                return {"status": "ok", "event_id": event_id, "cached": True, "impacts": [dict(r) for r in existing]}

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
        asset_lines.append(
            f"{i+1}. {ad['symbol']} ({ad['name']}) — sector: {ad['sector']}, "
            f"rank: #{ad['market_cap_rank']}, 24h: {ad['price_change_24h']:+.1f}%, "
            f"price: ${ad['current_price_usd']:,.2f}{profile_str}"
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
        client = _openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        result = json.loads(response.choices[0].message.content)
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

    return {
        "status": "ok",
        "event_id": event_id,
        "cached": False,
        "summary": result.get("summary", ""),
        "impacts": result.get("impacts", [])
    }


@app.get("/debug")
async def debug():
    import openai, os
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello in JSON: {\"message\": \"hello\"}"}],
            response_format={"type": "json_object"},
            max_tokens=50
        )
        return {"status": "ok", "response": response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "error": str(e), "type": type(e).__name__}
