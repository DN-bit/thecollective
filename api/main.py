# The Collective - Central API
# FastAPI application - Central Mind HTTP interface

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
# Database - asyncpg (async, no binary compilation issues)
# ---------------------------------------------------------------------------

async def get_db_conn():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    # Render sometimes gives postgres:// — asyncpg needs postgresql://
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
        """)
    finally:
        await conn.close()


@app.on_event("startup")
async def startup():
    try:
        await init_db()
        print("[Collective] Database initialized successfully")
    except Exception as e:
        print(f"[Collective] WARNING: DB init failed: {e}")


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
# MacroNode (inline until agents/ folder is built out)
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
