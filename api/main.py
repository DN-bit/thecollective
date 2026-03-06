# The Collective - Central API
# FastAPI application - Central Mind HTTP interface

import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor
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

def get_db_conn():
    """Get Postgres connection from DATABASE_URL env var."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)


def init_db():
    """Create corpus table if it doesn't exist."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
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
            conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def startup():
    """Initialize DB tables on startup."""
    try:
        init_db()
        print("[Collective] Database initialized successfully")
    except Exception as e:
        print(f"[Collective] WARNING: DB init failed: {e}")
        # Don't crash on startup — allows the API to run without DB for testing


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EventRequest(BaseModel):
    description: str = Field(..., min_length=10, description="Event description")
    impact: float = Field(..., ge=0.0, le=1.0, description="Impact score 0.0-1.0")
    domain: str = Field(default="macro", description="macro | defi | policy | sentiment")
    source: Optional[str] = Field(default="manual", description="Event source")


class JudgeRequest(BaseModel):
    event_id: str
    scores: Dict[str, float]
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Inline MacroNode (so deployment works before agents/ folder is built out)
# ---------------------------------------------------------------------------

class MacroNode(SpecialistNode):
    """MacroNode - market intelligence and geopolitical analysis."""

    def __init__(self):
        super().__init__(
            domain="macro",
            system_prompt="""You are MacroNode, a specialist in crypto macro intelligence.
You analyze geopolitical events, Fed policy, institutional flows, and market structure.
Always return structured JSON with chain-of-thought reasoning, scenarios, and citations.
Be specific, data-driven, and actionable.""",
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
        {{"outcome": "description", "probability": 0.X, "rationale": "..."}},
        {{"outcome": "description", "probability": 0.X, "rationale": "..."}},
        {{"outcome": "description", "probability": 0.X, "rationale": "..."}}
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
def health():
    """Health check — also verifies DB connectivity."""
    db_ok = False
    try:
        conn = get_db_conn()
        conn.close()
        db_ok = True
    except Exception as e:
        pass
    return {
        "status": "ok",
        "db_connected": db_ok,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/generate")
async def generate(request: EventRequest):
    """Generate intelligence from a Specialist Node."""

    # Build the event object
    event = IntelligenceEvent(
        id=f"evt_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        timestamp=datetime.now(),
        source=request.source or "manual",
        description=request.description,
        impact_score=request.impact,
        relevant_domains=[request.domain],
        raw_data=None
    )

    # Select node by domain (expand as other nodes are built)
    domain_map = {
        "macro": MacroNode,
    }
    NodeClass = domain_map.get(request.domain, MacroNode)
    node = NodeClass()

    try:
        # Generate variants (n=1 for API speed; increase for training runs)
        outputs = await node.generate(event, n_variants=1)

        if not outputs:
            raise HTTPException(
                status_code=422,
                detail="Node generated no outputs that passed quality threshold"
            )

        best = outputs[0]

        # Persist to Postgres corpus
        try:
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO corpus 
                        (event_id, description, impact, domain, intelligence, confidence, judged)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                """, (
                    event.id,
                    request.description,
                    request.impact,
                    request.domain,
                    json.dumps(best.output),
                    best.confidence,
                    False
                ))
                conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"[Collective] WARNING: DB write failed: {db_err}")
            # Don't fail the API response if DB write fails

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
def get_corpus(limit: int = 20, judged: Optional[bool] = None, domain: Optional[str] = None):
    """Retrieve corpus entries for review or training."""
    try:
        conn = get_db_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM corpus WHERE 1=1"
            params = []
            if judged is not None:
                query += " AND judged = %s"
                params.append(judged)
            if domain:
                query += " AND domain = %s"
                params.append(domain)
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            rows = cur.fetchall()
        conn.close()
        return {"status": "ok", "count": len(rows), "entries": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/judge/{event_id}")
def judge_entry(event_id: str, request: JudgeRequest):
    """Submit judge scores for a corpus entry (Consensus Chamber)."""
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE corpus
                SET judged = TRUE,
                    judge_scores = %s
                WHERE event_id = %s
            """, (json.dumps(request.scores), event_id))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
            conn.commit()
        conn.close()
        return {"status": "judged", "event_id": event_id, "scores": request.scores}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """Corpus statistics for monitoring the Synchronization Loop."""
    try:
        conn = get_db_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE judged = TRUE) AS judged,
                    COUNT(*) FILTER (WHERE judged = FALSE) AS pending_judgment,
                    AVG(confidence) AS avg_confidence,
                    COUNT(DISTINCT domain) AS active_domains
                FROM corpus
            """)
            row = dict(cur.fetchone())
        conn.close()
        return {"status": "ok", "corpus": row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
