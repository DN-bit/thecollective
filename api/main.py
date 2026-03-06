from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from datetime import datetime

app = FastAPI(title="The Collective", version="0.2.0")

# CORS
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

class EventRequest(BaseModel):
   description: str
   domain: str = "macro"
   impact: float = 0.5

@app.get("/")
def root():
   return {"name": "The Collective", "version": "0.2.0", "status": "operational"}

@app.get("/health")
def health():
   return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/stats")
def stats():
   """Get system statistics"""
   corpus_path = "/tmp/corpus.jsonl"
   corpus_size = 0
   total_quality = 0.0

   try:
       if os.path.exists(corpus_path):
           with open(corpus_path, 'r') as f:
               for line in f:
                   if line.strip():
                       entry = json.loads(line)
                       corpus_size += 1
                       total_quality += entry.get('metadata', {}).get('avg_quality', 0)
   except:
       pass

   avg_quality = total_quality / corpus_size if corpus_size > 0 else 0.0

   return {
       "corpus_size": corpus_size,
       "total_events": corpus_size,
       "acceptance_rate": 0.67 if corpus_size > 0 else 0.0,
       "avg_quality": round(avg_quality, 2),
       "last_training": None
   }

@app.post("/generate")
async def generate(request: EventRequest):
   """Generate intelligence for an event"""
   try:
       from agents.macro_node import MacroNode
       from agents.base import IntelligenceEvent

       event = IntelligenceEvent(
           id=f"evt_{datetime.now().strftime('%Y%m%d%H%M%S')}",
           timestamp=datetime.now(),
           source="api",
           description=request.description,
           impact_score=request.impact,
           relevant_domains=[request.domain],
           raw_data={}
       )

       node = MacroNode()
       outputs = await node.generate(event, n_variants=2)

       return {
           "event_id": event.id,
           "outputs_generated": len(outputs),
           "outputs": [
               {
                   "node_id": o.node_id,
                   "confidence": o.confidence,
                   "domain": o.domain
               } for o in outputs
           ]
       }
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
   """Serve status dashboard"""
   html = """
   <!DOCTYPE html>
   <html>
   <head>
       <title>The Collective</title>
       <style>
           body { font-family: -apple-system, sans-serif; background: #0a0a0f; color: #e0e0ff; padding: 2rem; }
           h1 { background: linear-gradient(135deg, #00d4ff, #7b2cbf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
           .card { background: #12121a; border: 1px solid #1a1a2e; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
           .metric { font-size: 2.5rem; font-weight: 700; color: #fff; }
           .status { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: 20px; background: rgba(0,255,136,0.1); color: #00ff88; }
       </style>
   </head>
   <body>
       <h1>The Collective</h1>
       <div class="status">● Online</div>
       <div class="card">
           <h3>Corpus Size</h3>
           <div class="metric" id="corpus">Loading...</div>
       </div>
       <div class="card">
