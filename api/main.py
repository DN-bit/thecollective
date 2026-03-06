from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
from datetime import datetime

app = FastAPI(title="The Collective", version="0.2.0")

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
   return {
       "event_id": f"evt_{datetime.now().strftime('%Y%m%d%H%M%S')}",
       "description": request.description,
       "domain": request.domain,
       "status": "queued"
   }
