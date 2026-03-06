from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from datetime import datetime

app = FastAPI()

class EventRequest(BaseModel):
   description: str
   domain: str = "macro"
   impact: float = 0.5

@app.get("/")
def root():
   return {"name": "The Collective", "version": "0.3.0"}

@app.get("/health")
def health():
   return {"status": "healthy"}

@app.get("/stats")
def stats():
   return {"corpus_size": 0, "acceptance_rate": 0.0, "avg_quality": 0.0}

@app.post("/generate")
def generate(request: EventRequest):
   return {
       "status": "generated",
       "event_id": "evt_001",
       "description": request.description,
       "intelligence": {
           "market_regime": "bull",
           "scenarios": [
               {"outcome": "Bull", "probability": 0.6},
               {"outcome": "Base", "probability": 0.3},
               {"outcome": "Bear", "probability": 0.1}
           ],
           "recommendation": "Accumulate on dips"
       }
   }

@app.get("/pending")
def pending():
   return {"pending": [], "count": 0}
