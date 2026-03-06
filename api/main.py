@app.post("/generate")
def generate(request: EventRequest):
   import openai
   import os

   client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

   prompt = f"""Analyze this crypto macro event and return structured intelligence:

Event: {request.description}
Impact: {request.impact}/1.0

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

   try:
       response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": prompt}],
           response_format={"type": "json_object"}
       )

       intelligence = json.loads(response.choices[0].message.content)

       # Save to corpus
       corpus_path = "/tmp/corpus.jsonl"
       entry = {
           "event_id": f"evt_{datetime.now().strftime('%Y%m%d%H%M%S')}",
           "description": request.description,
           "intelligence": intelligence,
           "timestamp": datetime.now().isoformat(),
           "judged": False
       }
       with open(corpus_path, 'a') as f:
           f.write(json.dumps(entry) + '\n')

       return {
           "status": "generated",
           "event_id": entry["event_id"],
           "intelligence": intelligence
       }

   except Exception as e:
       return {"status": "error", "error": str(e)}
