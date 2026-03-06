async def judge(event_id: str, evaluation: JudgeEvaluation):
   """Arca judges submit evaluations here"""
   corpus_path = "/tmp/corpus.jsonl"

   try:
       # Read corpus
       entries = []
       with open(corpus_path, 'r') as f:
           for line in f:
               entries.append(json.loads(line))

       # Find and update entry
       for entry in entries:
           if entry["event_id"] == event_id:
               entry["quality_score"] = evaluation.score
               entry["judged"] = True
               entry["judge_feedback"] = evaluation.feedback
               entry["accepted"] = evaluation.accepted

               # Rewrite corpus
               with open(corpus_path, 'w') as f:
                   for e in entries:
                       f.write(json.dumps(e) + '\n')

               return {
                   "status": "judged",
                   "event_id": event_id,
                   "score": evaluation.score,
                   "accepted": evaluation.accepted
               }

       return {"status": "error", "error": "Event not found"}

   except Exception as e:
       return {"status": "error", "error": str(e)}

@app.get("/pending")
def pending():
   """Get intelligence waiting for judge evaluation"""
   corpus_path = "/tmp/corpus.jsonl"
   pending_items = []

   try:
       if os.path.exists(corpus_path):
           with open(corpus_path, 'r') as f:
               for line in f:
                   entry = json.loads(line)
                   if not entry.get("judged", False):
                       pending_items.append({
                           "event_id": entry["event_id"],
                           "domain": entry["domain"],
                           "description": entry["intelligence"].get("input_event", "Unknown"),
                           "confidence": entry["confidence"]
                       })
   except:
       pass

   return {"pending": pending_items, "count": len(pending_items)}
