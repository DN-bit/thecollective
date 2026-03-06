from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "time": datetime.now().isoformat()}

@app.get("/health")
def health():
    return {"status": "healthy"}
