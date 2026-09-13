from fastapi import FastAPI
from pydantic import BaseModel

from agent.bulletin_agent import generate_bulletin

app = FastAPI(title="NewsFlow Bulletin Agent")


class BulletinRequest(BaseModel):
    topic: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/bulletin")
def bulletin(req: BulletinRequest):
    return generate_bulletin(req.topic)
