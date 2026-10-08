from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from agent.bulletin_agent import generate_bulletin

app = FastAPI(title="NewsFlow Bulletin Agent")

TOPIC_MAX_LENGTH = 200


class BulletinRequest(BaseModel):
    topic: str = Field(..., max_length=TOPIC_MAX_LENGTH)

    @field_validator("topic")
    @classmethod
    def topic_not_blank(cls, v: str) -> str:
        # Reject empty/whitespace-only topics before they reach the retriever.
        v = v.strip()
        if not v:
            raise ValueError("topic must not be empty")
        return v


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/bulletin")
def bulletin(req: BulletinRequest):
    return generate_bulletin(req.topic)
