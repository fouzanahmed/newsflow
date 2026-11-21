from typing import List, Literal, Optional

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


class BulletinItem(BaseModel):
    title: str
    summary: str


class BulletinResponse(BaseModel):
    """Response for POST /bulletin.

    Exactly one of the two payload fields is populated, depending on `source`:
      - "fallback_retrieval": `items` (retrieved corpus snippets)
      - "langchain_agent_claude": `bulletin_text` (LLM-written bulletin)
    """

    topic: str
    source: Literal["fallback_retrieval", "langchain_agent_claude"]
    headline: str
    items: Optional[List[BulletinItem]] = Field(
        default=None, description="Present when source is 'fallback_retrieval'."
    )
    bulletin_text: Optional[str] = Field(
        default=None, description="Present when source is 'langchain_agent_claude'."
    )


@app.get("/health")
def health():
    return {"ok": True}


# response_model_exclude_none keeps each path's wire shape unchanged: the unused
# payload field is omitted rather than serialized as null.
@app.post(
    "/bulletin",
    response_model=BulletinResponse,
    response_model_exclude_none=True,
)
def bulletin(req: BulletinRequest):
    return generate_bulletin(req.topic)
