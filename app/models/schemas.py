from pydantic import BaseModel, field_validator
from typing import Optional
from app.personality.prompts import VALID_PERSONALITIES


class ChatRequest(BaseModel):
    session_id: str
    message: str
    personality: str = "professional"

    @field_validator("personality")
    @classmethod
    def validate_personality(cls, v: str) -> str:
        if v not in VALID_PERSONALITIES:
            raise ValueError(f"personality must be one of {sorted(VALID_PERSONALITIES)}")
        return v


class ChatResponse(BaseModel):
    reply: str
    follow_ups: list[str]
    session_id: str
    personality: str


class SessionResponse(BaseModel):
    session_id: str
    message_count: int
    messages: list[dict]


class CrawlRequest(BaseModel):
    url: str
    label: str


class HealthResponse(BaseModel):
    status: str
    db: bool
    embedder: bool
    llm: bool
