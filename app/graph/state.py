from typing import TypedDict, Optional
from langchain_core.messages import BaseMessage


class BotState(TypedDict):
    messages: list[BaseMessage]
    personality: str
    session_id: str
    retrieved_chunks: list[str]
    rewritten_query: Optional[str]
    intent: Optional[str]
    follow_up_questions: list[str]
