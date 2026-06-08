import logging
import json
from groq import AsyncGroq
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.state import BotState
from app.personality.prompts import PERSONALITY_PROMPTS
from app.config import settings
from app.services.embedder import embed
from app.services.retriever import similarity_search

logger = logging.getLogger(__name__)

INTENT_LABELS = ["greeting", "experience", "projects", "skills", "contact", "availability", "education", "general"]
GROQ_MODEL = "llama-3.3-70b-versatile"


async def _chat_completion(system: str, user: str) -> str:
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


async def intent_classifier(state: BotState) -> BotState:
    messages = state["messages"]
    if not messages:
        return {**state, "intent": "general"}

    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        return {**state, "intent": "general"}

    query = last_human.content
    system = (
        f"Classify the intent of the user's question about Shloka Kulkarni. "
        f"Return exactly one label from: {', '.join(INTENT_LABELS)}. "
        "Reply with ONLY the label, no punctuation, no explanation."
    )
    try:
        intent = await _chat_completion(system, query)
        intent = intent.strip().lower()
        if intent not in INTENT_LABELS:
            intent = "general"
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}")
        intent = "general"

    return {**state, "intent": intent}


async def query_rewriter(state: BotState) -> BotState:
    messages = state["messages"]
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        return {**state, "rewritten_query": None}

    query = last_human.content
    system = (
        "You are a query rewriting assistant. Rewrite the user's question to be more specific "
        "and retrieval-friendly for a vector database about Shloka Kulkarni's professional profile. "
        "Return ONLY the rewritten query, nothing else."
    )
    try:
        rewritten = await _chat_completion(system, query)
        return {**state, "rewritten_query": rewritten.strip()}
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")
        return {**state, "rewritten_query": query}


async def retriever_node(state: BotState) -> BotState:
    query = state.get("rewritten_query") or ""
    if not query:
        messages = state["messages"]
        last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human.content if last_human else ""

    if not query:
        return {**state, "retrieved_chunks": []}

    try:
        embedding = await embed(query)
        results = await similarity_search(
            embedding=embedding,
            top_k=settings.TOP_K,
            min_score=settings.MIN_SCORE,
        )
        chunks = [text for text, _score, _meta in results]
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        chunks = []

    return {**state, "retrieved_chunks": chunks}


async def answer_generator(state: BotState) -> BotState:
    personality = state.get("personality", "professional")
    system_prompt = PERSONALITY_PROMPTS.get(personality, PERSONALITY_PROMPTS["professional"])

    messages = state["messages"]
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    query = last_human.content if last_human else ""

    chunks = state.get("retrieved_chunks", [])
    context = "\n\n---\n\n".join(chunks) if chunks else "No context retrieved."

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = await _chat_completion(system_prompt, user_message)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        answer = "I'm having trouble generating a response right now. Please try again."

    updated_messages = list(messages) + [AIMessage(content=answer)]
    return {**state, "messages": updated_messages}


async def follow_up_generator(state: BotState) -> BotState:
    messages = state["messages"]
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    answer = last_ai.content if last_ai else ""

    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    query = last_human.content if last_human else ""

    system = (
        "You are generating follow-up questions for a portfolio chatbot about Shloka Kulkarni. "
        "Based on the question and answer, generate exactly 3 short follow-up questions a visitor might want to ask next. "
        "Return a JSON array of exactly 3 strings, e.g. [\"Q1?\", \"Q2?\", \"Q3?\"]"
    )
    user = f"Question: {query}\nAnswer: {answer}"

    try:
        raw = await _chat_completion(system, user)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        follow_ups = json.loads(raw)
        if not isinstance(follow_ups, list):
            raise ValueError("Not a list")
        follow_ups = [str(q) for q in follow_ups[:3]]
        while len(follow_ups) < 3:
            follow_ups.append("What else would you like to know about Shloka?")
    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        follow_ups = [
            "What projects has she worked on?",
            "What is her technical stack?",
            "How can I contact Shloka?",
        ]

    return {**state, "follow_up_questions": follow_ups}


def should_rewrite(state: BotState) -> str:
    intent = state.get("intent", "general")
    if intent in ("greeting", "contact"):
        return "skip_rewrite"
    return "rewrite"
