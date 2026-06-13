import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from app.models.schemas import ChatRequest, ChatResponse, SessionResponse
from app.dependencies import get_bot_graph
from app.graph.builder import get_checkpointer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


async def invoke_graph_with_retry(graph, input_data: dict, config: dict, max_retries: int = 1):
    for attempt in range(max_retries + 1):
        try:
            return await graph.ainvoke(input_data, config)
        except Exception as e:
            error_str = str(e).lower()
            if attempt < max_retries and any(kw in error_str for kw in ("ssl", "connection", "closed")):
                logger.warning(f"Graph invocation failed (attempt {attempt + 1}), retrying: {e}")
                await asyncio.sleep(0.5)
                continue
            raise


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, graph=Depends(get_bot_graph)):
    config = {"configurable": {"thread_id": request.session_id}}

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "personality": request.personality,
        "session_id": request.session_id,
        "retrieved_chunks": [],
        "rewritten_query": None,
        "intent": None,
        "follow_up_questions": [],
    }

    try:
        result = await invoke_graph_with_retry(graph, initial_state, config=config)
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process your message.")

    messages = result.get("messages", [])
    from langchain_core.messages import AIMessage
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    reply = ai_messages[-1].content if ai_messages else "No response generated."

    follow_ups = result.get("follow_up_questions", [])

    return ChatResponse(
        reply=reply,
        follow_ups=follow_ups,
        session_id=request.session_id,
        personality=request.personality,
    )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    checkpointer = get_checkpointer()
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="Session memory not available.")

    config = {"configurable": {"thread_id": session_id}}
    try:
        checkpoint = await checkpointer.aget(config)
    except Exception as e:
        logger.error(f"Failed to retrieve session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session.")

    if checkpoint is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = checkpoint.get("channel_values", {})
    messages = state.get("messages", [])

    serialised = []
    for m in messages:
        serialised.append({
            "type": m.__class__.__name__,
            "content": m.content,
        })

    return SessionResponse(
        session_id=session_id,
        message_count=len(messages),
        messages=serialised,
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    checkpointer = get_checkpointer()
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="Session memory not available.")

    config = {"configurable": {"thread_id": session_id}}
    try:
        await checkpointer.adelete(config)
    except AttributeError:
        # Fallback: write an empty checkpoint to clear the thread
        try:
            from langgraph.checkpoint.base import empty_checkpoint
            await checkpointer.aput(config, empty_checkpoint(), {}, {})
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to clear session.")
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session.")

    return {"detail": f"Session {session_id} cleared."}
