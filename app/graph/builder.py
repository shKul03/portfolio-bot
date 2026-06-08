import logging
from langgraph.graph import StateGraph, END
from app.graph.state import BotState
from app.graph.nodes import (
    intent_classifier,
    query_rewriter,
    retriever_node,
    answer_generator,
    follow_up_generator,
    should_rewrite,
)
from app.config import settings

logger = logging.getLogger(__name__)

_graph = None
_checkpointer = None
_checkpointer_cm = None


async def build_graph():
    global _graph, _checkpointer, _checkpointer_cm

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        _checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL)
        _checkpointer = await _checkpointer_cm.__aenter__()
        await _checkpointer.setup()
        logger.info("LangGraph AsyncPostgresSaver initialised.")
    except Exception as e:
        logger.warning(f"Could not initialise AsyncPostgresSaver — session memory disabled: {e}")
        _checkpointer = None
        _checkpointer_cm = None

    builder = StateGraph(BotState)

    builder.add_node("intent_classifier", intent_classifier)
    builder.add_node("query_rewriter", query_rewriter)
    builder.add_node("retriever", retriever_node)
    builder.add_node("answer_generator", answer_generator)
    builder.add_node("follow_up_generator", follow_up_generator)

    builder.set_entry_point("intent_classifier")

    builder.add_conditional_edges(
        "intent_classifier",
        should_rewrite,
        {
            "rewrite": "query_rewriter",
            "skip_rewrite": "retriever",
        },
    )

    builder.add_edge("query_rewriter", "retriever")
    builder.add_edge("retriever", "answer_generator")
    builder.add_edge("answer_generator", "follow_up_generator")
    builder.add_edge("follow_up_generator", END)

    _graph = builder.compile(checkpointer=_checkpointer)
    return _graph


async def teardown_graph():
    global _checkpointer_cm
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error tearing down checkpointer: {e}")
        _checkpointer_cm = None


def get_graph():
    return _graph


def get_checkpointer():
    return _checkpointer
