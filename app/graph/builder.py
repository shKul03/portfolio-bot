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
_pool = None


async def build_graph():
    global _graph, _checkpointer, _pool

    try:
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # psycopg accepts standard postgresql:// URLs; strip channel_binding which Neon adds
        # but psycopg handles as a no-op and can cause parse issues on some drivers
        url = settings.DATABASE_URL.replace("postgres://", "postgresql://")

        _pool = AsyncConnectionPool(
            conninfo=url,
            max_size=5,
            kwargs={"autocommit": True},
        )
        await _pool.open()
        _checkpointer = AsyncPostgresSaver(_pool)
        await _checkpointer.setup()
        logger.info("LangGraph AsyncPostgresSaver initialised with connection pool.")
    except Exception as e:
        logger.warning(f"Could not initialise AsyncPostgresSaver — session memory disabled: {e}")
        _checkpointer = None
        _pool = None

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
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
        except Exception as e:
            logger.warning(f"Error closing checkpointer pool: {e}")
        _pool = None


def get_graph():
    return _graph


def get_checkpointer():
    return _checkpointer
