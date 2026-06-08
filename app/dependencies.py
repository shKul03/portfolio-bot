from app.graph.builder import get_graph, get_checkpointer


def get_bot_graph():
    graph = get_graph()
    if graph is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Graph not initialised.")
    return graph
