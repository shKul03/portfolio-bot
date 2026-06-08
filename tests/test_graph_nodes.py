import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.state import BotState
from app.graph import nodes as graph_nodes
from app.personality.prompts import PERSONALITY_PROMPTS


def _make_state(**overrides) -> BotState:
    base: BotState = {
        "messages": [HumanMessage(content="What has she built?")],
        "personality": "professional",
        "session_id": "test-session",
        "retrieved_chunks": [],
        "rewritten_query": None,
        "intent": None,
        "follow_up_questions": [],
    }
    base.update(overrides)
    return base


# ─── Intent classifier ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_classifier_returns_valid_intent():
    state = _make_state(messages=[HumanMessage(content="What projects has she built?")])
    with patch.object(graph_nodes, "_chat_completion", new=AsyncMock(return_value="projects")):
        result = await graph_nodes.intent_classifier(state)
    assert result["intent"] in graph_nodes.INTENT_LABELS


@pytest.mark.asyncio
async def test_intent_classifier_falls_back_on_invalid_label():
    state = _make_state(messages=[HumanMessage(content="foo")])
    with patch.object(graph_nodes, "_chat_completion", new=AsyncMock(return_value="garbage_label")):
        result = await graph_nodes.intent_classifier(state)
    assert result["intent"] == "general"


@pytest.mark.asyncio
async def test_intent_classifier_handles_empty_messages():
    state = _make_state(messages=[])
    result = await graph_nodes.intent_classifier(state)
    assert result["intent"] == "general"


@pytest.mark.asyncio
async def test_intent_classifier_falls_back_on_llm_error():
    state = _make_state(messages=[HumanMessage(content="test")])
    with patch.object(graph_nodes, "_chat_completion", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        result = await graph_nodes.intent_classifier(state)
    assert result["intent"] == "general"


# ─── Query rewriter ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_rewriter_returns_string():
    state = _make_state()
    with patch.object(graph_nodes, "_chat_completion", new=AsyncMock(return_value="Rewritten query")):
        result = await graph_nodes.query_rewriter(state)
    assert isinstance(result["rewritten_query"], str)
    assert len(result["rewritten_query"]) > 0


@pytest.mark.asyncio
async def test_query_rewriter_falls_back_to_original_on_error():
    state = _make_state()
    original_query = state["messages"][0].content
    with patch.object(graph_nodes, "_chat_completion", new=AsyncMock(side_effect=Exception("fail"))):
        result = await graph_nodes.query_rewriter(state)
    assert result["rewritten_query"] == original_query


# ─── Retriever ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retriever_uses_rewritten_query_if_available():
    captured = {}

    async def mock_embed(text):
        captured["query"] = text
        return [0.0] * 768

    state = _make_state(rewritten_query="Rewritten version of the question")
    with patch("app.graph.nodes.embed_text", new=mock_embed), \
         patch("app.graph.nodes.similarity_search", new=AsyncMock(return_value=[])):
        await graph_nodes.retriever_node(state)
    assert captured["query"] == "Rewritten version of the question"


@pytest.mark.asyncio
async def test_retriever_falls_back_to_original_message():
    captured = {}

    async def mock_embed(text):
        captured["query"] = text
        return [0.0] * 768

    state = _make_state(rewritten_query=None)
    with patch("app.graph.nodes.embed_text", new=mock_embed), \
         patch("app.graph.nodes.similarity_search", new=AsyncMock(return_value=[])):
        await graph_nodes.retriever_node(state)
    assert captured["query"] == "What has she built?"


@pytest.mark.asyncio
async def test_retriever_returns_empty_list_on_failure():
    state = _make_state()
    with patch("app.graph.nodes.embed_text", new=AsyncMock(side_effect=Exception("Ollama down"))):
        result = await graph_nodes.retriever_node(state)
    assert result["retrieved_chunks"] == []


# ─── Answer generator ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_answer_generator_selects_professional_prompt():
    captured_system = {}

    async def mock_chat(system, user):
        captured_system["system"] = system
        return "A professional answer."

    state = _make_state(personality="professional", retrieved_chunks=["Some context."])
    with patch.object(graph_nodes, "_chat_completion", new=mock_chat):
        await graph_nodes.answer_generator(state)
    assert captured_system["system"] == PERSONALITY_PROMPTS["professional"]


@pytest.mark.asyncio
async def test_answer_generator_selects_witty_prompt():
    captured_system = {}

    async def mock_chat(system, user):
        captured_system["system"] = system
        return "A witty answer."

    state = _make_state(personality="witty", retrieved_chunks=["Some context."])
    with patch.object(graph_nodes, "_chat_completion", new=mock_chat):
        await graph_nodes.answer_generator(state)
    assert captured_system["system"] == PERSONALITY_PROMPTS["witty"]


@pytest.mark.asyncio
async def test_answer_generator_appends_ai_message():
    state = _make_state(retrieved_chunks=["Context chunk."])
    with patch.object(graph_nodes, "_chat_completion", new=AsyncMock(return_value="AI reply")):
        result = await graph_nodes.answer_generator(state)
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].content == "AI reply"


# ─── Follow-up generator ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_follow_up_generator_returns_exactly_3():
    state = _make_state(
        messages=[HumanMessage(content="What has she built?"), AIMessage(content="She built X.")],
    )
    mock_json = '["What is her tech stack?", "Where is she based?", "How can I contact her?"]'
    with patch.object(graph_nodes, "_chat_completion", new=AsyncMock(return_value=mock_json)):
        result = await graph_nodes.follow_up_generator(state)
    assert len(result["follow_up_questions"]) == 3


@pytest.mark.asyncio
async def test_follow_up_generator_fallback_on_parse_error():
    state = _make_state(
        messages=[HumanMessage(content="test"), AIMessage(content="answer")],
    )
    with patch.object(graph_nodes, "_chat_completion", new=AsyncMock(return_value="not valid json")):
        result = await graph_nodes.follow_up_generator(state)
    assert len(result["follow_up_questions"]) == 3


# ─── All 4 personalities produce different system prompts ─────────────────────

@pytest.mark.asyncio
async def test_all_four_personalities_use_different_prompts():
    systems_used = []

    async def capture_system(system, user):
        systems_used.append(system)
        return "Answer."

    for personality in ("professional", "witty", "hype", "eli5"):
        state = _make_state(personality=personality, retrieved_chunks=["ctx"])
        with patch.object(graph_nodes, "_chat_completion", new=capture_system):
            await graph_nodes.answer_generator(state)

    assert len(set(systems_used)) == 4, "Each personality must use a distinct system prompt"


# ─── should_rewrite routing ───────────────────────────────────────────────────

def test_should_rewrite_routes_greeting_to_skip():
    state = _make_state(intent="greeting")
    assert graph_nodes.should_rewrite(state) == "skip_rewrite"


def test_should_rewrite_routes_contact_to_skip():
    state = _make_state(intent="contact")
    assert graph_nodes.should_rewrite(state) == "skip_rewrite"


def test_should_rewrite_routes_general_to_rewrite():
    state = _make_state(intent="general")
    assert graph_nodes.should_rewrite(state) == "rewrite"


def test_should_rewrite_routes_projects_to_rewrite():
    state = _make_state(intent="projects")
    assert graph_nodes.should_rewrite(state) == "rewrite"
