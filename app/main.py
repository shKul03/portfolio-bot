import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db, close_db
from app.graph.builder import build_graph, teardown_graph
from app.routers import chat, ingest, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising database...")
    await init_db()
    logger.info("Building LangGraph...")
    await build_graph()
    logger.info("Portfolio bot ready.")
    yield
    logger.info("Shutting down...")
    await teardown_graph()
    await close_db()


app = FastAPI(
    title="Portfolio Bot — SK",
    description="RAG chatbot answering questions about Shloka Kulkarni.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(ingest.router)
