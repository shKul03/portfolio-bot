from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    DATABASE_URL: str
    # Groq API key — used for both LLM generation (llama3)
    # and embeddings (nomic-embed-text-v1_5). Get yours free at console.groq.com
    GROQ_API_KEY: Optional[str] = None
    INGEST_API_KEY: str
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    TOP_K: int = 5
    MIN_SCORE: float = 0.42
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def asyncpg_url(self) -> str:
        """Convert Neon postgres:// URL to asyncpg-compatible format."""
        url = self.DATABASE_URL
        url = url.replace("postgresql://", "").replace("postgres://", "")
        url = url.split("?")[0]
        return f"postgresql://{url}"

    @property
    def asyncpg_ssl(self) -> bool:
        """Return True if the DATABASE_URL contains sslmode=require."""
        return "sslmode=require" in self.DATABASE_URL


settings = Settings()
