# Portfolio Bot — SK

A production-ready RAG chatbot that answers questions about Shloka Kulkarni, an AI Engineer based in Pune, India. Built with FastAPI, LangGraph, pgvector, and Groq/Ollama. The bot has four distinct personality modes, persistent session memory via PostgreSQL, and a Playwright-based web crawler for ingesting any website as a knowledge source.

---

## Personality Modes

| Mode | Description |
|------|-------------|
| `professional` | Clean, factual, recruiter-friendly — like a well-crafted cover letter response |
| `witty` | Sharp, dry, lightly sarcastic — competent and warm underneath |
| `hype` | Enthusiastic, energetic, TEDx-intro energy — genuinely proud |
| `eli5` | Simple analogies, no jargon — explains to a smart non-technical friend |

---

## Local Setup (Docker Compose)

**Prerequisites:** Docker, Docker Compose, an Ollama instance running `nomic-embed-text`, a Groq API key.

### 1. Clone and configure

```bash
git clone https://github.com/shKul03/portfolio-bot.git
cd portfolio-bot
cp .env.example .env
# Edit .env — fill in GROQ_API_KEY and INGEST_API_KEY at minimum
```

### 2. Start services

```bash
docker compose up --build -d
```

The API is available at `http://localhost:8000`.

### 3. Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok","db":true,"embedder":true,"llm":true}
```

---

## Ingest Knowledge Files

Re-ingest all `knowledge/*.md` files:

```bash
curl -X POST http://localhost:8000/ingest/knowledge \
  -H "X-API-Key: your-ingest-api-key"
```

Or run the script directly (outside Docker):

```bash
pip install -r requirements.txt
python scripts/ingest_knowledge.py
```

---

## Ingest a Website

```bash
curl -X POST http://localhost:8000/ingest/crawl \
  -H "X-API-Key: your-ingest-api-key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://shlokakulkarni.vercel.app", "label": "portfolio-site"}'
```

The crawl runs in the background. Check logs for progress.

---

## API Reference

### Chat

**POST /chat** — Send a message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-123",
    "message": "What has she built?",
    "personality": "witty"
  }'
```

Response:
```json
{
  "reply": "Oh, just a few things...",
  "follow_ups": ["What is LiveMind?", "Where is she based?", "How can I contact her?"],
  "session_id": "my-session-123",
  "personality": "witty"
}
```

**GET /chat/session/{session_id}** — Retrieve session history

```bash
curl http://localhost:8000/chat/session/my-session-123
```

**DELETE /chat/session/{session_id}** — Clear session memory

```bash
curl -X DELETE http://localhost:8000/chat/session/my-session-123
```

---

### Ingest (requires `X-API-Key` header)

**POST /ingest/crawl** — Crawl a website

```bash
curl -X POST http://localhost:8000/ingest/crawl \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "label": "example"}'
```

**POST /ingest/knowledge** — Re-ingest knowledge files

```bash
curl -X POST http://localhost:8000/ingest/knowledge \
  -H "X-API-Key: your-key"
```

**DELETE /ingest/content/{label}** — Remove all chunks for a label

```bash
curl -X DELETE http://localhost:8000/ingest/content/example \
  -H "X-API-Key: your-key"
```

---

### Health

**GET /health**

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "db": true, "embedder": true, "llm": true}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `GROQ_API_KEY` | Yes (groq) | — | Groq API key |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama host |
| `OLLAMA_MODEL` | No | `llama3` | Ollama LLM model name |
| `EMBEDDING_MODEL` | No | `nomic-embed-text` | Ollama embedding model |
| `LLM_PROVIDER` | No | `groq` | `groq` or `ollama` |
| `INGEST_API_KEY` | Yes | — | Key for all `/ingest` endpoints |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000` | Comma-separated CORS origins |
| `TOP_K` | No | `5` | Number of retrieved chunks |
| `MIN_SCORE` | No | `0.42` | Minimum cosine similarity score |
| `CHUNK_SIZE` | No | `700` | Words per chunk |
| `CHUNK_OVERLAP` | No | `100` | Overlap words between chunks |

---

## Railway Deployment

1. Push this repository to GitHub.
2. In Railway, create a new project → **Deploy from GitHub repo**.
3. Add a **PostgreSQL** plugin — Railway injects `DATABASE_URL` automatically.
4. Set all required environment variables in Railway's Variables tab.
5. Railway reads `railway.toml` and builds via the Dockerfile automatically.
6. Health check: `GET /health` (configured in `railway.toml`).

```bash
# One-time: ingest knowledge after deploy
curl -X POST https://your-app.railway.app/ingest/knowledge \
  -H "X-API-Key: your-ingest-api-key"
```

---

## Keeping Neon warm

Neon's free tier suspends compute after 5 minutes of inactivity. To prevent
cold starts during active use, set up a cron job to ping the bot every 4 minutes.

**Option A — cron-job.org (free, recommended)**
1. Go to cron-job.org and create a free account
2. Create a new cron job:
   - URL: `https://your-railway-url.up.railway.app/ping`
   - Schedule: every 4 minutes
   - Method: GET
3. Enable the job — that's it

**Option B — GitHub Actions (free)**

Create `.github/workflows/keepalive.yml`:

```yaml
name: Keep Neon warm
on:
  schedule:
    - cron: '*/4 * * * *'
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -f ${{ secrets.BOT_URL }}/ping
```

Add `BOT_URL` as a GitHub Actions secret with your Railway URL.
