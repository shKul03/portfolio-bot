# Projects — Shloka Kulkarni

## LiveMind (TechnoAI)
RAG chatbot deployable on any website. Playwright crawls the site (sitemap-first,
fallback paths), chunks content at 700 words with 100-word overlap, embeds using
nomic-embed-text via Ollama, stores in PostgreSQL/pgvector. Cosine similarity
retrieval. Session memory per UUID. Single API call to ingest. Re-ingestion clears
and rebuilds automatically. Fully configurable via env vars. Dockerised. New domain
live in under a day.
GitHub: github.com/shKul03/TechnoAI

## KnowledgeEngine
Enterprise RAG pipeline: ingestion → chunking → custom embedding → pgvector →
cosine retrieval → cross-encoder re-ranking → LLM generation. Modular — each
component swappable. Full document lifecycle (ingest, update, delete, re-embed).
Multi-tenant design. Documented FastAPI endpoints. Test suite for parser,
classifier, vector store.
GitHub: github.com/shKul03/Knowledge-Engine

## OnBoardIQ
AI BGV document pipeline. Classifies 10+ types (Aadhaar, PAN, Passport, Degree,
Resume, Offer Letter). Hybrid classifier: rules for obvious cases, LLM for
ambiguous. 3-layer dedup: byte hash → perceptual hash → semantic similarity.
Per-candidate folder trees. Excel audit reports with confidence scores.
Feature-flag phased rollout: PoC → Enhanced → Enterprise.
GitHub: github.com/shKul03/OnBoardIQ-Docs

## SentiCore
WhatsApp AI appointment system for hospitals. No app or website needed. AI handles:
symptom collection, specialist suggestion, availability check, slot booking, queue
management, live queue updates. Natural conversation flow. Shloka drove client
meetings, delivered technical walkthroughs and live demos to hospital stakeholders,
contributed to sales pitch, documentation, and direct AI feature additions.
GitHub: github.com/shKul03/SentiCure

## Smart Revenue Collector
AI debt-collection automation. ML defaulter scoring by likelihood-to-pay. Priority
queue. LLM personalised outreach per defaulter. Penalty/incentive engine.
React/TypeScript analytics dashboard. Live on GitHub Pages. Presented to
stakeholders with live demo.
GitHub: github.com/shKul03/SmartRevenueCollector

## VoiceBot PoC
Internal PoC at Technossus AI Studio. Python/FastAPI STT/TTS microservice +
C#/.NET orchestrator. Clean Architecture: API → Application → Domain →
Infrastructure. Phone-number capture flow. Session state management.
Extensible provider layer.
GitHub: github.com/Voice-Bot-poc/Voicebot-orchestrator-backend

## BillSage AI
Async OCR bill classification pipeline. Image and PDF uploads. Concurrent async
processing. Bill type classification (utility, grocery, medical, retail). Streamlit
dashboard with confidence scores and accuracy metrics. Async load tested.
GitHub: github.com/shKul03/BillSage

## Technossus Design System
Figma-to-code design system. W3C design tokens, CSS variables, custom Tailwind
preset. React components: Tag, Stats Card, Testimonial, SearchBar, Nav. Built full
pages from Figma and from content alone. Single source of truth for Technossus
frontend.
GitHub: github.com/shKul03/Technossus-Design-System
Live: technossus-design-system.vercel.app
