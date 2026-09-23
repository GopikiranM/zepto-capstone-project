# Zepto Support Assistant

## Module Overview

This module implements a policy-grounded Zepto Support Assistant using a
local document corpus, Sentence Transformers embeddings, ChromaDB retrieval,
LangGraph routing, and FastAPI.

The graded baseline uses `MOCK_LLM` in its default state. This mode is fully
deterministic and does not require an LLM API key or an external LLM network call.

---

## Project Components

- `docs/` — 8 Zepto policy documents used as the knowledge corpus
- `rag.py` — document ingestion, embeddings, ChromaDB retrieval, LangGraph
  workflow, intent routing, and response validation
- `main.py` — FastAPI application exposing the `/ask` endpoint
- `chroma_db/` — persistent ChromaDB vector store
- `Dockerfile` — container configuration for the FastAPI service

---

## RAG Pipeline Architecture

The complete pipeline is:

```text
Policy Documents
      |
      v
Ingestion + Embedding
      |
      v
ChromaDB Vector Store
      |
      v
LangGraph: classify_intent
      |
      +------------------------------+
      |                              |
      | policy_question              | general_question
      v                              v
retrieve_and_answer              direct_answer
      |                              |
      v                              v
Top-3 ChromaDB Retrieval        Fixed mock response
      |
      v
Mock/Optional LLM Generation
      |
      v
Pydantic Answer Schema
      |
      v
FastAPI /ask
{
  "query": "What is the delivery fee below INR 149?"
}
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
  "sources": [
    "doc_01",
    "doc_03",
    "doc_05"
  ],
  "confidence": 1.0
}
{
  "query": "What is the capital of India?"
}
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
POST /ask
{
  "query": "What is the delivery fee below INR 149?"
}
uvicorn main:app --host 0.0.0.0 --port 7860
curl -X POST "http://localhost:7860/ask" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is the delivery fee below INR 149?\"}"
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
http://localhost:7860/ask
MOCK_LLM=1
