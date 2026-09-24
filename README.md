# Zepto AI/ML Capstone Project

This repository contains the complete Zepto AI/ML Capstone Project with three integrated modules:

1. Data Pipeline
2. Titanic Analytics & Machine Learning
3. Zepto Support Assistant

---

## Project Structure

```text
zepto-capstone-project/
│
├── data_pipeline/
│   ├── data/
│   ├── scraper.py
│   ├── clean.py
│   ├── database.py
│   └── queries.py
│
├── analytics/
│   ├── artifacts/
│   ├── figures/
│   ├── titanic.csv
│   └── cleaned_titanic.csv
│
└── support_assistant/
    ├── docs/
    ├── chroma_db/
    ├── main.py
    ├── rag.py
    └── Dockerfile

```
## Installation

Clone the repository:

```bash
git clone https://github.com/GopikiranM/zepto-capstone-project.git
cd zepto-capstone-project



### Requirements

Python 3.10+ is recommended.
#### Data Pipeline
    pip install pandas requests beautifulsoup4 lxml

#### Analytics

    pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib

#### Support Assistant

    pip install fastapi uvicorn chromadb sentence-transformers pydantic langgraph

## How to Run

### 1. Data Pipeline

    cd data_pipeline
    python scraper.py
    python clean.py
    python database.py
    python queries.py

### 2. Analytics

Open the analytics notebook/script available in the `analytics` folder and run it.

The module performs data cleaning, EDA, classification, evaluation, class imbalance analysis, hyperparameter tuning, and regression analysis.

### 3. Support Assistant

    cd support_assistant
    uvicorn main:app --reload

Open:

    http://127.0.0.1:8000/docs

Use the `POST /ask` endpoint to test the assistant.

The default configuration uses:

    MOCK_LLM=1

No external LLM API call is required in mock mode.

## Design Decisions

### Data Pipeline

- Web data is collected and cleaned before database storage.
- SQLite is used for structured local storage.
- SQL queries are used for aggregation and analysis.

### Analytics

- Titanic data is cleaned before modelling.
- Multiple classification models are evaluated using standard metrics.
- Class imbalance is examined using baseline, class weighting, and SMOTE.
- Grid search is used for Random Forest hyperparameter tuning.
- Regression metrics are used for model evaluation.

### Support Assistant

- Eight Zepto policy documents are embedded and stored in ChromaDB.
- Retrieval uses cosine similarity and returns the top 3 relevant documents.
- LangGraph uses three nodes: `classify_intent`, `retrieve_and_answer`, and `direct_answer`.
- `MOCK_LLM=1` provides deterministic answers without an external LLM.
- Policy questions use retrieved context, while unrelated questions receive a fixed restricted response.
- Pydantic validates the final answer structure.

## End-to-End Flow

    Data Pipeline
        ↓
    Cleaned / Structured Data
        ↓
    Analytics & Machine Learning
        ↓
    Support Assistant
        ↓
    FastAPI /ask endpoint



