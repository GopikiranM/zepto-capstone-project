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
```

```text

Policy query → HTTP 200 → answer + sources + confidence
Non-policy query → HTTP 200 → restricted response
```
