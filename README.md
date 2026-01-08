# Retrieval-Augmented Generation (RAG) for Federal Reserve Press Conferences

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** to analyze U.S. Federal Reserve (FED) press conference transcripts from **2011 to 2025 (Q4)**.

The primary goal is **not** to build a production chatbot, but to **systematically evaluate how different chunking and retrieval strategies affect answer quality** in a high-stakes financial domain (monetary policy communication).

The system focuses on:
- Historical **sentiment analysis** (Hawkish / Neutral / Dovish)
- Multi-period comparisons (e.g. 2019 vs 2020, early vs late 2021)
- Hallucination detection in factual queries

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Rubenvalrom/Retrieval-Augmented_Generation_in_FED_press_releases)
---

## High-level Architecture

```
FED Website (PDFs)
        ↓
[Data Ingestion]
        ↓
[Cleaning & Metadata Normalization]
        ↓
[Chunking Experiments]
  ├─ Semantic Chunking (percentile-based)
  └─ Recursive Character Chunking
        ↓
[Embedding & Indexing]
        ↓
[Chroma Vector Store]
        ↓
[RAG Pipeline]
        ↓
[Evaluation with LLM Judge + MLflow]
        ↓
[Gradio Interface]
```

---

## Data Source

- **Source**: Official Federal Reserve website
- **Documents**: FOMC press conference transcripts (PDF)
- **Coverage**: 2011 → 2025 (Q4)
- **Exclusions**: FOMC minutes are intentionally excluded due to publication delay

All data is treated as **read-only historical text**; no external knowledge is injected during generation.

---

## Project Structure

```
.
├── data/
│   ├── import_data.py          # Scrapes and downloads FED PDFs
│   ├── clean_data.py           # Cleans metadata and serializes documents
│   ├── insert_data_to_chroma.py# Chunking + embedding + indexing experiments
│   └── raw/                    # Raw downloaded PDFs
│   └── clean/                  # Cleaned documents (pickle)
|   └── chroma/                 # Chroma database
|   └── mlflow/                 # Mlflow database (contains all runs, artifacts, etc.)
│
├── src/
│   ├── rag.py                  # Core RAG pipeline
│   ├── run_experiments.py      # Automated experiment runner
│   └── utils/
│       ├── llms.py             # LLM and embedding loaders
│       ├── prompts.py          # System + judge prompts
│       ├── format.py           # Context formatting & JSON fixing
│       └── evaluate.py         # Evaluation logic + MLflow logging
│
├── docker-compose.yml          # ChromaDB + MLflow services
├── environment.yml             # Conda environment (research-oriented)
├── requirements.txt            # Python dependencies
├── main.py                     # Gradio interface
└── README.md
```

---

## Chunking Strategies

This project explicitly **compares chunking methods**, treating them as experimental variables.

### 1. Semantic Chunking
- Based on embedding similarity breakpoints
- Percentile thresholds tested: `50, 75, 90, 97.5`
- Model: `BAAI/bge-small-en-v1.5`

**Goal**: Preserve semantic coherence across long policy statements.

### 2. Recursive Character Chunking
- Fixed chunk sizes with overlap
- Chunk sizes: `500, 1000, 1500`
- Overlap: `10%`, `15%`

**Goal**: Provide a strong lexical baseline.

Each configuration is indexed into a **separate Chroma collection**, making results directly comparable.

---

## Embeddings & Retrieval

- **Embedding model**: `BAAI/bge-large-en-v1.5`
- **Vector store**: ChromaDB (persistent)
- **Retrieval strategy**: Maximal Marginal Relevance (MMR)

Retrieval parameters explored:
- `k ∈ {10, 20, 30, 50}`
- `fetch_k = 5 × k`
- `lambda_mult = 0.7`

---

## RAG Pipeline

For each query:
1. Retrieve relevant chunks using MMR
2. Format context with strict metadata headers
3. Apply a **domain-specific system prompt** (FED monetary policy analyst)
4. Generate a **structured JSON response** with:
   - Answer
   - Sentiment classification
   - Key evidence with citations

Strict constraints are enforced:
- No external knowledge
- Explicit citation of source fragments
- Period-awareness (early vs late, multi-year comparisons)

---

## Evaluation Framework

### Test Queries
Three non-trivial evaluation queries are used:

1. **2021 Inflation Narrative**  
   Evolution of the term *"transitory"* and tone shift

2. **Crisis Comparison**  
   Unemployment urgency: post-2008 vs pandemic onset (2020)

3. **Hallucination Detection (2025)**  
   Interest rate decision + data availability claims

---

### LLM-as-a-Judge

A **separate, local LLM** evaluates generated answers using strict, query-specific criteria.

Each query has multiple boolean checks (e.g.:
- keyword presence
- timeline correctness
- explicit comparison
- factual grounding in evidence
)

### Overall Score (Important)

> **Overall score = total number of `True` evaluations across all criteria and all three questions.**

- Each boolean criterion contributes **+1** if satisfied
- Scores are **additive** across questions
- The score is used **only for relative comparison** between chunking + retrieval configurations

*Note: This is **not an absolute quality metric**. It is designed for **controlled experimentation**, not benchmarking against humans.*

All parameters, answers, and scores are tracked in **MLflow database**.

---

## Experiment Tracking (MLflow)

- Tracking server runs via Docker
- Each run logs:
  - Chunking parameters
  - Retrieval parameters
  - Generated answers (JSON)
  - Per-criterion scores
  - Overall score

This enables **post-hoc analysis**.

---

## Gradio Interface

A minimal Gradio UI is provided for interactive querying:

- Input: free-text question
- Output:
  - Sentiment (short)
  - Answer (long, cited)

The interface is intended for **exploration and demonstration**, not production use.

---

## Scope & Limitations

- Research-focused, not production-ready
- Environment is intentionally heavy for experimentation
- Latency and cost are not optimized (free groq layer used)
- Evaluation is not absolute, a small local llm judge was used

---

## Prerequisites

- **Python**: 3.11+
- **Docker**: For ChromaDB and MLflow services
- **Conda**: Recommended for environment management (optional)
- **Git**: For cloning the repository
- **System**: 8GB+ RAM recommended for embedding models

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Rubenvalrom/Retrieval-Augmented_Generation_in_FED_press_releases.git
cd Retrieval-Augmented_Generation_in_FED_press_releases
```

### 2. Create Environment

**Option A: Using Conda (recommended)**
```bash
conda env create -f environment.yml
conda activate rag-fed
```

**Option B: Using pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start Docker Services

Start ChromaDB and MLflow tracking servers:

```bash
docker-compose up -d
```
---

## Usage

### Pipeline Execution (Step-by-Step)

#### 1. Data Ingestion
Download FED press conference transcripts from the Federal Reserve website:
```bash
python data/import_data.py
```

#### 2. Data Cleaning
Clean and normalize metadata:
```bash
python data/clean_data.py
```

#### 3. Chunking & Indexing
Apply chunking strategies, create embeddings, and index documents into Chroma:
```bash
python data/insert_data_to_chroma.py
```

#### 4. Run Experiments
Execute automated chunking/retrieval experiments:
```bash
python src/run_experiments.py
```

Results are logged to MLflow (http://localhost:5000).

#### 5. Launch Gradio Interface
Start the interactive Q&A interface:
```bash
python main.py
```

Access the UI at the URL shown in the terminal (typically http://127.0.0.1:7860).

---

### Using MLflow Entry Points

Alternatively, run via MLflow:

```bash
mlflow run . -e ingest
mlflow run . -e clean
mlflow run . -e index
mlflow run . -e experiments
mlflow run . -e ui
```

---

## License

MIT License

