# NL2SPARQL — Multilingual Natural Language to SPARQL Query System

A thesis project that allows non-technical users to query a flight Knowledge Graph using natural language in **English**, **French**, or **Arabic** — without writing SPARQL.

---

## Overview

This system translates natural language questions into SPARQL queries through a structured pipeline:

1. **Language detection** — identifies whether the question is in English, French, or Arabic
2. **Entity extraction** — uses an LLM to extract the flight number and the requested property
3. **Hybrid Mapping Layer** — maps natural language expressions to Knowledge Graph URIs via a lexicon (exact match) or multilingual embeddings (semantic fallback)
4. **SPARQL generation** — injects the resolved URIs into an LLM prompt to generate a valid SPARQL query
5. **Execution** — runs the query against a local Apache Jena Fuseki triplestore
6. **Answer formatting** — reformulates the raw result into a natural language answer in the user's language

---

## Prerequisites

Before running the project, you must install and configure the following external tools.

### 1. Ollama (Local LLM Runtime)

Ollama is used to run the `llama3` model locally.

- Download and install from: https://ollama.com
- After installation, pull the required model:

```bash
ollama pull llama3
```

- Keep Ollama running in the background before launching the system.

### 2. Apache Jena Fuseki (SPARQL Triplestore)

Fuseki hosts the flight Knowledge Graph and exposes a SPARQL endpoint.

- Download from: https://jena.apache.org/download/
- Start the server and load the Knowledge Graph dataset named `flights`
- The system expects the endpoint at: `http://localhost:3030/flights/sparql`

---

## Installation

### Clone the repository

```bash
git clone <your-github-url>
cd <repository-folder>
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs the following libraries:

| Library | Purpose |
|---|---|
| `langdetect` | Detects the language of the input question |
| `ollama` | Python client to communicate with the local Ollama LLM |
| `sentence-transformers` | Multilingual embeddings for semantic property matching |
| `rdflib` | SPARQL syntax validation |

> Python 3.9 or higher is recommended.

---

## Running the System

Edit the `main.py` file to set your question and prompting strategy:

```python
question = "Where does flight OS235 depart from?"
condition = "zero-shot"  # options: "zero-shot", "few-shot", "cot"
```

Then run:

```bash
python main.py
```

---

## Project Structure

```
.
├── main.py                  # Entry point — configure question and strategy here
├── lexicon.json             # Multilingual lexicon mapping expressions to KG properties
├── requirements.txt         # Python dependencies
├── logs.jsonl               # Auto-generated evaluation logs (one JSON object per run)
└── pipeline/
    ├── language.py          # Language detection using langdetect
    ├── extractor.py         # LLM-based entity and property extraction
    ├── mapper.py            # Hybrid Mapping Layer (lexicon + embeddings)
    ├── generator.py         # SPARQL generation via URI injection
    └── executor.py          # SPARQL validation, execution, and answer formatting
```

---

## Prompting Strategies

The system supports three experimental conditions, selectable via the `condition` variable in `main.py`:

| Strategy | Description |
|---|---|
| `zero-shot` | The LLM receives the URIs and generates SPARQL with no examples |
| `few-shot` | The LLM is given example question–SPARQL pairs before generating |
| `cot` | Chain-of-Thought: the LLM reasons step by step before generating |

---

## Supported Languages

| Language | Code |
|---|---|
| English | `en` |
| French | `fr` |
| Arabic | `ar` |

Language is detected automatically. No manual configuration is required.

---

## Evaluation Logs

Every run appends a structured log entry to `logs.jsonl`. Each entry records:

- The condition and language
- The extracted entities
- The resolved URIs
- The generated SPARQL query
- Whether the query is syntactically valid
- The raw answer from the Knowledge Graph
- The final natural language answer

This file is used directly for thesis evaluation across the 12 experimental conditions.