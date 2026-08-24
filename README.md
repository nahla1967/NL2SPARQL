# NL2SPARQL — Trilingual Natural Language to SPARQL Pipeline

> **Master's Thesis Project** — Nahla Fersi, ESC Tunis (2025–2026)  
> Supervisors: Mr. Marwen Kachroudi, Mme Wiem Baazouzi

A trilingual (Arabic / French / English) natural-language-to-SPARQL pipeline, evaluated across three knowledge graphs (flights, airports, university/LUBM). The core contribution is a **hybrid 4-tier mapping cascade** that resolves property/entity URIs *before* injecting them into an LLM (LLaMA 3 via Ollama) for SPARQL generation — eliminating URI hallucination and substantially improving accuracy over raw-schema prompting.

---

## Key Results

| Condition | Exact Match | Notes |
|---|---|---|
| **Main pipeline** (full cascade + URI injection) | **95.9%** (284/296 scoreable rows) | All 3 languages, all strategies |
| **Baseline A** — no URI injection | **25.5%** | `single_kg1/2/3` + `cross_kg`, zero-shot only |
| **Baseline B** — no templates | **4.3%** | count/filter/ranking/comparison, zero-shot only |

The hybrid mapping layer is the difference between a system that works and one that doesn't.

---

## Architecture

```
User Question (ar/fr/en)
    ↓
Language Detection → Router (LLM classifier + deterministic overrides)
    ↓
Branch dispatch:
  • single_kg1/2/3    → Entity Extraction → Hybrid Mapping Layer → SPARQL Generation
  • cross_kg          → Entity Extraction → Hybrid Mapping → Cross-KG Resolver
  • template          → Param Extraction → Template Builder
  • open_kg           → Schema Injection → Fallback Generator
  • ask_query / out_of_scope → handled separately
    ↓
SPARQL Validation (syntax + URI checks)
    ↓
Execution (Apache Jena Fuseki)
    ↓
Entity Resolution + Answer Formatting → Natural Language Answer
```

### Hybrid Mapping Layer (`pipeline/mapper.py`)

Tried in order, first match wins:

1. **Pre-normalization** — case, whitespace, diacritic cleanup
2. **Exact lexicon match** — literal lookup against multilingual lexicons
3. **Fuzzy match** — RapidFuzz string similarity (typos, minor rewording)
4. **Semantic match** — MiniLM sentence-embedding similarity (paraphrases with no string overlap)

---

## Project Structure

```
NL2SPARQL/
├── app.py                          # Web/API entry point
├── main.py                         # CLI entry point
├── requirements.txt                # Python dependencies (pinned)
├── README.md                       # This file
│
├── pipeline/                       # Core NL2SPARQL pipeline
│   ├── executor.py                 # SPARQL execution, validation, formatting
│   ├── extractor.py                # Entity & property phrase extraction
│   ├── generator.py                # SPARQL generation (zero/few/CoT prompting)
│   ├── language.py                 # Language detection (langdetect)
│   ├── mapper.py                   # ⭐ Hybrid 4-tier mapping cascade
│   ├── cross_kg_resolver.py        # KG1↔KG2 bridge logic (IATA-based)
│   ├── orchestrator.py             # Pipeline orchestration
│   ├── template_resolver.py        # Hand-verified SPARQL template builder
│   └── kg_registry.py              # KG metadata, endpoints, namespaces
│
├── router/                         # Question routing & classification
│   ├── router.py                   # Main router (LLM classifier + overrides)
│   ├── classifier.py               # LLM-based query-type classification
│   ├── detectors.py                # Regex/heuristic fast-path detectors
│   └── rules.py                    # Deterministic override rules
│
├── evaluation/                     # Benchmark & evaluation scripts
│   ├── eval_runner.py              # Main evaluation harness
│   ├── eval_metrics.py             # Metrics computation + Excel export
│   ├── baseline_summary.py         # Baseline result aggregation
│   ├── build_dataset.py            # Dataset construction utilities
│   └── results/                    # All evaluation outputs
│       ├── eval_results.jsonl
│       ├── baseline_A_results.jsonl
│       ├── baseline_B_results.jsonl
│       ├── baseline_ablation_*.jsonl
│       ├── NL2SPARQL_Evaluation_Dataset.xlsx
│       └── NL2SPARQL_Evaluation_Results.xlsx
│
├── tests/                          # Test suites
│   ├── test_pipeline.py            # Per-branch pipeline tests
│   └── test.py                     # Additional tests
│
├── data/                           # Knowledge graphs & raw data
│   ├── flight_ontology-materialized.ttl      # KG1: Flights (123 flights, 8,262 triples)
│   ├── airport_ontology_kg1_aligned.ttl      # KG2: Airports (58 airports, 2,285 triples)
│   ├── university0_merged.ttl                # KG3: University/LUBM (100,545 triples)
│   ├── kg_sample_data.json
│   └── kg_airports/                # Raw CSV sources (OurAirports)
│       ├── airports.csv
│       ├── countries.csv
│       ├── regions.csv
│       └── runways.csv
│
└── lexicons/                       # Multilingual lexicons & embeddings
    ├── lexicon.json
    ├── lexicon_airports.json
    ├── lexicon_university.json
    ├── *_embeddings.npy             # Pre-computed sentence embeddings
    └── *_phrases.json               # Phrase lists for semantic tier
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with LLaMA 3
- Apache Jena Fuseki (for local SPARQL endpoints)

### Installation

```bash
git clone https://github.com/nahla1967/NL2SPARQL.git
cd NL2SPARQL
pip install -r requirements.txt
```

### Start Fuseki endpoints

Ensure KG1, KG2, and KG3 datasets are loaded and running on their respective endpoints (configured in `pipeline/kg_registry.py`).

---

## Usage

### Run a single question

```python
from pipeline.orchestrator import run_pipeline

result = run_pipeline("What is the departure city of flight OS295?", language="en")
print(result["answer"])
```

### Run the full evaluation

```bash
# Main pipeline (all 81 questions × 3 languages × strategies = 375 runs)
python -m evaluation.eval_runner

# Baseline A — no URI injection
# Edit BASELINE_MODE = "A" in evaluation/eval_runner.py, then rerun
python -m evaluation.eval_runner

# Baseline B — no templates
# Edit BASELINE_MODE = "B" in evaluation/eval_runner.py, then rerun
python -m evaluation.eval_runner

# Ablation study — vary cascade tiers in pipeline/mapper.py
# Set stages parameter in map_property_cascade(), then rerun
python -m evaluation.eval_runner
```

### Generate evaluation summary

```bash
python -m evaluation.eval_metrics
```

Produces `NL2SPARQL_Evaluation_Results.xlsx` with Summary + Raw sheets.

---

## Evaluation Methodology

- **Benchmark:** Custom 81-question dataset (inspired by QALD-9-plus methodology, but domain-specific). Covers:
  - Core branches: `single_kg1/2/3`, `cross_kg`
  - Template types: `count`, `filter_numeric`, `filter_string`, `ranking`, `compare_two_airports`
  - Robustness: `typo_fuzzy`, `property_ambiguity`, `multilingual_edge`, `out_of_scope`
- **Ground truth:** Independently verified via `rdflib` against raw `.ttl` files (not through the pipeline)
- **Metrics:** Exact Match (EM), F1, SPARQL Validity, Failure Type, Duration
- **Prompting strategies:** Zero-shot (primary), Few-shot, Chain-of-Thought — compared on `single_kg1/2/3`

---

## Baselines & Ablation

| Experiment | What it isolates | Population | Main vs. Baseline |
|---|---|---|---|
| **Baseline A** | Mapping cascade contribution | 34 q × 3 langs, `single_kg1/2/3` + `cross_kg` | 97.1% → **25.5%** |
| **Baseline B** | Template contribution | 23 q × 3 langs, template categories | 100% → **4.3%** |
| **Ablation** | Cascade tier necessity | Same as Baseline A | Full → exact+fuzzy → exact only |

Key ablation finding: French shows the steepest cumulative dependency on fuzzy+semantic tiers (−20.6 pts), not Arabic as originally hypothesized.

---

## Key Design Decisions

1. **No fine-tuning.** LLaMA 3 is used exactly as distributed. All accuracy comes from prompt design and the mapping layer, not model weights.
2. **No translation step.** Arabic and French questions are processed natively; the URI is language-neutral by construction.
3. **URI injection, not post-generation fix.** Resolving the correct URI *before* generation prevents silent failures (syntactically valid but semantically wrong queries).
4. **Template + open fallback.** Structured queries use hand-verified templates; genuinely open-ended questions fall back to schema-grounded LLM generation.

---

## Known Limitations

- **Sanitization gap:** `_sanitize_sparql_literal()` exists only in `template_resolver.py`; the `open_kg` fallback path (used by baselines and open production questions) has no sanitization.
- **Router fragility:** The LLM classifier required a growing set of deterministic fast-path overrides during development.
- **Single-LLM dependency:** Evaluated only on LLaMA 3 via Ollama. The architecture is model-agnostic, but this is not yet tested.
- **No combined "no injection + no templates" baseline:** Baselines A and B each remove one component while the other remains available. A true "everything unaided" condition was not measured.

See `NL2SPARQL_Thesis_Summary.md` (not in repo — thesis companion doc) for full evidentiary detail, bug log, and methodology notes.

---

## Thesis Context

.
- **Repo:** `https://github.com/nahla1967/NL2SPARQL`

---

## License

This project was developed as part of a Master's thesis at ESC Tunis. The code and benchmark dataset are provided for academic and research purposes.
