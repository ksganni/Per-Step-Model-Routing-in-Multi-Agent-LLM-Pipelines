# Per-Step Model Routing in Multi-Agent LLM Pipelines

**Can you swap in a cheaper model at individual steps of an LLM pipeline without hurting the final answer — and how far does a bad step's error spread?**

This project measures accuracy and cost across three model tiers at every step of two multi-agent pipelines (text-to-SQL and multi-hop QA), producing a per-step routing map that tells you where to save and where to spend.

## Model Tiers

| Tier | Model | Input cost |
|------|-------|-----------|
| Strong | Gemini 3.5 Flash | $1.50 / M tokens |
| Cheap | Gemini 3.5 Flash Lite | $0.30 / M tokens |
| Open-weight | Qwen3-8B-AWQ (4-bit, self-hosted) | ~$0 (free T4 on Kaggle) |

## Pipelines

### SQL (adapted from MAC-SQL)

Converts natural-language questions into SQLite queries against the [BIRD Mini-Dev](https://bird-bench.github.io/) benchmark (500 questions, 11 databases).

```
Question + Schema + Evidence
       │
   Selector  →  keeps only the relevant tables/columns
       │
  Decomposer →  breaks the question into sub-questions with SQL
       │
   Refiner   →  fixes errors using execution feedback
       │
   Final SQL
```

### QA (multi-hop question answering)

Answers multi-hop factual questions from the [HotpotQA](https://hotpotqa.github.io/) dataset by chaining retrieval and reasoning steps.

```
Question + Supporting Documents
       │
  Decomposer →  splits into atomic sub-questions
       │
  Hop Reader  →  retrieves and extracts answers per hop
       │
   Composer   →  combines hop answers into a final answer
       │
   Formatter  →  structures the response
       │
   Final Answer
```

## Repository Structure

```
├── run_pipeline.py              # Runs the full pipeline end-to-end via call()
├── harness/
│   ├── client.py                # LiteLLM wrapper — call(model, messages) with cost tracking
│   ├── cache.py                 # Disk cache for reproducible reruns
│   ├── pricing.py               # Per-million-token pricing and compute_cost()
│   ├── verifiers.py             # SQL and QA answer verification
│   ├── try_one.py               # Quick smoke test
│   ├── test_harness.py          # 11 offline unit tests
│   ├── test_run_pipeline.py     # 6 offline tests for run_pipeline
│   └── qwen_smoke_test.ipynb    # Kaggle notebook to run Qwen3-8B on a free T4
│
├── pipelines/
│   ├── sql/
│   │   ├── prompts/             # Few-shot prompt templates (selector, decomposer, refiner)
│   │   ├── schemas/             # JSON schemas for every step's input and output
│   │   └── steps.py             # Step definitions and parsing logic
│   ├── qa/
│   │   ├── prompts/             # Prompt templates (decomposer, hop_reader, composer, formatter)
│   │   ├── schemas/             # JSON schemas for every step's input and output
│   │   └── steps.py             # Step definitions and parsing logic
│   ├── common/text.py           # Shared text utilities
│   ├── manual_run.py            # CLI for paste-mode hand testing
│   ├── sample_tasks.py          # Stratified sampler (seed 42)
│   ├── download_data.py         # Download BIRD and HotpotQA datasets
│   └── tests/
│       └── test_pipelines.py    # 13 offline pytest cases
│
├── testkit/
│   ├── TASKS.md                 # 15 curated BIRD Mini-Dev tasks
│   ├── HOW_TO_TEST.md           # Step-by-step guide for paste-mode testing
│   ├── mini_dev_prompt.jsonl    # Full BIRD Mini-Dev prompts
│   └── pick_tasks.py            # Script that curated the 15 tasks
│
├── data/
│   ├── bird_sample_150.json     # 150 BIRD questions (stratified by difficulty)
│   └── hotpot_sample_150.json   # 150 HotpotQA questions (stratified by type)
│
├── experiments/                 # Experiment run outputs (generated)
├── reports/
│   └── data_summary.md          # Dataset statistics
├── .github/workflows/ci.yml    # GitHub Actions — runs all tests on every PR
├── .env.example                 # API key template
└── requirements.txt             # Python dependencies
```

## Quickstart

```bash
# Clone and install
git clone https://github.com/ksganni/Per-Step-Model-Routing-in-Multi-Agent-LLM-Pipelines.git
cd Per-Step-Model-Routing-in-Multi-Agent-LLM-Pipelines
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run offline tests (no API key needed)
python -m pytest pipelines/tests/ -v
python -m unittest discover -s harness -t . -p "test_*.py" -v

# Run a full SQL pipeline on one BIRD question
python run_pipeline.py --qid 1471 --model flash
```

## Running the Pipeline

`run_pipeline.py` wires `harness.call()` into each pipeline step so one function runs the full chain:

```python
from run_pipeline import run_pipeline, run_sql_trace

# Same model for all steps
sql = run_pipeline("sql", 1471, "flash")

# Different model per step
sql = run_pipeline("sql", 1471, {
    "sql.selector": "flash-lite",
    "sql.decomposer": "flash",
    "sql.refiner": "flash"
})

# Full trace with per-step token counts
trace = run_sql_trace(1471, "flash")
```

The refiner step executes the SQL against the BIRD database and retries on errors. It only runs when the databases are present under `data/bird/` (download with `python -m pipelines.download_data`).

## Running Qwen3-8B

The Qwen smoke-test notebook (`harness/qwen_smoke_test.ipynb`) runs on Kaggle with a free T4 GPU. It loads the 4-bit AWQ quantization via vLLM and exposes an OpenAI-compatible endpoint. Set `QWEN_API_BASE` in `.env` to point the harness at it.

## Evaluation Plan

Each pipeline step is tested with every model tier. For each combination we measure:

- **Accuracy** — execution accuracy for SQL (does the query return the gold result?), F1/EM for QA
- **Cost** — total input + output token cost per question
- **Error propagation** — when a step uses a weaker model, how often does its error survive to the final answer vs. get corrected downstream?

The experiment matrix is 3 models × 3 SQL steps × 150 sampled questions, plus 3 models × 4 QA steps × 150 sampled questions.

## Team

| Member | Role |
|--------|------|
| Gautam Govindarasan | Lead / Pipeline Developer |
| Krishna Sathvika Ganni | Harness / Infrastructure |
| Ashrith Reddy Katuru | Data / Evaluation |

## Course

YWCC 691 — Graduate Capstone, NJIT (Fall 2026)