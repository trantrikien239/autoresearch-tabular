# AutoResearch Tabular

Inspired by Andrej Karpathy's AutoResearch, I built this, an autonomous ML research agent that continuously experiments with features, hyperparameters, and data curation to maximize predictive performance on tabular binary classification tasks.

You provide the data. The agent runs experiments, logs results, learns from failures, and iterates — indefinitely — until you stop it.

The default model is **LightGBM**, chosen for its fast training time (seconds per run) which maximizes experiment throughput — more iterations means more discoveries. You can swap in any scikit-learn-compatible model by editing `model.py` (and adding relevant analysis primitives to `analysis.py` if needed).

## How it works

The agent operates in an infinite loop, alternating between two modes:

**Experiment mode** — The agent forms a hypothesis (e.g., "adding ratio features will capture trends"), edits `features.py` or `model.py`, runs the experiment, evaluates results, and keeps or reverts the change. Every experiment is logged.

**Analysis mode** — The agent writes custom analysis code to understand the data (univariate feature importance, correlations, null rates, error patterns), then uses those insights to inform the next experiment.

```
┌─────────────────────────────────────────────┐
│                 Agent Loop                  │
│                                             │
│  ┌──────────┐      ┌───────────────────┐    │
│  │ Analysis │─────>│ Form hypothesis   │    │
│  └──────────┘      └───────┬───────────┘    │
│       ^                    │                │
│       │              ┌─────v─────┐          │
│       │              │   Edit    │          │
│       │              │ features/ │          │
│       │              │  model    │          │
│       │              └─────┬─────┘          │
│       │                    │                │
│       │              ┌─────v─────┐          │
│       │              │    Run    │          │
│       │              │experiment │          │
│       │              └─────┬─────┘          │
│       │                    │                │
│       │              ┌─────v─────┐          │
│       └──────────────│  Evaluate │          │
│                      │ keep/     │          │
│                      │ revert    │          │
│                      └───────────┘          │
└─────────────────────────────────────────────┘
```

### Data assumptions

This system is designed for **business-oriented tabular datasets where labels are collected over time** — churn prediction, conversion modeling, conversion modeling, etc. These datasets share key properties:

- **Temporal structure**: rows arrive in chronological periods (months, weeks)
- **Delayed labels**: you only know the outcome after some observation window (e.g., "did the customer churned within 3 months?"), so recent periods may not yet have labels
- **Distribution drift**: the relationship between features and target can shift over time

### Evaluation: expanding time windows

The agent evaluates every experiment using **expanding time windows**, which simulate how the model would actually be deployed:

- For each cut period, train on **all historical data up to that period**, test on the **next 1-6 months**
- Per-window metric: configurable via `compute_metric()` in `prepare.py` (default: **ROC AUC**)
- Primary metric: **mean_score** across all windows (higher is better)
- Stability metric: **std_score** across windows (lower is better)

Why expanding windows instead of a single train/test split or k-fold?

- **Realistic**: mirrors production — you always train on the past and predict the future
- **Robust**: averaging scores across many windows prevents overfitting to one lucky split
- **Drift-aware**: std_score reveals whether performance is stable or degrading over time
- **No leakage**: strict temporal ordering means no future data leaks into training

### What the agent edits

| File | Purpose |
|------|---------|
| `features.py` | Feature engineering (ratios, interactions, transforms) and training data curation |
| `model.py` | Model choice and hyperparameters (default: LightGBM) |
| `analysis.py` | Custom analysis code using built-in primitives |
| `LOG.md` | Experiment log — every run gets an entry |
| `LEARNING.md` | Significant insights that change future approach |

### What stays fixed

| File | Purpose |
|------|---------|
| `prepare.py` | Data loading and time-window generation |
| `run_experiment.py` | Experiment orchestrator |
| `run_analysis.py` | Analysis orchestrator |
| `program.md` | Full agent instructions |

## Setup

### 1. Prepare your data

Place your parquet file(s) in `data/original/`. Any layout works — single file, multiple files, or hive-partitioned directories. The default `load_data()` reads all `.parquet` files recursively:

```
data/original/
  data.parquet              # single file works
  # or multiple files:
  part_001.parquet
  part_002.parquet
  # or hive-partitioned:
  year=2024/data.parquet
  year=2025/data.parquet
```

Your data must have:
- A **binary target column** (0/1) — rows where the target is null are automatically excluded
- A **time period column** in `YYYY-MM` format — used for temporal train/test splitting
- An **ID column** — row identifier (not used in modeling, just excluded from features)
- One or more **feature columns** (numeric) — everything not listed in `NON_FEATURE_COLS` is treated as a feature

If your time column uses a different format, update `load_data()` in `prepare.py` to parse it into `YYYY-MM`.

Also create a pickle file listing your feature column names:

```python
import pickle
feature_names = ["feat_1", "feat_2", ...]  # your feature column names
with open("data/original/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
```

### 2. Configure prepare.py

Edit the constants at the top of `prepare.py` to match your dataset:

```python
# ─── Column names ──────────────────────────────────────────────
TARGET_COL = "target"            # your binary target column
ID_COL = "id"                    # your row identifier
TIME_COL = "period"              # your time period column (YYYY-MM)

# ─── Time boundaries ──────────────────────────────────────────
MIN_TRAIN_PERIOD = "2024-01"     # earliest period to include in training
FIRST_CUT_PERIOD = "2025-08"    # first expanding window cut point
LAST_LABELED_PERIOD = "2026-02"  # last period with non-null labels

# ─── Agent constraints ─────────────────────────────────────────
MAX_FEATURES = 300               # max features per experiment
N_ESTIMATORS = 1000               # fixed boosting rounds

# ─── Non-feature columns ──────────────────────────────────────
NON_FEATURE_COLS = {
    "id", "period", "target",    # add any other non-feature columns
}
```

The time boundaries control how expanding windows are generated:
- `FIRST_CUT_PERIOD` to `LAST_LABELED_PERIOD` defines the evaluation range
- More months between them = more windows = more robust evaluation
- `LAST_LABELED_PERIOD` should be the last month where your target labels are available

Also update `load_data()` if your data needs special handling (e.g., CSV format, additional filtering, time column parsing).

### 3. Install dependencies

Using Docker:

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# Build and start the sandbox
docker compose build && docker compose up -d

# Enter the sandbox
docker compose exec sandbox bash
```

### 4. Run the agent

Inside the Docker sandbox:

```bash
claude --dangerously-skip-permissions
```

The agent reads `CLAUDE.md`, which points to `program.md` for full instructions. It will:

1. Create a research branch
2. Run a baseline experiment
3. Begin the autonomous experiment/analysis loop
4. Log everything to `LOG.md`, `LEARNING.md`, and `findings.md`

### 5. Monitor progress

While the agent runs, you can check:

```bash
# Latest experiment results
cat results.tsv

# Experiment log
cat LOG.md

# Significant learnings
cat LEARNING.md

# Analysis findings
cat findings.md

# Current run output
tail -f run.log
```

## Project structure

```
.
├── CLAUDE.md              # Agent quick reference (entry point)
├── program.md             # Full agent instructions
├── prepare.py             # Data loading + expanding windows (configure this)
├── run_experiment.py      # Experiment orchestrator (fixed)
├── run_analysis.py        # Analysis orchestrator (fixed)
├── features.py            # Feature engineering (agent edits)
├── model.py               # Model config (agent edits, default: LightGBM)
├── analysis.py            # Analysis code + primitives (agent edits)
├── selected_features.py   # Auto-generated feature selection list
├── LOG.md                 # Experiment log (agent writes)
├── LEARNING.md            # Insights journal (agent writes)
├── findings.md            # Analysis output (agent writes)
├── results.tsv            # Experiment results table
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker sandbox image
└── docker-compose.yml     # Docker sandbox config
```

## Built-in analysis primitives

The agent has access to these in `analysis.py`:

| Primitive | Description |
|-----------|-------------|
| `univariate_auc(df, features)` | ROC AUC of each feature vs target |
| `correlation_pairs(df, features, threshold)` | Highly correlated feature pairs |
| `null_rates(df, features)` | Null percentage per feature |
| `feature_importance_from_model()` | LightGBM gain/split importance |
| `error_analysis(threshold)` | FP/FN breakdown per window |
| `error_feature_patterns(top_n)` | Feature distribution diffs: FN vs TP |
| `error_by_segment(segment_col)` | Score by segment (e.g., time period) |

## Constraints

- **MAX_FEATURES** and **N_ESTIMATORS** are configured in `prepare.py` — the agent cannot change them, but you can
- The agent cannot install new packages or edit infrastructure files (`prepare.py`, `run_experiment.py`, `run_analysis.py`)
- All feature engineering must be stateless per-row transforms (applied independently to train and test)
- The agent can swap the model in `model.py` to any scikit-learn-compatible classifier (must implement `fit()` + `predict_proba()`)

## Requirements

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) with an Anthropic API key
- Your tabular dataset in parquet format

## License

MIT
