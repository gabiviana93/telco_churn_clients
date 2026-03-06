# Copilot Instructions

## Project Overview

This repository implements an end-to-end Machine Learning pipeline for **customer churn prediction in telecommunications**. It covers the full data science lifecycle: EDA → Feature Engineering → Modelling → Optimisation → API Deployment → Monitoring.

- **Dataset**: 7,043 customers, 26.5% churn rate (IBM Telco dataset).
- **Best model**: LightGBM — ROC-AUC 0.833, F1-Score 0.574, AUPRC 0.791.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Dependency manager | Poetry (`pyproject.toml` / `poetry.lock`) |
| ML frameworks | scikit-learn, XGBoost, LightGBM, CatBoost |
| Hyperparameter optimisation | Optuna |
| Class imbalance | imbalanced-learn (SMOTE) |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn + Pydantic v2 |
| Dashboard | Streamlit |
| Containerisation | Docker (multi-stage: builder → production → dashboard → development) |
| Linting | Ruff, Black, isort |
| Type checking | mypy |
| Testing | pytest + pytest-cov + pytest-asyncio |

---

## Project Structure

```
.
├── src/                  # Core ML library
│   ├── config.py         # All constants, paths, and pipeline step names
│   ├── enums.py          # ModelType, DriftSeverity, …
│   ├── preprocessing.py  # Scikit-learn transformers
│   ├── feature_engineering.py
│   ├── pipeline.py       # ChurnPipeline (trains & evaluates)
│   ├── evaluate.py       # evaluate() function
│   ├── inference.py      # Inference helpers
│   ├── interpret.py      # SHAP explainability
│   ├── monitoring.py     # PSI drift detection
│   ├── optimization.py   # HyperparameterOptimizer (Optuna)
│   ├── utils.py          # Shared utilities
│   └── logger.py         # Structured logging (structlog)
├── api/                  # FastAPI service
│   ├── main.py
│   ├── core/             # config, logging
│   ├── routes/           # health, prediction, interpret, drift
│   ├── schemas/          # Pydantic models
│   └── services/         # model_service.py
├── tests/                # pytest test suite
├── notebooks/            # Exploratory Jupyter notebooks
├── scripts/              # Utility scripts
├── config/               # YAML configuration files
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

---

## Development Setup

```bash
# Install dependencies (creates .venv automatically)
poetry install --no-interaction

# Activate the virtual environment
poetry shell

# Run the API locally
poetry run uvicorn api.main:app --reload --port 8000

# Run the Streamlit dashboard
poetry run streamlit run scripts/dashboard.py
```

---

## Testing

```bash
# Run the full test suite with coverage
poetry run pytest tests/ -v --cov=src --cov=api --cov-report=term-missing --cov-fail-under=70

# Run a specific test file
poetry run pytest tests/test_preprocessing.py -v

# Run tests by marker
poetry run pytest -m unit -v
poetry run pytest -m integration -v
```

Test markers: `unit`, `integration`, `slow`, `smoke`, `database`, `api`.  
Minimum coverage gate: **70%** (enforced in CI).

---

## Code Style

- **Line length**: 100 characters (Black + Ruff + isort all use `line-length = 100`).
- **Formatter**: Black (`poetry run black src/ api/ tests/ scripts/`).
- **Linter**: Ruff (`poetry run ruff check src/ api/ tests/ scripts/`).
- **Import order**: isort with `profile = "black"` (`poetry run isort --check-only src/ api/ tests/ scripts/`).
- **Type hints**: required on all public functions; mypy is configured but `ignore_missing_imports = true`.
- **Docstrings**: follow Google-style docstrings where present.

Run all checks at once with:
```bash
poetry run ruff check src/ api/ tests/ scripts/ && \
poetry run black --check src/ api/ tests/ scripts/ && \
poetry run isort --check-only src/ api/ tests/ scripts/
```

---

## Key Conventions

### Pipeline step names
Canonical sklearn pipeline step names are defined in `src/config.py` and **must not be changed**:
```python
STEP_PREPROCESSING = "preprocessing"
STEP_MODEL         = "model"
```
These are referenced in `src/utils.py`, `src/interpret.py`, `api/services/model_service.py`, and tests.

### Configuration
All constants (paths, thresholds, model parameters) live in `src/config.py`.  
Do not hard-code paths or magic numbers elsewhere — import from `src/config.py`.

### Enums
Use `src/enums.py` for typed enumerations (`ModelType`, `DriftSeverity`, etc.).  
Add new enums there, never as bare strings scattered through the code.

### Logging
Use `structlog` via the helpers in `src/logger.py`. Do not use `print()` in production code.

### Drift detection
`src/monitoring.py` exposes `population_stability_index()`.  
PSI < 0.10 → no drift; 0.10–0.25 → moderate drift; > 0.25 → significant drift.

---

## Docker

```bash
# Build production image
docker build --target production -t churn-api .

# Build dashboard image
docker build --target dashboard -t churn-dashboard .

# Run full stack (API + dashboard + MLflow)
docker compose up --build
```

---

## CI/CD (GitHub Actions)

Workflows live in `.github/workflows/`:

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | push/PR to main, develop, feature/** | lint → imports → tests → ml-validation → docker |
| `code-quality.yml` | push/PR | code quality checks |
| `pr-analysis.yml` | pull_request | PR analysis |

The CI pipeline enforces lint, import sanity, test coverage ≥ 70%, ML pipeline validation, and Docker build (on main/develop only).
