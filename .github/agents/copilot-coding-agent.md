# Copilot Coding Agent Instructions

## Project Overview

This is a **Customer Churn Prediction** end-to-end ML pipeline for a telecom company. The project covers data engineering, feature engineering, model training, experiment tracking (MLflow), a REST API (FastAPI), a Streamlit dashboard, and drift monitoring.

**Stack:** Python 3.12, FastAPI, Streamlit, LightGBM/XGBoost/CatBoost, Optuna, MLflow, Docker, pytest, Poetry.

---

## Repository Structure

```
├── api/                    # FastAPI application
│   ├── core/               # Settings (config.py) and structured logging (logging.py)
│   ├── routes/             # Routers: prediction.py, drift.py, health.py, interpret.py
│   ├── schemas/            # Pydantic request/response schemas
│   ├── services/           # model_service.py – singleton that loads and serves the model
│   └── main.py             # App factory, lifespan, CORS, router registration
├── src/                    # Core ML library (importable as `src`)
│   ├── config.py           # All project-wide constants and YAML-backed config
│   ├── features.py         # Feature engineering transformers
│   ├── inference.py        # load_model_package() and predict helpers
│   ├── monitoring.py       # Data drift detection (PSI + KS)
│   ├── utils.py            # Shared utilities; sklearn Pipeline step names defined here
│   └── ...
├── scripts/
│   ├── dashboard.py        # Streamlit dashboard – calls API via HTTP (requests library)
│   ├── train_pipeline.py   # End-to-end training script
│   └── run_pipeline.py     # Full pipeline runner
├── tests/                  # pytest test suite (~190 tests)
├── config/
│   ├── default.yaml        # Default project configuration
│   └── project.yaml        # Project-specific overrides
├── Dockerfile              # Multi-stage: builder → production → dashboard → development
├── docker-compose.yml      # Services: api, dashboard (default); api-dev, mlflow (profiles)
├── pyproject.toml          # Poetry project + tool configuration (ruff, black, isort, mypy)
└── requirements.txt        # Must be kept in sync with pyproject.toml
```

---

## Environment & Setup

```bash
# Install all dependencies (creates .venv automatically)
poetry install

# Install production-only dependencies
poetry install --only main

# Activate the virtual environment
poetry shell
```

---

## How to Build, Lint, and Test

### Linting and Formatting

```bash
# Run all linters (ruff, black, isort) – used in CI
make lint

# Auto-fix formatting
make format

# Individual tools
poetry run ruff check src/ api/ tests/ scripts/
poetry run black --check src/ api/ tests/ scripts/
poetry run isort --check-only src/ api/ tests/ scripts/
```

### Running Tests

```bash
# Full test suite with coverage
make test
# Equivalent to:
poetry run pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Fast run (no coverage)
make test-fast

# API tests only
make test-api

# Run a single test file
poetry run pytest tests/test_api.py -v

# Run tests matching a keyword
poetry run pytest -k "test_predict" -v
```

Test markers available: `unit`, `integration`, `slow`, `smoke`, `api`.

### Type Checking

```bash
make type-check
# Equivalent to: poetry run mypy src/ api/
```

---

## Running the Application

### API (FastAPI)

```bash
# Development mode (auto-reload)
make api
# Equivalent to: poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
make api-prod
```

API docs available at `http://localhost:8000/docs`.

### Streamlit Dashboard

```bash
make dashboard
# Equivalent to: poetry run streamlit run scripts/dashboard.py
```

Dashboard available at `http://localhost:8501`. It connects to the API at `http://{API_HOST}:{API_PORT}` (defaults: `localhost:8000`).

### Docker

```bash
# Build images
make docker-build

# Start API + Dashboard
make docker-up

# Stop everything
make docker-down

# View logs
make docker-logs
```

Docker services defined in `docker-compose.yml`:
- `api` – FastAPI on port 8000 (default profile)
- `dashboard` – Streamlit on port 8501 (default profile, depends on `api`)
- `api-dev` – Development API with reload (profile: `dev`)
- `mlflow` – MLflow UI on port 5000 (profile: `mlflow`)

### MLflow

```bash
make mlflow
# Equivalent to: poetry run mlflow ui --backend-store-uri mlruns/
```

---

## Training

```bash
# Standard training
make train

# Optimized training (ensemble, 100 Optuna trials)
make train-optimized

# Quick training
make train-quick
```

---

## Key Configuration

All project constants live in `src/config.py`, loaded from `config/default.yaml` and `config/project.yaml` (deep-merged). Environment variables take highest priority.

| Constant | Source | Default |
|---|---|---|
| `API_HOST` | `env API_HOST` / YAML | `0.0.0.0` |
| `API_PORT` | `env API_PORT` / YAML | `8000` |
| `MODEL_PATH` | YAML `model.default_name` | `models/model.joblib` |
| `STEP_PREPROCESSING` | hardcoded | `"preprocessing"` |
| `STEP_MODEL` | hardcoded | `"model"` |
| `RISK_THRESHOLD_LOW` | YAML | `0.4` |
| `RISK_THRESHOLD_HIGH` | YAML | `0.7` |

The dashboard imports these constants from `src.config` and uses `API_HOST`/`API_PORT` to construct the API URL.

---

## Code Conventions

- **Line length:** 100 characters (configured in `pyproject.toml` for black, ruff, isort).
- **Formatter:** `black` with `isort` (profile `"black"`).
- **Linter:** `ruff` (rules: E, W, F, I, B, C4, UP).
- **Imports:** stdlib → third-party → `src` / `api` (first-party). Use `from __future__ import annotations` for forward references.
- **Logging:** Use `structlog` via `from api.core.logging import get_logger`. Never use `print()` in production code.
- **Pydantic:** All API schemas use Pydantic v2 (`pydantic >= 2.9`).
- **sklearn Pipeline step names:** Always use `STEP_PREPROCESSING = "preprocessing"` and `STEP_MODEL = "model"` from `src/config.py`.
- **Tests:** Follow `test_<module>.py` naming, use `@pytest.mark.<marker>` decorators, and use the fixtures defined in `tests/conftest.py`.
- **Dependency files:** `requirements.txt` and `pyproject.toml` must be kept in sync (see header comment in `requirements.txt`).

---

## CI / GitHub Actions

Three workflows in `.github/workflows/`:

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | push/PR to main, develop, feature/** | lint, imports, tests, ML validation, code quality |
| `code-quality.yml` | push/PR | type checking, security scan |
| `pr-analysis.yml` | PR | PR summary / analysis |

The CI lint job runs: `ruff check`, `black --check`, `isort --check-only` on `src/`, `api/`, `tests/`, `scripts/`.

---

## Common Pitfalls

- **Model not loaded:** The API can start in "demo mode" if no trained model file exists at `MODEL_PATH`. Train a model with `make train` first.
- **Dashboard can't reach API:** Ensure the API is running. In Docker, `API_HOST` is set to `api` (the Docker service name). Locally, it defaults to `localhost`.
- **Config cache:** `get_config()` uses `@lru_cache`. Call `get_config.cache_clear()` in tests that need fresh config.
- **requirements.txt drift:** Always update both `pyproject.toml` and `requirements.txt` when adding/removing dependencies.
