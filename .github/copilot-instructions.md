# Copilot Instructions

## Project Overview

This is an end-to-end Machine Learning pipeline for **telecom customer churn prediction**. It covers the full data science lifecycle: EDA → Feature Engineering → Modeling → Optimization → Deploy → Monitoring.

- **Language**: Python 3.12+
- **Dependency manager**: Poetry (`pyproject.toml`)
- **Main frameworks**: FastAPI, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, MLflow, SHAP, Streamlit
- **Testing**: pytest with 70%+ coverage requirement
- **Code quality**: Ruff, Black (line length 100), isort, mypy

## Architecture

```
src/           # Core ML modules (config, pipeline, preprocessing, feature_engineering,
               #   evaluate, inference, interpret, monitoring, utils, logger)
api/           # FastAPI REST service (routes, schemas, services)
scripts/       # CLI scripts (train, inference, dashboard, monitoring)
notebooks/     # Jupyter notebooks for EDA and experimentation
tests/         # pytest test suite (190+ tests)
config/        # YAML configuration files
```

## Key Conventions

### Configuration
- All configurations are centralised in `src/config.py` — always import constants from there (e.g. `MODELS_DIR`, `TARGET`, `STEP_PREPROCESSING`, `STEP_MODEL`).
- The sklearn Pipeline uses two canonical step names: `STEP_PREPROCESSING = "preprocessing"` and `STEP_MODEL = "model"` (defined in `src/config.py`).
- Runtime configuration is loaded via `get_config()` from `config/project.yaml`.

### Code Style
- Line length: **100 characters** (enforced by Black and Ruff).
- Use **type hints** and **docstrings** on all public functions and classes.
- Follow Google-style docstrings with `Args:`, `Returns:`, and `Raises:` sections.
- Prefer `from __future__ import annotations` at the top of every module.
- Use `structlog` (via `src/logger.py`) for all structured logging — never use `print()` in production code.

### ML / Data Science
- All feature engineering is done through `FeatureEngineer` and `AdvancedFeatureEngineer` in `src/feature_engineering.py`.
- The full training pipeline lives in `src/pipeline.py` (`ChurnPipeline`).
- Hyperparameter tuning uses Optuna inside `src/optimization.py` (`HyperparameterOptimizer`).
- Model evaluation metrics (F1, ROC-AUC, AUPRC, Accuracy, Precision, Recall) are computed in `src/evaluate.py`.
- SHAP interpretability (model-agnostic) is in `src/interpret.py`.
- Data drift detection via PSI is in `src/monitoring.py`.
- Predictions in production go through `src/inference.py` and `api/services/model_service.py`.

### API
- All Pydantic schemas are in `api/schemas/`.
- Route handlers are thin — business logic stays in `api/services/`.
- The API exposes: `/predict/`, `/predict/batch`, `/health`, `/model/info`, `/interpret/...`, `/drift/...`.

### Testing
- Tests live in `tests/`; use the markers `unit`, `integration`, `slow`, `smoke`, `api` defined in `pyproject.toml`.
- Run tests with `poetry run pytest` or `make test`.
- Minimum coverage threshold is **70%**.
- Use `pytest-mock` (`mocker` fixture) for mocking; avoid patching internals unnecessarily.
- Fixtures shared across multiple test files belong in `tests/conftest.py`.

## Common Commands

```bash
# Install dependencies
poetry install

# Run tests with coverage
make test
# or
poetry run pytest

# Lint & format
make lint        # ruff check
make format      # black + isort

# Type check
make type-check  # mypy

# Train model (quick mode)
poetry run python scripts/train_pipeline.py --quick

# Start API
make api         # uvicorn api.main:app --reload

# Start Streamlit dashboard
make dashboard   # streamlit run scripts/dashboard.py
```

## Docker

The `Dockerfile` uses a **4-stage multi-stage build**: `builder → production → dashboard → development`. The `dashboard` stage inherits from `production`. Use `docker compose up api` to start the API service.

## What to Avoid

- Do **not** add `print()` statements in `src/` or `api/` — use the logger.
- Do **not** hardcode file paths — use the constants from `src/config.py` (e.g. `MODELS_DIR`).
- Do **not** modify `poetry.lock` manually — use `poetry add` / `poetry remove`.
- Do **not** commit model artifacts (`.joblib`) or `mlruns/` — they are git-ignored.
