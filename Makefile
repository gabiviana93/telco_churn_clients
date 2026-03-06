# =============================================================================
# Customer Churn Prediction - Makefile
# =============================================================================
# Common commands for development and deployment

.PHONY: help install dev test lint format api docker clean ci ci-lint ci-tests ci-docker ci-all

# Default target
help:
	@echo "Customer Churn Prediction - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install    - Install production dependencies"
	@echo "  make dev        - Install all dependencies (including dev)"
	@echo ""
	@echo "Development:"
	@echo "  make test       - Run all tests with coverage"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Format code with black and isort"
	@echo "  make check      - Run all checks (lint + test)"
	@echo ""
	@echo "CI/CD Local:"
	@echo "  make ci         - Run full CI pipeline locally (lint+imports+tests+ml+quality)"
	@echo "  make ci-lint    - Run only lint checks"
	@echo "  make ci-tests   - Run only tests with coverage"
	@echo "  make ci-docker  - Build Docker images locally"
	@echo "  make ci-all     - Full CI + Docker build"
	@echo ""
	@echo "API:"
	@echo "  make api        - Run API in development mode"
	@echo "  make api-prod   - Run API in production mode"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker images (api + dashboard)"
	@echo "  make docker-up    - Start API + Dashboard"
	@echo "  make docker-down  - Stop all containers"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow     - Start MLflow UI"
	@echo ""
	@echo "Training:"
	@echo "  make train      - Run training pipeline"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      - Remove cache and temporary files"

# =============================================================================
# Setup
# =============================================================================

install:
	poetry install --only main

dev:
	poetry install
	poetry run pre-commit install

# =============================================================================
# Development
# =============================================================================

test:
	poetry run pytest tests/ -v --cov=src --cov=api --cov-report=term-missing --cov-report=html

test-fast:
	poetry run pytest tests/ -v --tb=short -q

test-api:
	poetry run pytest tests/test_api.py -v

lint:
	poetry run ruff check src/ api/ tests/
	poetry run black --check src/ api/ tests/
	poetry run isort --check-only src/ api/ tests/

format:
	poetry run black src/ api/ tests/
	poetry run isort src/ api/ tests/
	poetry run ruff check --fix src/ api/ tests/

check: lint test

type-check:
	poetry run mypy src/ api/

# =============================================================================
# API
# =============================================================================

api:
	poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

api-prod:
	poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# =============================================================================
# CI/CD Local
# =============================================================================

ci:
	@bash scripts/test_ci_locally.sh ci

ci-lint:
	@bash scripts/test_ci_locally.sh lint

ci-tests:
	@bash scripts/test_ci_locally.sh tests

ci-docker:
	@bash scripts/test_ci_locally.sh docker

ci-all:
	@bash scripts/test_ci_locally.sh all

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build --target production -t churn-api .
	docker build --target dashboard -t churn-dashboard .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# =============================================================================
# MLflow
# =============================================================================

mlflow:
	poetry run mlflow ui --backend-store-uri mlruns/

# =============================================================================
# Dashboard
# =============================================================================

dashboard:
	poetry run streamlit run scripts/dashboard.py

# =============================================================================
# Notebooks
# =============================================================================

notebook:
	poetry run jupyter notebook notebooks/

lab:
	poetry run jupyter lab notebooks/

# =============================================================================
# Training
# =============================================================================

train:
	poetry run python scripts/train_pipeline.py

train-optimized:
	poetry run python scripts/train_pipeline.py --mode ensemble --trials 100

train-quick:
	poetry run python scripts/train_pipeline.py --quick

inference:
	poetry run python scripts/inference_pipeline.py --interactive

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .eggs *.egg-info 2>/dev/null || true

clean-all: clean
	rm -rf mlruns/ 2>/dev/null || true
	rm -rf models/*.joblib 2>/dev/null || true
