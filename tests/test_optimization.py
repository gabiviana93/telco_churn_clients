"""
Teste do Módulo de Otimização
==============================

Testes para otimização de hiperparâmetros com Optuna.
Usa trials mínimos para execução rápida dos testes.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    IMBLEARN_AVAILABLE,
    OPTUNA_AVAILABLE,
    OptimizationConfig,
    OptimizationResult,
)

# =============================================================================
# Testes de OptimizationConfig
# =============================================================================


class TestOptimizationConfig:
    """Testes para dataclass OptimizationConfig."""

    def test_default_values(self):
        """Testa se valores padrão de configuração são carregados do config YAML."""
        from src.config import (
            DEFAULT_N_TRIALS,
            DEFAULT_OPTIMIZATION_METRIC,
            DEFAULT_OPTIMIZATION_TIMEOUT,
        )

        config = OptimizationConfig()

        assert config.n_trials == DEFAULT_N_TRIALS
        assert config.n_cv_splits == 5
        assert config.metric == DEFAULT_OPTIMIZATION_METRIC
        assert config.direction == "maximize"
        assert config.timeout == DEFAULT_OPTIMIZATION_TIMEOUT
        assert config.n_jobs == -1
        assert config.use_smote is True
        assert config.optimize_threshold is True

    def test_custom_values(self):
        """Testa valores customizados de configuração."""
        config = OptimizationConfig(
            n_trials=50,
            n_cv_splits=3,
            metric="roc_auc",
            use_smote=False,
        )

        assert config.n_trials == 50
        assert config.n_cv_splits == 3
        assert config.metric == "roc_auc"
        assert config.use_smote is False


# =============================================================================
# Testes de OptimizationResult
# =============================================================================


class TestOptimizationResult:
    """Testes para dataclass OptimizationResult."""

    def test_basic_creation(self):
        """Testa criação básica de resultado."""
        result = OptimizationResult(
            best_params={"n_estimators": 200, "max_depth": 6},
            best_score=0.85,
            best_pipeline=None,
        )

        assert result.best_params["n_estimators"] == 200
        assert result.best_score == 0.85
        assert result.best_pipeline is None
        assert isinstance(result.timestamp, datetime)
        assert result.all_trials == []

    def test_summary(self):
        """Testa geração de sumário."""
        config = OptimizationConfig(metric="f1")
        result = OptimizationResult(
            best_params={"n_estimators": 200},
            best_score=0.85,
            best_pipeline=None,
            all_trials=[{}, {}],  # 2 trials
            config=config,
        )

        summary = result.summary()

        assert "OPTIMIZATION RESULTS" in summary
        assert "0.85" in summary or "0.8500" in summary
        assert "n_estimators: 200" in summary
        assert "Trials completed: 2" in summary


# =============================================================================
# Testes de HyperparameterOptimizer
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaOptimizer:
    """Testes para classe HyperparameterOptimizer."""

    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo para testes de otimização."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame(
            {
                "numeric_1": np.random.randn(n_samples),
                "numeric_2": np.random.randn(n_samples),
                "cat_1": np.random.choice(["A", "B", "C"], n_samples),
            }
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        return X, y

    def test_initialization(self, sample_data):
        """Testa inicialização do otimizador."""
        from src.optimization import HyperparameterOptimizer

        X, y = sample_data
        optimizer = HyperparameterOptimizer(X, y)

        assert optimizer.X is X
        assert optimizer.y is y
        assert len(optimizer.numeric_features) == 2
        assert len(optimizer.categorical_features) == 1
        assert optimizer._preprocessor is not None

    def test_initialization_with_config(self, sample_data):
        """Testa otimizador com config customizada."""
        from src.optimization import HyperparameterOptimizer

        X, y = sample_data
        config = OptimizationConfig(n_trials=10, n_cv_splits=2)
        optimizer = HyperparameterOptimizer(X, y, config=config)

        assert optimizer.config.n_trials == 10
        assert optimizer.config.n_cv_splits == 2

    def test_auto_feature_detection(self, sample_data):
        """Testa detecção automática de tipos de features."""
        from src.optimization import HyperparameterOptimizer

        X, y = sample_data
        optimizer = HyperparameterOptimizer(X, y)

        assert "numeric_1" in optimizer.numeric_features
        assert "numeric_2" in optimizer.numeric_features
        assert "cat_1" in optimizer.categorical_features

    def test_preprocessor_build(self, sample_data):
        """Testa se preprocessador é construído corretamente."""
        from sklearn.compose import ColumnTransformer

        from src.optimization import HyperparameterOptimizer

        X, y = sample_data
        optimizer = HyperparameterOptimizer(X, y)

        assert isinstance(optimizer._preprocessor, ColumnTransformer)

    def test_quick_optimize(self, sample_data):
        """Testa quick_optimize com trials mínimos."""
        from src.optimization import quick_optimize

        X, y = sample_data
        result = quick_optimize(X, y, n_trials=2, metric="f1")

        assert isinstance(result, OptimizationResult)
        assert result.best_score >= 0
        assert "n_estimators" in result.best_params
        assert result.best_pipeline is not None


# =============================================================================
# Testes Avançados de HyperparameterOptimizer
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestAdvancedOptimizer:
    """Testes para HyperparameterOptimizer com features avançadas."""

    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame(
            {
                "numeric_1": np.random.randn(n_samples),
                "numeric_2": np.random.randn(n_samples),
                "cat_1": np.random.choice(["A", "B", "C"], n_samples),
            }
        )
        # Target desbalanceado similar a churn
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
        return X, y

    def test_initialization(self, sample_data):
        """Testa inicialização do otimizador avançado."""
        from src.optimization import HyperparameterOptimizer

        X, y = sample_data
        optimizer = HyperparameterOptimizer(X, y)

        assert optimizer.X is X
        assert optimizer.y is y
        assert 0 < optimizer.churn_rate < 1

    def test_feature_detection(self, sample_data):
        """Testa detecção de tipos de features."""
        from src.optimization import HyperparameterOptimizer

        X, y = sample_data
        optimizer = HyperparameterOptimizer(X, y)

        assert len(optimizer.numeric_features) == 2
        assert len(optimizer.categorical_features) == 1


# =============================================================================
# Testes de Integração SMOTE
# =============================================================================


@pytest.mark.skipif(not IMBLEARN_AVAILABLE, reason="imbalanced-learn not installed")
class TestSMOTEIntegration:
    """Testes para integração de oversampling SMOTE."""

    def test_smote_available_flag(self):
        """Testa se IMBLEARN_AVAILABLE é True quando instalado."""
        assert IMBLEARN_AVAILABLE is True

    def test_config_smote_default(self):
        """Testa se SMOTE está habilitado por padrão."""
        config = OptimizationConfig()
        assert config.use_smote is True


# =============================================================================
# Casos Extremos
# =============================================================================


class TestEdgeCases:
    """Testes de casos extremos."""

    def test_config_valid_metrics(self):
        """Testa opções válidas de métrica."""
        for metric in ["f1", "roc_auc", "precision", "recall"]:
            config = OptimizationConfig(metric=metric)
            assert config.metric == metric

    def test_result_empty_trials(self):
        """Testa resultado sem trials."""
        result = OptimizationResult(
            best_params={},
            best_score=0.0,
            best_pipeline=None,
        )

        assert len(result.all_trials) == 0
        assert result.study is None
