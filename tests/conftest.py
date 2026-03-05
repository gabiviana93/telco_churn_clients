"""
Fixtures de Teste
=================

Fixtures compartilhadas para todos os módulos de teste.
Fornece dados de exemplo, modelos e configurações para testes.
"""

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Fixtures de Dados
# =============================================================================


@pytest.fixture
def sample_dataframe():
    """Fixture que fornece um DataFrame de exemplo para testes."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [30000, 50000, 60000, 75000, 100000],
            "city": ["São Paulo", "Rio de Janeiro", "São Paulo", "Brasília", "Salvador"],
            "target": [0, 1, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_X_y():
    """Fixture que fornece features (X) e target (y) separados."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        }
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), name="target")
    return X, y


@pytest.fixture
def sample_trained_model():
    """Fixture que fornece um modelo XGBoost simples treinado."""
    from sklearn.datasets import make_classification
    from xgboost import XGBClassifier

    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42, eval_metric="logloss")
    model.fit(X, y)
    return model


@pytest.fixture
def simple_test_dataframe():
    """Fixture que fornece um DataFrame simples para testes de preprocessing."""
    return pd.DataFrame(
        {"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40], "target": [0, 1, 0, 1]}
    )


@pytest.fixture
def psi_series_expected():
    """Fixture que fornece uma série para teste de PSI (distribuição esperada)."""
    return pd.Series([1, 2, 3, 4, 5])


@pytest.fixture
def psi_series_actual():
    """Fixture que fornece uma série para teste de PSI (distribuição atual)."""
    return pd.Series([2, 3, 4, 5, 6])


@pytest.fixture
def dummy_model():
    """Fixture que fornece um modelo dummy simples para testes de inference."""
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[1], [2], [3]])
    model.fit(X, [0, 1, 0])
    return model
