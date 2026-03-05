import os

import numpy as np
import pandas as pd
import pytest
from joblib import dump

from src.inference import (
    FeatureValidationError,
    load_model_package,
    predict_with_package,
    validate_input_features,
)


def test_validate_input_features_exact_match():
    """Testa validação com features exatas"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    expected = ["a", "b", "c"]

    is_valid, missing, extra = validate_input_features(df, expected, strict=True)

    assert is_valid
    assert len(missing) == 0
    assert len(extra) == 0


def test_validate_input_features_missing():
    """Testa validação com features faltando"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    expected = ["a", "b", "c", "d"]

    with pytest.raises(FeatureValidationError):
        validate_input_features(df, expected, strict=True)


def test_validate_input_features_extra_non_strict():
    """Testa validação com features extras em modo não-strict"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
    expected = ["a", "b", "c"]

    is_valid, missing, extra = validate_input_features(df, expected, strict=False)

    assert is_valid
    assert len(missing) == 0
    assert len(extra) == 1
    assert "d" in extra


def test_load_model_package_new_format(tmp_path):
    """Testa carregar modelo no novo formato"""
    from sklearn.dummy import DummyClassifier

    # Criar modelo mock no novo formato
    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    model.fit(X, y)

    package = {
        "model": model,
        "scaler": None,
        "model_name": "test_model",
        "features": {"input_features": ["feature1", "feature2"], "n_input_features": 2},
        "hyperparameters": {"strategy": "most_frequent"},
        "preprocessing_config": {"scaler_type": "none"},
        "metadata": {"training_date": "2024-01-01"},
    }

    model_path = tmp_path / "test_model.joblib"
    dump(package, model_path)

    # Carregar
    loaded = load_model_package(model_path)

    assert "model" in loaded
    assert "features" in loaded
    assert loaded["model_name"] == "test_model"
    assert loaded["features"]["n_input_features"] == 2


def test_load_model_package_legacy_format(tmp_path):
    """Testa carregar modelo em formato legado"""
    from sklearn.dummy import DummyClassifier

    # Criar modelo simples
    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    model.fit(X, y)

    model_path = tmp_path / "legacy_model.joblib"
    dump(model, model_path)

    # Carregar
    loaded = load_model_package(model_path)

    assert "model" in loaded
    assert loaded.get("is_legacy", False)


def test_predict_with_package_basic(tmp_path):
    """Testa predição com pacote completo"""
    from sklearn.dummy import DummyClassifier

    # Criar modelo
    model = DummyClassifier(strategy="most_frequent")
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    model.fit(X_train, y_train)

    # Criar pacote
    package = {
        "model": model,
        "scaler": None,
        "features": {"input_features": ["feat1", "feat2"], "n_input_features": 2},
    }

    # Dados de teste
    X_test = pd.DataFrame({"feat1": [1, 3], "feat2": [2, 4]})

    # Predição
    predictions = predict_with_package(package, X_test, validate_features=True)

    assert len(predictions) == 2
    assert isinstance(predictions, np.ndarray)


def test_predict_with_package_with_proba(tmp_path):
    """Testa predição com probabilidades"""
    from sklearn.dummy import DummyClassifier

    # Criar modelo
    model = DummyClassifier(strategy="stratified")
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    model.fit(X_train, y_train)

    # Criar pacote
    package = {
        "model": model,
        "scaler": None,
        "features": {"input_features": ["feat1", "feat2"], "n_input_features": 2},
    }

    # Dados de teste
    X_test = pd.DataFrame({"feat1": [1, 3], "feat2": [2, 4]})

    # Predição com probabilidades
    predictions, probas = predict_with_package(
        package, X_test, validate_features=True, return_proba=True
    )

    assert len(predictions) == 2
    assert probas is not None
    assert probas.shape == (2, 2)


def test_predict_with_package_feature_validation_error(tmp_path):
    """Testa erro quando features estão faltando"""
    from sklearn.dummy import DummyClassifier

    # Criar modelo
    model = DummyClassifier(strategy="most_frequent")
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model.fit(X_train, y_train)

    # Criar pacote
    package = {
        "model": model,
        "scaler": None,
        "features": {
            "input_features": ["feat1", "feat2", "feat3"],  # Espera 3 features
            "n_input_features": 3,
        },
    }

    # Dados de teste com apenas 2 features
    X_test = pd.DataFrame({"feat1": [1], "feat2": [2]})

    # Deve levantar erro
    with pytest.raises(FeatureValidationError):
        predict_with_package(package, X_test, validate_features=True)


def test_load_model_from_file(tmp_path):
    """Testa carregar modelo de arquivo"""
    from sklearn.dummy import DummyClassifier

    # Criar e salvar modelo temporário
    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    model.fit(X, y)

    model_path = os.path.join(tmp_path, "test_model.pkl")
    dump(model, model_path)

    # Carregar usando load_model_package
    loaded = load_model_package(model_path)
    assert loaded is not None
    assert "model" in loaded
    assert hasattr(loaded["model"], "predict")
