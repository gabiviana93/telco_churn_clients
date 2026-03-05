import pytest
import numpy as np
import os
from joblib import dump
from src.inference import predict, predict_proba, load_model


def test_predict_shape(dummy_model, inference_test_data):
    """Testa se predict retorna o número correto de predições"""
    preds = predict(dummy_model, inference_test_data)
    assert len(preds) == len(inference_test_data)


def test_predict_returns_array(dummy_model, inference_test_data):
    """Testa se predict retorna numpy array"""
    preds = predict(dummy_model, inference_test_data)
    assert isinstance(preds, np.ndarray)


def test_predict_consistency(dummy_model):
    """Testa consistência de predições"""
    X = np.array([[1], [2]])
    
    # Fazer predições múltiplas vezes
    pred1 = predict(dummy_model, X)
    pred2 = predict(dummy_model, X)
    
    # Devem ser idênticas
    np.testing.assert_array_equal(pred1, pred2)


def test_predict_with_different_data_types(dummy_model):
    """Testa predict com diferentes tipos de dados"""
    # Array 2D
    X_array = np.array([[1], [2], [3]])
    preds = predict(dummy_model, X_array)
    assert len(preds) == 3
    
    # Apenas certifique-se de que funciona
    assert preds is not None


def test_predict_proba_shape(dummy_model, inference_test_data):
    """Testa se predict_proba retorna o formato correto"""
    probas = predict_proba(dummy_model, inference_test_data)
    assert probas.shape[0] == len(inference_test_data)
    assert probas.shape[1] == 2  # Classificação binária


def test_predict_proba_values(dummy_model, inference_test_data):
    """Testa se as probabilidades somam 1"""
    probas = predict_proba(dummy_model, inference_test_data)
    np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(len(inference_test_data)))


def test_predict_proba_sum_to_one(dummy_model):
    """Testa se probabilidades somam 1 com dados variados"""
    X = np.array([[1], [2], [3]])
    probas = predict_proba(dummy_model, X)
    
    # Cada linha deve somar 1
    for row in probas:
        assert abs(sum(row) - 1.0) < 0.001


def test_predict_proba_consistency(dummy_model):
    """Testa consistência de probabilidades"""
    X = np.array([[1], [2]])
    
    # Fazer predições múltiplas vezes
    proba1 = predict_proba(dummy_model, X)
    proba2 = predict_proba(dummy_model, X)
    
    # Devem ser idênticas
    np.testing.assert_array_almost_equal(proba1, proba2)


def test_predict_proba_no_support():
    """Testa erro quando modelo não suporta predict_proba"""
    class MockModel:
        def predict(self, X):
            return np.array([0, 1])
    
    model = MockModel()
    X = np.array([[1], [2]])
    
    with pytest.raises(AttributeError, match="não suporta predição de probabilidades"):
        predict_proba(model, X)


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
    
    # Mock do MODEL_PATH
    import src.inference as inf_module
    original_path = inf_module.MODEL_PATH
    inf_module.MODEL_PATH = model_path
    
    try:
        loaded_model = load_model()
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
    finally:
        inf_module.MODEL_PATH = original_path
