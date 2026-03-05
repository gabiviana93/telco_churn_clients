import pytest
import numpy as np
import mlflow
from src.evaluate import evaluate
from src.config import MLFLOW_TRACKING_URI

@pytest.fixture(autouse=True)
def setup_mlflow():
    """Configura MLflow para todos os testes"""
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment("test_experiment")
    yield
    # Cleanup após os testes
    if mlflow.active_run():
        mlflow.end_run()

@pytest.fixture
def mock_model_for_eval():
    """Fixture que fornece um modelo mock para testes de avaliação"""
    from sklearn.dummy import DummyClassifier
    
    model = DummyClassifier(strategy="stratified", random_state=42)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 1, 0, 1, 0])
    model.fit(X, y)
    return model

def test_evaluate_returns_dict(mock_model_for_eval):
    """Testa se evaluate retorna um dicionário"""
    X_test = np.array([[1], [2], [3], [4]])
    y_test = np.array([0, 1, 0, 1])
    
    with mlflow.start_run():
        metrics = evaluate(mock_model_for_eval, X_test, y_test)
        assert isinstance(metrics, dict)
        assert 'roc_auc' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'report' in metrics

def test_evaluate_roc_auc_range(mock_model_for_eval):
    """Testa se o ROC-AUC está no intervalo válido [0, 1]"""
    X_test = np.array([[1], [2], [3], [4]])
    y_test = np.array([0, 1, 0, 1])
    
    with mlflow.start_run():
        metrics = evaluate(mock_model_for_eval, X_test, y_test)
        assert 0 <= metrics['roc_auc'] <= 1

def test_evaluate_report_structure(mock_model_for_eval):
    """Testa se o relatório de classificação tem a estrutura esperada"""
    X_test = np.array([[1], [2], [3], [4]])
    y_test = np.array([0, 1, 0, 1])
    
    with mlflow.start_run():
        metrics = evaluate(mock_model_for_eval, X_test, y_test)
        report = metrics['report']
        
        # Verifica se contém as chaves esperadas
        assert '0' in report or 'weighted avg' in report
        assert 'weighted avg' in report
        
        # Verifica métricas na média ponderada
        assert 'precision' in report['weighted avg']
        assert 'recall' in report['weighted avg']
        assert 'f1-score' in report['weighted avg']

def test_evaluate_logs_mlflow_metrics(mock_model_for_eval):
    """Testa se as métricas são logadas no MLflow"""
    X_test = np.array([[1], [2], [3], [4]])
    y_test = np.array([0, 1, 0, 1])
    
    with mlflow.start_run():
        evaluate(mock_model_for_eval, X_test, y_test)
        
        # Verificar se métricas foram logadas
        run = mlflow.active_run()
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run.info.run_id)
        
        assert 'roc_auc' in run_data.data.metrics

def test_evaluate_with_larger_dataset():
    """Testa avaliação com dataset maior"""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X[:80], y[:80])
    
    with mlflow.start_run():
        metrics = evaluate(model, X[80:], y[80:])
        assert metrics['roc_auc'] > 0
        assert 'report' in metrics

