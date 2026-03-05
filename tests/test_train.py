import pytest
import mlflow
import numpy as np
from src.train import train_model, cross_validate_model, save_model
from src.features import build_preprocessor
from src.config import MLFLOW_TRACKING_URI

@pytest.fixture(autouse=True)
def setup_mlflow():
    """Configura MLflow para todos os testes"""
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment("test_train_experiment")
    yield
    # Cleanup após os testes
    if mlflow.active_run():
        mlflow.end_run()

def test_pipeline_creation(preprocessor_features, model_params):
    preprocessor = build_preprocessor(
        numeric_features=preprocessor_features["numeric_features"],
        categorical_features=preprocessor_features["categorical_features"]
    )

    with mlflow.start_run():
        pipeline = train_model(preprocessor, params=model_params, experiment_name="test_train_experiment")

        assert "preprocessing" in pipeline.named_steps
        assert "model" in pipeline.named_steps

def test_train_model_without_data(preprocessor_features, model_params):
    """Testa criação do pipeline sem treinar"""
    preprocessor = build_preprocessor(
        numeric_features=preprocessor_features["numeric_features"],
        categorical_features=preprocessor_features["categorical_features"]
    )
    
    with mlflow.start_run():
        pipeline = train_model(
            preprocessor,
            params=model_params,
            experiment_name="test_train_experiment",
            run_name="test_run"
        )
        
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')

def test_train_model_default_params(preprocessor_features):
    """Testa treinamento com parâmetros padrão (None)"""
    preprocessor = build_preprocessor(
        numeric_features=preprocessor_features["numeric_features"],
        categorical_features=preprocessor_features["categorical_features"]
    )
    
    with mlflow.start_run():
        pipeline = train_model(
            preprocessor,
            params=None,  # Usar parâmetros padrão
            experiment_name="test_train_experiment"
        )
        
        assert pipeline is not None

def test_cross_validate_basic():
    """Testa validação cruzada básica"""
    from sklearn.datasets import make_classification
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(n_estimators=10, max_depth=2, random_state=42))
    ])
    
    with mlflow.start_run():
        cv_scores = cross_validate_model(
            pipeline,
            X,
            y,
            n_splits=3,
            log_mlflow=True
        )
        
        assert 'roc_auc' in cv_scores
        assert 'test_mean' in cv_scores['roc_auc']

def test_cross_validate_without_mlflow():
    """Testa validação cruzada sem logar no MLflow"""
    from sklearn.datasets import make_classification
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(n_estimators=10, max_depth=2, random_state=42))
    ])
    
    cv_scores = cross_validate_model(
        pipeline,
        X,
        y,
        n_splits=3,
        log_mlflow=False  # Não logar no MLflow
    )
    
    assert 'roc_auc' in cv_scores

def test_save_model_basic(tmp_path):
    """Testa salvamento básico de modelo"""
    import os
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(n_estimators=5, random_state=42))
    ])
    pipeline.fit(X, y)
    
    test_model_path = os.path.join(tmp_path, "test_model.pkl")
    
    with mlflow.start_run():
        save_model(pipeline, model_path=test_model_path, X_example=X[:3])
        
        assert os.path.exists(test_model_path)


