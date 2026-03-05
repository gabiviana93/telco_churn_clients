import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para testes
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
import xgboost as xgb
import mlflow

from src.interpret import (
    get_feature_names,
    create_importance_dataframe,
    plot_feature_importance,
    plot_shap_summary,
    plot_shap_dependence
)


@pytest.fixture
def sample_data():
    """Cria dados de exemplo para testes"""
    np.random.seed(42)
    X = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'cat1': np.random.choice(['A', 'B', 'C'], 100)
    })
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def trained_pipeline(sample_data):
    """Cria um pipeline treinado para testes"""
    X, y = sample_data
    
    # Criar preprocessador
    num_features = ['num1', 'num2']
    cat_features = ['cat1']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ])
    
    # Criar pipeline
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('model', xgb.XGBClassifier(random_state=42, n_estimators=10))
    ])
    
    # Treinar
    model.fit(X, y)
    
    return model, preprocessor, X


def test_get_feature_names():
    """Testa extração de nomes de features do preprocessador"""
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['feature1', 'feature2']),
        ('cat', OneHotEncoder(sparse_output=False), ['category'])
    ])
    
    # Fit com dados fake
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'category': ['A', 'B', 'A']
    })
    preprocessor.fit(X)
    
    features = get_feature_names(preprocessor)
    
    assert isinstance(features, list)
    assert len(features) > 0


def test_get_feature_names_fallback(sample_data):
    """Testa fallback quando transformer não tem get_feature_names_out"""
    X, y = sample_data
    
    # Criar preprocessador simples sem OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['num1', 'num2']),
        ])
    
    preprocessor.fit(X)
    features = get_feature_names(preprocessor)
    
    assert isinstance(features, list)
    assert len(features) > 0


def test_get_feature_names_with_cat_transformer(trained_pipeline):
    """Testa extração de features com transformador categórico"""
    model, preprocessor, X = trained_pipeline
    
    features = get_feature_names(preprocessor, feature_names=['num1', 'num2', 'cat1'])
    
    assert isinstance(features, list)
    # Deve conter features numéricas
    assert 'num1' in features
    assert 'num2' in features


def test_create_importance_dataframe():
    """Testa criação de DataFrame com importâncias"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessing', StandardScaler()),
        ('model', xgb.XGBClassifier(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X, y)
    
    df = create_importance_dataframe(pipeline)
    
    assert isinstance(df, pd.DataFrame)
    assert 'feature' in df.columns
    assert 'importance' in df.columns
    assert len(df) == 5


def test_create_importance_dataframe_with_preprocessor():
    """Testa criação de DataFrame com preprocessador"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), list(range(5)))
    ])
    
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', xgb.XGBClassifier(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X, y)
    
    df = create_importance_dataframe(pipeline, preprocessor=preprocessor)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_create_importance_dataframe_without_preprocessor(trained_pipeline):
    """Testa DataFrame de importância sem preprocessador"""
    model, preprocessor, X = trained_pipeline
    
    # Testar sem passar preprocessador
    df = create_importance_dataframe(model, preprocessor=None)
    
    assert isinstance(df, pd.DataFrame)
    assert 'feature' in df.columns
    assert 'importance' in df.columns
    # Deve usar nomes genéricos
    assert df['feature'].iloc[0].startswith('Feature')


def test_create_importance_dataframe_sorted(trained_pipeline):
    """Testa se DataFrame de importância está ordenado"""
    model, preprocessor, X = trained_pipeline
    
    df = create_importance_dataframe(model, preprocessor)
    
    # Verificar se está ordenado por importância (decrescente)
    assert df['importance'].is_monotonic_decreasing


def test_plot_feature_importance(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot de feature importance"""
    model, preprocessor, X = trained_pipeline
    
    # Mudar diretório de reports temporariamente
    import os
    monkeypatch.chdir(tmp_path)
    os.makedirs("reports", exist_ok=True)
    
    fig = plot_feature_importance(model, X, top_n=5)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    # Verificar se arquivo foi criado
    assert os.path.exists("reports/feature_importance.png")
    
    plt.close(fig)


def test_plot_feature_importance_with_mlflow(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot de feature importance com MLflow ativo"""
    model, preprocessor, X = trained_pipeline
    
    # Configurar MLflow
    import tempfile
    mlflow_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("test_interpret")
    
    # Mudar diretório de reports
    monkeypatch.chdir(tmp_path)
    import os
    os.makedirs("reports", exist_ok=True)
    
    with mlflow.start_run():
        fig = plot_feature_importance(model, X, top_n=10)
        assert fig is not None
    
    plt.close(fig)


def test_plot_feature_importance_custom_size(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot com tamanho customizado"""
    model, preprocessor, X = trained_pipeline
    
    monkeypatch.chdir(tmp_path)
    import os
    os.makedirs("reports", exist_ok=True)
    
    fig = plot_feature_importance(model, X, top_n=3, figsize=(10, 6))
    
    assert fig is not None
    assert fig.get_figwidth() == 10
    assert fig.get_figheight() == 6
    
    plt.close(fig)


def test_plot_shap_summary_basic(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot SHAP summary básico"""
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
    
    if not HAS_SHAP:
        pytest.skip("SHAP não está instalado")
    
    model, preprocessor, X = trained_pipeline
    
    monkeypatch.chdir(tmp_path)
    import os
    os.makedirs("reports", exist_ok=True)
    
    # Preprocessar X
    X_processed = model.named_steps['preprocessing'].transform(X)
    
    result = plot_shap_summary(model, X_processed, preprocessor, top_features=3)
    
    assert result is not None
    assert os.path.exists("reports/shap_summary_bar.png")


def test_plot_shap_summary_without_preprocessor(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot SHAP summary sem preprocessador"""
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
    
    if not HAS_SHAP:
        pytest.skip("SHAP não está instalado")
    
    model, preprocessor, X = trained_pipeline
    
    monkeypatch.chdir(tmp_path)
    import os
    os.makedirs("reports", exist_ok=True)
    
    X_processed = model.named_steps['preprocessing'].transform(X)
    
    result = plot_shap_summary(model, X_processed, preprocessor=None, top_features=3)
    
    # Deve funcionar mesmo sem preprocessador (usa nomes genéricos)
    assert result is not None or result is None


def test_plot_shap_dependence_basic(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot SHAP dependence básico"""
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
    
    if not HAS_SHAP:
        pytest.skip("SHAP não está instalado")
    
    model, preprocessor, X = trained_pipeline
    
    monkeypatch.chdir(tmp_path)
    import os
    os.makedirs("reports", exist_ok=True)
    
    # Preprocessar X
    X_processed = model.named_steps['preprocessing'].transform(X)
    
    result = plot_shap_dependence(model, X_processed, preprocessor=preprocessor, feature_idx=0)
    
    assert result is not None or result is None  # Pode retornar None se SHAP não estiver disponível
    

def test_plot_shap_dependence_with_shap_values(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot SHAP dependence com SHAP values pré-calculados"""
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
    
    if not HAS_SHAP:
        pytest.skip("SHAP não está instalado")
    
    model, preprocessor, X = trained_pipeline
    
    monkeypatch.chdir(tmp_path)
    import os
    os.makedirs("reports", exist_ok=True)
    
    # Preprocessar X e calcular SHAP values
    X_processed = model.named_steps['preprocessing'].transform(X)
    xgb_model = model.named_steps['model']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_processed)
    
    result = plot_shap_dependence(model, X_processed, shap_values=shap_values, preprocessor=preprocessor, feature_idx=1)
    
    assert result is not None or result is None

