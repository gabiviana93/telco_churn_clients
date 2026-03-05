import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Backend não-interativo para testes
import matplotlib.pyplot as plt
import mlflow
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import STEP_MODEL, STEP_PREPROCESSING
from src.interpret import (
    create_importance_dataframe,
    get_feature_names,
    plot_feature_importance,
    plot_shap_dependence,
    plot_shap_summary,
)

# Verificação de disponibilidade do SHAP em nível de módulo
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

_skip_no_shap = pytest.mark.skipif(not HAS_SHAP, reason="SHAP não está instalado")


@pytest.fixture
def sample_data():
    """Cria dados de exemplo para testes"""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "num1": np.random.randn(100),
            "num2": np.random.randn(100),
            "cat1": np.random.choice(["A", "B", "C"], 100),
        }
    )
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def trained_pipeline(sample_data):
    """Cria um pipeline treinado para testes"""
    X, y = sample_data

    # Criar preprocessador
    num_features = ["num1", "num2"]
    cat_features = ["cat1"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
        ]
    )

    # Criar pipeline
    model = Pipeline(
        [
            (STEP_PREPROCESSING, preprocessor),
            (STEP_MODEL, xgb.XGBClassifier(random_state=42, n_estimators=10)),
        ]
    )

    # Treinar
    model.fit(X, y)

    return model, preprocessor, X


def test_get_feature_names():
    """Testa extração de nomes de features do preprocessador"""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["feature1", "feature2"]),
            ("cat", OneHotEncoder(sparse_output=False), ["category"]),
        ]
    )

    # Fit com dados fake
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "category": ["A", "B", "A"]})
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
            ("num", StandardScaler(), ["num1", "num2"]),
        ]
    )

    preprocessor.fit(X)
    features = get_feature_names(preprocessor)

    assert isinstance(features, list)
    assert len(features) > 0


def test_get_feature_names_with_cat_transformer(trained_pipeline):
    """Testa extração de features com transformador categórico"""
    model, preprocessor, X = trained_pipeline

    features = get_feature_names(preprocessor, feature_names=["num1", "num2", "cat1"])

    assert isinstance(features, list)
    # Deve conter features numéricas
    assert "num1" in features
    assert "num2" in features


def test_create_importance_dataframe():
    """Testa criação de DataFrame com importâncias"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    pipeline = Pipeline(
        [
            (STEP_PREPROCESSING, StandardScaler()),
            (STEP_MODEL, xgb.XGBClassifier(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(X, y)

    df = create_importance_dataframe(pipeline)

    assert isinstance(df, pd.DataFrame)
    assert "feature" in df.columns
    assert "importance" in df.columns
    assert len(df) == 5


def test_create_importance_dataframe_with_preprocessor():
    """Testa criação de DataFrame com preprocessador"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    preprocessor = ColumnTransformer([("num", StandardScaler(), list(range(5)))])

    pipeline = Pipeline(
        [
            (STEP_PREPROCESSING, preprocessor),
            (STEP_MODEL, xgb.XGBClassifier(n_estimators=10, random_state=42)),
        ]
    )
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
    assert "feature" in df.columns
    assert "importance" in df.columns
    # Deve usar nomes genéricos
    assert df["feature"].iloc[0].startswith("Feature")


def test_create_importance_dataframe_sorted(trained_pipeline):
    """Testa se DataFrame de importância está ordenado"""
    model, preprocessor, X = trained_pipeline

    df = create_importance_dataframe(model, preprocessor)

    # Verificar se está ordenado por importância (decrescente)
    assert df["importance"].is_monotonic_decreasing


def test_plot_feature_importance(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot de feature importance"""
    model, preprocessor, X = trained_pipeline

    # Mudar REPORTS_DIR para diretório temporário
    import src.interpret as interpret_module

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(interpret_module, "REPORTS_DIR", reports_dir)

    fig = plot_feature_importance(model, X, top_n=5)

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    # Verificar se arquivo foi criado
    assert (reports_dir / "feature_importance.png").exists()

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


@_skip_no_shap
def test_plot_shap_summary_basic(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot SHAP summary básico"""
    model, preprocessor, X = trained_pipeline

    # Mudar REPORTS_DIR para diretório temporário
    import src.interpret as interpret_module

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(interpret_module, "REPORTS_DIR", reports_dir)

    # Preprocessar X
    X_processed = model.named_steps[STEP_PREPROCESSING].transform(X)

    result = plot_shap_summary(model, X_processed, preprocessor, top_features=3)

    assert result is not None
    assert (reports_dir / "shap_summary_bar.png").exists()


@_skip_no_shap
def test_plot_shap_summary_without_preprocessor(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot SHAP summary sem preprocessador"""
    model, preprocessor, X = trained_pipeline

    monkeypatch.chdir(tmp_path)
    import os

    os.makedirs("reports", exist_ok=True)

    X_processed = model.named_steps[STEP_PREPROCESSING].transform(X)

    result = plot_shap_summary(model, X_processed, preprocessor=None, top_features=3)

    # Sem preprocessador deve retornar SHAP values (array numpy)
    assert result is not None


@_skip_no_shap
def test_plot_shap_dependence_basic(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot SHAP dependence básico"""
    model, preprocessor, X = trained_pipeline

    monkeypatch.chdir(tmp_path)
    import os

    os.makedirs("reports", exist_ok=True)

    # Preprocessar X
    X_processed = model.named_steps[STEP_PREPROCESSING].transform(X)

    result = plot_shap_dependence(model, X_processed, preprocessor=preprocessor, feature_idx=0)

    # Deve retornar uma figura matplotlib
    assert result is not None


@_skip_no_shap
def test_plot_shap_dependence_with_shap_values(trained_pipeline, tmp_path, monkeypatch):
    """Testa plot SHAP dependence com SHAP values pré-calculados"""
    model, preprocessor, X = trained_pipeline

    monkeypatch.chdir(tmp_path)
    import os

    os.makedirs("reports", exist_ok=True)

    # Preprocessar X e calcular SHAP values
    X_processed = model.named_steps[STEP_PREPROCESSING].transform(X)
    xgb_model = model.named_steps[STEP_MODEL]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_processed)

    result = plot_shap_dependence(
        model, X_processed, shap_values=shap_values, preprocessor=preprocessor, feature_idx=1
    )

    # Deve retornar uma figura matplotlib
    assert result is not None
