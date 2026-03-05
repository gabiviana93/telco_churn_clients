"""
Módulo de interpretabilidade de modelos.

Fornece funções para análise de Feature Importance e SHAP values
para entendimento das predições do modelo.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    INTERPRET_TOP_N,
    REPORTS_DIR,
    STEP_MODEL,
    STEP_PREPROCESSING,
    VIZ_FIGSIZE_LARGE,
    VIZ_FIGSIZE_MEDIUM,
)
from src.logger import setup_logger

logger = setup_logger(__name__)


def _log_artifact_to_mlflow(path: str) -> None:
    """Registra artefato no MLflow se houver um run ativo."""
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_artifact(path, artifact_path="visualizations")
    except ImportError:
        pass


def get_feature_names(preprocessor, feature_names=None):
    """
    Extrai nomes de features após transformação do preprocessador.

    Args:
        preprocessor: ColumnTransformer treinado
        feature_names: Nomes das features originais (opcional)

    Returns:
        Lista de nomes de features após transformação
    """
    features = []

    # Processar cada transformer
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            # Features numéricas mantêm o mesmo nome
            features.extend(columns)
        elif name == "cat":
            # Features categóricas ganham nomes dos one-hot encoding
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    cat_features = transformer.get_feature_names_out(columns)
                    features.extend(cat_features)
                except (AttributeError, ValueError):
                    # Fallback para versões antigas do sklearn
                    features.extend(columns)
            else:
                features.extend(columns)

    return features


def plot_feature_importance(model, X_test, top_n=INTERPRET_TOP_N, figsize=VIZ_FIGSIZE_LARGE):
    """
    Plota feature importance do modelo tree-based (XGBoost, LightGBM, etc.).

    Args:
        model: Pipeline sklearn com modelo tree-based no step 'model'
        X_test: Dados de teste (para contexto de feature names)
        top_n: Número de top features a mostrar (padrão 20)
        figsize: Tamanho da figura

    Returns:
        fig: Figura matplotlib
    """
    tree_model = model.named_steps[STEP_MODEL]

    # Detect model type name dynamically
    model_type_name = type(tree_model).__name__

    # Obter importâncias
    importances = tree_model.feature_importances_

    # Obter nomes de features
    preprocessor = model.named_steps.get(STEP_PREPROCESSING)
    feature_names = get_feature_names(preprocessor)

    # Ordenar e pegar top N
    indices = np.argsort(importances)[-top_n:][::-1]

    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importância", fontsize=12)
    ax.set_title(f"Top {top_n} Features - {model_type_name}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")

    _log_artifact_to_mlflow(str(output_path))

    return fig


def plot_shap_summary(model, X_test, preprocessor=None, top_features=INTERPRET_TOP_N):
    """
    Plota SHAP summary plot para interpretabilidade global.

    Requer: pip install shap

    Args:
        model: Pipeline sklearn com modelo XGBoost
        X_test: Dados de teste (preprocessados)
        preprocessor: ColumnTransformer (opcional, para nomes de features)
        top_features: Número de top features a mostrar

    Returns:
        shap_values: Array com SHAP values
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP não está instalado. Instale com: pip install shap")
        return None

    tree_model = model.named_steps[STEP_MODEL]

    # Criar explainer
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test)

    # Obter nomes de features
    if preprocessor is not None:
        feature_names = get_feature_names(preprocessor)
    else:
        feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]

    # Summary plot (bar)
    plt.figure(figsize=VIZ_FIGSIZE_LARGE)
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names, max_display=top_features, plot_type="bar"
    )
    plt.tight_layout()
    output_path = REPORTS_DIR / "shap_summary_bar.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")

    _log_artifact_to_mlflow(str(output_path))

    return shap_values


def plot_shap_dependence(model, X_test, shap_values=None, preprocessor=None, feature_idx=None):
    """
    Plota SHAP dependence plot para uma feature específica.

    Mostra a relação entre o valor da feature e seu SHAP value.

    Args:
        model: Pipeline sklearn
        X_test: Dados de teste (preprocessados)
        shap_values: SHAP values (se None, calcula)
        preprocessor: ColumnTransformer (para nomes)
        feature_idx: Índice da feature a plotar (None = top feature por importância)

    Returns:
        fig: Figura matplotlib
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP não está instalado. Instale com: pip install shap")
        return None

    if shap_values is None:
        tree_model = model.named_steps[STEP_MODEL]
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_test)

    # Obter nomes de features
    if preprocessor is not None:
        feature_names = get_feature_names(preprocessor)
    else:
        feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]

    # Encontrar top feature se feature_idx is None
    if feature_idx is None:
        top_feature = np.argsort(np.abs(shap_values).mean(axis=0))[-1]
    else:
        top_feature = feature_idx

    plt.figure(figsize=VIZ_FIGSIZE_MEDIUM)
    shap.dependence_plot(top_feature, shap_values, X_test, feature_names=feature_names)
    plt.tight_layout()
    output_path = REPORTS_DIR / "shap_dependence.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")

    _log_artifact_to_mlflow(str(output_path))

    return plt.gcf()


def create_importance_dataframe(model, preprocessor=None):
    """
    Cria DataFrame com feature importance para exportação.

    Args:
        model: Pipeline com modelo XGBoost
        preprocessor: ColumnTransformer (opcional)

    Returns:
        DataFrame com features e importância
    """
    tree_model = model.named_steps[STEP_MODEL]
    importances = tree_model.feature_importances_

    if preprocessor is not None:
        feature_names = get_feature_names(preprocessor)
    else:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    return df
