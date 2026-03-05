"""
Módulo de interpretabilidade de modelos.

Fornece funções para análise de Feature Importance e SHAP values
para entendimento das predições do modelo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


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
        if name == 'num':
            # Features numéricas mantêm o mesmo nome
            features.extend(columns)
        elif name == 'cat':
            # Features categóricas ganham nomes dos one-hot encoding
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    cat_features = transformer.get_feature_names_out(columns)
                    features.extend(cat_features)
                except:
                    # Fallback
                    features.extend(columns)
            else:
                features.extend(columns)
    
    return features


def plot_feature_importance(model, X_test, top_n=20, figsize=(12, 8)):
    """
    Plota feature importance usando o modelo XGBoost.
    
    Args:
        model: Pipeline com modelo XGBoost
        X_test: Dados de teste (para context)
        top_n: Número de top features a mostrar (padrão 20)
        figsize: Tamanho da figura
    
    Returns:
        fig: Figura matplotlib
    """
    xgb_model = model.named_steps['model']
    
    # Obter importâncias
    importances = xgb_model.feature_importances_
    
    # Obter nomes de features
    preprocessor = model.named_steps['preprocessing']
    feature_names = get_feature_names(preprocessor)
    
    # Ordenar e pegar top N
    indices = np.argsort(importances)[-top_n:][::-1]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importância', fontsize=12)
    ax.set_title(f'Top {top_n} Features - XGBoost', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=100, bbox_inches='tight')
    
    # Log no MLflow
    if mlflow.active_run():
        mlflow.log_artifact("reports/feature_importance.png", artifact_path="visualizations")
    
    return fig


def plot_shap_summary(model, X_test, preprocessor=None, top_features=20):
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
        print("⚠️  SHAP não está instalado. Instale com: pip install shap")
        return None
    
    xgb_model = model.named_steps['model']
    
    # Criar explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    # Obter nomes de features
    if preprocessor is not None:
        feature_names = get_feature_names(preprocessor)
    else:
        feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]
    
    # Summary plot (bar)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=feature_names,
        max_display=top_features,
        plot_type="bar"
    )
    plt.tight_layout()
    plt.savefig("reports/shap_summary_bar.png", dpi=100, bbox_inches='tight')
    
    # Log no MLflow
    if mlflow.active_run():
        mlflow.log_artifact("reports/shap_summary_bar.png", artifact_path="visualizations")
    
    return shap_values


def plot_shap_dependence(model, X_test, shap_values=None, preprocessor=None, feature_idx=0):
    """
    Plota SHAP dependence plot para uma feature específica.
    
    Mostra a relação entre o valor da feature e seu SHAP value.
    
    Args:
        model: Pipeline sklearn
        X_test: Dados de teste (preprocessados)
        shap_values: SHAP values (se None, calcula)
        preprocessor: ColumnTransformer (para nomes)
        feature_idx: Índice da feature a plotar (padrão 0)
    
    Returns:
        fig: Figura matplotlib
    """
    try:
        import shap
    except ImportError:
        print("⚠️  SHAP não está instalado. Instale com: pip install shap")
        return None
    
    if shap_values is None:
        xgb_model = model.named_steps['model']
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
    
    # Obter nomes de features
    if preprocessor is not None:
        feature_names = get_feature_names(preprocessor)
    else:
        feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]
    
    # Encontrar top feature se feature_idx = 0
    if feature_idx == 0:
        top_feature = np.argsort(np.abs(shap_values).mean(axis=0))[-1]
    else:
        top_feature = feature_idx
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feature, 
        shap_values, 
        X_test,
        feature_names=feature_names
    )
    plt.tight_layout()
    plt.savefig("reports/shap_dependence.png", dpi=100, bbox_inches='tight')
    
    # Log no MLflow
    if mlflow.active_run():
        mlflow.log_artifact("reports/shap_dependence.png", artifact_path="visualizations")
    
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
    xgb_model = model.named_steps['model']
    importances = xgb_model.feature_importances_
    
    if preprocessor is not None:
        feature_names = get_feature_names(preprocessor)
    else:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df
