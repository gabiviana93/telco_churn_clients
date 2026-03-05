"""
Core Utilities
==============

Funções utilitárias essenciais para o projeto de predição de churn.
Incluí apenas funções usadas pelo código de produção (API, scripts, pipeline).

Para funções de análise e plotagem usadas em notebooks, veja `notebook_utils.py`.

Usage:
    from src.utils import (
        generate_model_filename,
        parse_model_filename,
        convert_target_to_binary,
        validate_target,
        classify_risk,
        normalize_metrics_keys,
    )
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW, STEP_MODEL, STEP_PREPROCESSING
from src.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# FEATURE AUTO-DETECTION
# =============================================================================


def auto_detect_features(X: pd.DataFrame) -> tuple:
    """
    Auto-detect numeric and categorical features from a DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple of (list[str], list[str])
        (numeric_features, categorical_features)

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    >>> numeric, categorical = auto_detect_features(df)
    >>> numeric
    ['a']
    >>> categorical
    ['b']
    """
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


# =============================================================================
# RISK CLASSIFICATION
# =============================================================================


def classify_risk(probability: float) -> str:
    """
    Classify churn risk based on probability using centralized thresholds.

    Uses RISK_THRESHOLD_LOW and RISK_THRESHOLD_HIGH from src.config.

    Parameters
    ----------
    probability : float
        Churn probability (0-1)

    Returns
    -------
    str
        Risk category: "HIGH", "MEDIUM", or "LOW"

    Examples
    --------
    >>> classify_risk(0.8)
    'HIGH'
    >>> classify_risk(0.5)
    'MEDIUM'
    >>> classify_risk(0.2)
    'LOW'
    """
    if probability >= RISK_THRESHOLD_HIGH:
        return "HIGH"
    elif probability >= RISK_THRESHOLD_LOW:
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# METRICS NORMALIZATION
# =============================================================================

# Canonical metric key mapping: maps common variations to snake_case keys
_METRICS_KEY_MAP = {
    "F1-Score": "f1_score",
    "f1-score": "f1_score",
    "f1": "f1_score",
    "f1_optimal": "f1_score",
    "ROC-AUC": "roc_auc",
    "roc-auc": "roc_auc",
    "AUC": "roc_auc",
    "auc": "roc_auc",
    "Precision": "precision",
    "Recall": "recall",
    "Accuracy": "accuracy",
}


def normalize_metrics_keys(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize metric keys to canonical snake_case format.

    Maps common variations (e.g., 'F1-Score', 'ROC-AUC') to their
    canonical snake_case equivalents ('f1_score', 'roc_auc').

    Parameters
    ----------
    metrics : dict
        Dictionary with potentially inconsistent metric keys.

    Returns
    -------
    dict
        Dictionary with normalized keys. Unknown keys are passed through.

    Examples
    --------
    >>> normalize_metrics_keys({"F1-Score": 0.85, "ROC-AUC": 0.90})
    {'f1_score': 0.85, 'roc_auc': 0.90}
    """
    normalized = {}
    for key, value in metrics.items():
        canonical = _METRICS_KEY_MAP.get(key, key)
        normalized[canonical] = value
    return normalized


# =============================================================================
# MODEL ARTIFACT NAMING
# =============================================================================


def generate_model_filename(
    algorithm: str = "xgb",
    metrics: dict[str, float] | None = None,
    prefix: str = "churn",
    include_timestamp: bool = True,
    include_metrics: bool = True,
    extension: str = ".joblib",
) -> str:
    """
    Gera nome descritivo para artefatos de modelo.

    Parameters
    ----------
    algorithm : str, default='xgb'
        Nome do algoritmo (xgb, lgbm, catboost, rf, etc.)
    metrics : dict, optional
        Dicionário com métricas (f1_score, roc_auc, etc.)
    prefix : str, default='churn'
        Prefixo do projeto
    include_timestamp : bool, default=True
        Se deve incluir timestamp no nome
    include_metrics : bool, default=True
        Se deve incluir métricas no nome
    extension : str, default='.joblib'
        Extensão do arquivo

    Returns
    -------
    str
        Nome do arquivo formatado

    Examples
    --------
    >>> generate_model_filename('lgbm', {'f1_score': 0.85, 'roc_auc': 0.88})
    'churn_lgbm_f1-85_auc-88_20260303_120000.joblib'
    """
    parts = [prefix, algorithm.lower()]

    if include_metrics and metrics:
        if "f1_score" in metrics:
            f1_pct = int(metrics["f1_score"] * 100)
            parts.append(f"f1-{f1_pct}")
        if "roc_auc" in metrics:
            auc_pct = int(metrics["roc_auc"] * 100)
            parts.append(f"auc-{auc_pct}")

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)

    return "_".join(parts) + extension


def parse_model_filename(filename: str) -> dict[str, Any]:
    """
    Extrai informações do nome do arquivo de modelo.

    Parameters
    ----------
    filename : str
        Nome do arquivo de modelo

    Returns
    -------
    dict
        Dicionário com informações extraídas
    """
    result = {
        "algorithm": "unknown",
        "f1_score": None,
        "roc_auc": None,
        "timestamp": None,
        "prefix": None,
    }

    name = Path(filename).stem
    parts = name.split("_")

    if not parts:
        return result

    result["prefix"] = parts[0]

    known_algorithms = ["xgb", "lgbm", "lightgbm", "catboost", "rf", "gb", "model", "optimized"]

    for part in parts[1:]:
        part_lower = part.lower()

        if part_lower in known_algorithms:
            result["algorithm"] = part_lower
        elif part_lower.startswith("f1-"):
            try:
                result["f1_score"] = int(part[3:]) / 100.0
            except ValueError:
                pass
        elif part_lower.startswith("auc-"):
            try:
                result["roc_auc"] = int(part[4:]) / 100.0
            except ValueError:
                pass

    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                try:
                    result["timestamp"] = datetime.strptime(f"{part}_{parts[i+1]}", "%Y%m%d_%H%M%S")
                except ValueError:
                    pass

    return result


# =============================================================================
# TARGET CONVERSION
# =============================================================================


def convert_target_to_binary(y, positive_class: str = "Yes") -> pd.Series:
    """
    Converte target categórico para binário (0/1).

    Parameters
    ----------
    y : pd.Series, np.ndarray, or list
        Target a ser convertido
    positive_class : str, default='Yes'
        Valor que representa a classe positiva (churn=1)

    Returns
    -------
    pd.Series
        Target convertido para 0/1 (int)

    Examples
    --------
    >>> y = pd.Series(['Yes', 'No', 'Yes', 'No'])
    >>> convert_target_to_binary(y)
    0    1
    1    0
    2    1
    3    0
    dtype: int64
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    if pd.api.types.is_numeric_dtype(y):
        unique_vals = set(y.dropna().unique())
        if unique_vals.issubset({0, 1}):
            logger.debug("Target já é binário (0/1)")
            return y.astype(int)

    if y.dtype == "object" or isinstance(y.dtype, pd.CategoricalDtype):
        y_lower = y.str.lower() if hasattr(y.str, "lower") else y
        positive_lower = positive_class.lower()
        y_binary = (y_lower == positive_lower).astype(int)

        logger.info(
            f"Target convertido: '{positive_class}'->1, outros->0 | "
            f"Distribuição: {y_binary.value_counts().to_dict()}"
        )
        return y_binary

    raise ValueError(f"Tipo de target não suportado: {y.dtype}")


def validate_target(y) -> bool:
    """
    Valida se o target está no formato binário correto (0/1).

    Parameters
    ----------
    y : pd.Series or array-like
        Target a ser validado

    Returns
    -------
    bool
        True se válido

    Raises
    ------
    ValueError
        Se o target contém valores fora de {0, 1}
    """
    unique_values = pd.Series(y).dropna().unique()
    valid_values = {0, 1}

    if not set(unique_values).issubset(valid_values):
        raise ValueError(
            f"Target deve conter apenas valores 0 e 1. "
            f"Valores encontrados: {sorted(unique_values)}"
        )

    logger.debug("Target validado: formato binário correto")
    return True


# =============================================================================
# MODEL/PIPELINE EXTRACTION
# =============================================================================


def extract_model_components(model) -> tuple:
    """
    Extract tree model, preprocessor and feature engineering from a pipeline or model object.

    Supports sklearn Pipeline, ChurnPipeline, and raw model objects.
    Uses the canonical step names defined in ``src.config``
    (``STEP_MODEL`` and ``STEP_PREPROCESSING``).

    Parameters
    ----------
    model : object
        A trained model, sklearn Pipeline, or ChurnPipeline instance.

    Returns
    -------
    tuple of (tree_model, preprocessor, feature_engineering)
        Any component may be None if not found.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> tree, prep, fe = extract_model_components(pipeline)
    """
    tree_model = None
    preprocessor = None
    feature_engineering = None

    if hasattr(model, "named_steps"):
        steps = model.named_steps
        tree_model = steps.get(STEP_MODEL)
        preprocessor = steps.get(STEP_PREPROCESSING)

        if "feature_engineering" in steps:
            feature_engineering = steps["feature_engineering"]

    elif hasattr(model, "best_model"):
        tree_model = model.best_model
        preprocessor = getattr(model, "preprocessor", None)
    else:
        tree_model = model

    return tree_model, preprocessor, feature_engineering
