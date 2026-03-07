"""
Módulo de Inferência para Churn Prediction.

Carrega modelos treinados (ChurnPipeline, dict-package ou sklearn Pipeline)
e fornece validação de features e predições com classificação de risco.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from src.config import MODEL_PATH
from src.logger import setup_logger

logger = setup_logger(__name__)


class FeatureValidationError(Exception):
    """Exceção customizada para erros de validação de features."""

    pass


def validate_input_features(
    X: pd.DataFrame, expected_features: list[str], strict: bool = True
) -> tuple[bool, list[str], list[str]]:
    """
    Valida se o DataFrame de entrada possui as features esperadas.

    Args:
        X: DataFrame com os dados de entrada
        expected_features: Lista de features que o modelo espera
        strict: Se True, exige correspondência exata. Se False, permite features extras.

    Returns:
        Tupla (is_valid, missing_features, extra_features)

    Raises:
        FeatureValidationError: Se validação falhar em modo strict
    """
    input_features = set(X.columns)
    expected_features_set = set(expected_features)

    missing = list(expected_features_set - input_features)
    extra = list(input_features - expected_features_set)

    is_valid = len(missing) == 0 and (not strict or len(extra) == 0)

    if not is_valid:
        error_msg = []
        if missing:
            error_msg.append(f"Features faltando: {missing}")
        if strict and extra:
            error_msg.append(f"Features extras não esperadas: {extra}")

        logger.error(
            "Validação de features falhou",
            extra={
                "missing_features": missing,
                "extra_features": extra,
                "expected_count": len(expected_features),
                "received_count": len(input_features),
            },
        )

        if strict:
            raise FeatureValidationError(" | ".join(error_msg))

    return is_valid, missing, extra


def load_model_package(model_path: str | Path = None) -> dict[str, Any]:
    """
    Carrega pacote completo do modelo com validações.

    Suporta dois formatos:
    1. Novo formato (dict com 'model', 'scaler', 'features', 'hyperparameters', etc.)
    2. Formato legado (apenas o pipeline/modelo)

    Args:
        model_path: Caminho para o arquivo .joblib do modelo

    Returns:
        Dict contendo o modelo e metadados

    Raises:
        FileNotFoundError: Se arquivo não existir
        ValueError: Se formato do arquivo for inválido
    """
    if model_path is None:
        model_path = MODEL_PATH

    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Modelo não encontrado: {model_path}")
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

    logger.info("Carregando modelo", extra={"model_path": str(model_path)})

    try:
        package = load(model_path)

        # Verificar se é novo formato (dict com campos específicos)
        if isinstance(package, dict) and "model" in package and "features" in package:
            logger.info(
                "Modelo carregado (novo formato)",
                extra={
                    "model_name": package.get("model_name", "unknown"),
                    "n_input_features": package["features"].get("n_input_features", "unknown"),
                    "training_date": package.get("metadata", {}).get("training_date", "unknown"),
                },
            )
            return package

        # Formato com 'pipeline' em vez de 'model' (ex: churn_model_lightgbm.joblib)
        elif isinstance(package, dict) and "pipeline" in package:
            logger.info(
                "Modelo carregado (formato pipeline)",
                extra={"model_name": package.get("model_name", "unknown")},
            )
            return {
                "model": package["pipeline"],
                "scaler": None,
                "features": {"input_features": None, "transformed_features": None},
                "metrics": package.get("metrics"),
                "params": package.get("params"),
                "is_legacy": True,
            }

        # Formato com 'model' mas sem 'features' (ex: model_optimized.joblib)
        elif isinstance(package, dict) and "model" in package:
            logger.info(
                "Modelo carregado (formato model sem features)",
                extra={"has_preprocessor": "preprocessor" in package},
            )
            return {
                "model": package["model"],
                "scaler": package.get("preprocessor"),
                "feature_engineer": package.get("feature_engineer"),
                "features": {"input_features": None, "transformed_features": None},
                "threshold": package.get("threshold"),
                "metrics": package.get("metrics"),
                "is_legacy": True,
            }

        # Formato legado: apenas pipeline/modelo com método predict
        elif hasattr(package, "predict"):
            logger.warning("Modelo carregado em formato legado (sem metadados)")
            return {
                "model": package,
                "scaler": None,
                "features": {"input_features": None, "transformed_features": None},
                "is_legacy": True,
            }

        else:
            raise ValueError(f"Formato de modelo não reconhecido: {type(package)}")

    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise


def predict_with_package(
    model_package: dict[str, Any],
    X: pd.DataFrame,
    validate_features: bool = True,
    return_proba: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Realiza predição usando o pacote completo do modelo.

    Args:
        model_package: Pacote retornado por load_model_package()
        X: DataFrame com os dados de entrada
        validate_features: Se True, valida features antes de predizer
        return_proba: Se True, retorna também as probabilidades

    Returns:
        predictions ou (predictions, probabilities) se return_proba=True

    Raises:
        FeatureValidationError: Se validação de features falhar
    """
    # Validar features se necessário
    if validate_features and model_package["features"]["input_features"] is not None:
        expected = model_package["features"]["input_features"]
        is_valid, missing, extra = validate_input_features(X, expected, strict=False)

        if missing:
            raise FeatureValidationError(
                f"Dados de entrada estão faltando {len(missing)} features: {missing[:5]}..."
            )

        # Reordenar colunas para match esperado
        X = X[expected]

    # Aplicar feature engineering
    feature_engineer = model_package.get("feature_engineer")
    if feature_engineer is not None:
        X = feature_engineer.transform(X)

    # Aplicar preprocessing
    scaler = model_package.get("scaler")
    if scaler is not None:
        logger.info(
            f"Aplicando preprocessing com scaler: {model_package.get('preprocessing_config', {}).get('scaler_type', 'unknown')}"
        )
        X_transformed = scaler.transform(X)
    else:
        X_transformed = X

    # Predição
    model = model_package["model"]
    logger.info(f"Realizando inferência em {X.shape[0]} amostras")

    predictions = model.predict(X_transformed)
    logger.info("Inferência concluída")

    if return_proba:
        if hasattr(model, "predict_proba"):
            logger.info(f"Calculando probabilidades para {X.shape[0]} amostras")
            probas = model.predict_proba(X_transformed)
            logger.info("Cálculo de probabilidades concluído")
            return predictions, probas
        else:
            logger.warning("Modelo não suporta predict_proba")
            return predictions, None

    return predictions
