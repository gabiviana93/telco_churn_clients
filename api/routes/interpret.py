"""
Rotas de Interpretação
=====================

Endpoints da API para interpretabilidade do modelo (SHAP, Importância de Features).
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.core.logging import get_logger
from api.schemas.prediction import FeatureImportance
from api.services.model_service import ModelService, get_model_service
from src.config import DATA_DIR_RAW, FILENAME, ID_COL, RANDOM_STATE, TARGET
from src.interpret import get_feature_names
from src.utils import extract_model_components

logger = get_logger(__name__)

router = APIRouter(prefix="/interpret", tags=["Interpretability"])


# ============================================================================
# Schemas (Esquemas)
# ============================================================================

# FeatureImportance imported from api.schemas.prediction


class InterpretFeatureImportanceResponse(BaseModel):
    """Resposta para endpoint de importância de features."""

    success: bool = True
    feature_importances: list[FeatureImportance]
    model_type: str = Field(..., description="Tipo de modelo")
    total_features: int = Field(..., description="Número total de features")


class ShapValue(BaseModel):
    """Valor SHAP para uma única feature."""

    feature: str
    shap_value: float
    feature_value: float | None = None


class ShapExplanation(BaseModel):
    """Explicação SHAP para uma única predição."""

    customer_id: str | None = None
    base_value: float = Field(..., description="Valor esperado do modelo")
    prediction_value: float = Field(..., description="Valor do modelo para esta instância")
    shap_values: list[ShapValue]
    top_positive_contributors: list[ShapValue] = Field(
        ..., description="Features que aumentam a probabilidade de churn"
    )
    top_negative_contributors: list[ShapValue] = Field(
        ..., description="Features que diminuem a probabilidade de churn"
    )


class ShapExplanationResponse(BaseModel):
    """Resposta para endpoint de explicação SHAP."""

    success: bool = True
    explanation: ShapExplanation


class ShapExplainRequest(BaseModel):
    """Corpo da requisição para endpoint de explicação SHAP."""

    customer_id: str | None = Field(None, description="Identificador opcional do cliente")
    features: dict[str, float | int | str] = Field(
        ..., description="Features do cliente para explicação SHAP"
    )


class GlobalShapResponse(BaseModel):
    """Resposta para resumo SHAP global."""

    success: bool = True
    mean_abs_shap: list[FeatureImportance] = Field(
        ..., description="Valores SHAP absolutos médios por feature"
    )
    sample_size: int = Field(..., description="Número de amostras utilizadas")


# ============================================================================
# Funções Auxiliares
# ============================================================================


def _extract_model_and_preprocessor(
    model_service: ModelService,
) -> tuple:
    """Extrai modelo e pré-processador do ModelService.

    Returns:
        Tupla de (model, preprocessor). preprocessor pode ser None.

    Raises:
        HTTPException: Se o modelo não está carregado.
    """
    if model_service.pipeline is not None:
        pipeline = model_service.pipeline
        if hasattr(pipeline, "pipeline") and pipeline.pipeline is not None:
            model, preprocessor, _ = extract_model_components(pipeline.pipeline)
        else:
            model = model_service.model
            preprocessor = None
    elif model_service.model is not None:
        model, preprocessor, _ = extract_model_components(model_service.model)
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "MODEL_NOT_LOADED", "message": "Model not loaded"},
        )

    # Fallback: usar preprocessor do ModelService para modelos legados
    if preprocessor is None and model_service.preprocessor is not None:
        preprocessor = model_service.preprocessor

    return model, preprocessor


def _is_tree_model(model) -> bool:
    """Verifica se o modelo é baseado em árvore."""
    tree_attrs = ("tree_", "estimators_", "get_booster")
    tree_names = (
        "LGBM",
        "XGB",
        "CatBoost",
        "Forest",
        "GradientBoosting",
        "DecisionTree",
        "ExtraTrees",
    )
    if any(hasattr(model, attr) for attr in tree_attrs):
        return True
    model_name = type(model).__name__
    return any(name.lower() in model_name.lower() for name in tree_names)


def _is_linear_model(model) -> bool:
    """Verifica se o modelo é linear."""
    return hasattr(model, "coef_")


def _create_shap_explainer(model, X_background=None):
    """Cria o explainer SHAP adequado para o tipo de modelo.

    Seleciona automaticamente TreeExplainer, LinearExplainer ou
    KernelExplainer conforme o tipo do modelo.
    """
    import shap

    if _is_tree_model(model):
        return shap.TreeExplainer(model)
    if _is_linear_model(model):
        if X_background is None:
            raise ValueError("X_background é necessário para o LinearExplainer")
        return shap.LinearExplainer(model, X_background)
    # Fallback genérico para qualquer modelo
    if X_background is None:
        raise ValueError("X_background é necessário para o KernelExplainer")

    def predict_fn(x):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]
        return model.decision_function(x)

    return shap.KernelExplainer(predict_fn, X_background)


def _extract_shap_values(shap_values, expected_value, *, single_instance: bool = False):
    """Normaliza a saída do SHAP para a classe positiva.

    Retorna (shap_array, base_value) onde shap_array tem shape
    (n_features,) se single_instance=True ou (n_samples, n_features) caso contrário.
    """
    if isinstance(shap_values, list):
        sv = shap_values[1]
        ev = expected_value[1] if isinstance(expected_value, list | np.ndarray) else expected_value
    else:
        sv = shap_values
        ev = (
            expected_value
            if not isinstance(expected_value, list | np.ndarray)
            else expected_value[0]
        )

    if single_instance and sv.ndim > 1:
        sv = sv[0]
    return sv, float(ev)


def _transform_and_get_names(preprocessor, df: pd.DataFrame, feature_engineer=None) -> tuple:
    """Transforma dados e obtém nomes de features.

    Returns:
        Tupla de (X_transformado, nomes_features)
    """
    if feature_engineer is not None:
        df = feature_engineer.transform(df)
    if preprocessor is not None:
        X_transformed = preprocessor.transform(df)
        feature_names = get_feature_names(preprocessor)
    else:
        X_transformed = df.values
        feature_names = list(df.columns)
    return X_transformed, feature_names


def _load_background_sample(preprocessor, n: int = 100, feature_engineer=None) -> np.ndarray:
    """Carrega amostra de background dos dados de referência para explainers não-tree."""
    data_path = DATA_DIR_RAW / FILENAME
    df = pd.read_csv(data_path)
    df = df.drop(columns=[ID_COL, TARGET], errors="ignore")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
    X, _ = _transform_and_get_names(preprocessor, df, feature_engineer=feature_engineer)
    return X


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/feature-importance",
    response_model=InterpretFeatureImportanceResponse,
    summary="Obter Importância de Features",
    description="Retorna a importância de features do modelo treinado.",
)
async def get_feature_importance(
    top_n: int = Query(default=20, ge=1, le=100, description="Número de features principais"),
    model_service: ModelService = Depends(get_model_service),
) -> InterpretFeatureImportanceResponse:
    """Obtém importância de features do modelo."""
    logger.info("Obtendo importância de features", top_n=top_n)

    try:
        # Use existing get_feature_importance method
        importances_list = model_service.get_feature_importance()

        if not importances_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "FEATURE_IMPORTANCE_NOT_AVAILABLE",
                    "message": "Importância de features não disponível para este modelo",
                },
            )

        # Convert to our format and limit to top_n
        feature_list = [
            FeatureImportance(
                feature_name=item.get("feature_name", f"feature_{i}"),
                importance=round(item.get("importance", 0), 6),
                rank=item.get("rank", i + 1),
            )
            for i, item in enumerate(importances_list[:top_n])
        ]

        return InterpretFeatureImportanceResponse(
            success=True,
            feature_importances=feature_list,
            model_type=model_service.get_model_info().get("model_name", "unknown"),
            total_features=len(importances_list),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get feature importance", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "FEATURE_IMPORTANCE_ERROR",
                "message": f"Falha ao obter importância de features: {str(e)}",
            },
        ) from e


@router.post(
    "/shap/explain",
    response_model=ShapExplanationResponse,
    summary="Explicar Predição Individual com SHAP",
    description="""
    Obtém explicação baseada em SHAP para uma predição individual.

    Retorna os valores SHAP para cada feature, mostrando como cada
    feature contribui para a predição.
    """,
)
async def explain_prediction(
    request: ShapExplainRequest, model_service: ModelService = Depends(get_model_service)
) -> ShapExplanationResponse:
    """Explica uma predição individual usando valores SHAP."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"error_code": "SHAP_NOT_INSTALLED", "message": "Biblioteca SHAP não instalada"},
        ) from exc

    logger.info("Gerando explicação SHAP")

    try:
        customer_id = request.customer_id
        features = dict(request.features)

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Get model, preprocessor and transform data
        model, preprocessor = _extract_model_and_preprocessor(model_service)
        fe = model_service.feature_engineer
        X_transformed, feature_names = _transform_and_get_names(
            preprocessor, df, feature_engineer=fe
        )

        # For non-tree models, load background sample from reference data
        X_background = None
        if not _is_tree_model(model):
            X_background = _load_background_sample(preprocessor, n=100, feature_engineer=fe)

        # Create appropriate SHAP explainer for the model type
        explainer = _create_shap_explainer(model, X_background=X_background)
        shap_values = explainer.shap_values(X_transformed)

        # Normalize SHAP output to positive class
        sample_shap, base_value = _extract_shap_values(
            shap_values, explainer.expected_value, single_instance=True
        )
        prediction_value = float(base_value + np.sum(sample_shap))

        # Build SHAP value list
        shap_list = []
        for i, (feat, sv) in enumerate(zip(feature_names, sample_shap, strict=True)):
            feat_val = float(X_transformed[0, i]) if isinstance(X_transformed, np.ndarray) else None
            shap_list.append(
                ShapValue(
                    feature=feat,
                    shap_value=round(float(sv), 6),
                    feature_value=round(feat_val, 4) if feat_val is not None else None,
                )
            )

        # Sort for top contributors
        sorted_shap = sorted(shap_list, key=lambda x: x.shap_value, reverse=True)
        top_positive = [s for s in sorted_shap if s.shap_value > 0][:5]
        top_negative = [s for s in sorted_shap if s.shap_value < 0][-5:][::-1]

        explanation = ShapExplanation(
            customer_id=customer_id,
            base_value=round(base_value, 6),
            prediction_value=round(prediction_value, 6),
            shap_values=shap_list,
            top_positive_contributors=top_positive,
            top_negative_contributors=top_negative,
        )

        logger.info("Explicação SHAP gerada com sucesso")

        return ShapExplanationResponse(success=True, explanation=explanation)

    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "error_code": "SHAP_DEPENDENCY_ERROR",
                "message": "SHAP requer dependências adicionais",
            },
        ) from exc
    except Exception as e:
        logger.error("Erro na explicação SHAP", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "SHAP_EXPLANATION_ERROR",
                "message": f"Erro na explicação SHAP: {str(e)}",
            },
        ) from e


@router.get(
    "/shap/global",
    response_model=GlobalShapResponse,
    summary="Obter Resumo SHAP Global",
    description="Retorna valores SHAP absolutos médios de uma amostra de dados.",
)
async def get_global_shap(
    sample_size: int = Query(
        default=100, ge=10, le=500, description="Tamanho da amostra para SHAP"
    ),
    top_n: int = Query(default=20, ge=1, le=50, description="Número de principais features"),
    model_service: ModelService = Depends(get_model_service),
) -> GlobalShapResponse:
    """Obtém estatísticas resumidas SHAP globais."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"error_code": "SHAP_NOT_INSTALLED", "message": "Biblioteca SHAP não instalada"},
        ) from exc

    logger.info("Calculando resumo SHAP global", sample_size=sample_size)

    try:
        # Load sample data
        data_path = DATA_DIR_RAW / FILENAME
        df = pd.read_csv(data_path)

        # Prepare data
        df = df.drop(columns=[ID_COL, TARGET], errors="ignore")
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna()
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)

        # Get model, preprocessor and transform data
        model, preprocessor = _extract_model_and_preprocessor(model_service)
        fe = model_service.feature_engineer
        X_transformed, feature_names = _transform_and_get_names(
            preprocessor, df, feature_engineer=fe
        )

        # Create appropriate SHAP explainer for the model type
        explainer = _create_shap_explainer(model, X_background=X_transformed)
        shap_values = explainer.shap_values(X_transformed)

        # Normalize SHAP output to positive class
        shap_values, _ = _extract_shap_values(
            shap_values, explainer.expected_value, single_instance=False
        )

        # Calculate mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Build response
        shap_importance = []
        for i, (feat, val) in enumerate(
            sorted(
                zip(feature_names, mean_abs_shap, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )[:top_n]
        ):
            shap_importance.append(
                FeatureImportance(feature_name=feat, importance=round(float(val), 6), rank=i + 1)
            )

        logger.info("Resumo SHAP global calculado")

        return GlobalShapResponse(success=True, mean_abs_shap=shap_importance, sample_size=len(df))

    except Exception as e:
        logger.error("Erro no cálculo do SHAP global", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "GLOBAL_SHAP_ERROR",
                "message": f"Erro no cálculo do SHAP global: {str(e)}",
            },
        ) from e
