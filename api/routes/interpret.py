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
    model_type: str = Field(..., description="Type of model")
    total_features: int = Field(..., description="Total number of features")


class ShapValue(BaseModel):
    """Valor SHAP para uma única feature."""

    feature: str
    shap_value: float
    feature_value: float | None = None


class ShapExplanation(BaseModel):
    """Explicação SHAP para uma única predição."""

    customer_id: str | None = None
    base_value: float = Field(..., description="Expected model output")
    prediction_value: float = Field(..., description="Model output for this instance")
    shap_values: list[ShapValue]
    top_positive_contributors: list[ShapValue] = Field(
        ..., description="Features that increase churn probability"
    )
    top_negative_contributors: list[ShapValue] = Field(
        ..., description="Features that decrease churn probability"
    )


class ShapExplanationResponse(BaseModel):
    """Resposta para endpoint de explicação SHAP."""

    success: bool = True
    explanation: ShapExplanation


class ShapExplainRequest(BaseModel):
    """Corpo da requisição para endpoint de explicação SHAP."""

    customer_id: str | None = Field(None, description="Optional customer identifier")
    features: dict[str, float | int | str] = Field(
        ..., description="Customer features for explanation"
    )


class GlobalShapResponse(BaseModel):
    """Resposta para resumo SHAP global."""

    success: bool = True
    mean_abs_shap: list[FeatureImportance] = Field(
        ..., description="Mean absolute SHAP values per feature"
    )
    sample_size: int = Field(..., description="Number of samples used")


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not loaded")
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
            raise ValueError("X_background is required for LinearExplainer")
        return shap.LinearExplainer(model, X_background)
    # Fallback genérico para qualquer modelo
    if X_background is None:
        raise ValueError("X_background is required for KernelExplainer")

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
        ev = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        sv = shap_values
        ev = (
            expected_value
            if not isinstance(expected_value, (list, np.ndarray))
            else expected_value[0]
        )

    if single_instance and sv.ndim > 1:
        sv = sv[0]
    return sv, float(ev)


def _transform_and_get_names(preprocessor, df: pd.DataFrame) -> tuple:
    """Transforma dados e obtém nomes de features.

    Returns:
        Tupla de (X_transformado, nomes_features)
    """
    if preprocessor is not None:
        X_transformed = preprocessor.transform(df)
        feature_names = get_feature_names(preprocessor)
    else:
        X_transformed = df.values
        feature_names = list(df.columns)
    return X_transformed, feature_names


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
    logger.info("Getting feature importance", top_n=top_n)

    try:
        # Use existing get_feature_importance method
        importances_list = model_service.get_feature_importance()

        if not importances_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature importance not available for this model",
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
            detail=f"Failed to get feature importance: {str(e)}",
        )


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
        import shap
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="SHAP library not installed"
        )

    logger.info("Generating SHAP explanation")

    try:
        customer_id = request.customer_id
        features = dict(request.features)

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Get model, preprocessor and transform data
        model, preprocessor = _extract_model_and_preprocessor(model_service)
        X_transformed, feature_names = _transform_and_get_names(preprocessor, df)

        # Create appropriate SHAP explainer for the model type
        explainer = _create_shap_explainer(model, X_background=X_transformed)
        shap_values = explainer.shap_values(X_transformed)

        # Normalize SHAP output to positive class
        sample_shap, base_value = _extract_shap_values(
            shap_values, explainer.expected_value, single_instance=True
        )
        prediction_value = float(base_value + np.sum(sample_shap))

        # Build SHAP value list
        shap_list = []
        for i, (feat, sv) in enumerate(
            zip(feature_names[: len(sample_shap)], sample_shap, strict=False)
        ):
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

        logger.info("SHAP explanation generated successfully")

        return ShapExplanationResponse(success=True, explanation=explanation)

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SHAP requires additional dependencies",
        )
    except Exception as e:
        logger.error("SHAP explanation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation failed: {str(e)}",
        )


@router.get(
    "/shap/global",
    response_model=GlobalShapResponse,
    summary="Obter Resumo SHAP Global",
    description="Retorna valores SHAP absolutos médios de uma amostra de dados.",
)
async def get_global_shap(
    sample_size: int = Query(default=100, ge=10, le=500, description="Sample size for SHAP"),
    top_n: int = Query(default=20, ge=1, le=50, description="Number of top features"),
    model_service: ModelService = Depends(get_model_service),
) -> GlobalShapResponse:
    """Obtém estatísticas resumidas SHAP globais."""
    try:
        import shap
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="SHAP library not installed"
        )

    logger.info("Calculating global SHAP summary", sample_size=sample_size)

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
        X_transformed, feature_names = _transform_and_get_names(preprocessor, df)

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
                zip(feature_names[: len(mean_abs_shap)], mean_abs_shap, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )[:top_n]
        ):
            shap_importance.append(
                FeatureImportance(feature_name=feat, importance=round(float(val), 6), rank=i + 1)
            )

        logger.info("Global SHAP summary calculated")

        return GlobalShapResponse(success=True, mean_abs_shap=shap_importance, sample_size=len(df))

    except Exception as e:
        logger.error("Global SHAP calculation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Global SHAP calculation failed: {str(e)}",
        )
