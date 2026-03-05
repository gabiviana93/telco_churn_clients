"""
Rotas de Saúde e Monitoramento
==============================

Endpoints da API para verificação de saúde e monitoramento.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.core.config import settings
from api.core.logging import get_logger
from api.schemas.prediction import (
    HealthResponse,
    HealthStatus,
    ModelInfo,
    ModelMetrics,
)
from api.services.model_service import ModelService, get_model_service

logger = get_logger(__name__)

router = APIRouter(tags=["Health & Monitoring"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificação de Saúde",
    description="Verifica o status de saúde da API e do modelo.",
)
async def health_check(model_service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """
    Realiza verificação de saúde.

    Returns:
    - **status**: Status geral de saúde (healthy/degraded/unhealthy)
    - **model_loaded**: Se o modelo ML está carregado
    - **uptime_seconds**: Tempo de atividade da API
    """
    if model_service.is_loaded:
        status = HealthStatus.HEALTHY
    else:
        status = HealthStatus.DEGRADED

    return HealthResponse(
        status=status,
        model_loaded=model_service.is_loaded,
        model_version=model_service.model_version,
        uptime_seconds=model_service.uptime_seconds,
    )


@router.get(
    "/health/live", summary="Probe de Liveness", description="Endpoint de probe de liveness para Kubernetes."
)
async def liveness():
    """Verificação simples de liveness para Kubernetes."""
    return {"status": "alive"}


@router.get(
    "/health/ready", summary="Probe de Readiness", description="Endpoint de probe de readiness para Kubernetes."
)
async def readiness(model_service: ModelService = Depends(get_model_service)):
    """Verificação de readiness - confirma que o modelo está carregado."""
    if model_service.is_loaded:
        return {"status": "ready"}
    return JSONResponse(
        status_code=503,
        content={"status": "not ready", "reason": "model not loaded"},
    )


@router.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Informações do Modelo",
    description="Obtém informações detalhadas sobre o modelo carregado.",
)
async def model_info(model_service: ModelService = Depends(get_model_service)) -> ModelInfo:
    """
    Obtém metadados e configuração do modelo.

    Returns:
    - Nome e versão do modelo
    - Data de treinamento
    - Informações de features
    - Métricas de performance
    - Hiperparâmetros
    """
    info = model_service.get_model_info()
    features = info.get("features", {})

    # Construir métricas se disponíveis
    metrics_data = info.get("metrics", {})
    metrics = None
    if metrics_data:
        metrics = ModelMetrics(
            roc_auc=metrics_data.get("roc_auc", 0.0),
            accuracy=metrics_data.get("accuracy", 0.0),
            precision=metrics_data.get("precision", 0.0),
            recall=metrics_data.get("recall", 0.0),
            f1_score=metrics_data.get("f1_score", 0.0),
        )

    # Processar data de treinamento
    trained_at = None
    if info.get("trained_at"):
        try:
            trained_at = datetime.fromisoformat(str(info["trained_at"]))
        except (ValueError, TypeError):
            pass

    return ModelInfo(
        model_name=info.get("model_name", "Unknown"),
        model_version=info.get("version", settings.MODEL_VERSION),
        trained_at=trained_at,
        features_count=features.get("n_input_features", 0),
        feature_names=features.get("input_features", []),
        metrics=metrics,
        hyperparameters=info.get("hyperparameters"),
    )
