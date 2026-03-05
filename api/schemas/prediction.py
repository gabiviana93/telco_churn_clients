"""
Schemas Pydantic para API de Predição de Churn
===============================================

Modelos de requisição e resposta com validação abrangente
para os endpoints de predição de churn.

CustomerFeatures é gerado dinamicamente a partir de config/project.yaml,
então mudar datasets requer apenas editar o YAML — sem alterações no código.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model, field_validator

from src.config import API_BATCH_MAX_SIZE, API_METRIC_PRECISION, get_config


def _utc_now() -> datetime:
    """Retorna datetime UTC atual (evita datetime.utcnow depreciado)."""
    return datetime.now(UTC)


# =============================================================================
# Geração dinâmica de CustomerFeatures a partir do config YAML
# =============================================================================


def _build_customer_features_model() -> type[BaseModel]:
    """Constrói modelo CustomerFeatures dinamicamente do config YAML.

    Lê features.numeric, features.categorical, features.valid_values
    e features.numeric_ranges do config do projeto para gerar
    um modelo Pydantic com anotações de tipo e validação adequadas.
    """
    cfg = get_config()
    features_cfg = cfg.get("features", {})

    numeric = features_cfg.get("numeric", [])
    categorical = features_cfg.get("categorical", [])
    valid_values = features_cfg.get("valid_values", {})
    numeric_ranges = features_cfg.get("numeric_ranges", {})

    fields: dict[str, Any] = {}
    example: dict[str, Any] = {}

    # Features numéricas: int ou float com restrições ge/le de numeric_ranges
    for feat in numeric:
        ranges = numeric_ranges.get(feat, {})
        py_type = int if ranges.get("type") == "int" else float

        field_kwargs: dict[str, Any] = {"description": feat}
        if "min" in ranges:
            field_kwargs["ge"] = ranges["min"]
        if "max" in ranges:
            field_kwargs["le"] = ranges["max"]

        fields[feat] = (py_type, Field(..., **field_kwargs))
        example[feat] = ranges.get("default", ranges.get("min", 0))

    # Features categóricas: tipos Literal de valid_values
    for feat in categorical:
        vals = valid_values.get(feat)
        if vals:
            all_int = all(isinstance(v, int) for v in vals)
            if all_int:
                # Categórica inteira (ex.: SeniorCitizen: [0, 1])
                literal_type = Literal.__getitem__(tuple(vals))
            else:
                # Categórica string
                literal_type = Literal.__getitem__(tuple(str(v) for v in vals))
            fields[feat] = (literal_type, Field(..., description=feat))
            example[feat] = vals[0]

    model = create_model("CustomerFeatures", **fields)
    model.model_config["json_schema_extra"] = {"example": example}

    return model


# Construir uma vez no import (config é cacheada via @lru_cache)
CustomerFeatures = _build_customer_features_model()


# =============================================================================
# Modelos de Requisição
# =============================================================================


class PredictionRequest(BaseModel):
    """Requisição de predição para um único cliente."""

    customer_id: str | None = Field(
        None, description="Optional customer identifier for tracking", examples=["CUST-001"]
    )
    features: CustomerFeatures = Field(..., description="Customer features for prediction")


class BatchPredictionRequest(BaseModel):
    """Requisição de predição em lote para múltiplos clientes."""

    customers: list[PredictionRequest] = Field(
        ..., min_length=1, max_length=API_BATCH_MAX_SIZE, description="List of customers for batch prediction"
    )


# =============================================================================
# Modelos de Resposta
# =============================================================================


class PredictionResult(BaseModel):
    """Resultado de predição individual."""

    customer_id: str | None = Field(None, description="Customer identifier")
    churn_prediction: int = Field(
        ..., ge=0, le=1, description="Binary churn prediction (0=No, 1=Yes)"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of churn (0.0 to 1.0)"
    )
    churn_risk: str = Field(..., description="Risk category based on probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in the prediction")

    @field_validator("churn_probability", "confidence", mode="before")
    @classmethod
    def round_float(cls, v: float) -> float:
        """Arredonda valores float para casas decimais configuradas."""
        return round(v, API_METRIC_PRECISION) if isinstance(v, float) else v


class PredictionResponse(BaseModel):
    """Resposta para requisição de predição individual."""

    success: bool = Field(True, description="Whether the prediction was successful")
    prediction: PredictionResult = Field(..., description="Prediction result")
    model_version: str = Field(..., description="Version of the model used")
    timestamp: datetime = Field(default_factory=_utc_now, description="Timestamp of prediction")


class BatchPredictionResponse(BaseModel):
    """Resposta para requisição de predição em lote."""

    success: bool = Field(True, description="Whether all predictions were successful")
    predictions: list[PredictionResult] = Field(..., description="List of prediction results")
    total_customers: int = Field(..., description="Total number of customers processed")
    high_risk_count: int = Field(..., description="Number of customers with high churn risk")
    model_version: str = Field(..., description="Version of the model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=_utc_now, description="Timestamp of prediction")


# =============================================================================
# Modelos de Saúde e Monitoramento
# =============================================================================


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Resposta de verificação de saúde."""

    status: HealthStatus = Field(..., description="Overall health status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    model_version: str = Field(..., description="Loaded model version")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    timestamp: datetime = Field(default_factory=_utc_now, description="Health check timestamp")


class ModelMetrics(BaseModel):
    """Métricas de performance do modelo."""

    roc_auc: float = Field(..., description="ROC-AUC score")
    accuracy: float = Field(..., description="Accuracy score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")


class ModelInfo(BaseModel):
    """Informações e metadados do modelo."""

    model_name: str = Field(..., description="Model algorithm name")
    model_version: str = Field(..., description="Model version")
    trained_at: datetime | None = Field(None, description="Training timestamp")
    features_count: int = Field(..., description="Number of input features")
    feature_names: list[str] = Field(..., description="List of feature names")
    metrics: ModelMetrics | None = Field(None, description="Model performance metrics")
    hyperparameters: dict[str, Any] | None = Field(None, description="Model hyperparameters")


class FeatureImportance(BaseModel):
    """Scores de importância de features."""

    feature_name: str = Field(..., description="Name of the feature")
    importance: float = Field(..., description="Importance score")
    rank: int = Field(..., description="Rank by importance")


# =============================================================================
# Modelos de Erro
# =============================================================================


class ErrorResponse(BaseModel):
    """Resposta padrão de erro."""

    success: bool = Field(False, description="Always False for errors")
    error_code: str = Field(..., description="Error code for debugging")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=_utc_now, description="Error timestamp")
