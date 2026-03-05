"""
Rotas de Predição
=================

Endpoints da API para predições de churn.
"""

import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from api.core.logging import get_logger
from api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from api.services.model_service import ModelService, get_model_service

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Predictions"])


def features_to_dataframe(features) -> pd.DataFrame:
    """Converte CustomerFeatures para DataFrame."""
    return pd.DataFrame([features.model_dump()])


@router.post(
    "/",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction error"},
    },
    summary="Prever Churn de Cliente",
    description="""
    Prevê se um cliente vai cancelar com base em suas features.

    Retorna a predição binária, probabilidade de churn, categoria de risco
    e confiança do modelo.
    """,
)
async def predict_single(
    request: PredictionRequest, model_service: ModelService = Depends(get_model_service)
) -> PredictionResponse:
    """
    Realiza predição de churn para um único cliente.

    - **customer_id**: Identificador opcional para rastreamento
    - **features**: Features do cliente para predição
    """
    logger.info("Processing prediction request", customer_id=request.customer_id)

    try:
        # Convert features to DataFrame
        df = features_to_dataframe(request.features)

        # Make prediction
        predictions, probabilities = model_service.predict(df)

        prediction = int(predictions[0])
        probability = float(probabilities[0])

        result = PredictionResult(
            customer_id=request.customer_id,
            churn_prediction=prediction,
            churn_probability=probability,
            churn_risk=model_service.classify_risk(probability),
            confidence=model_service.calculate_confidence(probability),
        )

        logger.info(
            "Prediction completed",
            customer_id=request.customer_id,
            churn_prediction=prediction,
            churn_probability=probability,
        )

        return PredictionResponse(
            success=True, prediction=result, model_version=model_service.model_version
        )

    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "PREDICTION_ERROR",
                "message": f"Failed to make prediction: {str(e)}",
            },
        ) from e


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    responses={
        200: {"description": "Successful batch prediction"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Batch prediction error"},
    },
    summary="Predição em Lote de Churn",
    description="""
    Prevê churn para múltiplos clientes em uma única requisição.

    Tamanho máximo do lote: 1000 clientes.
    Retorna predições para todos os clientes com estatísticas resumidas.
    """,
)
async def predict_batch(
    request: BatchPredictionRequest, model_service: ModelService = Depends(get_model_service)
) -> BatchPredictionResponse:
    """
    Realiza predições de churn em lote para múltiplos clientes.

    - **customers**: Lista de requisições de predição de clientes (máx 1000)
    """
    start_time = time.time()

    logger.info("Processing batch prediction", batch_size=len(request.customers))

    try:
        # Convert all features to DataFrames and concatenate
        dfs = []
        customer_ids = []

        for customer in request.customers:
            df = features_to_dataframe(customer.features)
            dfs.append(df)
            customer_ids.append(customer.customer_id)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Make batch predictions
        predictions, probabilities = model_service.predict(combined_df)

        # Build results
        results: list[PredictionResult] = []
        high_risk_count = 0

        for _i, (pred, prob, cust_id) in enumerate(
            zip(predictions, probabilities, customer_ids, strict=False)
        ):
            prediction = int(pred)
            probability = float(prob)
            risk = model_service.classify_risk(probability)

            if risk == "HIGH":
                high_risk_count += 1

            results.append(
                PredictionResult(
                    customer_id=cust_id,
                    churn_prediction=prediction,
                    churn_probability=probability,
                    churn_risk=risk,
                    confidence=model_service.calculate_confidence(probability),
                )
            )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "Batch prediction completed",
            batch_size=len(request.customers),
            high_risk_count=high_risk_count,
            processing_time_ms=processing_time,
        )

        return BatchPredictionResponse(
            success=True,
            predictions=results,
            total_customers=len(results),
            high_risk_count=high_risk_count,
            model_version=model_service.model_version,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "BATCH_PREDICTION_ERROR",
                "message": f"Failed to make batch prediction: {str(e)}",
            },
        ) from e
