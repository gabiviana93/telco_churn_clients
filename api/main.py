"""
API de Predição de Churn de Clientes
====================================

Aplicação FastAPI para predições de churn de clientes em tempo real.

Esta API fornece:
- Predições de churn individuais e em lote
- Monitoramento de saúde do modelo
- Análise de importância de features
- Documentação abrangente via OpenAPI
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.config import settings
from api.core.logging import get_logger, setup_logging
from api.routes import drift, health, interpret, models, prediction
from api.services.model_service import model_service

# Configurar logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handler de ciclo de vida da aplicação.

    Gerencia eventos de startup e shutdown.
    """
    # Inicialização
    logger.info(
        "Starting Customer Churn Prediction API", version=settings.API_VERSION, debug=settings.DEBUG
    )

    # Garantir que o modelo está carregado
    if model_service.is_loaded:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded - running in demo mode")

    yield

    # Encerramento
    logger.info("Shutting down API")


# Criar aplicação FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "Predictions", "description": "Customer churn prediction endpoints"},
        {"name": "Model Management", "description": "List and switch active models"},
        {"name": "Data Drift", "description": "Data drift detection and monitoring"},
        {"name": "Health & Monitoring", "description": "API health checks and model monitoring"},
    ],
)


# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# Incluir routers
app.include_router(prediction.router)
app.include_router(models.router)
app.include_router(drift.router)
app.include_router(health.router)
app.include_router(interpret.router)


# Endpoint raiz
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz da API com informações básicas."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "documentation": {"swagger": "/docs", "redoc": "/redoc", "openapi": "/openapi.json"},
        "endpoints": {
            "predict": "/predict/",
            "batch_predict": "/predict/batch",
            "drift_detect": "/drift/detect",
            "drift_status": "/drift/status",
            "health": "/health",
            "models_list": "/models/",
            "models_switch": "/models/switch",
            "model_info": "/model/info",
            "feature_importance": "/interpret/feature-importance",
        },
    }


# Handler global de exceções
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Trata exceções não capturadas."""
    logger.error(
        "Unhandled exception", error=str(exc), path=request.url.path, method=request.method
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


# Entry point for running directly
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
