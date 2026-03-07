"""
Rotas de Gerenciamento de Modelos
=================================

Endpoints para listar modelos disponíveis e trocar o modelo ativo.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.core.config import settings
from api.core.logging import get_logger
from api.services.model_service import ModelService, get_model_service
from src.config import list_available_models

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Model Management"])


class AvailableModel(BaseModel):
    """Informações de um modelo disponível."""

    name: str = Field(..., description="Nome do modelo (sem extensão)")
    filename: str = Field(..., description="Nome do arquivo")
    size_mb: float = Field(..., description="Tamanho em MB")
    is_default: bool = Field(..., description="Se é o modelo padrão")


class ModelsListResponse(BaseModel):
    """Resposta com lista de modelos disponíveis."""

    models: list[AvailableModel] = Field(..., description="Modelos disponíveis")
    active_model: str = Field(..., description="Nome do modelo ativo")
    total: int = Field(..., description="Total de modelos")


class SwitchModelRequest(BaseModel):
    """Requisição para trocar o modelo ativo."""

    model_name: str = Field(..., description="Nome do modelo a ativar (sem .joblib)")


class SwitchModelResponse(BaseModel):
    """Resposta de troca de modelo."""

    success: bool = Field(..., description="Se a troca foi bem-sucedida")
    active_model: str = Field(..., description="Stem do arquivo do modelo ativo")
    estimator_name: str = Field(..., description="Nome legível do estimador")
    message: str = Field(..., description="Mensagem de status")


@router.get(
    "/",
    response_model=ModelsListResponse,
    summary="Listar Modelos Disponíveis",
    description="Lista todos os modelos .joblib disponíveis no diretório de modelos.",
)
async def list_models(
    model_service: ModelService = Depends(get_model_service),
) -> ModelsListResponse:
    """Retorna todos os modelos disponíveis e qual está ativo."""
    models_raw = list_available_models()

    models = [
        AvailableModel(
            name=m["name"],
            filename=m["filename"],
            size_mb=m["size_mb"],
            is_default=m["is_default"],
        )
        for m in models_raw
    ]

    info = model_service.get_model_info()
    active = info.get("file_model_name") or settings.model_path_resolved.stem

    return ModelsListResponse(models=models, active_model=active, total=len(models))


@router.post(
    "/switch",
    response_model=SwitchModelResponse,
    summary="Trocar Modelo Ativo",
    description="Troca o modelo ativo da API para outro modelo disponível.",
)
async def switch_model(
    request: SwitchModelRequest,
    model_service: ModelService = Depends(get_model_service),
) -> SwitchModelResponse:
    """Troca o modelo ativo pelo nome informado."""
    model_name = request.model_name
    logger.info("Switching model", target_model=model_name)

    # Verifica se o modelo existe na lista
    available = list_available_models()
    available_names = [m["name"] for m in available]
    if model_name not in available_names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "MODEL_NOT_FOUND",
                "message": f"Modelo '{model_name}' não encontrado. Disponíveis: {available_names}",
            },
        )

    try:
        model_service.reload_model(model_name)
    except Exception as e:
        logger.error("Failed to switch model", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "MODEL_RELOAD_ERROR",
                "message": f"Erro ao carregar modelo: {e}",
            },
        ) from e

    info = model_service.get_model_info()
    active = info.get("file_model_name") or model_name
    return SwitchModelResponse(
        success=True,
        active_model=active,
        estimator_name=info.get("model_name", "Unknown"),
        message=f"Modelo trocado para '{model_name}' com sucesso.",
    )
