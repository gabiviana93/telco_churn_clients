"""
Enumerações Compartilhadas
==========================

Tipos de enumeração centralizados para configuração do pipeline e monitoramento.
Valores categóricos específicos do domínio (gênero, contrato, etc.) são agora
definidos em config/project.yaml sob features.valid_values e gerados
dinamicamente em api/schemas/prediction.py.

Tipos de modelo disponíveis são definidos em ``model.registry`` (YAML).
O enum ``ModelType`` abaixo lista os modelos embutidos, mas modelos customizados
podem ser adicionados via registro sem modificar este arquivo.

Uso:
    from src.enums import ModelType, DriftSeverity
"""

from enum import Enum

# =============================================================================
# ENUMS DE CONFIGURAÇÃO DO PIPELINE
# =============================================================================


class ModelType(str, Enum):
    """Tipos de modelo embutidos.

    Modelos customizados podem ser adicionados via ``model.registry`` no
    config YAML sem modificar este enum. O pipeline e otimizador aceitam
    qualquer string registrada no registro YAML.
    """

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class DriftSeverity(str, Enum):
    """Níveis de severidade de drift baseados em limiares PSI."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
