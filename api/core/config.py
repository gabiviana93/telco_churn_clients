"""
Configurações da API
===================

Configuração centralizada usando Pydantic settings para
gerenciamento e validação de variáveis de ambiente.

Importa valores padrão de src.config para consistência.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Importar valores centralizados de configuração
from src.config import (
    API_HOST,
    API_PORT,
    API_TITLE,
    API_VERSION,
    DEFAULT_MODEL_NAME,
    get_config,
    get_model_path,
    list_available_models,
)

# Carregar config YAML para valores não exportados por src.config
_yaml_config = get_config()
_api_config = _yaml_config.get("api", {})
_logging_config = _yaml_config.get("logging", {})
_cors_config = _api_config.get("cors", {})
_rate_limit_config = _api_config.get("rate_limit", {})


class Settings(BaseSettings):
    """Configurações da aplicação com suporte a variáveis de ambiente."""

    # Configurações da API (padrões de src.config)
    API_TITLE: str = API_TITLE
    API_DESCRIPTION: str = _api_config.get(
        "description",
        """
    **Customer Churn Prediction API**

    Machine Learning API for predicting customer churn in telecommunications.
    """,
    )
    API_VERSION: str = API_VERSION
    DEBUG: bool = _api_config.get("debug", False)

    # Configurações do Servidor (padrões de src.config)
    HOST: str = API_HOST
    PORT: int = API_PORT
    WORKERS: int = _api_config.get("workers", 4)

    # Configurações CORS
    CORS_ORIGINS: list[str] = (
        _cors_config.get("origins", ["*"]) if isinstance(_cors_config, dict) else ["*"]
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    # Configurações do Modelo
    MODEL_NAME: str = DEFAULT_MODEL_NAME  # Nome do modelo (sem .joblib)
    MODEL_PATH: str | None = None  # Sobrescrever caminho completo se necessário
    MODEL_VERSION: str = API_VERSION

    # Logging
    LOG_LEVEL: str = _logging_config.get("level", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Limitação de taxa
    RATE_LIMIT_REQUESTS: int = (
        _rate_limit_config.get("requests", 100) if isinstance(_rate_limit_config, dict) else 100
    )
    RATE_LIMIT_PERIOD: int = (
        _rate_limit_config.get("period", 60) if isinstance(_rate_limit_config, dict) else 60
    )

    # Monitoramento
    ENABLE_METRICS: bool = True
    METRICS_PATH: str = "/metrics"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @property
    def model_path_resolved(self) -> Path:
        """Obtém o caminho resolvido do modelo."""
        if self.MODEL_PATH:
            return Path(self.MODEL_PATH)
        # Usar função centralizada para obter caminho do modelo
        return get_model_path(self.MODEL_NAME)

    @property
    def available_models(self) -> list[dict]:
        """Lista modelos disponíveis."""
        return list_available_models()


@lru_cache
def get_settings() -> Settings:
    """
    Obtém instância de configurações com cache.

    Usa LRU cache para evitar leitura de variáveis de ambiente
    múltiplas vezes durante o tratamento de requisições.
    """
    return Settings()


settings = get_settings()
