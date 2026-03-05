"""
Configuração de Logging
======================

Configuração de logging estruturado para a API com formatação JSON
para ambientes de produção.
"""

import logging
import sys
from datetime import UTC, datetime

import structlog
from structlog.types import EventDict, Processor

from api.core.config import settings


def add_timestamp(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Adiciona timestamp em formato ISO ao evento de log."""
    event_dict["timestamp"] = datetime.now(UTC).isoformat()
    return event_dict


def add_service_info(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Adiciona metadados do serviço ao evento de log."""
    event_dict["service"] = "churn-prediction-api"
    event_dict["version"] = settings.API_VERSION
    return event_dict


def setup_logging() -> None:
    """Configura logging estruturado para a aplicação."""

    # Definir processadores para structlog
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        add_timestamp,
        add_service_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.DEBUG:
        # Desenvolvimento: Saída formatada no console
        processors = shared_processors + [structlog.dev.ConsoleRenderer(colors=True)]
    else:
        # Produção: Saída JSON
        processors = shared_processors + [structlog.processors.JSONRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configurar logging padrão para usar structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.LOG_LEVEL),
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Obtém uma instância de logger configurada."""
    return structlog.get_logger(name)
