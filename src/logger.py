"""
Sistema genérico de logging estruturado para aplicações ML.

Fornece logging em formato JSON para fácil parsing e análise.
Combina console (legível) e arquivo (estruturado).

Cada módulo pode usar o logger para registrar eventos específicos do seu domínio.

Uso:
    from src.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("Mensagem de teste")
    logger.error("Erro durante processamento", extra={'feature': 'value'})
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Formata logs em JSON para fácil parse e análise."""
    
    def format(self, record):
        """Converte log record em JSON estruturado."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Adicionar campos extras se presentes
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ('name', 'msg', 'args', 'created', 'filename', 'funcName', 
                               'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                               'message', 'pathname', 'process', 'processName', 'relativeCreated',
                               'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info'):
                    log_data[key] = value
        
        # Adicionar exceção se houver
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(name, log_file='logs/pipeline.log', level=logging.INFO):
    """
    Configura logger genérico com saída dupla (console + arquivo JSON).
    
    Args:
        name: Nome do logger (geralmente __name__)
        log_file: Caminho do arquivo de log (padrão logs/pipeline.log)
        level: Nível de logging (padrão INFO)
    
    Returns:
        logger: Logger configurado
    
    Exemplo:
        logger = setup_logger(__name__)
        logger.info("Processamento iniciado")
        logger.error("Erro durante treinamento", extra={'step': 'validation'})
    """
    
    # Criar logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar duplicação de handlers
    if logger.hasHandlers():
        return logger
    
    # Criar diretório de logs se não existir
    log_dir = os.path.dirname(log_file)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # ==================== FILE HANDLER (JSON) ====================
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"⚠️  Erro ao criar handler de arquivo: {e}")
    
    # ==================== CONSOLE HANDLER (Legível) ====================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formato legível para console
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


# ==================== EXEMPLO DE USO ====================
if __name__ == "__main__":
    # Setup
    logger = setup_logger(__name__)
    
    # Testes básicos
    logger.info("Pipeline iniciado com sucesso")
    logger.warning("Atenção: versão de dependência pode estar desatualizada")
    logger.error("Erro simulado para teste")
    
    # Com campos extras (aparecerão no JSON)
    logger.info("Treinamento iniciado", extra={
        'n_samples': 5000,
        'n_features': 20,
        'model_type': 'xgboost'
    })
    
    print("\n✅ Logs salvos em: logs/pipeline.log")
    print("Verifique o arquivo para ver o formato JSON")
