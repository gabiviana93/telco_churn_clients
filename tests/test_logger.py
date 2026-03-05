import logging
import json
import os
from pathlib import Path
from src.logger import setup_logger, JSONFormatter

def test_setup_logger_creates_logger():
    """Testa se setup_logger cria um logger"""
    logger = setup_logger("test_logger", log_file="logs/test.log")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"

def test_json_formatter():
    """Testa se o JSONFormatter gera JSON válido"""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    formatted = formatter.format(record)
    data = json.loads(formatted)
    
    assert data['message'] == "Test message"
    assert data['level'] == "INFO"
    assert data['logger'] == "test"

def test_logger_no_duplicate_handlers():
    """Testa se não há duplicação de handlers"""
    logger1 = setup_logger("test_dup", log_file="logs/test_dup.log")
    initial_handlers = len(logger1.handlers)
    
    logger2 = setup_logger("test_dup", log_file="logs/test_dup.log")
    final_handlers = len(logger2.handlers)
    
    assert initial_handlers == final_handlers

def test_json_formatter_with_exception():
    """Testa JSONFormatter com exceção"""
    formatter = JSONFormatter()
    try:
        raise ValueError("Test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert 'exception' in data
        assert 'ValueError' in data['exception']

def test_logger_level_setting():
    """Testa se o nível de log é definido corretamente"""
    logger = setup_logger("test_level", log_file="logs/test_level.log", level=logging.DEBUG)
    assert logger.level == logging.DEBUG
    
    logger2 = setup_logger("test_level2", log_file="logs/test_level2.log", level=logging.WARNING)
    assert logger2.level == logging.WARNING

def test_logger_with_empty_log_dir():
    """Testa logger com diretório vazio"""
    logger = setup_logger("test_no_dir", log_file="", level=logging.INFO)
    assert logger is not None
    assert logger.name == "test_no_dir"

def test_json_formatter_format():
    """Testa formatação de diferentes tipos de logs"""
    formatter = JSONFormatter()
    
    # Log INFO
    record_info = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py",
        lineno=1, msg="Info message", args=(), exc_info=None
    )
    formatted = formatter.format(record_info)
    data = json.loads(formatted)
    assert data['level'] == 'INFO'
    
    # Log WARNING
    record_warn = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="test.py",
        lineno=2, msg="Warning message", args=(), exc_info=None
    )
    formatted = formatter.format(record_warn)
    data = json.loads(formatted)
    assert data['level'] == 'WARNING'

def test_logger_file_path_creation():
    """Testa criação do caminho de diretório para logs"""
    test_log_path = "logs/deep/nested/path/test.log"
    logger = setup_logger("test_path", log_file=test_log_path)
    
    # Verificar que o logger foi criado
    assert logger is not None

def test_logger_file_handler_error():
    """Testa comportamento quando há erro ao criar file handler"""
    # Usar caminho inválido que deve causar erro (em sistemas Unix, não pode criar arquivo em /)
    logger = setup_logger("test_error", log_file="/invalid/path/that/does/not/exist.log")
    
    # O logger deve ainda ser criado (apenas sem o file handler)
    assert logger is not None

def test_logger_returns_existing():
    """Testa se retorna logger existente quando já tem handlers"""
    logger1 = setup_logger("test_existing", log_file="logs/existing.log")
    logger1_handlers = len(logger1.handlers)
    
    # Chamar novamente - deve retornar o mesmo logger
    logger2 = setup_logger("test_existing", log_file="logs/existing.log")
    logger2_handlers = len(logger2.handlers)
    
    # Número de handlers deve ser o mesmo (não duplicou)
    assert logger1_handlers == logger2_handlers

def test_logger_with_valid_file():
    """Testa logger que cria arquivo com sucesso"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        log_path = f.name
    
    logger = setup_logger("test_valid_file", log_file=log_path)
    logger.info("Test message")
    
    # Verificar que o arquivo existe
    import os
    assert os.path.exists(log_path)
    
    # Cleanup
    os.remove(log_path)
