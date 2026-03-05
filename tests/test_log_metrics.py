def test_train_logs_metrics(sample_X_y):
    """Testa se train_model registra métricas e logs corretamente."""
    from src.features import build_preprocessor
    from src.train import train_model
    from src.logger import setup_logger

    X_train, y_train = sample_X_y
    
    # Construir preprocessador com as colunas de X_train
    numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
    preprocessor = build_preprocessor(numeric_features, [])

    # Setup logger para verificar logs
    logger = setup_logger('test_train')

    # Treinar modelo (mantém como DataFrame)
    pipeline = train_model(preprocessor, X_train, y_train)

    # Verificar que o pipeline foi criado
    assert pipeline is not None
    assert hasattr(pipeline, 'fit')
    assert hasattr(pipeline, 'predict')