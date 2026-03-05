from joblib import load
from src.config import MODEL_PATH
from src.logger import setup_logger

logger = setup_logger(__name__)

def load_model():
    logger.info("Carregando modelo", extra={'model_path': MODEL_PATH})
    model = load(MODEL_PATH)
    logger.info("Modelo carregado com sucesso")
    return model

def predict(model, X):
    logger.info(f"Realizando inferência em {X.shape[0]} amostras")
    predictions = model.predict(X)
    logger.info("Inferência concluída")
    return predictions

def predict_proba(model, X):
    if hasattr(model, 'predict_proba'):
        logger.info(f"Calculando probabilidades para {X.shape[0]} amostras")
        probas = model.predict_proba(X)
        logger.info("Cálculo de probabilidades concluído")
        return probas
    else:
        logger.error("Modelo não suporta predict_proba")
        raise AttributeError("Esse modelo não suporta predição de probabilidades.")