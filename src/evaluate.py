import mlflow
from sklearn.metrics import roc_auc_score, classification_report
from src.logger import setup_logger

logger = setup_logger(__name__)

def evaluate(model, X_test, y_test):
    logger.info(f"Iniciando avaliação com {X_test.shape[0]} amostras")
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)

    mlflow.log_metric("roc_auc", roc_auc)
    
    # Extrair apenas as métricas numéricas para logar
    metrics_to_log = {}
    for class_label in ['0', '1']:
        if class_label in report:
            for metric_name in ['precision', 'recall', 'f1-score']:
                key = f"{class_label}_{metric_name}"
                metrics_to_log[key] = report[class_label][metric_name]
    
    # Log das métricas agregadas
    if 'macro avg' in report:
        for metric_name in ['precision', 'recall', 'f1-score']:
            key = f"macro_avg_{metric_name}"
            metrics_to_log[key] = report['macro avg'][metric_name]
    
    if 'weighted avg' in report:
        for metric_name in ['precision', 'recall', 'f1-score']:
            key = f"weighted_avg_{metric_name}"
            metrics_to_log[key] = report['weighted avg'][metric_name]
    
    mlflow.log_metrics(metrics_to_log)
    
    logger.info("Avaliação concluída", extra={
        'roc_auc': roc_auc,
        'precision_weighted': report['weighted avg']['precision'],
        'recall_weighted': report['weighted avg']['recall'],
        'f1_weighted': report['weighted avg']['f1-score']
    })

    metrics = {
        "roc_auc": roc_auc,
        "accuracy": report['accuracy'],
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1_score": report['weighted avg']['f1-score'],
        "report": report
    }

    return metrics