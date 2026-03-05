"""
Módulo de Avaliação de Modelo
=============================

Fornece métricas e relatórios abrangentes de avaliação de modelo.
Integra com MLflow para rastreamento de experimentos.
"""

from typing import Any

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.logger import setup_logger

logger = setup_logger(__name__)


def evaluate(
    model, X_test: pd.DataFrame, y_test: pd.Series, log_to_mlflow: bool = True
) -> dict[str, Any]:
    """
    Avalia performance do modelo nos dados de teste.

    Calcula métricas abrangentes incluindo ROC-AUC, precisão, recall,
    F1-score, e registra no MLflow se ativo.

    Args:
        model: Modelo ou pipeline treinado
        X_test: Features de teste
        y_test: Labels de teste
        log_to_mlflow: Se deve registrar métricas no MLflow

    Returns:
        Dicionário contendo todas as métricas de avaliação
    """
    logger.info(f"Starting evaluation with {X_test.shape[0]} samples")

    # Gerar predições
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    roc_auc = roc_auc_score(y_test, proba)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auprc = average_precision_score(y_test, proba)

    # Relatório de classificação
    report = classification_report(y_test, preds, output_dict=True)

    # Matriz de confusão
    cm = confusion_matrix(y_test, preds)

    # Registrar no MLflow
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auprc", auprc)

        # Registrar métricas por classe
        metrics_to_log = {}
        for class_label in ["0", "1"]:
            if class_label in report:
                for metric_name in ["precision", "recall", "f1-score"]:
                    key = f"class_{class_label}_{metric_name.replace('-', '_')}"
                    metrics_to_log[key] = report[class_label][metric_name]

        # Registrar métricas agregadas
        if "macro avg" in report:
            for metric_name in ["precision", "recall", "f1-score"]:
                key = f"macro_avg_{metric_name.replace('-', '_')}"
                metrics_to_log[key] = report["macro avg"][metric_name]

        if "weighted avg" in report:
            for metric_name in ["precision", "recall", "f1-score"]:
                key = f"weighted_avg_{metric_name.replace('-', '_')}"
                metrics_to_log[key] = report["weighted avg"][metric_name]

        mlflow.log_metrics(metrics_to_log)

    logger.info(
        "Evaluation completed",
        extra={
            "roc_auc": round(roc_auc, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auprc": round(auprc, 4),
        },
    )

    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auprc": auprc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
