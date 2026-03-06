"""Pipeline de treino e avaliação do modelo de churn.

Executa o pipeline completo: carrega dados, treina modelo,
avalia métricas e salva artefatos.

Uso:
    poetry run python scripts/run_pipeline.py
"""

import json

import mlflow
import pandas as pd

from src.config import (
    DATA_DIR_RAW,
    FILENAME,
    ID_COL,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET,
)
from src.pipeline import ChurnPipeline
from src.preprocessing import split_data


def main():
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        # 1. Carregar dados
        df = pd.read_csv(DATA_DIR_RAW / FILENAME)
        df = df.drop(columns=[ID_COL])

        # 2. Split treino/teste
        X_train, X_test, y_train, y_test = split_data(df, TARGET)

        # 3. Treinar pipeline (com cross-validation)
        pipeline = ChurnPipeline()
        pipeline.fit(X_train, y_train, validate=True)

        # 4. Avaliar no conjunto de teste
        metrics = pipeline.evaluate(X_test, y_test)

        # Adicionar métricas de CV ao resultado (prefixadas para evitar colisão)
        for key, value in pipeline.metrics.items():
            if key not in metrics:
                metrics[key] = value

        # Logar métricas no MLflow
        mlflow.log_metrics(metrics)

        # 5. Salvar métricas em JSON
        with open(REPORTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # 6. Salvar modelo
        model_path = MODELS_DIR / "model.joblib"
        pipeline.save(model_path)

        print(f"Pipeline concluído. Métricas: {metrics}")


if __name__ == "__main__":
    main()
