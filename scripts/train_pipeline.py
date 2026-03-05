import mlflow
import pandas as pd

from src.config import (
    DATA_DIR_RAW,
    FILENAME,
    ID_COL,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    TARGET,
    ModelConfig,
)
from src.pipeline import ChurnPipeline
from src.preprocessing import split_data


def main():
    # 1. Carregar dados
    df = pd.read_csv(DATA_DIR_RAW / FILENAME)
    df = df.drop(columns=[ID_COL])

    # 2. Split treino/teste
    X_train, X_test, y_train, y_test = split_data(df, TARGET)

    # 3. Configurar modelo (parâmetros do YAML ou customizados)
    config = ModelConfig()

    # 4. Configurar MLflow
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        # 5. Treinar com cross-validation
        pipeline = ChurnPipeline(config=config)
        pipeline.fit(X_train, y_train, validate=True)

        # Logar parâmetros e métricas no MLflow
        mlflow.log_params(config.to_dict())
        mlflow.log_metrics(pipeline.metrics)

        # 6. Avaliar no conjunto de teste
        metrics = pipeline.evaluate(X_test, y_test)
        mlflow.log_metrics(metrics)

        # 7. Salvar modelo
        model_path = MODELS_DIR / "model.joblib"
        pipeline.save(model_path)

        print(f"CV Metrics: {pipeline.metrics}")
        print(f"Test Metrics: {metrics}")
        print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
