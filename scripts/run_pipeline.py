import json
import mlflow
import pandas as pd
from src.config import *
from src.preprocessing import split_data
from src.features import build_preprocessor
from src.train import train_model, save_model
from src.evaluate import evaluate

def main():
    # Configurar o tracking URI do MLflow para garantir que salve no diretório correto
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    
    # Configurar o experimento MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    # Iniciar uma run do MLflow
    with mlflow.start_run():
        df = pd.read_csv("data/processed/data.csv")

        num_features = df.select_dtypes(include="number").columns.drop(TARGET)
        cat_features = df.select_dtypes(include="object").columns

        X_train, X_test, y_train, y_test = split_data(
            df, TARGET, TEST_SIZE, RANDOM_STATE
        )

        preprocessor = build_preprocessor(num_features, cat_features)
        pipeline = train_model(preprocessor)

        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)

        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Passar um exemplo do X_test para o save_model registrar a signature
        save_model(pipeline, X_example=X_test)

if __name__ == "__main__":
    main()
