import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
from joblib import dump

from src.config import MODEL_PATH, MODEL_PARAMS
from src.logger import setup_logger

logger = setup_logger(__name__)


def train_model(
    preprocessor,
    X_train=None,
    y_train=None,
    params=None,
    experiment_name="default",
    run_name=None,
):
    """
    Treina um modelo XGBoost com pipeline de pré-processamento e MLflow tracking.
    """

    if params is None:
        params = MODEL_PARAMS

    mlflow.set_experiment(experiment_name)
    
    # Fechar qualquer run ativa anterior
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model = XGBClassifier(**params)

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model),
            ]
        )

        if X_train is not None and y_train is not None:
            logger.info(
                "Iniciando treinamento",
                extra={
                    "n_samples": X_train.shape[0],
                    "n_features": X_train.shape[1],
                    "model_type": "XGBClassifier",
                },
            )

            pipeline.fit(X_train, y_train)
            mlflow.log_metric("training_samples", X_train.shape[0])

            logger.info("Treinamento concluído")

        return pipeline


def cross_validate_model(
    pipeline,
    X_train,
    y_train,
    n_splits=5,
    log_mlflow=True,
):
    """
    Realiza validação cruzada estratificada e loga métricas no MLflow.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "f1_weighted": "f1_weighted",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
    }

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    summary = {}

    for metric in scoring:
        summary[metric] = {
            "test_mean": float(cv_results[f"test_{metric}"].mean()),
            "test_std": float(cv_results[f"test_{metric}"].std()),
            "train_mean": float(cv_results[f"train_{metric}"].mean()),
            "train_std": float(cv_results[f"train_{metric}"].std()),
        }

    if log_mlflow and mlflow.active_run():
        for metric, stats in summary.items():
            mlflow.log_metric(
                f"cv_{metric}_test_mean", stats["test_mean"]
            )
            mlflow.log_metric(
                f"cv_{metric}_test_std", stats["test_std"]
            )

    return summary


def save_model(
    model,
    model_path=MODEL_PATH,
    model_name="xgboost-pipeline",
    X_example=None,
):
    """
    Salva o pipeline treinado localmente e registra no MLflow.
    """

    dump(model, model_path)

    if mlflow.active_run():
        mlflow.set_tag("model_type", "xgboost_classifier")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("pipeline", "preprocessing + model")

    mlflow.log_artifact(model_path, artifact_path="model")

    mlflow.sklearn.log_model(
        model,
        artifact_path="sklearn-model",
        registered_model_name=model_name,
        input_example=(
            X_example.iloc[:1]
            if hasattr(X_example, "iloc")
            else X_example[:1]
            if X_example is not None
            else None
        ),
    )
    logger.info("Modelo salvo e registrado no MLflow", extra={'model_path': model_path})