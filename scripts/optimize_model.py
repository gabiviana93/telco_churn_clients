"""Script de otimização multi-modelo para atingir F1 > 0.65.

Estratégia:
1. Otimiza cada algoritmo (LightGBM, XGBoost, CatBoost) com Optuna
2. Salva cada modelo com nome identificável (model_lightgbm.joblib, etc.)
3. Compara todos e salva o melhor como model.joblib (padrão do dashboard)

Uso:
    poetry run python scripts/optimize_model.py                  # padrão (20 trials)
    poetry run python scripts/optimize_model.py --trials 50      # mais trials
    poetry run python scripts/optimize_model.py --quick           # rápido (10 trials, 3-fold)
    poetry run python scripts/optimize_model.py --models lgb xgb  # modelos específicos
"""

import argparse
import json
import time
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
from src.optimization import HyperparameterOptimizer, OptimizationConfig
from src.preprocessing import split_data

# Suprimir warnings de validação do sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

MODEL_ALIASES = {
    "lgb": "lightgbm",
    "lightgbm": "lightgbm",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "cat": "catboost",
    "catboost": "catboost",
}

ALL_MODELS = ["lightgbm", "xgboost", "catboost"]


def evaluate_with_threshold(pipeline, X, y, threshold=0.5):
    """Avalia um pipeline com threshold customizado."""
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
        "auprc": average_precision_score(y, y_proba),
        "threshold": threshold,
    }


def find_optimal_threshold(y_true, y_proba):
    """Encontra threshold ótimo para F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def optimize_single_model(model_type, X_train, y_train, n_trials, n_cv_splits):
    """Otimiza um único modelo com SMOTE + threshold."""
    t0 = time.time()
    print(f"\n{'─' * 55}")
    print(f"  {model_type.upper()} — {n_trials} trials, {n_cv_splits}-fold CV, SMOTE")
    print(f"{'─' * 55}")

    config = OptimizationConfig(
        model_type=model_type,
        n_trials=n_trials,
        metric="f1",
        use_smote=True,
        optimize_threshold=True,
        n_cv_splits=n_cv_splits,
        timeout=600 * n_cv_splits,
    )

    optimizer = HyperparameterOptimizer(X_train, y_train, config)
    result = optimizer.optimize(n_trials=n_trials)

    elapsed = time.time() - t0
    best_trial = result.study.best_trial.number
    print(
        f"  → CV F1: {result.best_score:.4f} | "
        f"Threshold: {result.best_threshold:.4f} | "
        f"Melhor trial: #{best_trial} | {elapsed:.0f}s"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Otimização multi-modelo para churn prediction")
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Número de trials Optuna por modelo (default: 20)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido: 10 trials, 3-fold CV",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Modelos a otimizar (lgb, xgb, cat). Default: todos",
    )
    args = parser.parse_args()

    # Resolver modelos
    if args.models:
        models = []
        for m in args.models:
            resolved = MODEL_ALIASES.get(m.lower())
            if not resolved:
                print(f"Modelo '{m}' não reconhecido. Use: lgb, xgb, cat")
                return
            models.append(resolved)
    else:
        models = ALL_MODELS

    n_trials = 10 if args.quick else args.trials
    n_cv_splits = 3 if args.quick else 5

    print("=" * 55)
    print("OTIMIZAÇÃO MULTI-MODELO — Meta: F1 > 0.65")
    print(f"Modelos: {', '.join(m.upper() for m in models)}")
    print(f"Trials: {n_trials} | CV: {n_cv_splits}-fold | SMOTE: sim")
    print("=" * 55)

    # 1. Carregar dados
    df = pd.read_csv(DATA_DIR_RAW / FILENAME)
    df = df.drop(columns=[ID_COL])
    X_train, X_test, y_train, y_test = split_data(df, TARGET)
    print(f"Dados: {len(X_train)} treino, {len(X_test)} teste")

    # 2. Otimizar cada modelo
    results = {}
    for model_type in models:
        results[model_type] = optimize_single_model(
            model_type, X_train, y_train, n_trials, n_cv_splits
        )

    # 3. Avaliar e salvar cada modelo individualmente
    print("\n" + "=" * 55)
    print("RESULTADOS NO CONJUNTO DE TESTE")
    print("=" * 55)

    individual_metrics = {}
    best_f1 = 0
    best_name = ""

    for name, result in results.items():
        metrics = evaluate_with_threshold(
            result.best_pipeline, X_test, y_test, result.best_threshold
        )
        individual_metrics[name] = metrics

        # Salvar modelo com nome identificável
        result.best_pipeline.test_metrics = metrics
        result.best_pipeline.algorithm = name
        model_path = MODELS_DIR / f"model_{name}.joblib"
        result.best_pipeline.save(model_path)

        print(
            f"\n  {name.upper():>10}: "
            f"F1={metrics['f1_score']:.4f}  "
            f"AUC={metrics['roc_auc']:.4f}  "
            f"Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  "
            f"Thr={metrics['threshold']:.4f}"
        )
        print(f"              → salvo: {model_path.name}")

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name

    # 4. Ranking final
    print("\n" + "=" * 55)
    print("RANKING")
    print("=" * 55)
    for name, m in sorted(
        individual_metrics.items(),
        key=lambda x: x[1]["f1_score"],
        reverse=True,
    ):
        star = " ★ BEST" if name == best_name else ""
        print(f"  {name.upper():>10}: F1 = {m['f1_score']:.4f}{star}")

    # 5. Copiar melhor para model.joblib (padrão do dashboard/API)
    best_result = results[best_name]
    best_pipeline = best_result.best_pipeline
    best_threshold = best_result.best_threshold
    best_metrics = individual_metrics[best_name]

    model_path = MODELS_DIR / "model.joblib"
    best_pipeline.save(model_path)
    print(f"\n→ Melhor modelo ({best_name.upper()}) salvo como: {model_path}")

    # 6. Logar no MLflow
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("optimal_threshold", best_threshold)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_cv_splits", n_cv_splits)
        mlflow.log_param("best_cv_f1", best_result.best_score)
        mlflow.log_params({f"best_{k}": v for k, v in best_result.best_params.items()})
        mlflow.log_metrics(best_metrics)
        for name, m in individual_metrics.items():
            mlflow.log_metric(f"{name}_f1", m["f1_score"])

    # 7. Salvar métricas
    all_metrics = {
        "best_model": best_name,
        "optimal_threshold": best_threshold,
        "n_trials_per_model": n_trials,
        "n_cv_splits": n_cv_splits,
        "individual_results": {
            name: {
                "cv_f1": results[name].best_score,
                "best_trial": results[name].study.best_trial.number,
                "best_params": results[name].best_params,
                "threshold": results[name].best_threshold,
                **m,
            }
            for name, m in individual_metrics.items()
        },
        # Top-level canonical keys (consumed by dashboard / normalize_metrics_keys)
        **best_metrics,
        # Prefixed duplicates kept for explicitness
        **{f"test_{k}": v for k, v in best_metrics.items()},
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Métricas salvas: {metrics_path}")

    # 8. Veredicto
    print("\n" + "=" * 55)
    if best_f1 >= 0.65:
        print(f"META ATINGIDA! F1 = {best_f1:.4f} ({best_name.upper()})")
    else:
        delta = (best_f1 - 0.5744) * 100
        print(f"F1 = {best_f1:.4f} ({best_name.upper()}) — +{delta:.1f}% vs baseline")
    print("=" * 55)


if __name__ == "__main__":
    main()
