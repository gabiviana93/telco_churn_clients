"""
Módulo de Pipeline de Churn
===========================

Este módulo contém a classe principal de pipeline para predição de churn.
Fornece uma interface unificada para pré-processamento de dados, treinamento
de modelo, otimização de hiperparâmetros e inferência.

Classes:
    ChurnPipeline: Pipeline completo combinando pré-processamento e modelagem

Exemplo:
    >>> from src.pipeline import ChurnPipeline
    >>> pipeline = ChurnPipeline()
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
    STEP_MODEL,
    STEP_PREPROCESSING,
    ModelConfig,
    create_model,
)
from src.logger import setup_logger
from src.preprocessing import build_preprocessor
from src.utils import auto_detect_features

logger = setup_logger(__name__)


class ChurnPipeline:
    """Pipeline completo para predição de churn.

    Combina pré-processamento e treinamento de modelo em uma única interface.
    Suporta validação cruzada, rastreamento de métricas e persistência de modelo.

    Atributos:
        config: Configuração do modelo
        pipeline: Pipeline sklearn ajustado
        metrics: Dicionário de métricas de avaliação

    Exemplo:
        >>> config = ModelConfig(n_estimators=100, max_depth=5)
        >>> pipeline = ChurnPipeline(config)
        >>> pipeline.fit(X_train, y_train)
        >>> metrics = pipeline.evaluate(X_test, y_test)
        >>> pipeline.save("models/model.joblib")
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ):
        """Inicializa o pipeline de churn.

        Args:
            config: Configuração do modelo. Usa padrões se None.
            numeric_features: Lista de nomes de colunas numéricas.
            categorical_features: Lista de nomes de colunas categóricas.
        """
        self.config = config or ModelConfig()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.pipeline: Pipeline | None = None
        self.cv_metrics: dict[str, float] = {}
        self.test_metrics: dict[str, float] = {}
        self.algorithm: str | None = None
        self._is_fitted = False

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Constrói o pipeline sklearn com pré-processador e modelo.

        Usa o registro de modelos (``model.registry`` no YAML) para instanciar
        o estimador correto para ``self.config.model_type``.
        """
        # Auto-detectar features se não fornecidas
        if self.numeric_features is None or self.categorical_features is None:
            numeric, categorical = auto_detect_features(X)
            if self.numeric_features is None:
                self.numeric_features = numeric
            if self.categorical_features is None:
                self.categorical_features = categorical

        # Construir pré-processador usando função centralizada
        preprocessor = build_preprocessor(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )

        # Construir modelo via fábrica do registro
        model_params = self.config.to_dict()
        algorithm = str(self.config.model_type)
        model = create_model(algorithm, model_params)

        return Pipeline([(STEP_PREPROCESSING, preprocessor), (STEP_MODEL, model)])

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validate: bool = True,
        n_splits: int = N_SPLITS,
    ) -> ChurnPipeline:
        """Ajusta o pipeline aos dados de treinamento.

        Args:
            X: Features de treinamento
            y: Target de treinamento
            validate: Se deve realizar validação cruzada
            n_splits: Número de folds do CV (padrão da config)

        Returns:
            Self para encadeamento de métodos
        """
        logger.info(
            "Starting pipeline training", extra={"n_samples": len(X), "n_features": X.shape[1]}
        )

        # Build and fit pipeline
        self.pipeline = self._build_pipeline(X)
        self.pipeline.fit(X, y)
        self._is_fitted = True

        # Cross-validation if requested
        if validate:
            self.cv_metrics = self._cross_validate(X, y, n_splits)

        logger.info("Pipeline training completed", extra={"cv_metrics": self.cv_metrics})
        return self

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = N_SPLITS,
    ) -> dict[str, float]:
        """Realiza validação cruzada estratificada.

        Usa cross_validate com múltiplas métricas de scoring em uma única
        passagem para evitar reajuste do pipeline para cada métrica.

        Args:
            X: Features
            y: Target
            n_splits: Número de folds (padrão da config)

        Returns:
            Dicionário de médias e desvios padrão das métricas
        """
        from sklearn.model_selection import cross_validate as sklearn_cross_validate

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        scoring = {"f1": "f1", "roc_auc": "roc_auc"}
        cv_results = sklearn_cross_validate(self.pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        return {
            "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
            "cv_f1_std": float(np.std(cv_results["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predições.

        Args:
            X: Features para predição

        Returns:
            Array de predições binárias
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Obtém probabilidades de predição.

        Args:
            X: Features para predição

        Returns:
            Array de probabilidades com shape (n_samples, 2)
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        return self.pipeline.predict_proba(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Avalia modelo nos dados de teste.

        Args:
            X: Features de teste
            y: Target de teste
            threshold: Limiar de classificação

        Returns:
            Dicionário de métricas de avaliação
        """
        y_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "f1_score": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
            "auprc": average_precision_score(y, y_proba),
            "threshold": threshold,
        }

        self.test_metrics = metrics
        logger.info("Evaluation completed", extra=metrics)
        return metrics

    def find_optimal_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: Literal["f1", "precision", "recall"] = "f1",
    ) -> tuple[float, float]:
        """Encontra limiar de classificação ótimo para a métrica dada.

        Args:
            X: Features de validação
            y: Target de validação
            metric: Métrica a otimizar

        Returns:
            Tupla de (limiar_ótimo, melhor_score)
        """
        from sklearn.metrics import precision_recall_curve

        y_proba = self.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_proba)

        if metric == "f1":
            scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        elif metric == "precision":
            scores = precisions
        else:  # recall
            scores = recalls

        best_idx = np.argmax(scores[:-1])
        return float(thresholds[best_idx]), float(scores[best_idx])

    def save(self, path: str | Path) -> None:
        """Salva pipeline no disco.

        Args:
            path: Caminho para salvar o pipeline
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        package = {
            "pipeline": self.pipeline,
            "config": self.config,
            "metrics": {**self.cv_metrics, **self.test_metrics},
            "cv_metrics": self.cv_metrics,
            "test_metrics": self.test_metrics,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "algorithm": self.algorithm
            or (
                self.config.model_type.value
                if hasattr(self.config.model_type, "value")
                else str(self.config.model_type)
            ),
        }

        dump(package, path)
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ChurnPipeline:
        """Carrega pipeline do disco.

        Args:
            path: Caminho do pipeline salvo

        Returns:
            Instância de ChurnPipeline carregada
        """
        path = Path(path)
        package = load(path)

        instance = cls(
            config=package.get("config"),
            numeric_features=package.get("numeric_features"),
            categorical_features=package.get("categorical_features"),
        )
        instance.pipeline = package["pipeline"]
        instance.cv_metrics = package.get("cv_metrics", {})
        instance.test_metrics = package.get("test_metrics", {})
        instance.algorithm = package.get("algorithm")
        # Backward compatibility: old packages without separated metrics
        if not instance.cv_metrics and not instance.test_metrics:
            all_metrics = package.get("metrics", {})
            instance.cv_metrics = {k: v for k, v in all_metrics.items() if k.startswith("cv_")}
            instance.test_metrics = {
                k: v for k, v in all_metrics.items() if not k.startswith("cv_")
            }
        instance._is_fitted = True

        logger.info(f"Pipeline loaded from {path}")
        return instance

    @property
    def metrics(self) -> dict[str, float]:
        """Métricas combinadas (CV + teste) para backward compatibility."""
        return {**self.cv_metrics, **self.test_metrics}

    @metrics.setter
    def metrics(self, value: dict[str, float]) -> None:
        """Setter para backward compatibility: separa cv_ e test metrics."""
        self.cv_metrics = {k: v for k, v in value.items() if k.startswith("cv_")}
        self.test_metrics = {k: v for k, v in value.items() if not k.startswith("cv_")}

    @property
    def sklearn_pipeline(self) -> Pipeline:
        """Obtém o Pipeline sklearn subjacente."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted")
        return self.pipeline
