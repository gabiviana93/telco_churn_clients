"""
Serviço de Modelo
=================

Camada de serviço para carregamento, predição e gerenciamento do modelo.
Implementa padrão singleton para cache eficiente do modelo.

Utiliza a mesma classe ChurnPipeline dos scripts para consistência.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from api.core.config import settings
from api.core.logging import get_logger
from src.config import STEP_MODEL
from src.pipeline import ChurnPipeline
from src.utils import classify_risk, normalize_metrics_keys

logger = get_logger(__name__)


def _detect_model_name(model: Any) -> str:
    """Detecta nome legível do modelo a partir de um objeto de modelo.

    Trata sklearn Pipelines, VotingClassifiers e estimadores simples.

    Returns:
        Nome descritivo do modelo (ex.: 'LGBMClassifier', 'VotingClassifier').
    """
    # sklearn Pipeline — extrai o estimador final
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get(STEP_MODEL)
        if inner is not None:
            return type(inner).__name__
    # Classificador Voting / Stacking
    if hasattr(model, "estimators_"):
        return type(model).__name__
    return type(model).__name__


class ModelService:
    """
    Serviço para operações de modelo ML.

    Gerencia carregamento, cache e lógica de predição do modelo.
    Usa padrão singleton para garantir apenas uma instância do modelo.

    Integra com ChurnPipeline para consistência com os scripts.
    """

    _instance: ModelService | None = None

    def __new__(cls) -> ModelService:
        """Garante instância singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Inicializa o serviço de modelo."""
        if self._initialized:
            return
        self._initialized = True
        self._model = None
        self._pipeline: ChurnPipeline | None = None
        self._preprocessor = None
        self._feature_engineer = None
        self._threshold: float = 0.5
        self._model_info: dict[str, Any] = {}
        self._start_time: float = time.time()
        self._load_model()

    def _load_model(self) -> None:
        """Carrega o modelo ML do disco.

        Suporta tanto formato ChurnPipeline quanto pipelines sklearn legados.
        """
        model_path = settings.model_path_resolved

        logger.info("Loading model", model_path=str(model_path))

        try:
            if not model_path.exists():
                logger.warning(
                    "Model file not found, service will run in demo mode",
                    model_path=str(model_path),
                )
                self._model = None
                self._pipeline = None
                self._model_info = {"model_name": "Demo Model", "version": "demo", "loaded": False}
                return

            # Tenta carregar como ChurnPipeline primeiro
            try:
                self._pipeline = ChurnPipeline.load(model_path)
                self._model = self._pipeline.pipeline  # Access sklearn Pipeline

                # Detecta nome real do modelo do pipeline
                actual_model = self._pipeline.pipeline.named_steps.get(STEP_MODEL)
                model_name = type(actual_model).__name__ if actual_model else "Unknown"

                numeric = self._pipeline.numeric_features or []
                categorical = self._pipeline.categorical_features or []

                self._model_info = {
                    "model_name": f"{model_name} (ChurnPipeline)",
                    "version": settings.MODEL_VERSION,
                    "features": {
                        "numeric": numeric,
                        "categorical": categorical,
                        "n_input_features": len(numeric) + len(categorical),
                        "input_features": numeric + categorical,
                    },
                    "hyperparameters": {},
                    "metrics": {},
                    "trained_at": None,
                    "loaded": True,
                    "pipeline_type": "ChurnPipeline",
                }
                logger.info(
                    "ChurnPipeline loaded successfully",
                    model_name=self._model_info["model_name"],
                    version=self._model_info["version"],
                )
                return
            except Exception as e:
                logger.debug(f"ChurnPipeline load failed, trying legacy format: {e}")

            # Fallback para formato legado
            package = load(model_path)

            # Trata diferentes formatos de pacote
            if isinstance(package, dict):
                # Formato novo com chave "model"
                if "model" in package:
                    self._model = package["model"]
                    self._preprocessor = package.get("preprocessor")
                    self._feature_engineer = package.get("feature_engineer")
                    self._threshold = package.get("threshold", 0.5)
                    detected_name = _detect_model_name(package["model"])
                    self._model_info = {
                        "model_name": package.get("model_name", detected_name),
                        "version": package.get("version", settings.MODEL_VERSION),
                        "features": package.get("features", {}),
                        "hyperparameters": package.get(
                            "params", package.get("hyperparameters", {})
                        ),
                        "metrics": package.get("metrics", {}),
                        "trained_at": package.get(
                            "timestamp", package.get("metadata", {}).get("training_date")
                        ),
                        "threshold": self._threshold,
                        "threshold_strategy": package.get("threshold_strategy", ""),
                        "loaded": True,
                        "pipeline_type": "dict",
                    }
                # Formato otimizado com chave "pipeline"
                elif "pipeline" in package:
                    self._model = package["pipeline"]
                    detected_name = _detect_model_name(package["pipeline"])
                    raw_metrics = package.get("metrics", {})
                    self._model_info = {
                        "model_name": package.get("model_name", detected_name),
                        "version": settings.MODEL_VERSION,
                        "features": {},
                        "hyperparameters": package.get("params", {}),
                        "metrics": normalize_metrics_keys(raw_metrics),
                        "trained_at": package.get("timestamp"),
                        "loaded": True,
                        "pipeline_type": "optimized_pipeline",
                    }
                else:
                    # Formato dict desconhecido
                    self._model = package
                    self._model_info = {
                        "model_name": "Unknown Model",
                        "version": settings.MODEL_VERSION,
                        "features": {},
                        "hyperparameters": {},
                        "metrics": {},
                        "trained_at": None,
                        "loaded": True,
                        "pipeline_type": "unknown_dict",
                    }
            else:
                # Formato legado - apenas o pipeline
                self._model = package
                detected_name = _detect_model_name(package)
                self._model_info = {
                    "model_name": detected_name,
                    "version": settings.MODEL_VERSION,
                    "features": {},
                    "hyperparameters": {},
                    "metrics": {},
                    "trained_at": None,
                    "loaded": True,
                    "pipeline_type": "legacy",
                }

            logger.info(
                "Model loaded successfully",
                model_name=self._model_info["model_name"],
                version=self._model_info["version"],
            )

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            self._model = None
            self._model_info = {
                "model_name": "Error",
                "version": "N/A",
                "loaded": False,
                "error": str(e),
            }

    @property
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        return self._model is not None or self._pipeline is not None

    @property
    def model_version(self) -> str:
        """Obtém a versão do modelo."""
        return self._model_info.get("version", settings.MODEL_VERSION)

    @property
    def uptime_seconds(self) -> float:
        """Obtém o tempo de atividade do serviço em segundos."""
        return time.time() - self._start_time

    @property
    def model(self) -> Any:
        """Obtém o modelo ML subjacente.

        Retorna o modelo independente de como foi carregado (ChurnPipeline ou legado).
        """
        return self._model

    @property
    def pipeline(self) -> ChurnPipeline | None:
        """Obtém o ChurnPipeline se disponível."""
        return self._pipeline

    def get_model_info(self) -> dict[str, Any]:
        """Obtém metadados e informações do modelo."""
        return self._model_info.copy()

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Realiza predições para as features fornecidas.

        Usa ChurnPipeline se disponível, caso contrário faz fallback para modelo legado.

        Args:
            features: DataFrame com features dos clientes

        Returns:
            Tupla de (predições, probabilidades)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Cannot make predictions without a trained model.")

        try:
            # Usa ChurnPipeline se disponível
            if self._pipeline is not None:
                predictions = self._pipeline.predict(features)
                probabilities = self._pipeline.predict_proba(features)[:, 1]
                return predictions, probabilities

            # Formato legado — aplica preprocessor/feature_engineer armazenados
            X = features
            if self._feature_engineer is not None:
                X = self._feature_engineer.transform(X)
            if self._preprocessor is not None:
                X = self._preprocessor.transform(X)

            # Obtém probabilidades se disponível
            if hasattr(self._model, "predict_proba"):
                probabilities = self._model.predict_proba(X)[:, 1]
                threshold = getattr(self, "_threshold", 0.5)
                predictions = (probabilities >= threshold).astype(int)
            else:
                predictions = self._model.predict(X)
                probabilities = predictions.astype(float)

            return predictions, probabilities

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise

    def get_feature_importance(self) -> list[dict[str, Any]]:
        """
        Obtém scores de importância de features.

        Usa nomes de features pós-preprocessamento (após one-hot encoding)
        para corresponder corretamente ao array feature_importances_ do modelo.

        Returns:
            Lista de dicionários de importância de features ordenada por importância
        """
        if not self.is_loaded:
            return []

        try:
            # Tenta obter importância de features do modelo
            importances = None
            model_obj = self._model

            if hasattr(model_obj, "named_steps"):
                model_obj = model_obj.named_steps.get(STEP_MODEL, model_obj)

            if hasattr(model_obj, "feature_importances_"):
                importances = model_obj.feature_importances_
            elif hasattr(model_obj, "coef_"):
                import numpy as np

                coef = np.asarray(model_obj.coef_)
                importances = np.abs(coef.ravel())

            if importances is None:
                return []

            # Obtém nomes de features do preprocessador (nomes pós-transformação)
            # Garante que nomes correspondam ao array feature_importances_ (após one-hot encoding)
            feature_names = None

            if hasattr(self._model, "named_steps"):
                from src.config import STEP_PREPROCESSING
                from src.interpret import get_feature_names

                preprocessor = self._model.named_steps.get(STEP_PREPROCESSING)
                if preprocessor is not None:
                    try:
                        feature_names = get_feature_names(preprocessor)
                    except Exception:
                        pass

            # Fallback para nomes genéricos se extração do preprocessador falhar
            if feature_names is None or len(feature_names) != len(importances):
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            # Cria lista ordenada
            importance_list = [
                {"feature_name": name, "importance": float(imp)}
                for name, imp in zip(feature_names, importances, strict=False)
            ]
            importance_list.sort(key=lambda x: x["importance"], reverse=True)

            # Adiciona ranking
            for rank, item in enumerate(importance_list, 1):
                item["rank"] = rank

            return importance_list

        except Exception as e:
            logger.error("Failed to get feature importance", error=str(e))
            return []

    def classify_risk(self, probability: float) -> str:
        """Classifica risco de churn baseado na probabilidade."""
        return classify_risk(probability)

    def calculate_confidence(self, probability: float) -> float:
        """
        Calcula confiança do modelo baseada na probabilidade.

        Maior confiança quando probabilidade está mais próxima de 0 ou 1.

        Args:
            probability: Probabilidade de churn (0-1)

        Returns:
            Score de confiança (0-1)
        """
        return abs(probability - 0.5) * 2


# Instância global do serviço
model_service = ModelService()


def get_model_service() -> ModelService:
    """Helper de injeção de dependência para FastAPI."""
    return model_service
