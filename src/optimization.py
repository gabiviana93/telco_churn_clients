"""
Módulo de Otimização de Hiperparâmetros
========================================

Implementa otimização bayesiana de hiperparâmetros usando Optuna,
focada em maximizar o F1-Score para predição de churn.

Inclui:
- Otimização padrão e avançada com XGBoost
- Oversampling opcional com SMOTE para desbalanceamento de classes
- Otimização opcional de threshold para maximização de F1
- Análise de importância de features

Classes:
    HyperparameterOptimizer: Otimizador unificado com SMOTE e threshold configuráveis
    OptimizationConfig: Configuração para otimização
    OptimizationResult: Resultados da otimização

Exemplo:
    >>> from src.optimization import HyperparameterOptimizer, OptimizationConfig
    >>> optimizer = HyperparameterOptimizer(X_train, y_train)
    >>> result = optimizer.optimize(n_trials=100)
    >>> print(f"F1: {result.best_score:.4f}")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    DEFAULT_MODEL_TYPE,
    DEFAULT_N_TRIALS,
    DEFAULT_OPTIMIZATION_METRIC,
    DEFAULT_OPTIMIZATION_TIMEOUT,
    DEFAULT_OPTIMIZE_THRESHOLD,
    MODEL_REGISTRY,
    N_SPLITS,
    RANDOM_STATE,
    SEARCH_SPACE,
    STEP_MODEL,
    STEP_PREPROCESSING,
    create_model,
)
from src.logger import setup_logger
from src.pipeline import ChurnPipeline, ModelConfig
from src.preprocessing import build_preprocessor
from src.utils import auto_detect_features

logger = setup_logger(__name__)

# Verificar se Optuna está disponível
try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna não instalado. Recursos de otimização serão limitados.")

# Verificar se imbalanced-learn está disponível
try:
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn não instalado. SMOTE indisponível.")


@dataclass
class OptimizationConfig:
    """Configuração para otimização de hiperparâmetros.

    Ranges de espaço de busca e padrões são carregados da configuração YAML
    (``config/project.yaml``). Todos os valores podem ser sobrescritos na construção.

    Atributos:
        model_type: Tipo de modelo a otimizar (XGBOOST, LIGHTGBM, CATBOOST)
        n_trials: Número de trials de otimização
        n_cv_splits: Número de folds de validação cruzada
        metric: Métrica principal a otimizar
        direction: Direção da otimização (maximize/minimize)
        timeout: Tempo máximo de otimização em segundos
        n_jobs: Número de jobs paralelos para CV
        random_state: Seed aleatória
        use_smote: Se deve usar oversampling SMOTE
        optimize_threshold: Se deve otimizar o limiar de classificação
    """

    model_type: str = field(default_factory=lambda: DEFAULT_MODEL_TYPE)
    n_trials: int = DEFAULT_N_TRIALS
    n_cv_splits: int = N_SPLITS
    metric: Literal["f1", "roc_auc", "precision", "recall"] = DEFAULT_OPTIMIZATION_METRIC
    direction: Literal["maximize", "minimize"] = "maximize"
    timeout: int | None = DEFAULT_OPTIMIZATION_TIMEOUT
    n_jobs: int = -1
    random_state: int = RANDOM_STATE
    use_smote: bool = True
    optimize_threshold: bool = DEFAULT_OPTIMIZE_THRESHOLD


@dataclass
class OptimizationResult:
    """Resultados da otimização de hiperparâmetros.

    Atributos:
        best_params: Dicionário dos melhores hiperparâmetros
        best_score: Melhor score de métrica alcançado
        best_pipeline: Pipeline ajustado com os melhores parâmetros
        best_threshold: Limiar ótimo de classificação (0.5 se não otimizado)
        study: Objeto de estudo Optuna (se disponível)
        all_trials: Lista de todos os resultados de trials
        timestamp: Hora de conclusão da otimização
        config: Configuração de otimização utilizada
    """

    best_params: dict[str, Any]
    best_score: float
    best_pipeline: ChurnPipeline | None
    best_threshold: float = 0.5
    study: Any | None = None  # optuna.Study
    all_trials: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    config: OptimizationConfig | None = None

    def summary(self) -> str:
        """Gera um resumo dos resultados da otimização."""
        lines = [
            "\n" + "=" * 60,
            "OPTIMIZATION RESULTS",
            "=" * 60,
            f"Best {self.config.metric if self.config else 'score'}: {self.best_score:.4f} ({self.best_score*100:.2f}%)",
            f"Optimal Threshold: {self.best_threshold:.4f}",
            f"Trials completed: {len(self.all_trials)}",
        ]
        if self.config:
            lines.append(f"SMOTE: {'Yes' if self.config.use_smote else 'No'}")
        lines.append(f"Timestamp: {self.timestamp.isoformat()}")
        lines.append("")
        lines.append("Best Parameters:")
        for param, value in self.best_params.items():
            if isinstance(value, float):
                lines.append(f"  {param}: {value:.6f}")
            else:
                lines.append(f"  {param}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def predict_with_threshold(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Realiza predições usando limiar ótimo.

        Args:
            X: Features para predição

        Returns:
            Predições binárias usando limiar ótimo
        """
        if self.best_pipeline is None:
            raise ValueError("No pipeline available")

        proba = self.best_pipeline.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)


class HyperparameterOptimizer:
    """Otimizador unificado de hiperparâmetros usando Optuna.

    Realiza otimização bayesiana usando sampler TPE (Tree-structured Parzen Estimator)
    para encontrar hiperparâmetros ótimos para o classificador.

    Suporta oversampling SMOTE e otimização de threshold opcionais
    via flags de configuração.

    Atributos:
        X: Features de treinamento
        y: Target de treinamento
        config: Configuração de otimização

    Exemplo:
        Otimização básica:
        >>> optimizer = HyperparameterOptimizer(X_train, y_train)
        >>> result = optimizer.optimize(n_trials=50)
        >>> print(result.summary())

        Otimização avançada com SMOTE e threshold:
        >>> config = OptimizationConfig(use_smote=True, optimize_threshold=True)
        >>> optimizer = HyperparameterOptimizer(X_train, y_train, config)
        >>> result = optimizer.optimize(n_trials=100)
        >>> predictions = result.predict_with_threshold(X_test)
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: OptimizationConfig | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ):
        """Inicializa o otimizador.

        Args:
            X: Features de treinamento
            y: Target de treinamento
            config: Configuração de otimização
            numeric_features: Lista de nomes de colunas numéricas (auto-detectadas se None)
            categorical_features: Lista de nomes de colunas categóricas (auto-detectadas se None)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for optimization. " "Install with: pip install optuna"
            )

        self.X = X
        self.y = y
        self.config = config or OptimizationConfig()

        # Auto-detectar features
        detected_numeric, detected_categorical = auto_detect_features(X)
        self.numeric_features = numeric_features or detected_numeric
        self.categorical_features = categorical_features or detected_categorical

        # Construir pré-processador uma vez
        self._preprocessor = build_preprocessor(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )

        # Info de desbalanceamento de classes
        self.churn_rate = y.mean()
        self._scale_pos_weight = (1 - self.churn_rate) / self.churn_rate

        logger.info(
            "Optimizer initialized",
            extra={
                "n_samples": len(X),
                "n_numeric": len(self.numeric_features),
                "n_categorical": len(self.categorical_features),
                "churn_rate": f"{self.churn_rate:.2%}",
                "use_smote": self.config.use_smote,
                "optimize_threshold": self.config.optimize_threshold,
            },
        )

    def _create_objective(self) -> Callable:
        """Cria a função objetivo do Optuna.

        Lê os ``search_params`` do modelo do registro YAML para
        determinar quais hiperparâmetros ajustar. Usa ``create_model()``
        para instanciação, suportando qualquer modelo registrado.

        Returns:
            Função objetivo para otimização Optuna
        """
        algorithm = str(self.config.model_type)
        algo_cfg = MODEL_REGISTRY.get(algorithm, {})
        search_param_names: list[str] = algo_cfg.get("search_params", [])
        use_smote = self.config.use_smote and IMBLEARN_AVAILABLE

        def _suggest(trial: optuna.Trial, name: str) -> int | float | None:
            """Sugere valor de hiperparâmetro com base no tipo do SEARCH_SPACE."""
            if name not in SEARCH_SPACE:
                return None
            # scale_pos_weight é redundante quando SMOTE lida com desbalanceamento
            if name == "scale_pos_weight" and use_smote:
                return None
            low, high = SEARCH_SPACE[name]
            if isinstance(low, int) and isinstance(high, int):
                return trial.suggest_int(name, low, high)
            if name == "learning_rate":
                return trial.suggest_float(name, float(low), float(high), log=True)
            return trial.suggest_float(name, float(low), float(high))

        def objective(trial: optuna.Trial) -> float:
            # Sugerir hiperparâmetros listados nos search_params do registro
            params: dict[str, Any] = {}
            for p_name in search_param_names:
                val = _suggest(trial, p_name)
                if val is not None:
                    params[p_name] = val

            # Adicionar random_state
            params["random_state"] = self.config.random_state

            # Criar modelo via fábrica do registro
            model = create_model(algorithm, params)

            # Construir pipeline (com ou sem SMOTE)
            if use_smote:
                pipeline = ImbPipeline(
                    [
                        (STEP_PREPROCESSING, self._preprocessor),
                        ("smote", SMOTETomek(random_state=self.config.random_state)),
                        (STEP_MODEL, model),
                    ]
                )
            else:
                pipeline = Pipeline(
                    [
                        (STEP_PREPROCESSING, self._preprocessor),
                        (STEP_MODEL, model),
                    ]
                )

            # Validação cruzada
            cv = StratifiedKFold(
                n_splits=self.config.n_cv_splits,
                shuffle=True,
                random_state=self.config.random_state,
            )

            if self.config.optimize_threshold:
                y_proba = cross_val_predict(
                    pipeline, self.X, self.y, cv=cv, method="predict_proba"
                )[:, 1]

                precision, recall, thresholds = precision_recall_curve(self.y, y_proba)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                best_idx = np.argmax(f1_scores[:-1])
                best_f1 = f1_scores[best_idx]
                best_threshold = thresholds[best_idx]

                trial.set_user_attr("threshold", float(best_threshold))
                return best_f1
            else:
                scores = cross_val_score(
                    pipeline,
                    self.X,
                    self.y,
                    cv=cv,
                    scoring=self.config.metric,
                    n_jobs=self.config.n_jobs,
                )
                return np.mean(scores)

        return objective

    def optimize(
        self,
        n_trials: int | None = None,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """Executa otimização de hiperparâmetros.

        Args:
            n_trials: Número de trials (sobrescreve config)
            show_progress: Se deve mostrar barra de progresso

        Returns:
            OptimizationResult com melhores parâmetros e pipeline
        """
        n_trials = n_trials or self.config.n_trials

        logger.info(
            "Starting optimization",
            extra={
                "n_trials": n_trials,
                "metric": self.config.metric,
                "cv_splits": self.config.n_cv_splits,
                "use_smote": self.config.use_smote,
                "optimize_threshold": self.config.optimize_threshold,
            },
        )

        # Criar estudo com pruning opcional
        sampler = TPESampler(seed=self.config.random_state)

        study_kwargs = {
            "direction": self.config.direction,
            "sampler": sampler,
        }

        study = optuna.create_study(**study_kwargs)

        # Otimizar
        optuna.logging.set_verbosity(
            optuna.logging.INFO if show_progress else optuna.logging.WARNING
        )

        study.optimize(
            self._create_objective(),
            n_trials=n_trials,
            timeout=self.config.timeout,
            show_progress_bar=show_progress,
        )

        # Extrair resultados
        best_params = study.best_params
        best_score = study.best_value
        best_threshold = study.best_trial.user_attrs.get("threshold", 0.5)

        # Treinar modelo final com melhores parâmetros
        best_pipeline = self._train_best_model(best_params)

        # Coletar todos os resultados de trials
        all_trials = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "threshold": trial.user_attrs.get("threshold", 0.5),
            }
            for trial in study.trials
            if trial.value is not None
        ]

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_pipeline=best_pipeline,
            best_threshold=best_threshold,
            study=study,
            all_trials=all_trials,
            config=self.config,
        )

        logger.info(result.summary())

        return result

    def _train_best_model(self, params: dict[str, Any]) -> ChurnPipeline:
        """Treina um ChurnPipeline com os melhores parâmetros.

        Usa ``self.config.model_type`` para que o modelo final corresponda
        ao avaliado durante a busca Optuna.

        Args:
            params: Melhores hiperparâmetros da otimização

        Returns:
            ChurnPipeline ajustado
        """
        model_type = str(self.config.model_type)

        # Build a ModelConfig with the best params.
        # Use .get() to avoid KeyError for model-specific params.
        scale_pos_weight = params.get("scale_pos_weight", self._scale_pos_weight)
        if self.config.use_smote and IMBLEARN_AVAILABLE:
            scale_pos_weight = 1.0

        # Map CatBoost l2_leaf_reg back to reg_lambda for ModelConfig
        reg_lambda = params.get("reg_lambda", params.get("l2_leaf_reg", 1.0))

        config = ModelConfig(
            model_type=model_type,
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 1),
            gamma=params.get("gamma", 0.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            num_leaves=params.get("num_leaves"),
            min_child_samples=params.get("min_child_samples"),
        )

        pipeline = ChurnPipeline(
            config=config,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )

        # Fit with SMOTE if enabled
        if self.config.use_smote and IMBLEARN_AVAILABLE:
            preprocessor = build_preprocessor(
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
            )
            X_processed = preprocessor.fit_transform(self.X)
            smote = SMOTETomek(random_state=self.config.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_processed, self.y)

            # Build pipeline structure and reuse already-fitted preprocessor
            pipeline.pipeline = pipeline._build_pipeline(self.X)
            pipeline.pipeline.steps[0] = (STEP_PREPROCESSING, preprocessor)
            pipeline.pipeline.named_steps[STEP_MODEL].fit(X_resampled, y_resampled)
            pipeline._is_fitted = True
        else:
            pipeline.fit(self.X, self.y, validate=True)

        return pipeline

    def get_feature_importance(
        self,
        result: OptimizationResult,
    ) -> pd.DataFrame:
        """Obtém importância de features do melhor modelo.

        Args:
            result: Resultado da otimização contendo melhor pipeline

        Returns:
            DataFrame com nomes de features e scores de importância
        """
        if result.best_pipeline is None:
            raise ValueError("No best pipeline available")

        pipeline = result.best_pipeline.sklearn_pipeline
        model = pipeline.named_steps[STEP_MODEL]
        preprocessor = pipeline.named_steps[STEP_PREPROCESSING]

        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        return df


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================


def quick_optimize(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    metric: str = DEFAULT_OPTIMIZATION_METRIC,
    model_type: str | None = None,
) -> OptimizationResult:
    """Otimização rápida com padrões sensíveis (sem SMOTE, sem threshold).

    Args:
        X: Features de treinamento
        y: Target de treinamento
        n_trials: Número de trials de otimização
        metric: Métrica a otimizar
        model_type: Tipo de modelo a otimizar (padrão da config YAML)

    Returns:
        OptimizationResult com melhor modelo

    Exemplo:
        >>> result = quick_optimize(X_train, y_train, n_trials=50)
        >>> model = result.best_pipeline
    """
    config = OptimizationConfig(
        model_type=model_type or DEFAULT_MODEL_TYPE,
        n_trials=n_trials,
        metric=metric,
        n_cv_splits=N_SPLITS,
        use_smote=False,
        optimize_threshold=False,
    )

    optimizer = HyperparameterOptimizer(X, y, config)
    return optimizer.optimize()


def optimize_for_f1(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = DEFAULT_N_TRIALS,
    use_smote: bool = True,
    model_type: str | None = None,
) -> OptimizationResult:
    """Otimiza modelo especificamente para F1-Score com todas as técnicas.

    Usa oversampling SMOTE + otimização de threshold + busca estendida.

    Args:
        X: Features de treinamento
        y: Target de treinamento
        n_trials: Número de trials de otimização
        use_smote: Se deve usar SMOTE
        model_type: Tipo de modelo a otimizar (padrão da config YAML)

    Returns:
        OptimizationResult com melhor modelo e limiar ótimo

    Exemplo:
        >>> result = optimize_for_f1(X_train, y_train, n_trials=100)
        >>> print(f"F1: {result.best_score:.2%}")
        >>> predictions = result.predict_with_threshold(X_test)
    """
    config = OptimizationConfig(
        model_type=model_type or DEFAULT_MODEL_TYPE,
        n_trials=n_trials,
        metric="f1",
        use_smote=use_smote and IMBLEARN_AVAILABLE,
        optimize_threshold=True,
        n_cv_splits=N_SPLITS,
    )

    optimizer = HyperparameterOptimizer(X, y, config)
    return optimizer.optimize()
