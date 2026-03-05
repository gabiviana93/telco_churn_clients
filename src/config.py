"""
Módulo de Configuração
======================

Configuração centralizada para o projeto de Predição de Churn.
Todas as configurações, caminhos e constantes do projeto são definidos aqui.

A configuração é carregada de arquivos YAML com a seguinte prioridade:
1. config/project.yaml (overrides específicos do projeto)
2. config/default.yaml (valores padrão do framework)
3. Variáveis de ambiente (maior prioridade)

Uso:
    from src.config import get_config, MODELS_DIR, TARGET

    # Acessar caminhos
    model_path = MODELS_DIR / "model.joblib"

    # Obter configuração completa
    config = get_config()
    target = config["data"]["target_column"]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# CONFIGURAÇÃO DE CAMINHOS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR_RAW = BASE_DIR / "src" / "data" / "raw"
DATA_DIR_PROCESSED = BASE_DIR / "src" / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Garante que os diretórios existam
for _dir in [MODELS_DIR, DATA_DIR_PROCESSED, REPORTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CARREGADOR DE CONFIGURAÇÃO YAML
# =============================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge profundo de dois dicionários, com override tendo precedência."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Carrega um arquivo YAML de forma segura."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """
    Carrega e mescla configuração dos arquivos YAML.

    Prioridade (maior para menor):
    1. Variáveis de ambiente (MODEL_NAME, etc.)
    2. config/project.yaml
    3. config/default.yaml

    Returns:
        Dicionário de configuração mesclado
    """
    # Carregar configuração padrão
    default_config = _load_yaml_file(CONFIG_DIR / "default.yaml")

    # Carregar configuração do projeto (sobrescreve padrões)
    project_config = _load_yaml_file(CONFIG_DIR / "project.yaml")

    # Mesclar configurações
    config = _deep_merge(default_config, project_config)

    # Aplicar overrides de variáveis de ambiente
    env_overrides = {
        "model": {
            "default_name": os.getenv(
                "MODEL_NAME", config.get("model", {}).get("default_name", "model")
            ),
        },
        "api": {
            "host": os.getenv("API_HOST", config.get("api", {}).get("host", "0.0.0.0")),
            "port": int(os.getenv("API_PORT", config.get("api", {}).get("port", 8000))),
        },
        "logging": {
            "level": os.getenv("LOG_LEVEL", config.get("logging", {}).get("level", "INFO")),
        },
    }

    config = _deep_merge(config, env_overrides)

    return config


# =============================================================================
# CONSTANTES DE CONVENIÊNCIA (carregadas do YAML com fallbacks)
# =============================================================================


def _get_safe(config: dict, *keys, default=None):
    """Obtém valor aninhado da configuração de forma segura."""
    for key in keys:
        if isinstance(config, dict):
            config = config.get(key, default)
        else:
            return default
    return config


# Carregar configuração inicial
_config = get_config()

# Configuração de Dados
FILENAME = _get_safe(_config, "data", "filename", default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
TARGET = _get_safe(_config, "data", "target_column", default="Churn")
ID_COL = _get_safe(_config, "data", "id_column", default="customerID")
TEST_SIZE = _get_safe(_config, "data", "test_size", default=0.2)
RANDOM_STATE = _get_safe(_config, "data", "random_state", default=42)
N_SPLITS = _get_safe(_config, "data", "cv_folds", default=5)

# Métrica principal
PRIMARY_METRIC = _get_safe(_config, "optimization", "metric", default="f1")

# Configuração de Features
NUMERIC_FEATURES = _get_safe(
    _config, "features", "numeric", default=["tenure", "MonthlyCharges", "TotalCharges"]
)
CATEGORICAL_FEATURES = _get_safe(
    _config,
    "features",
    "categorical",
    default=[
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ],
)

# Bins de Feature Engineering
TENURE_BINS = _get_safe(
    _config, "features", "engineering", "binning", "tenure", "bins", default=[0, 6, 12, 24, 48, 72]
)
TENURE_LABELS = _get_safe(
    _config,
    "features",
    "engineering",
    "binning",
    "tenure",
    "labels",
    default=["0-6m", "7-12m", "1-2a", "2-4a", "4-6a"],
)
MONTHLY_CAT_LABELS = _get_safe(
    _config,
    "features",
    "engineering",
    "binning",
    "MonthlyCharges",
    "labels",
    default=["muito_baixo", "baixo", "medio", "alto"],
)

# Nomes dos passos do Pipeline (canônicos — usados em todo o projeto)
STEP_PREPROCESSING = "preprocessing"
STEP_MODEL = "model"

# Configuração do Modelo
DEFAULT_MODEL_NAME = _get_safe(_config, "model", "default_name", default="model")
MODEL_PATH = MODELS_DIR / f"{DEFAULT_MODEL_NAME}.joblib"
DEFAULT_MODEL_TYPE = _get_safe(_config, "model", "algorithm", default="lightgbm")

# Configuração do MLflow
MLFLOW_TRACKING_URI = str(
    BASE_DIR / _get_safe(_config, "tracking", "mlflow", "tracking_uri", default="mlruns")
)
MLFLOW_EXPERIMENT = _get_safe(
    _config, "tracking", "mlflow", "experiment_name", default="churn_prediction"
)

# Configuração da API
API_HOST = _get_safe(_config, "api", "host", default="0.0.0.0")
API_PORT = _get_safe(_config, "api", "port", default=8000)
API_TITLE = _get_safe(_config, "api", "title", default="Customer Churn Prediction API")
API_VERSION = _get_safe(_config, "api", "version", default="1.0.0")

# Configuração do Dashboard
DASHBOARD_HOST = _get_safe(_config, "dashboard", "host", default="0.0.0.0")
DASHBOARD_PORT = _get_safe(_config, "dashboard", "port", default=8501)
DASHBOARD_TITLE = _get_safe(_config, "dashboard", "title", default="Customer Churn Dashboard")

# Limiares de Classificação de Risco
RISK_THRESHOLD_LOW = _get_safe(_config, "risk", "threshold_low", default=0.4)
RISK_THRESHOLD_HIGH = _get_safe(_config, "risk", "threshold_high", default=0.7)

# Classe positiva para conversão do target
POSITIVE_CLASS = _get_safe(_config, "data", "positive_class", default="Yes")

# Limiares de Feature Engineering
FE_NEW_CUSTOMER_TENURE_MAX = _get_safe(
    _config, "feature_engineering", "thresholds", "new_customer_tenure_max", default=6
)
FE_CHARGES_DIVERGENCE = _get_safe(
    _config, "feature_engineering", "thresholds", "charges_divergence", default=0.3
)
FE_EARLY_CHURN_TENURE = _get_safe(
    _config, "feature_engineering", "thresholds", "early_churn_tenure", default=12
)
FE_LOW_ENGAGEMENT_MAX = _get_safe(
    _config, "feature_engineering", "thresholds", "low_engagement_max", default=2
)
FE_HIGH_ENGAGEMENT_MIN = _get_safe(
    _config, "feature_engineering", "thresholds", "high_engagement_min", default=5
)
FE_RISK_SCORE_HIGH = _get_safe(
    _config, "feature_engineering", "thresholds", "risk_score_high", default=4
)
FE_RISK_SCORE_VERY_HIGH = _get_safe(
    _config, "feature_engineering", "thresholds", "risk_score_very_high", default=6
)

# Pesos de risco para score composto
FE_RISK_WEIGHTS = {
    "monthly_contract": _get_safe(
        _config, "feature_engineering", "risk_weights", "monthly_contract", default=2.0
    ),
    "short_tenure": _get_safe(
        _config, "feature_engineering", "risk_weights", "short_tenure", default=2.0
    ),
    "fiber_no_security": _get_safe(
        _config, "feature_engineering", "risk_weights", "fiber_no_security", default=1.5
    ),
    "low_engagement": _get_safe(
        _config, "feature_engineering", "risk_weights", "low_engagement", default=1.0
    ),
    "paperless_billing": _get_safe(
        _config, "feature_engineering", "risk_weights", "paperless_billing", default=0.5
    ),
    "electronic_check": _get_safe(
        _config, "feature_engineering", "risk_weights", "electronic_check", default=1.0
    ),
}

# Limiares de ciclo de vida
FE_LIFECYCLE_NEW_MAX = _get_safe(_config, "feature_engineering", "lifecycle", "new_max", default=3)
FE_LIFECYCLE_EARLY_MAX = _get_safe(
    _config, "feature_engineering", "lifecycle", "early_max", default=12
)
FE_LIFECYCLE_MID_MAX = _get_safe(_config, "feature_engineering", "lifecycle", "mid_max", default=36)

# Colunas de serviço (para AdvancedFeatureEngineer)
SERVICE_COLS_ALL = _get_safe(_config, "features", "service_columns", "all", default=[])
SERVICE_COLS_STREAMING = _get_safe(_config, "features", "service_columns", "streaming", default=[])
SERVICE_COLS_SECURITY = _get_safe(_config, "features", "service_columns", "security", default=[])
INACTIVE_SERVICE_VALUES = _get_safe(
    _config,
    "features",
    "inactive_service_values",
    default=["No", "No phone service", "No internet service"],
)

# Valores válidos para features categóricas (de features.valid_values)
VALID_VALUES = _get_safe(_config, "features", "valid_values", default={})

# Valores de tipo de contrato (usados em feature engineering)
CONTRACT_VALUES = _get_safe(
    _config,
    "features",
    "valid_values",
    "Contract",
    default=["Month-to-month", "One year", "Two year"],
)

# Padrões de pré-processamento
CATEGORICAL_FILL_VALUE = _get_safe(
    _config, "preprocessing", "imputation", "categorical_fill_value", default="missing"
)
SCALER_TYPE = _get_safe(_config, "preprocessing", "scaler", "type", default="robust")
ENCODER_HANDLE_UNKNOWN = _get_safe(
    _config, "preprocessing", "encoder", "handle_unknown", default="ignore"
)
ENCODER_DROP_FIRST = _get_safe(_config, "preprocessing", "encoder", "drop_first", default=False)
MISSING_NUMERIC_STRATEGY = _get_safe(
    _config, "features", "missing", "numeric_strategy", default="median"
)
MISSING_CATEGORICAL_STRATEGY = _get_safe(
    _config, "features", "missing", "categorical_strategy", default="most_frequent"
)

# Interpretabilidade / Visualização
INTERPRET_TOP_N = _get_safe(_config, "interpretability", "feature_importance", "top_n", default=20)
INTERPRET_TOP_FEATURES = _get_safe(_config, "interpretability", "top_features", default=10)
VIZ_FIGSIZE_LARGE = tuple(
    _get_safe(_config, "interpretability", "visualization", "figsize_large", default=[12, 8])
)
VIZ_FIGSIZE_MEDIUM = tuple(
    _get_safe(_config, "interpretability", "visualization", "figsize_medium", default=[10, 6])
)
VIZ_FIGSIZE_SMALL = tuple(
    _get_safe(_config, "interpretability", "visualization", "figsize_small", default=[8, 6])
)
VIZ_TOP_FACTORS = _get_safe(_config, "interpretability", "visualization", "top_factors", default=3)
SHAP_SAMPLE_SIZE = _get_safe(
    _config, "interpretability", "visualization", "shap_sample_size", default=500
)
SHAP_N_SAMPLES = _get_safe(_config, "interpretability", "shap", "n_samples", default=1000)

# Monitoramento de drift
DRIFT_PSI_BINS = _get_safe(_config, "drift", "psi_bins", default=10)
DRIFT_PSI_THRESHOLDS = {
    "none": _get_safe(_config, "drift", "psi_thresholds", "none", default=0.10),
    "low": _get_safe(_config, "drift", "psi_thresholds", "low", default=0.20),
    "moderate": _get_safe(_config, "drift", "psi_thresholds", "moderate", default=0.25),
}
DRIFT_ESCALATION_MODERATE_COUNT = _get_safe(
    _config, "drift", "escalation", "moderate_feature_count", default=2
)
DRIFT_ESCALATION_LOW_FRACTION = _get_safe(
    _config, "drift", "escalation", "low_feature_fraction", default=0.3
)
DRIFT_QUANTILE_LOWER = _get_safe(_config, "drift", "quantile_clip", "lower", default=0.01)
DRIFT_QUANTILE_UPPER = _get_safe(_config, "drift", "quantile_clip", "upper", default=0.99)

# Batch/resposta da API
API_BATCH_MAX_SIZE = _get_safe(_config, "api", "batch", "max_size", default=1000)
API_METRIC_PRECISION = _get_safe(_config, "api", "response", "metric_precision", default=4)

# Espaços de busca de otimização (carregados do YAML)
_search_space = _get_safe(_config, "optimization", "search_space", default={})
SEARCH_SPACE = {
    "n_estimators": tuple(_search_space.get("n_estimators", [50, 400])),
    "max_depth": tuple(_search_space.get("max_depth", [3, 10])),
    "learning_rate": tuple(_search_space.get("learning_rate", [0.01, 0.3])),
    "subsample": tuple(_search_space.get("subsample", [0.6, 1.0])),
    "colsample_bytree": tuple(_search_space.get("colsample_bytree", [0.5, 1.0])),
    "min_child_weight": tuple(_search_space.get("min_child_weight", [1, 15])),
    "gamma": tuple(_search_space.get("gamma", [0.0, 2.0])),
    "reg_alpha": tuple(_search_space.get("reg_alpha", [0.0, 2.0])),
    "reg_lambda": tuple(_search_space.get("reg_lambda", [0.0, 2.0])),
    "scale_pos_weight": tuple(_search_space.get("scale_pos_weight", [1.0, 5.0])),
    # Específicos do LightGBM
    "num_leaves": tuple(_search_space.get("num_leaves", [20, 150])),
    "min_child_samples": tuple(_search_space.get("min_child_samples", [5, 50])),
    # Específicos do CatBoost
    "l2_leaf_reg": tuple(_search_space.get("l2_leaf_reg", [1.0, 10.0])),
}

# Padrões de otimização (carregados do YAML)
DEFAULT_N_TRIALS = _get_safe(_config, "optimization", "n_trials", default=50)
DEFAULT_OPTIMIZATION_METRIC = _get_safe(_config, "optimization", "metric", default="f1")
DEFAULT_OPTIMIZE_THRESHOLD = _get_safe(_config, "optimization", "optimize_threshold", default=True)
DEFAULT_OPTIMIZATION_TIMEOUT = _get_safe(_config, "optimization", "timeout", default=None)


# =============================================================================
# REGISTRO DE MODELOS
# =============================================================================

MODEL_REGISTRY: dict[str, dict[str, Any]] = _get_safe(_config, "model", "registry", default={})


def create_model(algorithm: str, model_params: dict[str, Any]) -> Any:
    """Cria uma instância de modelo usando a configuração do registro YAML.

    O registro (``model.registry`` no YAML) mapeia nomes de algoritmos para suas
    classes Python, filtros de parâmetros, mapeamentos e padrões. Isso substitui
    todos os branches hardcoded ``if model_type == …``.

    Args:
        algorithm: Chave no ``model.registry`` (ex.: ``"xgboost"``, ``"lightgbm"``).
        model_params: Dict de parâmetros brutos (ex.: de ``ModelConfig.to_dict()``).

    Returns:
        Classificador instanciado pronto para ``.fit()``.

    Raises:
        ValueError: Se *algorithm* não for encontrado no registro.
        ImportError: Se o pacote do modelo não estiver instalado.
    """
    import importlib

    if algorithm not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "(nenhum)"
        raise ValueError(
            f"Algoritmo '{algorithm}' não encontrado no registry. "
            f"Disponíveis: {available}. "
            f"Adicione-o em model.registry no config/project.yaml."
        )

    algo_cfg = MODEL_REGISTRY[algorithm]
    class_path: str = algo_cfg["class"]

    # Import dinâmico
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Pacote '{module_path}' não instalado. "
            f"Instale com: pip install {module_path.split('.')[0]}"
        ) from exc
    model_class = getattr(module, class_name)

    # 1. Filtrar parâmetros excluídos
    exclude = set(algo_cfg.get("exclude_params", []))
    filtered = {k: v for k, v in model_params.items() if k not in exclude}

    # 2. Aplicar mapeamento de parâmetros (dos parâmetros originais, mesmo se excluídos)
    for old_name, new_name in algo_cfg.get("param_mapping", {}).items():
        if old_name in model_params:
            filtered[new_name] = model_params[old_name]
            filtered.pop(old_name, None)

    # 3. Aplicar parâmetros padrão (maior prioridade — sempre sobrescrevem)
    filtered.update(algo_cfg.get("default_params", {}))

    return model_class(**filtered)


# =============================================================================
# DATACLASSES DE CONFIGURAÇÃO
# =============================================================================


@dataclass
class ModelConfig:
    """Configuração para treinamento de modelo.

    Os padrões são carregados de ``config/project.yaml`` > ``config/default.yaml``.
    Todos os campos podem ser sobrescritos na construção.
    """

    model_type: str = field(default_factory=lambda: DEFAULT_MODEL_TYPE)
    n_estimators: int = field(
        default_factory=lambda: _get_safe(_config, "model", "params", "n_estimators", default=300)
    )
    max_depth: int = field(
        default_factory=lambda: _get_safe(_config, "model", "params", "max_depth", default=6)
    )
    learning_rate: float = field(
        default_factory=lambda: _get_safe(_config, "model", "params", "learning_rate", default=0.05)
    )
    subsample: float = field(
        default_factory=lambda: _get_safe(_config, "model", "params", "subsample", default=0.8)
    )
    colsample_bytree: float = field(
        default_factory=lambda: _get_safe(
            _config, "model", "params", "colsample_bytree", default=0.8
        )
    )
    min_child_weight: int = field(
        default_factory=lambda: _get_safe(_config, "model", "params", "min_child_weight", default=1)
    )
    gamma: float = field(
        default_factory=lambda: _get_safe(_config, "model", "params", "gamma", default=0.0)
    )
    reg_alpha: float = field(
        default_factory=lambda: _get_safe(_config, "model", "params", "reg_alpha", default=0.0)
    )
    reg_lambda: float = field(
        default_factory=lambda: _get_safe(_config, "model", "params", "reg_lambda", default=1.0)
    )
    scale_pos_weight: float = field(
        default_factory=lambda: _get_safe(
            _config, "model", "params", "scale_pos_weight", default=1.0
        )
    )
    random_state: int = RANDOM_STATE
    eval_metric: str = field(
        default_factory=lambda: _get_safe(
            _config, "model", "params", "eval_metric", default="logloss"
        )
    )

    # Específicos do LightGBM (incluídos no to_dict apenas quando definidos)
    num_leaves: int | None = None
    min_child_samples: int | None = None

    def __post_init__(self) -> None:
        # Aceita tanto enum ModelType quanto string para compatibilidade
        if hasattr(self.model_type, "value"):
            self.model_type = self.model_type.value

    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionário para inicialização do modelo."""
        d = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
            "verbosity": 0,
        }
        # Parâmetros específicos do LightGBM (incluídos apenas quando definidos)
        if self.num_leaves is not None:
            d["num_leaves"] = self.num_leaves
        if self.min_child_samples is not None:
            d["min_child_samples"] = self.min_child_samples
        return d


# =============================================================================
# MODEL PATH UTILITIES
# =============================================================================


def get_model_path(name: str | None = None, version: str | None = None) -> Path:
    """
    Get path for model file.

    Args:
        name: Model name (without extension). Defaults to DEFAULT_MODEL_NAME.
        version: Optional version suffix.

    Returns:
        Path to model file.
    """
    name = name or DEFAULT_MODEL_NAME
    if version:
        return MODELS_DIR / f"{name}_v{version}.joblib"
    return MODELS_DIR / f"{name}.joblib"


def list_available_models() -> list[dict[str, Any]]:
    """
    List all available models in the models directory.

    Returns:
        List of dictionaries with model info (name, path, size, modified).
    """
    models = []

    if not MODELS_DIR.exists():
        return models

    for model_file in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            stat = model_file.stat()
            models.append(
                {
                    "name": model_file.stem,
                    "filename": model_file.name,
                    "path": str(model_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "is_default": model_file.stem == DEFAULT_MODEL_NAME,
                }
            )
        except OSError:
            continue

    return models
