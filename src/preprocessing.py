"""
Módulo de Pré-processamento
============================

Utilitários de pré-processamento de dados incluindo divisão treino-teste,
pipelines de transformação de features e encoding.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.config import (
    CATEGORICAL_FILL_VALUE,
    ENCODER_DROP_FIRST,
    ENCODER_HANDLE_UNKNOWN,
    MISSING_CATEGORICAL_STRATEGY,
    MISSING_NUMERIC_STRATEGY,
    POSITIVE_CLASS,
    RANDOM_STATE,
    SCALER_TYPE,
    TEST_SIZE,
)
from src.logger import setup_logger
from src.utils import convert_target_to_binary

logger = setup_logger(__name__)


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide dados em conjuntos de treinamento e teste.

    Args:
        df: DataFrame contendo features e target
        target_column: Nome da coluna target
        test_size: Proporção de dados para teste (padrão: 0.2)
        random_state: Seed aleatória para reprodutibilidade
        stratify: Se deve usar amostragem estratificada

    Returns:
        Tupla de (X_train, X_test, y_train, y_test)
    """
    logger.info(
        "Splitting data",
        extra={"total_samples": len(df), "test_size": test_size, "stratify": stratify},
    )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Converter target para binário se necessário (usando função centralizada)
    if y.dtype == "object":
        y = convert_target_to_binary(y, positive_class=POSITIVE_CLASS)

    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    logger.info(
        "Data split completed",
        extra={
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_churn_rate": y_train.mean(),
            "test_churn_rate": y_test.mean(),
        },
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numeric_features,
    categorical_features,
    scaler_type=SCALER_TYPE,
    numeric_impute_strategy=MISSING_NUMERIC_STRATEGY,
    categorical_impute_strategy=MISSING_CATEGORICAL_STRATEGY,
    handle_unknown=ENCODER_HANDLE_UNKNOWN,
    drop_first=ENCODER_DROP_FIRST,
    remainder="drop",
):
    """
    Constrói preprocessador robusto com imputation e normalização.

    Processa features numéricas e categóricas separadamente:
    - Numéricas: Imputation → Normalização (StandardScaler/MinMaxScaler/RobustScaler)
    - Categóricas: Imputation → One-Hot Encoding

    Parameters
    ----------
    numeric_features : list
        Lista de nomes de features numéricas
    categorical_features : list
        Lista de nomes de features categóricas
    scaler_type : str, default='standard'
        Tipo de scaler: 'standard', 'minmax', 'robust', ou None (sem scaling)
    numeric_impute_strategy : str, default='median'
        Estratégia de imputation numérica: 'mean', 'median', 'most_frequent'
    categorical_impute_strategy : str, default='constant'
        Estratégia de imputation categórica: 'most_frequent', 'constant'
    handle_unknown : str, default='ignore'
        Como lidar com categorias desconhecidas no OneHotEncoder:
        - 'ignore': ignora (recomendado)
        - 'error': lança erro
    drop_first : bool, default=False
        Se True, remove primeira categoria (evita multicolinearidade)
    remainder : str, default='drop'
        Como tratar colunas não especificadas: 'drop', 'passthrough'

    Returns
    -------
    ColumnTransformer
        Preprocessador configurado e pronto para fit/transform

    Raises
    ------
    ValueError
        Se listas de features estiverem vazias ou scaler_type for inválido

    """
    # ============================================================
    # Validações de entrada
    # ============================================================
    if not numeric_features and not categorical_features:
        raise ValueError(
            "Pelo menos uma lista (numeric_features ou categorical_features) "
            "deve conter features"
        )

    if numeric_features and not isinstance(numeric_features, list | tuple):
        raise TypeError("numeric_features deve ser uma lista ou tupla")

    if categorical_features and not isinstance(categorical_features, list | tuple):
        raise TypeError("categorical_features deve ser uma lista ou tupla")

    # Validar scaler_type
    valid_scalers = {"standard", "minmax", "robust", None}
    if scaler_type not in valid_scalers:
        raise ValueError(
            f"scaler_type '{scaler_type}' inválido. " f"Opções válidas: {valid_scalers}"
        )

    logger.info(
        "Construindo preprocessador",
        extra={
            "numeric_features": len(numeric_features) if numeric_features else 0,
            "categorical_features": len(categorical_features) if categorical_features else 0,
            "scaler_type": scaler_type,
            "numeric_impute_strategy": numeric_impute_strategy,
            "categorical_impute_strategy": categorical_impute_strategy,
        },
    )

    # ============================================================
    # Construir transformers
    # ============================================================
    transformers = []

    # Pipeline para features numéricas
    if numeric_features:
        numeric_steps = [("imputer", SimpleImputer(strategy=numeric_impute_strategy))]

        # Adicionar scaler se especificado
        if scaler_type == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
            logger.debug("Scaler: StandardScaler")
        elif scaler_type == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
            logger.debug("Scaler: MinMaxScaler")
        elif scaler_type == "robust":
            numeric_steps.append(("scaler", RobustScaler()))
            logger.debug("Scaler: RobustScaler")
        elif scaler_type is None:
            logger.debug("Scaler: Nenhum (apenas imputation)")

        numeric_transformer = Pipeline(steps=numeric_steps)
        transformers.append(("num", numeric_transformer, numeric_features))

        logger.info(f"Scaler selecionado: {scaler_type}")
        logger.info(f"Pipeline numérico configurado: {len(numeric_features)} features")

    # Pipeline para features categóricas
    if categorical_features:
        categorical_steps = []

        # Imputation categórica
        if categorical_impute_strategy == "constant":
            categorical_steps.append(
                ("imputer", SimpleImputer(strategy="constant", fill_value=CATEGORICAL_FILL_VALUE))
            )
        elif categorical_impute_strategy == "most_frequent":
            categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

        # One-Hot Encoding
        categorical_steps.append(
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown=handle_unknown,
                    sparse_output=False,
                    drop="first" if drop_first else None,
                ),
            )
        )

        categorical_transformer = Pipeline(steps=categorical_steps)
        transformers.append(("cat", categorical_transformer, categorical_features))

        logger.info(f"Pipeline categórico configurado: {len(categorical_features)} features")

    # ============================================================
    # Combinar transformers
    # ============================================================
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
        n_jobs=-1,  # Paralelização
        verbose=False,
    )

    logger.info("Preprocessador construído com sucesso")

    return preprocessor
