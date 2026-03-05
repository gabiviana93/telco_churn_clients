import json

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_DIR_RAW,
    FILENAME,
    ID_COL,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET,
)
from src.monitoring import detect_drift
from src.preprocessing import split_data


def main():
    # 1. Carregar dados de referência (treino)
    df = pd.read_csv(DATA_DIR_RAW / FILENAME)
    df = df.drop(columns=[ID_COL])

    X_train, X_test, _, _ = split_data(df, TARGET)

    # 2. Detectar drift entre treino e teste (ou novos dados)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # Filtrar apenas features presentes nos dois DataFrames
    features = [f for f in features if f in X_train.columns and f in X_test.columns]

    report = detect_drift(
        reference_df=X_train,
        current_df=X_test,
        features=features,
        categorical_features=CATEGORICAL_FEATURES,
    )

    # 3. Salvar relatório
    with open(REPORTS_DIR / "drift.json", "w") as f:
        json.dump(report.to_dict(), f, indent=4)

    print(f"Drift detection concluído: {report.overall_severity.value}")
    print(f"Features com drift: {report.features_with_drift}/{report.features_checked}")


if __name__ == "__main__":
    main()
