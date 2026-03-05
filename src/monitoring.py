import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.logger import setup_logger

logger = setup_logger(__name__)

def population_stability_index(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    psi = np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
    )
    return psi


def detect_drift(train_df, new_df, features):
    logger.info(f"Iniciando detecção de drift para {len(features)} features")
    drift_report = {}

    for col in features:
        psi = population_stability_index(
            train_df[col].dropna(),
            new_df[col].dropna()
        )
        drift_report[col] = psi
    
    # Identificar features com drift significativo
    significant_drift = {k: v for k, v in drift_report.items() if v > 0.25}
    
    logger.info("Detecção de drift concluída", extra={
        'features_checked': len(features),
        'features_with_drift': len(significant_drift),
        'max_psi': max(drift_report.values()) if drift_report else 0
    })

    return drift_report
