"""
Módulo de Monitoramento de Modelo
==================================

Fornece ferramentas para detecção de drift de dados e degradação de performance.
Implementa testes PSI (Population Stability Index) e KS (Kolmogorov-Smirnov).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import (
    DRIFT_ESCALATION_LOW_FRACTION,
    DRIFT_ESCALATION_MODERATE_COUNT,
    DRIFT_PSI_BINS,
    DRIFT_PSI_THRESHOLDS,
)
from src.enums import DriftSeverity
from src.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DriftResult:
    """Resultado da detecção de drift para uma única feature."""

    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    severity: DriftSeverity

    @property
    def has_drift(self) -> bool:
        """Verifica se a feature tem drift significativo."""
        return self.severity in (DriftSeverity.MODERATE, DriftSeverity.HIGH)


@dataclass
class DriftReport:
    """Relatório completo de detecção de drift."""

    timestamp: datetime
    features_checked: int
    features_with_drift: int
    severity_counts: dict[str, int]
    feature_results: list[DriftResult]
    overall_severity: DriftSeverity

    def to_dict(self) -> dict[str, Any]:
        """Converte relatório para dicionário."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "features_checked": self.features_checked,
            "features_with_drift": self.features_with_drift,
            "severity_counts": self.severity_counts,
            "overall_severity": self.overall_severity.value,
            "features": [
                {
                    "name": r.feature_name,
                    "psi": round(r.psi, 4),
                    "ks_statistic": round(r.ks_statistic, 4),
                    "ks_pvalue": round(r.ks_pvalue, 4),
                    "severity": r.severity.value,
                    "has_drift": r.has_drift,
                }
                for r in self.feature_results
            ],
        }


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = DRIFT_PSI_BINS
) -> float:
    """
    Calcula o Population Stability Index (PSI).

    PSI mede o quanto uma distribuição mudou entre dois conjuntos de dados.

    Args:
        expected: Distribuição de referência/treinamento
        actual: Distribuição atual/produção
        bins: Número de bins para histograma

    Returns:
        Valor PSI (0 = sem mudança, >0.25 = mudança significativa)

    Interpretação PSI (ver classify_drift_severity):
        - PSI < 0.10: Sem mudança significativa (NONE)
        - 0.10 <= PSI < 0.20: Mudança baixa, monitoramento recomendado (LOW)
        - 0.20 <= PSI < 0.25: Mudança moderada, investigação recomendada (MODERATE)
        - PSI >= 0.25: Mudança significativa, ação necessária (HIGH)
    """
    # Criar bins a partir da distribuição de referência
    _, bin_edges = np.histogram(expected, bins=bins)

    expected_perc, _ = np.histogram(expected, bins=bin_edges)
    actual_perc, _ = np.histogram(actual, bins=bin_edges)

    # Normalizar para porcentagens (proteger contra divisão por zero)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    # Adicionar valor pequeno para evitar log(0)
    epsilon = 1e-6
    expected_perc = np.clip(expected_perc, epsilon, 1)
    actual_perc = np.clip(actual_perc, epsilon, 1)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


def classify_drift_severity(psi: float) -> DriftSeverity:
    """
    Classifica severidade do drift com base no valor PSI.

    Args:
        psi: Valor do Population Stability Index

    Returns:
        Valor do enum DriftSeverity
    """
    if psi < DRIFT_PSI_THRESHOLDS["none"]:
        return DriftSeverity.NONE
    elif psi < DRIFT_PSI_THRESHOLDS["low"]:
        return DriftSeverity.LOW
    elif psi < DRIFT_PSI_THRESHOLDS["moderate"]:
        return DriftSeverity.MODERATE
    else:
        return DriftSeverity.HIGH


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    categorical_features: list[str] | None = None,
) -> DriftReport:
    """
    Detecta drift de dados entre datasets de referência e atual.

    Usa PSI (Population Stability Index) para todas as features
    e teste KS (Kolmogorov-Smirnov) para validação adicional.

    Args:
        reference_df: Dataset de referência/treinamento
        current_df: Dataset atual/produção
        features: Lista de features a verificar
        categorical_features: Reservado para uso futuro (não utilizado atualmente)

    Returns:
        DriftReport com resultados detalhados
    """
    logger.info(f"Starting drift detection for {len(features)} features")

    categorical_features = categorical_features or []
    results: list[DriftResult] = []
    severity_counts = {s.value: 0 for s in DriftSeverity}

    for col in features:
        if col not in reference_df.columns or col not in current_df.columns:
            logger.warning(f"Feature {col} not found in one of the datasets")
            continue

        ref_data = reference_df[col].dropna()
        curr_data = current_df[col].dropna()

        if len(ref_data) == 0 or len(curr_data) == 0:
            continue

        # Calculate PSI (works for both numeric and encoded categorical)
        try:
            psi = population_stability_index(ref_data.values, curr_data.values)
        except Exception as e:
            logger.warning(f"PSI calculation failed for {col}: {e}")
            psi = 0.0

        # KS test for numeric features
        try:
            ks_stat, ks_pvalue = ks_2samp(ref_data.values, curr_data.values)
        except Exception as e:
            logger.warning(f"KS test failed for {col}: {e}")
            ks_stat, ks_pvalue = 0.0, 1.0

        severity = classify_drift_severity(psi)
        severity_counts[severity.value] += 1

        results.append(
            DriftResult(
                feature_name=col,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pvalue,
                severity=severity,
            )
        )

    # Determine overall severity
    features_with_drift = sum(1 for r in results if r.has_drift)

    if severity_counts[DriftSeverity.HIGH.value] > 0:
        overall_severity = DriftSeverity.HIGH
    elif severity_counts[DriftSeverity.MODERATE.value] > DRIFT_ESCALATION_MODERATE_COUNT:
        overall_severity = DriftSeverity.MODERATE
    elif severity_counts[DriftSeverity.LOW.value] > len(results) * DRIFT_ESCALATION_LOW_FRACTION:
        overall_severity = DriftSeverity.LOW
    else:
        overall_severity = DriftSeverity.NONE

    report = DriftReport(
        timestamp=datetime.now(UTC),
        features_checked=len(results),
        features_with_drift=features_with_drift,
        severity_counts=severity_counts,
        feature_results=sorted(results, key=lambda x: x.psi, reverse=True),
        overall_severity=overall_severity,
    )

    logger.info(
        "Drift detection completed",
        extra={
            "features_checked": len(results),
            "features_with_drift": features_with_drift,
            "overall_severity": overall_severity.value,
            "max_psi": max((r.psi for r in results), default=0),
        },
    )

    return report
