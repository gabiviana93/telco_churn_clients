"""
Data Drift Routes
=================

API endpoints for data drift detection and monitoring.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from api.core.logging import get_logger
from api.schemas.drift import DriftErrorResponse, DriftReportResponse, FeatureDriftResult
from src.config import DATA_DIR_RAW, FILENAME, get_config
from src.monitoring import detect_drift
from src.utils import convert_target_to_binary

logger = get_logger(__name__)

router = APIRouter(prefix="/drift", tags=["Data Drift"])


def _load_reference_data() -> pd.DataFrame:
    """Load reference (training) dataset."""
    data_path = DATA_DIR_RAW / FILENAME
    if not data_path.exists():
        raise FileNotFoundError(f"Reference data not found at {data_path}")
    df = pd.read_csv(data_path)
    if "Churn" in df.columns:
        config = get_config()
        positive_class = config.get("data", {}).get("positive_class", "Yes")
        df["Churn"] = convert_target_to_binary(df["Churn"], positive_class)
    return df


def _get_feature_lists() -> tuple[list[str], list[str]]:
    """Get numeric and categorical feature lists from config."""
    config = get_config()
    numeric = config.get("features", {}).get("numeric", [])
    categorical = config.get("features", {}).get("categorical", [])
    return numeric, categorical


def _encode_categoricals(
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical columns as integer codes for both DataFrames."""
    ref_encoded = ref_df.copy()
    prod_encoded = prod_df.copy()
    for col in categorical_features:
        if col in ref_encoded.columns:
            ref_encoded[col] = ref_encoded[col].astype("category").cat.codes
        if col in prod_encoded.columns:
            cat_type = pd.CategoricalDtype(categories=ref_df[col].dropna().unique())
            prod_encoded[col] = prod_encoded[col].astype(cat_type).cat.codes
    return ref_encoded, prod_encoded


def _get_recommendation(severity: str) -> str:
    """Return an actionable recommendation based on overall drift severity."""
    recommendations = {
        "none": (
            "No significant drift detected. Production data is aligned with "
            "training data. Continue periodic monitoring."
        ),
        "low": (
            "Minor drift detected. Monitor affected features more frequently. "
            "Evaluate if changes reflect real business trends. "
            "Retraining is not yet necessary."
        ),
        "moderate": (
            "Moderate drift detected. Investigate root cause of changes. "
            "Assess impact on model performance with recent data. "
            "Consider partial or full retraining."
        ),
        "high": (
            "Severe drift detected! Immediate action required. "
            "Retrain the model with recent data. Investigate data source changes. "
            "Validate current model performance against business metrics."
        ),
    }
    return recommendations.get(severity, "Unknown severity level.")


@router.post(
    "/detect",
    response_model=DriftReportResponse,
    responses={
        200: {"description": "Drift detection completed"},
        400: {"model": DriftErrorResponse, "description": "Invalid data"},
        500: {"model": DriftErrorResponse, "description": "Detection error"},
    },
    summary="Detect Data Drift",
    description="""
    Detect data drift between reference (training) data and uploaded production data.

    Upload a CSV file containing production data. The API compares it against the
    reference training dataset using PSI (Population Stability Index) and KS
    (Kolmogorov-Smirnov) tests.
    """,
)
async def detect_data_drift(
    file: UploadFile = File(..., description="CSV file with production data"),
    features: str | None = Query(
        None,
        description=(
            "Comma-separated list of features to check. "
            "If omitted, uses all configured features."
        ),
    ),
) -> DriftReportResponse:
    """Detect drift between reference data and uploaded production CSV."""
    logger.info("Processing drift detection request", filename=file.filename)

    try:
        # Load reference data
        reference_df = _load_reference_data()

        # Read uploaded production data
        content = await file.read()
        try:
            production_df = pd.read_csv(StringIO(content.decode("utf-8")))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "INVALID_CSV",
                    "message": f"Failed to parse CSV file: {str(e)}",
                },
            )

        # Get feature lists
        numeric_features, categorical_features = _get_feature_lists()

        # Use specified features or default to config
        all_features = numeric_features + categorical_features
        if features:
            requested = [f.strip() for f in features.split(",")]
            all_features = [f for f in requested if f in all_features]
            if not all_features:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_code": "NO_VALID_FEATURES",
                        "message": "None of the requested features are valid.",
                    },
                )

        # Validate columns
        valid_features = [
            f
            for f in all_features
            if f in reference_df.columns and f in production_df.columns
        ]
        if not valid_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "MISSING_FEATURES",
                    "message": (
                        "No matching features found between reference and "
                        "production data."
                    ),
                },
            )

        # Encode categoricals
        ref_encoded, prod_encoded = _encode_categoricals(
            reference_df, production_df, categorical_features
        )

        # Run drift detection
        drift_report = detect_drift(
            reference_df=ref_encoded,
            current_df=prod_encoded,
            features=valid_features,
            categorical_features=categorical_features,
        )

        report_dict = drift_report.to_dict()
        severity = drift_report.overall_severity.value
        drift_pct = (
            drift_report.features_with_drift / drift_report.features_checked * 100
            if drift_report.features_checked > 0
            else 0.0
        )

        logger.info(
            "Drift detection completed",
            features_checked=drift_report.features_checked,
            features_with_drift=drift_report.features_with_drift,
            overall_severity=severity,
        )

        return DriftReportResponse(
            success=True,
            timestamp=report_dict["timestamp"],
            features_checked=drift_report.features_checked,
            features_with_drift=drift_report.features_with_drift,
            drift_percentage=round(drift_pct, 2),
            severity_counts=drift_report.severity_counts,
            overall_severity=severity,
            features=[
                FeatureDriftResult(**feat) for feat in report_dict["features"]
            ],
            recommendation=_get_recommendation(severity),
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error("Reference data not found", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "REFERENCE_DATA_NOT_FOUND",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.error("Drift detection failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "DRIFT_DETECTION_ERROR",
                "message": f"Failed to detect drift: {str(e)}",
            },
        )


@router.get(
    "/status",
    response_model=dict[str, Any],
    summary="Drift Monitoring Status",
    description="Get the current status of drift monitoring configuration.",
)
async def drift_status() -> dict[str, Any]:
    """Return drift monitoring configuration and readiness status."""
    numeric_features, categorical_features = _get_feature_lists()

    try:
        ref_df = _load_reference_data()
        ref_available = True
        ref_rows = len(ref_df)
    except FileNotFoundError:
        ref_available = False
        ref_rows = 0

    return {
        "monitoring_enabled": True,
        "reference_data_available": ref_available,
        "reference_data_rows": ref_rows,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "total_features": len(numeric_features) + len(categorical_features),
        "psi_thresholds": {
            "none": "< 0.10",
            "low": "0.10 - 0.20",
            "moderate": "0.20 - 0.25",
            "high": ">= 0.25",
        },
    }
