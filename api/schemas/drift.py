"""
Pydantic Schemas for Data Drift API
====================================

Request and response models for drift detection endpoints.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


# =============================================================================
# Response Models
# =============================================================================


class FeatureDriftResult(BaseModel):
    """Drift result for a single feature."""

    name: str = Field(..., description="Feature name")
    psi: float = Field(..., description="Population Stability Index value")
    ks_statistic: float = Field(..., description="KS test statistic")
    ks_pvalue: float = Field(..., description="KS test p-value")
    severity: str = Field(..., description="Drift severity: none, low, moderate, high")
    has_drift: bool = Field(..., description="Whether significant drift was detected")


class DriftReportResponse(BaseModel):
    """Complete drift detection report response."""

    success: bool = Field(default=True, description="Whether drift detection succeeded")
    timestamp: str = Field(..., description="Report timestamp (ISO format)")
    features_checked: int = Field(..., description="Number of features analyzed")
    features_with_drift: int = Field(..., description="Number of features with drift")
    drift_percentage: float = Field(..., description="Percentage of features with drift")
    severity_counts: dict[str, int] = Field(..., description="Count per severity level")
    overall_severity: str = Field(
        ..., description="Overall drift severity: none, low, moderate, high"
    )
    features: list[FeatureDriftResult] = Field(
        ..., description="Per-feature drift results sorted by PSI (descending)"
    )
    recommendation: str = Field(..., description="Actionable recommendation")


class DriftErrorResponse(BaseModel):
    """Error response for drift detection."""

    success: bool = Field(default=False)
    error_code: str
    message: str
    timestamp: str = Field(default_factory=lambda: _utc_now().isoformat())
