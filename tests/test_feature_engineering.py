"""
Testes para src/feature_engineering.py
======================================

Cobre FeatureEngineer e AdvancedFeatureEngineer: fit, transform,
features criadas, correções de dados, e pipeline completo.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import AdvancedFeatureEngineer, FeatureEngineer


@pytest.fixture
def telco_df():
    """DataFrame representativo do dataset Telco Churn."""
    return pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 0, 1, 0],
            "Partner": ["Yes", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
            "tenure": [0, 1, 12, 24, 36, 48, 60, 72],
            "PhoneService": ["Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": [
                "No",
                "Yes",
                "No",
                "No phone service",
                "Yes",
                "No",
                "Yes",
                "No",
            ],
            "InternetService": [
                "Fiber optic",
                "Fiber optic",
                "DSL",
                "No",
                "DSL",
                "Fiber optic",
                "Fiber optic",
                "DSL",
            ],
            "OnlineSecurity": [
                "No",
                "No",
                "Yes",
                "No internet service",
                "Yes",
                "No",
                "No",
                "Yes",
            ],
            "OnlineBackup": [
                "No",
                "No",
                "Yes",
                "No internet service",
                "No",
                "Yes",
                "No",
                "Yes",
            ],
            "DeviceProtection": [
                "No",
                "Yes",
                "Yes",
                "No internet service",
                "No",
                "No",
                "No",
                "Yes",
            ],
            "TechSupport": [
                "No",
                "No",
                "Yes",
                "No internet service",
                "Yes",
                "No",
                "No",
                "Yes",
            ],
            "StreamingTV": [
                "No",
                "Yes",
                "No",
                "No internet service",
                "Yes",
                "Yes",
                "Yes",
                "No",
            ],
            "StreamingMovies": [
                "No",
                "Yes",
                "Yes",
                "No internet service",
                "No",
                "Yes",
                "Yes",
                "No",
            ],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
                "One year",
                "Month-to-month",
                "Month-to-month",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "Yes", "No", "No", "Yes", "Yes", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Electronic check",
                "Bank transfer (automatic)",
                "Mailed check",
                "Credit card (automatic)",
                "Electronic check",
                "Electronic check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 95.0, 50.0, 20.0, 75.0, 90.0, 105.0, 35.0],
            "TotalCharges": [
                "29.85",
                "95.0",
                "600.0",
                "480.0",
                "2700.0",
                "4320.0",
                "6300.0",
                "2520.0",
            ],
            "Churn": ["Yes", "Yes", "No", "No", "No", "Yes", "Yes", "No"],
        }
    )


@pytest.fixture
def telco_X_y(telco_df):
    """Retorna X e y separados."""
    X = telco_df.drop(columns=["Churn", "customerID"])
    y = telco_df["Churn"]
    return X, y


# =============================================================================
# FeatureEngineer
# =============================================================================


class TestFeatureEngineer:
    def test_fit_returns_self(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit(X, y)
        assert result is fe

    def test_fit_stores_statistics(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        fe.fit(X, y)
        assert fe.monthly_bin_edges_ is not None
        assert fe.avg_monthly_charge_median_ is not None
        assert fe.monthly_median_ is not None
        assert fe.contract_avg_tenure_ is not None
        assert fe.global_avg_tenure_ is not None
        assert fe.contract_order_map_ is not None
        assert fe.contract_churn_rate_map_ is not None
        assert fe.global_churn_rate_ is not None

    def test_transform_creates_features(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        fe.fit(X, y)
        result = fe.transform(X)
        assert result.shape[1] > X.shape[1]
        # Verifica features esperadas
        expected_features = [
            "new_customer_flag",
            "tenure_group",
            "avg_monthly_charge",
            "is_monthly_contract",
            "long_term_contract",
            "early_churn_risk",
            "tenure_vs_contract_avg",
        ]
        for feat in expected_features:
            assert feat in result.columns, f"Feature {feat} não foi criada"

    def test_fit_transform_equivalent(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result_separate = fe.fit(X, y).transform(X)
        fe2 = FeatureEngineer()
        result_combined = fe2.fit_transform(X, y)
        pd.testing.assert_frame_equal(result_separate, result_combined)

    def test_fix_total_charges(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        fe.fit(X, y)
        result = fe.transform(X)
        # TotalCharges deve ser numérico
        assert pd.api.types.is_numeric_dtype(result["TotalCharges"])

    def test_new_customer_flag(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit_transform(X, y)
        # Primeiro cliente tem tenure=0, deve ser flagged
        assert result.iloc[0]["new_customer_flag"] == 1
        assert result.iloc[2]["new_customer_flag"] == 0

    def test_monthly_contract_flags(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit_transform(X, y)
        # Primeiro cliente é Month-to-month
        assert result.iloc[0]["is_monthly_contract"] == 1
        assert result.iloc[0]["long_term_contract"] == 0
        # Quarto cliente é Two year
        assert result.iloc[3]["is_monthly_contract"] == 0
        assert result.iloc[3]["long_term_contract"] == 1

    def test_nonlinear_transforms(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit_transform(X, y)
        assert "MonthlyCharges_log" in result.columns
        assert "MonthlyCharges_squared" in result.columns
        assert "TotalCharges_log" in result.columns

    def test_supervised_encodings(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit_transform(X, y)
        assert "contract_risk_ordinal" in result.columns
        assert "contract_churn_rate" in result.columns

    def test_fit_without_target(self, telco_X_y):
        X, _ = telco_X_y
        fe = FeatureEngineer()
        fe.fit(X)  # sem y
        result = fe.transform(X)
        # Supervised encodings não devem existir
        assert fe.contract_order_map_ is None
        assert result.shape[1] > X.shape[1]

    def test_charges_mismatch_flag(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit_transform(X, y)
        assert "charges_mismatch_flag" in result.columns

    def test_price_sensitivity(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit_transform(X, y)
        assert "is_high_MonthlyCharges" in result.columns
        assert "price_sensitive_contract" in result.columns

    def test_segment_comparisons(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit_transform(X, y)
        assert "tenure_vs_contract_avg" in result.columns

    def test_monthly_bins(self, telco_X_y):
        X, y = telco_X_y
        fe = FeatureEngineer()
        result = fe.fit_transform(X, y)
        assert "MonthlyCharges_binned" in result.columns


# =============================================================================
# AdvancedFeatureEngineer
# =============================================================================


class TestAdvancedFeatureEngineer:
    def test_creates_more_features_than_base(self, telco_X_y):
        X, y = telco_X_y
        base = FeatureEngineer()
        adv = AdvancedFeatureEngineer()
        base_result = base.fit_transform(X, y)
        adv_result = adv.fit_transform(X, y)
        assert adv_result.shape[1] > base_result.shape[1]

    def test_service_count_features(self, telco_X_y):
        X, y = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X, y)
        expected = [
            "total_services",
            "service_density",
            "streaming_services",
            "has_streaming",
            "security_services",
            "has_security",
        ]
        for feat in expected:
            assert feat in result.columns, f"Feature {feat} não criada"

    def test_engagement_features(self, telco_X_y):
        X, y = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X, y)
        assert "low_engagement" in result.columns
        assert "high_engagement" in result.columns
        assert "digital_only" in result.columns
        assert "family_account" in result.columns
        assert "family_stability" in result.columns

    def test_risk_scores(self, telco_X_y):
        X, y = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X, y)
        assert "churn_risk_score" in result.columns
        assert "high_risk_flag" in result.columns
        assert "very_high_risk_flag" in result.columns
        assert "internet_risk" in result.columns
        assert "payment_risk" in result.columns

    def test_lifecycle_features(self, telco_X_y):
        X, y = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X, y)
        assert "lifecycle_new" in result.columns
        assert "lifecycle_early" in result.columns
        assert "lifecycle_mid" in result.columns
        assert "lifecycle_mature" in result.columns
        assert "tenure_years" in result.columns
        assert "tenure_sqrt" in result.columns

    def test_value_features(self, telco_X_y):
        X, y = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X, y)
        assert "estimated_clv" in result.columns
        assert "value_per_month" in result.columns
        assert "cost_per_service" in result.columns

    def test_multi_service_interactions(self, telco_X_y):
        X, y = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X, y)
        assert "fiber_no_security" in result.columns
        assert "streamer_unprotected" in result.columns
        assert "senior_monthly" in result.columns
        assert "new_high_value" in result.columns

    def test_fit_without_target(self, telco_X_y):
        X, _ = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X)
        # Sem target, risk maps supervisionados não são criados
        assert adv.internet_service_risk_ is None
        assert result.shape[1] > X.shape[1]

    def test_sklearn_compatible(self, telco_X_y):
        """Testa compatibilidade com sklearn (get_params, set_params)."""
        adv = AdvancedFeatureEngineer()
        params = adv.get_params()
        assert "convert_target" in params

    def test_high_risk_customer_scenario(self, telco_X_y):
        """Cliente com fiber+mensal+sem segurança deve ter alto risk score."""
        X, y = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X, y)
        # Segundo cliente: Fiber optic, Month-to-month, tenure=1, no security
        assert result.iloc[1]["churn_risk_score"] > 0

    def test_new_customer_with_zero_tenure(self, telco_X_y):
        """Cliente novo (tenure=0) não deve causar divisão por zero."""
        X, y = telco_X_y
        adv = AdvancedFeatureEngineer()
        result = adv.fit_transform(X, y)
        # avg_monthly_charge para tenure=0 deve ser 0 (não NaN/inf)
        assert np.isfinite(result.iloc[0]["avg_monthly_charge"])
