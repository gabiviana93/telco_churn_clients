"""
Testes para src/utils.py
========================

Cobre: generate_model_filename, parse_model_filename, convert_target_to_binary,
validate_target, normalize_metrics_keys, classify_risk, auto_detect_features,
extract_model_components.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from src.utils import (
    auto_detect_features,
    classify_risk,
    convert_target_to_binary,
    extract_model_components,
    generate_model_filename,
    normalize_metrics_keys,
    parse_model_filename,
    validate_target,
)

# =============================================================================
# auto_detect_features
# =============================================================================


class TestAutoDetectFeatures:
    def test_basic_detection(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.5, 2.5]})
        num, cat = auto_detect_features(df)
        assert "a" in num
        assert "c" in num
        assert "b" in cat

    def test_all_numeric(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        num, cat = auto_detect_features(df)
        assert len(num) == 2
        assert len(cat) == 0

    def test_all_categorical(self):
        df = pd.DataFrame({"x": ["a", "b"], "y": pd.Categorical(["c", "d"])})
        num, cat = auto_detect_features(df)
        assert len(num) == 0
        assert len(cat) == 2


# =============================================================================
# classify_risk
# =============================================================================


class TestClassifyRisk:
    def test_high_risk(self):
        assert classify_risk(0.9) == "HIGH"
        assert classify_risk(0.7) == "HIGH"

    def test_medium_risk(self):
        assert classify_risk(0.5) == "MEDIUM"

    def test_low_risk(self):
        assert classify_risk(0.1) == "LOW"
        assert classify_risk(0.0) == "LOW"

    def test_boundary_values(self):
        from src.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW

        assert classify_risk(RISK_THRESHOLD_HIGH) == "HIGH"
        assert classify_risk(RISK_THRESHOLD_LOW) == "MEDIUM"
        assert classify_risk(RISK_THRESHOLD_LOW - 0.001) == "LOW"


# =============================================================================
# normalize_metrics_keys
# =============================================================================


class TestNormalizeMetricsKeys:
    def test_standard_normalization(self):
        result = normalize_metrics_keys({"F1-Score": 0.85, "ROC-AUC": 0.90})
        assert result == {"f1_score": 0.85, "roc_auc": 0.90}

    def test_lowercase_variants(self):
        result = normalize_metrics_keys({"f1-score": 0.8, "roc-auc": 0.9})
        assert "f1_score" in result
        assert "roc_auc" in result

    def test_auprc_variants(self):
        result = normalize_metrics_keys({"AUPRC": 0.75, "average_precision": 0.76})
        assert result["auprc"] == 0.76  # last one wins

    def test_unknown_keys_passthrough(self):
        result = normalize_metrics_keys({"custom_metric": 0.5, "F1-Score": 0.8})
        assert result["custom_metric"] == 0.5
        assert result["f1_score"] == 0.8

    def test_cv_means(self):
        result = normalize_metrics_keys({"cv_f1_mean": 0.82, "cv_roc_auc_mean": 0.91})
        assert result["f1_score"] == 0.82
        assert result["roc_auc"] == 0.91

    def test_empty_dict(self):
        assert normalize_metrics_keys({}) == {}

    def test_precision_recall_accuracy(self):
        result = normalize_metrics_keys({"Precision": 0.8, "Recall": 0.7, "Accuracy": 0.85})
        assert result["precision"] == 0.8
        assert result["recall"] == 0.7
        assert result["accuracy"] == 0.85


# =============================================================================
# generate_model_filename / parse_model_filename
# =============================================================================


class TestModelFilename:
    def test_generate_with_metrics(self):
        name = generate_model_filename("lgbm", {"f1_score": 0.85, "roc_auc": 0.88})
        assert name.startswith("churn_lgbm_f1-85_auc-88_")
        assert name.endswith(".joblib")

    def test_generate_without_metrics(self):
        name = generate_model_filename("xgb", include_metrics=False)
        assert "f1-" not in name
        assert name.startswith("churn_xgb_")

    def test_generate_without_timestamp(self):
        name = generate_model_filename("rf", {"f1_score": 0.7}, include_timestamp=False)
        assert name == "churn_rf_f1-70.joblib"

    def test_generate_custom_prefix(self):
        name = generate_model_filename(
            "xgb", prefix="myproject", include_timestamp=False, include_metrics=False
        )
        assert name == "myproject_xgb.joblib"

    def test_parse_basic(self):
        result = parse_model_filename("churn_lgbm_f1-85_auc-88_20250101_120000.joblib")
        assert result["algorithm"] == "lgbm"
        assert result["f1_score"] == 0.85
        assert result["roc_auc"] == 0.88
        assert result["prefix"] == "churn"
        assert result["timestamp"] is not None

    def test_parse_no_metrics(self):
        result = parse_model_filename("churn_xgb.joblib")
        assert result["algorithm"] == "xgb"
        assert result["f1_score"] is None

    def test_parse_unknown_format(self):
        result = parse_model_filename("model.joblib")
        assert result["prefix"] == "model"

    def test_roundtrip(self):
        name = generate_model_filename(
            "catboost", {"f1_score": 0.90, "roc_auc": 0.95}, include_timestamp=False
        )
        parsed = parse_model_filename(name)
        assert parsed["algorithm"] == "catboost"
        assert parsed["f1_score"] == 0.90
        assert parsed["roc_auc"] == 0.95

    def test_parse_with_invalid_metric(self):
        result = parse_model_filename("churn_xgb_f1-abc_auc-xyz.joblib")
        assert result["f1_score"] is None
        assert result["roc_auc"] is None


# =============================================================================
# convert_target_to_binary
# =============================================================================


class TestConvertTarget:
    def test_yes_no_conversion(self):
        y = pd.Series(["Yes", "No", "Yes", "No"])
        result = convert_target_to_binary(y)
        assert list(result) == [1, 0, 1, 0]

    def test_already_binary(self):
        y = pd.Series([0, 1, 0, 1])
        result = convert_target_to_binary(y)
        assert list(result) == [0, 1, 0, 1]

    def test_list_input(self):
        result = convert_target_to_binary(["Yes", "No", "Yes"])
        assert list(result) == [1, 0, 1]

    def test_numpy_input(self):
        result = convert_target_to_binary(np.array([0, 1, 1, 0]))
        assert list(result) == [0, 1, 1, 0]

    def test_custom_positive_class(self):
        y = pd.Series(["Churn", "NoChurn", "Churn"])
        result = convert_target_to_binary(y, positive_class="Churn")
        assert list(result) == [1, 0, 1]

    def test_unsupported_type_raises(self):
        y = pd.Series([1.5, 2.5, 3.5])
        with pytest.raises(ValueError, match="não suportado"):
            convert_target_to_binary(y)

    def test_categorical_dtype(self):
        y = pd.Categorical(["Yes", "No", "Yes"])
        result = convert_target_to_binary(y)
        assert list(result) == [1, 0, 1]


# =============================================================================
# validate_target
# =============================================================================


class TestValidateTarget:
    def test_valid_binary(self):
        assert validate_target(pd.Series([0, 1, 0, 1])) is True

    def test_invalid_values_raises(self):
        with pytest.raises(ValueError, match="0 e 1"):
            validate_target(pd.Series([0, 1, 2]))

    def test_numpy_input(self):
        assert validate_target(np.array([0, 1, 1, 0])) is True

    def test_list_input(self):
        assert validate_target([0, 1, 0]) is True


# =============================================================================
# extract_model_components
# =============================================================================


class TestExtractModelComponents:
    def test_raw_model(self):
        model = DummyClassifier()
        tree, prep, fe = extract_model_components(model)
        assert tree is model
        assert prep is None
        assert fe is None

    def test_sklearn_pipeline(self):
        pipe = SklearnPipeline([("preprocessing", StandardScaler()), ("model", DummyClassifier())])
        tree, prep, fe = extract_model_components(pipe)
        assert isinstance(tree, DummyClassifier)
        assert isinstance(prep, StandardScaler)
        assert fe is None

    def test_pipeline_with_feature_engineering(self):
        pipe = SklearnPipeline(
            [
                ("feature_engineering", StandardScaler()),
                ("preprocessing", StandardScaler()),
                ("model", DummyClassifier()),
            ]
        )
        tree, prep, fe = extract_model_components(pipe)
        assert isinstance(tree, DummyClassifier)
        assert isinstance(prep, StandardScaler)
        assert fe is not None

    def test_object_with_best_model(self):
        class MockOptimizer:
            best_model = DummyClassifier()
            preprocessor = StandardScaler()

        tree, prep, fe = extract_model_components(MockOptimizer())
        assert isinstance(tree, DummyClassifier)
        assert isinstance(prep, StandardScaler)
