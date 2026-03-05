"""
Testes para src/notebook_utils.py
=================================

Cobre funções de cálculo de IV, detecção de outliers, métricas
de classificação e funções de avaliação (com plots desabilitados).
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend não-interativo para testes

from src.notebook_utils import (
    calculate_classification_metrics,
    calculate_iv_categorical,
    calculate_iv_numeric,
    detect_outliers_iqr,
    detect_outliers_mad,
    evaluate_model,
    plot_confusion_matrix,
    plot_distribution_analysis,
    plot_precision_recall_curve,
    plot_roc_curve,
)

# =============================================================================
# IV Calculations
# =============================================================================


class TestIVCategorical:
    def test_basic_iv(self):
        df = pd.DataFrame(
            {
                "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
                "target": [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            }
        )
        iv = calculate_iv_categorical(df, "category", "target")
        assert isinstance(iv, float)
        assert iv >= 0

    def test_perfectly_predictive(self):
        df = pd.DataFrame(
            {
                "category": ["A"] * 50 + ["B"] * 50,
                "target": [1] * 50 + [0] * 50,
            }
        )
        iv = calculate_iv_categorical(df, "category", "target")
        assert iv > 1.0  # Should be very high

    def test_no_predictive_power(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "category": ["A"] * 100 + ["B"] * 100,
                "target": np.random.randint(0, 2, 200),
            }
        )
        iv = calculate_iv_categorical(df, "category", "target")
        assert iv < 0.1  # Should be low


class TestIVNumeric:
    def test_basic_iv(self):
        np.random.seed(42)
        n = 500
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [np.random.normal(5, 1, n // 2), np.random.normal(3, 1, n // 2)]
                ),
                "target": [1] * (n // 2) + [0] * (n // 2),
            }
        )
        iv = calculate_iv_numeric(df, "value", "target")
        assert isinstance(iv, float)
        assert iv >= 0

    def test_with_different_bins(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "value": np.random.randn(200),
                "target": np.random.randint(0, 2, 200),
            }
        )
        iv5 = calculate_iv_numeric(df, "value", "target", bins=5)
        iv20 = calculate_iv_numeric(df, "value", "target", bins=20)
        assert isinstance(iv5, float)
        assert isinstance(iv20, float)


# =============================================================================
# Outlier Detection
# =============================================================================


class TestDetectOutliersIQR:
    def test_no_outliers(self):
        series = pd.Series([1, 2, 3, 4, 5])
        result = detect_outliers_iqr(series)
        assert result["total_outliers"] == 0

    def test_with_outliers(self):
        series = pd.Series([1, 2, 3, 4, 5, 100])
        result = detect_outliers_iqr(series)
        assert result["total_outliers"] >= 1
        assert result["outliers_upper"] >= 1

    def test_result_structure(self):
        series = pd.Series([10, 20, 30, 40, 50])
        result = detect_outliers_iqr(series)
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "outliers_lower" in result
        assert "outliers_upper" in result
        assert "total_outliers" in result
        assert "outliers_mask" in result

    def test_outliers_mask_is_boolean(self):
        series = pd.Series([1, 2, 3, 100])
        result = detect_outliers_iqr(series)
        assert result["outliers_mask"].dtype == bool


class TestDetectOutliersMAD:
    def test_no_outliers(self):
        series = pd.Series([10, 11, 12, 13, 14])
        result = detect_outliers_mad(series)
        assert result["outliers_count"] == 0

    def test_with_outliers(self):
        series = pd.Series([10, 11, 12, 13, 14, 1000])
        result = detect_outliers_mad(series)
        assert result["outliers_count"] >= 1

    def test_all_same_values(self):
        series = pd.Series([5, 5, 5, 5, 5])
        result = detect_outliers_mad(series)
        assert result["mad_value"] == 0

    def test_custom_threshold(self):
        series = pd.Series([1, 2, 3, 4, 5, 50])
        result_strict = detect_outliers_mad(series, threshold=2.0)
        result_lenient = detect_outliers_mad(series, threshold=5.0)
        assert result_strict["outliers_count"] >= result_lenient["outliers_count"]


# =============================================================================
# Classification Metrics
# =============================================================================


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 0, 1]
        metrics = calculate_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_with_proba(self):
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 0, 1]
        y_proba = [0.1, 0.9, 0.2, 0.8, 0.15, 0.85]
        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0.9

    def test_confusion_matrix_values(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        metrics = calculate_classification_metrics(y_true, y_pred)
        assert metrics["true_negatives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["true_positives"] == 1
        assert metrics["false_negatives"] == 1

    def test_specificity(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1, 0]
        metrics = calculate_classification_metrics(y_true, y_pred)
        assert "specificity" in metrics
        assert 0 <= metrics["specificity"] <= 1


# =============================================================================
# Plotting Functions (backend Agg, sem display)
# =============================================================================


class TestPlotFunctions:
    def test_confusion_matrix_returns_figure(self):
        import matplotlib.pyplot as plt

        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        fig = plot_confusion_matrix(y_true, y_pred, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_confusion_matrix_no_normalize(self):
        import matplotlib.pyplot as plt

        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        fig = plot_confusion_matrix(y_true, y_pred, normalize=False, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_roc_curve_returns_figure(self):
        import matplotlib.pyplot as plt

        y_true = [0, 0, 1, 1]
        y_proba = [0.1, 0.4, 0.6, 0.9]
        fig = plot_roc_curve(y_true, y_proba, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_precision_recall_curve_returns_figure(self):
        import matplotlib.pyplot as plt

        y_true = [0, 0, 1, 1]
        y_proba = [0.1, 0.4, 0.6, 0.9]
        fig = plot_precision_recall_curve(y_true, y_proba, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_distribution_analysis(self):
        import matplotlib.pyplot as plt

        data = pd.Series(np.random.randn(100))
        fig, axes = plot_distribution_analysis(data, "test_col")
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3
        plt.close(fig)

    def test_roc_curve_with_string_labels(self):
        import matplotlib.pyplot as plt

        y_true = pd.Series(["No", "No", "Yes", "Yes"])
        y_proba = [0.1, 0.4, 0.6, 0.9]
        fig = plot_roc_curve(y_true, y_proba, pos_label="Yes", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# evaluate_model
# =============================================================================


class TestEvaluateModel:
    def test_full_evaluation(self):
        import matplotlib.pyplot as plt
        from sklearn.dummy import DummyClassifier

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        model = DummyClassifier(strategy="stratified", random_state=42)
        model.fit(X, y)

        results = evaluate_model(model, X, y, model_name="Test", show_plots=False)
        assert "metrics" in results
        assert "fig_confusion_matrix" in results
        assert "fig_roc" in results
        assert "fig_precision_recall" in results
        assert results["metrics"]["accuracy"] >= 0
        plt.close("all")
