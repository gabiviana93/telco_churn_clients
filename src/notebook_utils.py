"""
Notebook Utilities
==================

Funções auxiliares para notebooks de análise exploratória (EDA) e modelagem.
Inclui funções de plotagem, cálculo de IV, detecção de outliers e avaliação de modelos.

Este módulo é usado exclusivamente pelos notebooks e NÃO deve ser importado
no código de produção (API, scripts de treino, etc.).

Usage:
    from src.notebook_utils import (
        plot_distribution_analysis,
        calculate_iv_categorical,
        calculate_iv_numeric,
        plot_confusion_matrix,
        evaluate_model,
    )
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, median_abs_deviation, probplot
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import ID_COL, TARGET
from src.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# INFORMATION VALUE (IV)
# =============================================================================


def calculate_iv_categorical(
    df: pd.DataFrame, feature_col: str, target_col: str, smoothing: float = 0.5
) -> float:
    """Calcula Information Value para feature categórica."""
    grouped = df.groupby(feature_col, observed=False)[target_col].agg(["sum", "count"])
    grouped.columns = ["event", "total"]
    grouped["non_event"] = grouped["total"] - grouped["event"]

    total_event = grouped["event"].sum()
    total_non_event = grouped["non_event"].sum()
    k = grouped.shape[0]

    grouped["event_dist"] = (grouped["event"] + smoothing) / (total_event + smoothing * k)
    grouped["non_event_dist"] = (grouped["non_event"] + smoothing) / (
        total_non_event + smoothing * k
    )
    grouped["woe"] = np.log(grouped["event_dist"] / grouped["non_event_dist"])
    grouped["iv"] = (grouped["event_dist"] - grouped["non_event_dist"]) * grouped["woe"]

    return grouped["iv"].sum()


def calculate_iv_numeric(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    bins: int = 10,
    min_bin_size: float = 0.05,
    smoothing: float = 0.5,
) -> float:
    """Calcula Information Value para feature numérica com binning."""
    data = df[[feature_col, target_col]].dropna().copy()

    lower, upper = data[feature_col].quantile([0.01, 0.99])
    data = data[(data[feature_col] >= lower) & (data[feature_col] <= upper)]

    data["bin"] = pd.qcut(data[feature_col], q=bins, duplicates="drop")
    grouped = data.groupby("bin", observed=True)[target_col].agg(["sum", "count"])
    grouped.columns = ["event", "total"]
    grouped["non_event"] = grouped["total"] - grouped["event"]

    total_count = grouped["total"].sum()
    grouped = grouped[grouped["total"] / total_count >= min_bin_size]

    total_event = grouped["event"].sum()
    total_non_event = grouped["non_event"].sum()
    k = grouped.shape[0]

    grouped["event_dist"] = (grouped["event"] + smoothing) / (total_event + smoothing * k)
    grouped["non_event_dist"] = (grouped["non_event"] + smoothing) / (
        total_non_event + smoothing * k
    )
    grouped["woe"] = np.log(grouped["event_dist"] / grouped["non_event_dist"])
    grouped["iv"] = (grouped["event_dist"] - grouped["non_event_dist"]) * grouped["woe"]

    return grouped["iv"].sum()


def plot_iv_ranking_and_summary(iv_df: pd.DataFrame, type_stats=None, top_n: int = 20):
    """Plota ranking de IV com visualização e resumo."""
    if type_stats is None:
        type_stats = iv_df.groupby("Type")["IV"].agg(["mean", "max", "count"])

    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    topn = iv_df.head(top_n).copy()
    colors = [
        "#2ecc71" if iv >= 0.5 else "#3498db" if iv >= 0.1 else "#95a5a6" for iv in topn["IV"]
    ]
    axes.barh(range(len(topn)), topn["IV"], color=colors)
    axes.set_yticks(range(len(topn)))
    axes.set_yticklabels(topn["Feature"], fontsize=10)
    axes.invert_yaxis()
    axes.set_xlabel("Information Value (IV)", fontsize=12, fontweight="bold")
    axes.set_title(f"Top {top_n} Features por IV", fontsize=14, fontweight="bold")
    axes.axvline(x=0.5, color="green", linestyle="--", alpha=0.7, label="IV > 0.5 (Forte)")
    axes.axvline(x=0.1, color="orange", linestyle="--", alpha=0.7, label="IV > 0.1 (Médio)")
    axes.axvline(x=0.02, color="red", linestyle="--", alpha=0.7, label="IV > 0.02 (Fraco)")
    axes.legend(loc="lower right", fontsize=9)
    axes.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# OUTLIERS
# =============================================================================


def detect_outliers_iqr(series: pd.Series) -> dict:
    """Detecta outliers usando método IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_lower = (series < lower_bound).sum()
    outliers_upper = (series > upper_bound).sum()
    outliers_mask = (series < lower_bound) | (series > upper_bound)

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers_lower": outliers_lower,
        "outliers_upper": outliers_upper,
        "total_outliers": outliers_lower + outliers_upper,
        "outliers_mask": outliers_mask,
    }


def detect_outliers_mad(series: pd.Series, threshold: float = 3.5) -> dict:
    """Detecta outliers usando método MAD (Median Absolute Deviation)."""
    median = np.median(series)
    mad = median_abs_deviation(series)

    if mad == 0:
        modified_z_scores = np.zeros(len(series))
    else:
        modified_z_scores = 0.6745 * (series - median) / mad

    outliers_mask = np.abs(modified_z_scores) > threshold
    outliers_count = outliers_mask.sum()

    return {
        "threshold": threshold,
        "outliers_count": outliers_count,
        "outliers_mask": outliers_mask,
        "mad_value": mad,
    }


# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================


def plot_distribution_analysis(data: pd.Series, col_name: str, figsize: tuple = (15, 5)) -> tuple:
    """Cria visualização completa de distribuição (histograma, boxplot, Q-Q plot)."""
    clean_data = data.dropna()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Histograma + KDE
    axes[0].hist(clean_data, bins=50, edgecolor="black", alpha=0.7, density=True)
    kde = gaussian_kde(clean_data)
    x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
    axes[0].plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
    axes[0].axvline(
        clean_data.mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Média: {clean_data.mean():.1f}",
    )
    axes[0].axvline(
        clean_data.median(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {clean_data.median():.1f}",
    )
    axes[0].set_title(f"{col_name} - Histograma + KDE")
    axes[0].legend()

    # Box Plot
    bp = axes[1].boxplot(clean_data, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    axes[1].set_title(f"{col_name} - Box Plot")

    # Q-Q Plot
    probplot(clean_data, dist="norm", plot=axes[2])
    axes[2].set_title(f"{col_name} - Q-Q Plot")

    plt.tight_layout()
    return fig, axes


# =============================================================================
# CATEGORICAL ANALYSIS
# =============================================================================


def plot_churn_by_column(dataset, column):
    """Plota percentual de churn por categoria."""
    df_plot = dataset.groupby([column, TARGET]).size().reset_index(name="count")
    df_plot["percent"] = df_plot.groupby(column)["count"].transform(lambda x: x / x.sum() * 100)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_plot, x=column, y="percent", hue=TARGET, palette=["#4C72B0", "#DD8452"])
    plt.ylabel("Percentual (%)")
    plt.title(f"Percentual de Churn por {column}")
    plt.xticks(rotation=45)
    plt.show()


def panel_categoric_features(dataset, target_col=TARGET, id_col=ID_COL):
    """Cria painel de visualizações para variáveis categóricas."""
    plt.style.use("seaborn-v0_8-whitegrid")

    categorical_cols = dataset.select_dtypes(include="object").columns.tolist()
    cols_plot = [col for col in categorical_cols if col not in [id_col, target_col]]
    n_cols = len(cols_plot)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(cols_plot):
        dist = dataset[col].value_counts(normalize=True).sort_values(ascending=False).head(10)
        ax = axes[idx]
        bars = ax.bar(dist.index.astype(str), dist.values, color="#4C72B0", alpha=0.85)

        for bar, val in zip(bars, dist.values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(f"Distribuição de {col}", fontsize=13, fontweight="bold")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=40)

    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# =============================================================================
# MODEL EVALUATION
# =============================================================================


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[str] | None = None,
    model_name: str = "Model",
    normalize: bool = True,
    figsize: tuple[int, int] = (12, 5),
    cmap: str = "Blues",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plota matriz de confusão (absoluta e normalizada)."""
    if labels is None:
        labels = sorted(pd.Series(y_true).unique().tolist())

    cm_abs = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    n_plots = 2 if normalize else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    sns.heatmap(
        cm_abs,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title(f"Matriz de Confusão - {model_name}\n(Valores Absolutos)")
    axes[0].set_xlabel("Predito")
    axes[0].set_ylabel("Real")

    if normalize:
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[1],
        )
        axes[1].set_title(f"Matriz de Confusão - {model_name}\n(Normalizada)")
        axes[1].set_xlabel("Predito")
        axes[1].set_ylabel("Real")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    pos_label: str | int = 1,
    model_name: str = "Model",
    figsize: tuple[int, int] = (8, 6),
    show: bool = True,
) -> plt.Figure:
    """Plota curva ROC."""
    if isinstance(pos_label, str):
        y_true_binary = (y_true == pos_label).astype(int)
    else:
        y_true_binary = y_true

    fpr, tpr, _ = roc_curve(y_true_binary, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    pos_label: str | int = 1,
    model_name: str = "Model",
    figsize: tuple[int, int] = (8, 6),
    show: bool = True,
) -> plt.Figure:
    """Plota curva Precision-Recall."""
    if isinstance(pos_label, str):
        y_true_binary = (y_true == pos_label).astype(int)
    else:
        y_true_binary = y_true

    precision, recall, _ = precision_recall_curve(y_true_binary, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color="blue", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def calculate_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series | None = None,
    pos_label: str | int = 1,
) -> dict[str, float]:
    """Calcula métricas de classificação."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

    return metrics


def evaluate_model(
    model,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    labels: list[str] | None = None,
    pos_label: str | int = 1,
    model_name: str = "Model",
    show_plots: bool = True,
) -> dict[str, Any]:
    """Avaliação completa de modelo com métricas e visualizações."""
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba_full = model.predict_proba(X_test)
        y_proba = y_proba_full[:, 1]

    metrics = calculate_classification_metrics(y_test, y_pred, y_proba, pos_label)

    results = {"metrics": metrics}

    fig_cm = plot_confusion_matrix(
        y_test, y_pred, labels=labels, model_name=model_name, show=show_plots
    )
    results["fig_confusion_matrix"] = fig_cm

    if y_proba is not None:
        fig_roc = plot_roc_curve(
            y_test, y_proba, pos_label=pos_label, model_name=model_name, show=show_plots
        )
        results["fig_roc"] = fig_roc

        fig_pr = plot_precision_recall_curve(
            y_test, y_proba, pos_label=pos_label, model_name=model_name, show=show_plots
        )
        results["fig_precision_recall"] = fig_pr

    logger.info("\nCLASSIFICATION REPORT:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=labels))

    return results
