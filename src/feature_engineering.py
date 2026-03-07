"""
Módulo de Feature Engineering para Churn Prediction.

Contém FeatureEngineer (features básicas) e AdvancedFeatureEngineer
(features avançadas com 67+ transformações) para o dataset Telco Churn.
Todas as estatísticas são calculadas no treino (fit) e aplicadas
em transform, evitando data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    CONTRACT_VALUES,
    FE_CHARGES_DIVERGENCE,
    FE_EARLY_CHURN_TENURE,
    FE_HIGH_ENGAGEMENT_MIN,
    FE_LIFECYCLE_EARLY_MAX,
    FE_LIFECYCLE_MID_MAX,
    FE_LIFECYCLE_NEW_MAX,
    FE_LOW_ENGAGEMENT_MAX,
    FE_RISK_SCORE_HIGH,
    FE_RISK_SCORE_VERY_HIGH,
    FE_RISK_WEIGHTS,
    INACTIVE_SERVICE_VALUES,
    MONTHLY_CAT_LABELS,
    POSITIVE_CLASS,
    SERVICE_COLS_ALL,
    SERVICE_COLS_SECURITY,
    SERVICE_COLS_STREAMING,
    TENURE_BINS,
    TENURE_LABELS,
)
from src.logger import setup_logger
from src.utils import convert_target_to_binary, validate_target

logger = setup_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer customizado para Feature Engineering do dataset Telco Churn.

    Cria features derivadas sem vazamento de informação (data leakage),
    calculando estatísticas apenas no conjunto de treino durante fit()
    e aplicando-as em transform().

    Features criadas:
    - Flags de cliente novo e tipo de contrato
    - Binning de MonthlyCharges em quartis
    - Métricas de gasto médio mensal
    - Flags de sensibilidade a preço
    - Interações comportamentais (early churn risk, price sensitivity)
    - Comparativos com média do segmento
    - Codificações ordinais e por risco
    - Transformações logarítmicas e quadráticas

    Parameters
    ----------
    convert_target : bool, default=True
        Se True, converte automaticamente target categórico para binário no fit()

    Attributes
    ----------
    monthly_bin_edges_ : ndarray
        Edges dos bins de MonthlyCharges calculados no treino
    avg_monthly_charge_median_ : float
        Mediana de avg_monthly_charge no treino
    monthly_median_ : float
        Mediana de MonthlyCharges no treino
    contract_avg_tenure_ : dict
        Tenure médio por tipo de contrato no treino
    contract_order_map_ : dict
        Mapeamento ordinal de Contract baseado em churn rate
    contract_churn_rate_map_ : dict
        Taxa de churn por tipo de contrato
    global_churn_rate_ : float
        Taxa global de churn no treino
    global_avg_tenure_ : float
        Tenure médio global no treino
    """

    def __init__(self, convert_target=True):
        self.convert_target = convert_target
        self.monthly_bin_edges_ = None
        self.avg_monthly_charge_median_ = None
        self.monthly_median_ = None
        self.contract_avg_tenure_ = None
        self.contract_order_map_ = None
        self.contract_churn_rate_map_ = None
        self.global_churn_rate_ = None
        self.global_avg_tenure_ = None

    def fit(self, X, y=None):
        """
        Calcula estatísticas no conjunto de treino para uso posterior.

        Parameters
        ----------
        X : pd.DataFrame
            Features de entrada
        y : pd.Series or array-like or None, optional
            Target (necessário para mapeamentos supervisionados).
            Pode ser categórico ('Yes'/'No') ou binário (0/1).
            Se categórico, será automaticamente convertido para binário.

        Returns
        -------
        self
        """
        df = X.copy()

        logger.info("Iniciando fit do FeatureEngineer")
        logger.info(f"Shape de entrada: {df.shape}")

        # Converter target se necessário
        if y is not None and self.convert_target:
            y = convert_target_to_binary(y, positive_class=POSITIVE_CLASS)
            validate_target(y)

        # Correções de base
        df = self._fix_total_charges(df)

        # Bins de MonthlyCharges (quantis em treino)
        if "MonthlyCharges" in df.columns:
            self.monthly_bin_edges_ = self._calculate_monthly_bins(df["MonthlyCharges"])
            logger.info(f"Bins de MonthlyCharges: {self.monthly_bin_edges_}")

        # Medianas em treino
        if {"TotalCharges", "tenure"}.issubset(df.columns):
            avg_monthly_charge = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0)
            self.avg_monthly_charge_median_ = float(pd.Series(avg_monthly_charge).median())
            logger.info(f"Mediana avg_monthly_charge: {self.avg_monthly_charge_median_:.2f}")

        if "MonthlyCharges" in df.columns:
            self.monthly_median_ = float(df["MonthlyCharges"].median())
            logger.info(f"Mediana MonthlyCharges: {self.monthly_median_:.2f}")

        # Médias por contrato
        if {"Contract", "tenure"}.issubset(df.columns):
            self.contract_avg_tenure_ = df.groupby("Contract")["tenure"].mean().to_dict()
            self.global_avg_tenure_ = float(df["tenure"].mean())
            logger.info(f"Tenure médio por contrato: {self.contract_avg_tenure_}")
        else:
            self.contract_avg_tenure_ = {}
            self.global_avg_tenure_ = None

        # Mapeamentos supervisionados (se y fornecido)
        if (y is not None) and ("Contract" in df.columns):
            self._fit_supervised_encodings(df, y)
        else:
            self.contract_order_map_ = None
            self.contract_churn_rate_map_ = None
            self.global_churn_rate_ = None

        logger.info("Fit do FeatureEngineer concluído")
        return self

    def transform(self, X):
        """
        Aplica transformações de feature engineering usando estatísticas do fit.

        Parameters
        ----------
        X : pd.DataFrame
            Features de entrada

        Returns
        -------
        pd.DataFrame
            DataFrame transformado com features derivadas
        """
        logger.info(f"Iniciando transform - shape inicial: {X.shape}")

        df = (
            X.copy()
            # Correções de base
            .pipe(self._fix_total_charges)
            # Features de tenure
            .pipe(self._create_tenure_features)
            # Features de charges
            .pipe(self._create_charge_features)
            # Features de contrato
            .pipe(self._create_contract_features)
            # Binning de MonthlyCharges
            .pipe(self._create_monthly_bins)
            # Sensibilidade a preço
            .pipe(self._create_price_sensitivity_features)
            # Interações comportamentais
            .pipe(self._create_behavioral_interactions)
            # Comparativos com segmento
            .pipe(self._create_segment_comparisons)
            # Mapeamentos supervisionados
            .pipe(self._create_supervised_encodings)
            # Transformações não-lineares
            .pipe(self._create_nonlinear_transforms)
        )

        logger.info(f"Transform concluído - shape final: {df.shape}")
        logger.info(f"Features criadas: {df.shape[1] - X.shape[1]} novas features")
        return df

    def _fix_total_charges(self, df):
        """Corrige TotalCharges para numérico e preenche nulos de clientes novos. Necessário antes de gerar outras features."""
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        if "tenure" in df.columns and "TotalCharges" in df.columns:
            cond = (df["tenure"] == 0) & (df["TotalCharges"].isna())
            n_fixed = cond.sum()
            if n_fixed > 0:
                df.loc[cond, "TotalCharges"] = 0
                logger.debug(f"Corrigidos {n_fixed} valores de TotalCharges para clientes novos")

        return df

    def _calculate_monthly_bins(self, monthly_charges):
        """Calcula bins de MonthlyCharges usando quantis."""
        quantiles = monthly_charges.quantile([0, 0.25, 0.5, 0.75, 1]).values
        bin_edges = np.unique(quantiles)

        if len(bin_edges) < 5:
            bin_edges = np.linspace(monthly_charges.min(), monthly_charges.max(), 5)

        # Extend edges to -inf/+inf so unseen values in test data are binned
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        return bin_edges

    def _fit_supervised_encodings(self, df, y):
        """Calcula mapeamentos supervisionados baseados no target."""
        # Alinhar índices para evitar mismatch após train_test_split
        y_aligned = pd.Series(np.asarray(y), index=df.index)
        churn_by_contract = y_aligned.groupby(df["Contract"]).mean()
        order = churn_by_contract.sort_values(ascending=False).index

        self.contract_order_map_ = {name: i for i, name in enumerate(order)}
        self.contract_churn_rate_map_ = churn_by_contract.to_dict()
        self.global_churn_rate_ = float(y_aligned.mean())

        logger.info(f"Mapeamento ordinal: {self.contract_order_map_}")
        logger.info(f"Taxa de churn por contrato: {self.contract_churn_rate_map_}")

    def _create_tenure_features(self, df):
        """Cria features relacionadas ao tempo de permanência."""
        if "tenure" in df.columns:
            df["new_customer_flag"] = (df["tenure"] == 0).astype(int)

            df["tenure_group"] = pd.cut(
                df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
            )

        return df

    def _create_charge_features(self, df):
        """Cria features derivadas de charges."""
        if {"TotalCharges", "tenure"}.issubset(df.columns):
            df["avg_monthly_charge"] = np.where(
                df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0
            )

            if "MonthlyCharges" in df.columns:
                expected_total = df["MonthlyCharges"] * df["tenure"]
                actual_total = df["TotalCharges"]
                divergence = abs(actual_total - expected_total) / (actual_total + 1e-6)
                df["charges_mismatch_flag"] = (divergence > FE_CHARGES_DIVERGENCE).astype(int)

        return df

    def _create_contract_features(self, df):
        """Cria features relacionadas ao tipo de contrato."""
        if "Contract" in df.columns:
            df["is_monthly_contract"] = (df["Contract"] == CONTRACT_VALUES[0]).astype(int)
            df["long_term_contract"] = df["Contract"].isin(CONTRACT_VALUES[1:]).astype(int)

        return df

    def _create_monthly_bins(self, df):
        """Cria binning de MonthlyCharges usando edges do fit."""
        if "MonthlyCharges" in df.columns and self.monthly_bin_edges_ is not None:
            labels = MONTHLY_CAT_LABELS[: len(self.monthly_bin_edges_) - 1]
            df["MonthlyCharges_binned"] = pd.cut(
                df["MonthlyCharges"],
                bins=self.monthly_bin_edges_,
                labels=labels,
                include_lowest=True,
            )

        return df

    def _create_price_sensitivity_features(self, df):
        """Cria features de sensibilidade a preço."""
        if "avg_monthly_charge" in df.columns and self.avg_monthly_charge_median_ is not None:
            df["is_high_MonthlyCharges"] = (
                df["avg_monthly_charge"] > self.avg_monthly_charge_median_
            ).astype(int)

        if "MonthlyCharges" in df.columns and self.monthly_median_ is not None:
            df["price_sensitive_contract"] = (
                (df.get("is_monthly_contract", 0) == 1)
                & (df["MonthlyCharges"] > self.monthly_median_)
            ).astype(int)

        return df

    def _create_behavioral_interactions(self, df):
        """Cria interações comportamentais entre features."""
        if {"tenure", "is_monthly_contract"}.issubset(df.columns):
            df["early_churn_risk"] = (
                (df["is_monthly_contract"] == 1) & (df["tenure"] < FE_EARLY_CHURN_TENURE)
            ).astype(int)

        return df

    def _create_segment_comparisons(self, df):
        """Cria comparações com médias do segmento."""
        if {"Contract", "tenure"}.issubset(df.columns):
            avg_map = self.contract_avg_tenure_ or {}
            if self.global_avg_tenure_ is None:
                logger.warning("global_avg_tenure_ não disponível - usando 1.0 como fallback")
            default_avg = self.global_avg_tenure_ if self.global_avg_tenure_ is not None else 1.0
            contract_avg = df["Contract"].map(avg_map).fillna(default_avg)
            df["tenure_vs_contract_avg"] = df["tenure"] / contract_avg.replace(0, 1)

        return df

    def _create_supervised_encodings(self, df):
        """Cria codificações baseadas no target (usando estatísticas do fit)."""
        if "Contract" in df.columns and self.contract_order_map_ is not None:
            df["contract_risk_ordinal"] = (
                df["Contract"].map(self.contract_order_map_).fillna(0).astype(int)
            )

        if "Contract" in df.columns and self.contract_churn_rate_map_ is not None:
            if self.global_churn_rate_ is None:
                logger.warning("global_churn_rate_ não disponível - usando 0.5 como fallback")
            global_rate = self.global_churn_rate_ if self.global_churn_rate_ is not None else 0.5
            df["contract_churn_rate"] = (
                df["Contract"].map(self.contract_churn_rate_map_).fillna(global_rate)
            )

        return df

    def _create_nonlinear_transforms(self, df):
        """Cria transformações logarítmicas e quadráticas."""
        transform_cols = ["MonthlyCharges", "TotalCharges", "avg_monthly_charge"]

        for col in transform_cols:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
                df[f"{col}_squared"] = df[col] ** 2

        return df


class AdvancedFeatureEngineer(FeatureEngineer):
    """
    Feature Engineering avançado para maximizar F1-Score.

    Adiciona features mais sofisticadas além do FeatureEngineer base:
    - Contagem e scores de serviços
    - Indicadores de engagement
    - Features de interação multi-serviço
    - Risk scores compostos
    - Features de ciclo de vida do cliente

    Estas features são projetadas especificamente para melhorar
    a predição de churn em datasets de telecomunicações.
    """

    def __init__(self, convert_target=True):
        super().__init__(convert_target)
        self.service_cols_ = None
        self.streaming_cols_ = None
        self.security_cols_ = None
        self.internet_service_risk_ = None
        self.payment_method_risk_ = None

    def fit(self, X, y=None):
        """Fit com cálculos adicionais para features avançadas."""
        super().fit(X, y)

        df = X.copy()
        df = self._fix_total_charges(df)

        # Identificar colunas de serviço (via config)
        self.service_cols_ = SERVICE_COLS_ALL or [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        self.streaming_cols_ = SERVICE_COLS_STREAMING or ["StreamingTV", "StreamingMovies"]
        self.security_cols_ = SERVICE_COLS_SECURITY or [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
        ]

        # Risk scores por categoria
        if y is not None:
            y_binary = convert_target_to_binary(y, positive_class=POSITIVE_CLASS)
            # Alinhar índices para evitar KeyError após train_test_split
            y_binary = pd.Series(y_binary.values, index=df.index)

            if "InternetService" in df.columns:
                self.internet_service_risk_ = (
                    df.groupby("InternetService")
                    .apply(lambda x: y_binary.loc[x.index].mean(), include_groups=False)
                    .to_dict()
                )

            if "PaymentMethod" in df.columns:
                self.payment_method_risk_ = (
                    df.groupby("PaymentMethod")
                    .apply(lambda x: y_binary.loc[x.index].mean(), include_groups=False)
                    .to_dict()
                )

        logger.info("AdvancedFeatureEngineer fit concluído")
        return self

    def transform(self, X):
        """Transform com features avançadas adicionais."""
        # Aplicar transformações base
        df = super().transform(X)

        # Features avançadas
        df = (
            df.pipe(self._create_service_count_features)
            .pipe(self._create_engagement_features)
            .pipe(self._create_multi_service_interactions)
            .pipe(self._create_risk_scores)
            .pipe(self._create_lifecycle_features)
            .pipe(self._create_value_features)
        )

        logger.info(f"AdvancedFeatureEngineer transform - shape final: {df.shape}")
        return df

    def _create_service_count_features(self, df):
        """Cria features de contagem de serviços."""
        # Contar serviços ativos
        service_binary = []
        for col in self.service_cols_ or []:
            if col in df.columns:
                # "Yes" ou valores específicos indicam serviço ativo
                is_active = ~df[col].isin(INACTIVE_SERVICE_VALUES)
                service_binary.append(is_active.astype(int))

        if service_binary:
            df["total_services"] = sum(service_binary)
            df["service_density"] = df["total_services"] / len(service_binary)

        # Serviços de streaming
        streaming_count = 0
        for col in self.streaming_cols_ or []:
            if col in df.columns:
                streaming_count += (~df[col].isin(INACTIVE_SERVICE_VALUES)).astype(int)
        df["streaming_services"] = streaming_count
        df["has_streaming"] = (streaming_count > 0).astype(int)

        # Serviços de segurança
        security_count = 0
        for col in self.security_cols_ or []:
            if col in df.columns:
                security_count += (~df[col].isin(INACTIVE_SERVICE_VALUES)).astype(int)
        df["security_services"] = security_count
        df["has_security"] = (security_count > 0).astype(int)

        return df

    def _create_engagement_features(self, df):
        """Cria features de engagement do cliente."""
        # Combinação de serviços indica engagement
        if "total_services" in df.columns:
            df["low_engagement"] = (df["total_services"] <= FE_LOW_ENGAGEMENT_MAX).astype(int)
            df["high_engagement"] = (df["total_services"] >= FE_HIGH_ENGAGEMENT_MIN).astype(int)

        # Paperless billing + electronic check = baixo engagement tradicional
        if {"PaperlessBilling", "PaymentMethod"}.issubset(df.columns):
            df["digital_only"] = (
                (df["PaperlessBilling"] == "Yes")
                & (df["PaymentMethod"].str.contains("Electronic|automatic", case=False, na=False))
            ).astype(int)

        # Combinação de partner/dependents indica estabilidade
        if {"Partner", "Dependents"}.issubset(df.columns):
            df["family_account"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(
                int
            )
            df["family_stability"] = (
                (df["Partner"] == "Yes") & (df["Dependents"] == "Yes")
            ).astype(int)

        return df

    def _create_multi_service_interactions(self, df):
        """Cria interações entre múltiplos serviços."""
        # Fiber optic + sem serviços de segurança = alto risco
        if {"InternetService", "security_services"}.issubset(df.columns):
            df["fiber_no_security"] = (
                (df["InternetService"] == "Fiber optic") & (df["security_services"] == 0)
            ).astype(int)

        # Internet + streaming mas sem proteção
        if {"InternetService", "has_streaming", "security_services"}.issubset(df.columns):
            df["streamer_unprotected"] = (
                (df["InternetService"] != "No")
                & (df["has_streaming"] == 1)
                & (df["security_services"] == 0)
            ).astype(int)

        # Senior citizen com contrato mensal
        if {"SeniorCitizen", "is_monthly_contract"}.issubset(df.columns):
            df["senior_monthly"] = (
                (df["SeniorCitizen"] == 1) & (df["is_monthly_contract"] == 1)
            ).astype(int)

        # Novo cliente com alto valor (usa mediana calculada no fit)
        if {"new_customer_flag", "MonthlyCharges"}.issubset(df.columns):
            # Usa APENAS self.monthly_median_ do fit - NUNCA calcular em dados de teste
            if not hasattr(self, "monthly_median_") or self.monthly_median_ is None:
                logger.warning("monthly_median_ não disponível - feature new_high_value será 0")
                df["new_high_value"] = 0
            else:
                df["new_high_value"] = (
                    (df["new_customer_flag"] == 1) & (df["MonthlyCharges"] > self.monthly_median_)
                ).astype(int)

        return df

    def _create_risk_scores(self, df):
        """Cria scores de risco compostos."""
        # Score de risco baseado em múltiplos fatores
        risk_score = 0

        w = FE_RISK_WEIGHTS

        if "is_monthly_contract" in df.columns:
            risk_score += df["is_monthly_contract"] * w["monthly_contract"]

        if "tenure" in df.columns:
            risk_score += (df["tenure"] < FE_EARLY_CHURN_TENURE).astype(int) * w["short_tenure"]

        if "fiber_no_security" in df.columns:
            risk_score += df["fiber_no_security"] * w["fiber_no_security"]

        if "low_engagement" in df.columns:
            risk_score += df["low_engagement"] * w["low_engagement"]

        if "PaperlessBilling" in df.columns:
            risk_score += (df["PaperlessBilling"] == "Yes").astype(int) * w["paperless_billing"]

        if "PaymentMethod" in df.columns:
            risk_score += (df["PaymentMethod"] == "Electronic check").astype(int) * w[
                "electronic_check"
            ]

        df["churn_risk_score"] = risk_score
        df["high_risk_flag"] = (risk_score >= FE_RISK_SCORE_HIGH).astype(int)
        df["very_high_risk_flag"] = (risk_score >= FE_RISK_SCORE_VERY_HIGH).astype(int)

        # Risk scores por categoria (do fit)
        if "InternetService" in df.columns and self.internet_service_risk_:
            default_risk = sum(self.internet_service_risk_.values()) / len(
                self.internet_service_risk_
            )
            df["internet_risk"] = (
                df["InternetService"].map(self.internet_service_risk_).fillna(default_risk)
            )

        if "PaymentMethod" in df.columns and self.payment_method_risk_:
            default_risk = sum(self.payment_method_risk_.values()) / len(self.payment_method_risk_)
            df["payment_risk"] = (
                df["PaymentMethod"].map(self.payment_method_risk_).fillna(default_risk)
            )

        return df

    def _create_lifecycle_features(self, df):
        """Cria features de ciclo de vida do cliente."""
        if "tenure" in df.columns:
            # Fases do ciclo de vida
            df["lifecycle_new"] = (df["tenure"] <= FE_LIFECYCLE_NEW_MAX).astype(int)
            df["lifecycle_early"] = (
                (df["tenure"] > FE_LIFECYCLE_NEW_MAX) & (df["tenure"] <= FE_LIFECYCLE_EARLY_MAX)
            ).astype(int)
            df["lifecycle_mid"] = (
                (df["tenure"] > FE_LIFECYCLE_EARLY_MAX) & (df["tenure"] <= FE_LIFECYCLE_MID_MAX)
            ).astype(int)
            df["lifecycle_mature"] = (df["tenure"] > FE_LIFECYCLE_MID_MAX).astype(int)

            # Tenure em anos
            df["tenure_years"] = df["tenure"] / 12

            # Raiz quadrada de tenure (captura não-linearidade)
            df["tenure_sqrt"] = np.sqrt(df["tenure"])

        return df

    def _create_value_features(self, df):
        """Cria features de valor do cliente."""
        if {"MonthlyCharges", "tenure", "TotalCharges"}.issubset(df.columns):
            # Customer Lifetime Value estimado
            df["estimated_clv"] = df["MonthlyCharges"] * df["tenure"]

            # Valor por mês de tenure
            df["value_per_month"] = np.where(
                df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
            )

        if "total_services" in df.columns and "MonthlyCharges" in df.columns:
            # Custo por serviço
            df["cost_per_service"] = np.where(
                df["total_services"] > 0,
                df["MonthlyCharges"] / df["total_services"],
                df["MonthlyCharges"],
            )

        return df
