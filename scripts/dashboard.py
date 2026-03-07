"""
Dashboard de Predição de Churn
================================

Dashboard interativo Streamlit para:
- Testar o modelo de predição de churn
- Visualizar performance do modelo
- Análise de interpretabilidade SHAP
- Rastreamento de experimentos MLflow

Uso:
    streamlit run scripts/dashboard.py
    # OU
    poetry run streamlit run scripts/dashboard.py
"""

# Garante que o diretório raiz do projeto esteja no sys.path
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Predição de Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .prediction-high {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
    .prediction-low {
        background-color: #00cc66;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Usa config centralizado
from src.config import (  # noqa: E402
    API_HOST,
    API_PORT,
    DATA_DIR_RAW,
    FILENAME,
    INTERPRET_TOP_FEATURES,
    MODEL_PATH,
    REPORTS_DIR,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
    SHAP_SAMPLE_SIZE,
    STEP_MODEL,
    VIZ_TOP_FACTORS,
    get_config,
    list_available_models,
)
from src.inference import load_model_package  # noqa: E402
from src.utils import (  # noqa: E402
    classify_risk,
    convert_target_to_binary,
    extract_model_components,
    normalize_metrics_keys,
)

# Constrói URL da API a partir do config centralizado
API_URL = f"http://{API_HOST}:{API_PORT}"


# =============================================================================
# Funções auxiliares (extraídas para eliminar duplicação)
# =============================================================================


def _encode_categoricals(df: pd.DataFrame) -> np.ndarray:
    """Codifica colunas categóricas como códigos inteiros e retorna array numpy."""
    X_numeric = df.copy()
    for col in X_numeric.select_dtypes(include=["object"]).columns:
        X_numeric[col] = X_numeric[col].astype("category").cat.codes
    return X_numeric.values


def _get_preprocessor_feature_names(preprocessor, n_features: int) -> list[str]:
    """Obtém nomes de features do preprocessador, com fallbacks."""
    if preprocessor is None:
        return [f"Feature {i}" for i in range(n_features)]
    try:
        from src.interpret import get_feature_names

        return get_feature_names(preprocessor)
    except Exception:
        pass
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        pass
    return [f"Feature {i}" for i in range(n_features)]


def get_available_models() -> list[dict[str, Any]]:
    """Obtém lista de modelos disponíveis no diretório de modelos."""
    models = list_available_models()

    # Converte para formato do dashboard com objetos datetime
    result = []
    for m in models:
        result.append(
            {
                "name": m["name"],
                "path": Path(m["path"]),
                "size_mb": m["size_mb"],
                "modified": datetime.fromtimestamp(m["modified"]),
                "is_default": m["is_default"],
            }
        )

    # Ordena por data de modificação (mais recente primeiro)
    result.sort(key=lambda x: x["modified"], reverse=True)
    return result


@st.cache_resource
def load_model_by_path(model_path: Path):
    """Carrega um modelo específico pelo caminho."""
    try:
        package = load_model_package(model_path)
        return package
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None


@st.cache_resource
def load_model():
    """Carrega o modelo treinado usando loader centralizado."""
    # Usa modelo selecionado do session state se disponível
    if "selected_model_path" in st.session_state:
        return load_model_by_path(st.session_state.selected_model_path)

    try:
        package = load_model_package(MODEL_PATH)
        return package  # Return full package, not just model
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60)
def load_sample_data():
    """Carrega dados de exemplo para visualização."""
    data_path = DATA_DIR_RAW / FILENAME
    if data_path.exists():
        config = get_config()
        positive_class = config.get("data", {}).get("positive_class", "Yes")
        df = pd.read_csv(data_path)
        df["Churn"] = convert_target_to_binary(df["Churn"], positive_class)
        return df
    return None


def check_api_health() -> bool:
    """Verifica se a API está rodando."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except (requests.RequestException, ConnectionError):
        return False


def predict_via_api(customer_data: dict) -> dict | None:
    """Realiza predição via API."""
    try:
        payload = {"customer_id": "dashboard-user", "features": customer_data}
        response = requests.post(f"{API_URL}/predict/", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prediction = data.get("prediction", {})
            return {
                "prediction": prediction.get("churn_prediction", 0),
                "probability": prediction.get("churn_probability", 0.0),
                "churn_risk": prediction.get("churn_risk", "UNKNOWN"),
            }
        else:
            st.error(f"API retornou status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


def explain_via_api(customer_data: dict, customer_id: str = "dashboard-user") -> dict | None:
    """Obtém explicação SHAP via API."""
    try:
        payload = {"customer_id": customer_id, "features": customer_data}
        response = requests.post(f"{API_URL}/interpret/shap/explain", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("explanation")
    except Exception:
        pass
    return None


def get_feature_importance_via_api(top_n: int = 20) -> list[dict] | None:
    """Obtém importância de features via API."""
    try:
        response = requests.get(
            f"{API_URL}/interpret/feature-importance", params={"top_n": top_n}, timeout=10
        )
        if response.status_code == 200:
            return response.json().get("feature_importances", [])
    except Exception:
        pass
    return None


def get_global_shap_via_api(sample_size: int = 100, top_n: int = 20) -> dict | None:
    """Obtém resumo SHAP global via API."""
    try:
        response = requests.get(
            f"{API_URL}/interpret/shap/global",
            params={"sample_size": sample_size, "top_n": top_n},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def detect_drift_via_api(production_df: pd.DataFrame) -> dict | None:
    """Envia dados de produção para detecção de drift via API."""
    try:
        import io

        csv_buffer = io.StringIO()
        production_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"file": ("production_data.csv", csv_buffer, "text/csv")}
        response = requests.post(f"{API_URL}/drift/detect", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def predict_locally(model_or_package, customer_data: dict) -> dict | None:
    """Realiza predição usando modelo local.

    Args:
        model_or_package: Objeto de modelo ou dict de pacote contendo o modelo
        customer_data: Dicionário com features do cliente

    Returns:
        Dicionário com resultados de predição
    """
    try:
        # Extrai modelo do pacote se necessário
        if isinstance(model_or_package, dict):
            model = model_or_package.get("model")
            threshold = model_or_package.get("threshold", 0.5)
            feature_engineer = model_or_package.get("feature_engineer")
            scaler = model_or_package.get("scaler")
        else:
            model = model_or_package
            threshold = 0.5
            feature_engineer = None
            scaler = None

        if model is None:
            return None

        df = pd.DataFrame([customer_data])

        if feature_engineer is not None:
            df = feature_engineer.transform(df)
        if scaler is not None:
            df = pd.DataFrame(scaler.transform(df))

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            prediction = int(proba[1] >= threshold)
            probability = float(proba[1])
        else:
            prediction = int(model.predict(df)[0])
            probability = float(prediction)

        return {
            "prediction": prediction,
            "probability": probability,
            "churn_risk": classify_risk(probability),
        }
    except Exception as e:
        st.error(f"Prediction Error: {e}")
    return None


def list_models_via_api() -> dict[str, Any] | None:
    """Lista modelos disponíveis via API."""
    try:
        response = requests.get(f"{API_URL}/models/", timeout=5)
        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, ConnectionError):
        pass
    return None


def switch_model_via_api(model_name: str) -> bool:
    """Troca o modelo ativo na API. Retorna True se bem-sucedido."""
    try:
        response = requests.post(
            f"{API_URL}/models/switch",
            json={"model_name": model_name},
            timeout=15,
        )
        return response.status_code == 200
    except (requests.RequestException, ConnectionError):
        return False


def render_sidebar():
    """Renderiza sidebar com navegação."""
    st.sidebar.markdown("## 📊 Navigation")

    page = st.sidebar.radio(
        "Selecione a página:",
        [
            "🏠 Home",
            "🔮 Predição",
            "📈 Performance",
            "⚖️ Comparação",
            "🧠 Interpretabilidade",
            "📡 Data Drift",
            "📋 MLflow",
        ],
    )

    st.sidebar.markdown("---")

    # Verifica status da API antes da seleção de modelo
    api_online = check_api_health()

    # Seleção de Modelo
    st.sidebar.markdown("### 🤖 Seleção de Modelo")

    # Se API está online, busca modelos e modelo ativo da API
    api_models_data = list_models_via_api() if api_online else None

    if api_models_data is not None:
        # --- Seleção via API ---
        model_names = [m["name"] for m in api_models_data["models"]]

        # Detecta índice do modelo ativo na API
        current_idx = 0
        if "selected_model_name" in st.session_state:
            try:
                current_idx = model_names.index(st.session_state.selected_model_name)
            except ValueError:
                current_idx = 0
        else:
            # Primeiro render: inicializa com o modelo ativo na API (sem disparar switch)
            active_model = api_models_data.get("active_model")
            if active_model and active_model in model_names:
                current_idx = model_names.index(active_model)
            st.session_state.selected_model_name = model_names[current_idx]

        selected_model_name = st.sidebar.selectbox(
            "Modelo (via API):",
            model_names,
            index=current_idx,
            help="Selecione o modelo — a troca é aplicada diretamente na API",
        )

        # Troca modelo na API somente se o usuário de fato mudou a seleção
        prev_selected = st.session_state.get("selected_model_name")
        if prev_selected is not None and selected_model_name != prev_selected:
            with st.sidebar.status("Trocando modelo na API..."):
                ok = switch_model_via_api(selected_model_name)
            if ok:
                st.sidebar.success(f"Modelo na API trocado para {selected_model_name}")
                load_model_by_path.clear()  # limpa cache local
            else:
                st.sidebar.error("Falha ao trocar modelo na API")

        st.session_state.selected_model_name = selected_model_name

        # Também atualiza path local para fallback
        selected_info = next(
            (m for m in api_models_data["models"] if m["name"] == selected_model_name), None
        )
        if selected_info:
            from src.config import get_model_path

            st.session_state.selected_model_path = get_model_path(selected_model_name)
            st.sidebar.caption(f"📁 Tamanho: {selected_info['size_mb']:.1f} MB")
    else:
        # --- Seleção local (fallback) ---
        available_models = get_available_models()

        if available_models:
            model_names = [m["name"] for m in available_models]

            current_idx = 0
            if "selected_model_name" in st.session_state:
                try:
                    current_idx = model_names.index(st.session_state.selected_model_name)
                except ValueError:
                    current_idx = 0

            selected_model_name = st.sidebar.selectbox(
                "Modelo:",
                model_names,
                index=current_idx,
                help="Selecione o modelo a ser usado para predições",
            )

            selected_model = next(
                (m for m in available_models if m["name"] == selected_model_name), None
            )
            if selected_model:
                st.session_state.selected_model_name = selected_model_name
                st.session_state.selected_model_path = selected_model["path"]

                st.sidebar.caption(
                    f"📁 Tamanho: {selected_model['size_mb']:.1f} MB\\n"
                    f"📅 Modificado: {selected_model['modified'].strftime('%d/%m/%Y %H:%M')}"
                )
        else:
            st.sidebar.warning("Nenhum modelo encontrado em models/")

    st.sidebar.markdown("---")

    # Status da API
    if api_online:
        st.sidebar.success("✅ API Online")
    else:
        st.sidebar.warning("⚠️ API Offline - Usando modelo local")

    # Status do Modelo
    if "selected_model_path" in st.session_state:
        package = load_model_by_path(st.session_state.selected_model_path)
        if package is not None:
            st.sidebar.success(f"✅ Modelo: {st.session_state.selected_model_name}")
            # Mostra métricas se disponível
            if isinstance(package, dict) and "metadata" in package:
                metadata = package.get("metadata", {})
                metrics = normalize_metrics_keys(metadata.get("metrics", {}))
                if metrics:
                    f1 = metrics.get("f1_score", 0)
                    auc = metrics.get("roc_auc", 0)
                    st.sidebar.caption(f"F1: {f1:.2%} | AUC: {auc:.2%}")
        else:
            st.sidebar.error("❌ Erro ao carregar modelo")
    else:
        model = load_model()
        if model is not None:
            st.sidebar.success("✅ Modelo Carregado")
        else:
            st.sidebar.error("❌ Modelo não encontrado")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Sobre")
    st.sidebar.info(
        "Dashboard para predição de churn de clientes de telecomunicações. "
        "Selecione diferentes modelos para comparar resultados."
    )

    return page


def _get_active_model_name() -> str:
    """Retorna o nome legível do modelo ativo."""
    if "selected_model_name" in st.session_state:
        return st.session_state.selected_model_name
    return Path(MODEL_PATH).stem


def _load_metrics_from_report() -> dict:
    """Carrega métricas de reports/metrics.json ou fallback do pacote do modelo."""
    import json

    metrics_json_path = REPORTS_DIR / "metrics.json"
    if metrics_json_path.exists():
        with open(metrics_json_path) as f:
            return normalize_metrics_keys(json.load(f))

    package = load_model()
    if isinstance(package, dict):
        raw = package.get("metrics", {})
        if not raw:
            meta = package.get("metadata", {})
            raw = meta.get("metrics", {}) if isinstance(meta, dict) else {}
        return normalize_metrics_keys(raw)
    return {}


def render_home():
    """Renderiza página inicial."""
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction</h1>', unsafe_allow_html=True)

    st.markdown(
        """
    ### Bem-vindo ao Dashboard de Predição de Churn

    Este sistema utiliza **Machine Learning** para prever quais clientes têm maior
    probabilidade de cancelar seus serviços.
    """
    )

    # Modelo ativo
    model_name = _get_active_model_name()
    st.info(f"📦 **Modelo ativo:** {model_name}")

    # Métricas do modelo
    metrics = _load_metrics_from_report()
    package = load_model()

    f1 = metrics.get("f1_score")
    auc = metrics.get("roc_auc")
    prec = metrics.get("precision")
    rec = metrics.get("recall")
    auprc = metrics.get("auprc")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="F1-Score", value=f"{f1:.0%}" if f1 else "N/A")
    with col2:
        st.metric(label="AUC-ROC", value=f"{auc:.0%}" if auc else "N/A")
    with col3:
        st.metric(label="Precision", value=f"{prec:.0%}" if prec else "N/A")
    with col4:
        st.metric(label="Recall", value=f"{rec:.0%}" if rec else "N/A")
    with col5:
        st.metric(label="AUPRC", value=f"{auprc:.0%}" if auprc else "N/A")

    st.markdown("---")

    # Visão Geral Rápida
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔧 Tecnologias")
        # Detecta algoritmo do modelo dinamicamente
        model_algo = "N/A"
        if isinstance(package, dict):
            model_obj = package.get("model") or package.get("pipeline")
            if model_obj is not None:
                if hasattr(model_obj, "named_steps"):
                    inner = model_obj.named_steps.get(STEP_MODEL)
                    model_algo = type(inner).__name__ if inner else type(model_obj).__name__
                else:
                    model_algo = type(model_obj).__name__
        elif package is not None and hasattr(package, "pipeline"):
            inner = getattr(package.pipeline, "named_steps", {}).get(STEP_MODEL)
            model_algo = type(inner).__name__ if inner else "N/A"

        st.markdown(
            f"""
        - **Modelo**: {model_algo}
        - **Balanceamento**: SMOTETomek (configurável)
        - **Otimização**: Optuna TPE Sampler
        """
        )

    with col2:
        st.markdown("### 📊 Dataset")
        df = load_sample_data()
        if df is not None:
            n_features = len([c for c in df.columns if c != "Churn"])
            st.markdown(
                f"""
            - **Total de clientes**: {len(df):,}
            - **Taxa de churn**: {df['Churn'].mean():.1%}
            - **Features originais**: {n_features}
            """
            )

    # Gráfico de Distribuição de Churn
    st.markdown("---")
    st.markdown("### 📈 Distribuição de Churn")

    df = load_sample_data()
    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                df,
                names=df["Churn"].map({0: "Não Churn", 1: "Churn"}),
                title="Distribuição de Churn",
                color_discrete_sequence=["#00cc66", "#ff4b4b"],
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            fig = px.histogram(
                df,
                x="tenure",
                color=df["Churn"].map({0: "Não Churn", 1: "Churn"}),
                title="Distribuição de Tenure por Churn",
                barmode="overlay",
                opacity=0.7,
            )
            st.plotly_chart(fig, width="stretch")


def render_shap_explanation(customer_data: dict[str, Any]) -> None:
    """
    Renderiza a explicação SHAP para um cliente específico.
    Usa a API quando disponível, senão calcula localmente.
    """
    # Tenta via API primeiro
    if check_api_health():
        with st.spinner("Obtendo explicação SHAP via API..."):
            explanation = explain_via_api(customer_data)
        if explanation is not None:
            shap_values_list = explanation.get("shap_values", [])
            feature_names = [sv["feature"] for sv in shap_values_list]
            shap_values = np.array([sv["shap_value"] for sv in shap_values_list])

            _render_shap_chart(feature_names, shap_values)
            return

    # Fallback: calcula localmente
    _render_shap_local(customer_data)


def _render_shap_local(customer_data: dict[str, Any]) -> None:
    """Calcula e renderiza explicação SHAP localmente."""
    try:
        import shap

        if "selected_model_path" in st.session_state:
            package = load_model_by_path(st.session_state.selected_model_path)
        else:
            package = load_model_package(MODEL_PATH)

        if package is None:
            st.warning("Modelo não carregado para análise SHAP.")
            return

        if isinstance(package, dict):
            model = package.get("model")
        else:
            model = package

        tree_model, preprocessor, feature_engineering = extract_model_components(model)

        if tree_model is None:
            st.warning("Não foi possível extrair o modelo para análise SHAP.")
            return

        df = pd.DataFrame([customer_data])

        if feature_engineering is not None:
            try:
                df = feature_engineering.transform(df)
            except Exception:
                pass

        if preprocessor is not None:
            try:
                X_transformed = preprocessor.transform(df)
                feature_names = _get_preprocessor_feature_names(
                    preprocessor, X_transformed.shape[1]
                )
            except Exception:
                X_transformed = _encode_categoricals(df)
                feature_names = list(df.columns)
        else:
            X_transformed = _encode_categoricals(df)
            feature_names = list(df.columns)

        with st.spinner("Calculando explicação SHAP localmente..."):
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            if shap_values.ndim > 1:
                shap_values = shap_values[0]

        if len(feature_names) != len(shap_values):
            feature_names = [f"Feature {i}" for i in range(len(shap_values))]

        _render_shap_chart(feature_names, shap_values)

    except ImportError:
        st.error("SHAP não está instalado. Execute: `poetry add shap`")
    except Exception as e:
        st.warning(f"Não foi possível gerar explicação SHAP: {e}")


def _render_shap_chart(feature_names: list[str], shap_values: np.ndarray) -> None:
    """Renderiza gráficos SHAP a partir de nomes/valores já calculados."""
    shap_df = pd.DataFrame(
        {"Feature": feature_names, "SHAP Value": shap_values, "Abs SHAP": np.abs(shap_values)}
    ).sort_values("Abs SHAP", ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        top_features = shap_df.head(INTERPRET_TOP_FEATURES).copy()
        top_features = top_features.sort_values("SHAP Value", ascending=True)

        colors = ["#ff4b4b" if v > 0 else "#00cc66" for v in top_features["SHAP Value"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=top_features["Feature"],
                x=top_features["SHAP Value"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in top_features["SHAP Value"]],
                textposition="outside",
            )
        )

        fig.update_layout(
            title="Top 10 Fatores que Influenciaram a Predição",
            xaxis_title="Impacto SHAP (+ aumenta chance de churn)",
            yaxis_title="",
            height=400,
            showlegend=False,
        )

        # Adiciona linha zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### 📖 Interpretação")

        # Mostra fatores positivos e negativos principais
        positive_factors = shap_df[shap_df["SHAP Value"] > 0].head(VIZ_TOP_FACTORS)
        negative_factors = shap_df[shap_df["SHAP Value"] < 0].head(VIZ_TOP_FACTORS)

        if len(positive_factors) > 0:
            st.markdown("**Fatores que AUMENTAM risco de churn:**")
            for _, row in positive_factors.iterrows():
                st.markdown(f"- 🔴 **{row['Feature']}**: +{row['SHAP Value']:.3f}")

        if len(negative_factors) > 0:
            st.markdown("**Fatores que DIMINUEM risco de churn:**")
            for _, row in negative_factors.iterrows():
                st.markdown(f"- 🟢 **{row['Feature']}**: {row['SHAP Value']:.3f}")

        st.markdown("---")
        st.info(
            """
        **Como ler:**
        - 🔴 Valores positivos → aumentam probabilidade de churn
        - 🟢 Valores negativos → diminuem probabilidade de churn
        - Barras maiores = maior impacto na decisão
        """
        )

    # Tabela detalhada
    with st.expander("📋 Ver todos os valores SHAP"):
        st.dataframe(
            shap_df[["Feature", "SHAP Value"]].style.format({"SHAP Value": "{:+.4f}"}),
            width="stretch",
            hide_index=True,
        )

    st.success("✅ Explicação SHAP gerada com sucesso!")


def render_prediction():
    """Renderiza página de predição."""
    st.markdown("## 🔮 Predição de Churn")

    model_name = _get_active_model_name()
    st.caption(f"📦 Modelo ativo: **{model_name}**")

    st.markdown(
        """
    Preencha os dados do cliente abaixo para obter uma predição de churn.
    """
    )

    col1, col2, col3 = st.columns(3)

    # Carrega ranges numéricos e valores válidos do config
    config = get_config()
    valid_values = config.get("features", {}).get("valid_values", {})
    numeric_ranges = config.get("features", {}).get("numeric_ranges", {})
    tenure_range = numeric_ranges.get("tenure", {})
    monthly_range = numeric_ranges.get("MonthlyCharges", {})
    total_range = numeric_ranges.get("TotalCharges", {})

    with col1:
        st.markdown("### 👤 Dados Pessoais")
        gender = st.selectbox("Gênero", valid_values.get("gender", ["Male", "Female"]))
        senior = st.selectbox("É Idoso?", ["Não", "Sim"])
        partner = st.selectbox("Tem Parceiro?", valid_values.get("Partner", ["Yes", "No"]))
        dependents = st.selectbox("Tem Dependentes?", valid_values.get("Dependents", ["Yes", "No"]))

    with col2:
        st.markdown("### 📞 Serviços")
        phone_service = st.selectbox(
            "Serviço Telefônico", valid_values.get("PhoneService", ["Yes", "No"])
        )
        multiple_lines = st.selectbox(
            "Linhas Múltiplas", valid_values.get("MultipleLines", ["Yes", "No", "No phone service"])
        )
        internet_service = st.selectbox(
            "Serviço de Internet", valid_values.get("InternetService", ["DSL", "Fiber optic", "No"])
        )
        online_security = st.selectbox(
            "Segurança Online",
            valid_values.get("OnlineSecurity", ["Yes", "No", "No internet service"]),
        )
        online_backup = st.selectbox(
            "Backup Online", valid_values.get("OnlineBackup", ["Yes", "No", "No internet service"])
        )
        device_protection = st.selectbox(
            "Proteção de Dispositivo",
            valid_values.get("DeviceProtection", ["Yes", "No", "No internet service"]),
        )
        tech_support = st.selectbox(
            "Suporte Técnico", valid_values.get("TechSupport", ["Yes", "No", "No internet service"])
        )
        streaming_tv = st.selectbox(
            "Streaming TV", valid_values.get("StreamingTV", ["Yes", "No", "No internet service"])
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            valid_values.get("StreamingMovies", ["Yes", "No", "No internet service"]),
        )

    with col3:
        st.markdown("### 💰 Contrato e Pagamento")
        tenure = st.slider(
            "Meses como cliente (tenure)",
            int(tenure_range.get("min", 0)),
            int(tenure_range.get("max", 72)),
            int(tenure_range.get("default", 12)),
        )
        contract = st.selectbox(
            "Tipo de Contrato",
            valid_values.get("Contract", ["Month-to-month", "One year", "Two year"]),
        )
        paperless = st.selectbox(
            "Fatura Digital", valid_values.get("PaperlessBilling", ["Yes", "No"])
        )
        payment_method = st.selectbox(
            "Método de Pagamento",
            valid_values.get(
                "PaymentMethod",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            ),
        )
        monthly_charges = st.slider(
            "Cobrança Mensal ($)",
            float(monthly_range.get("min", 18.0)),
            float(monthly_range.get("max", 120.0)),
            float(monthly_range.get("default", 70.0)),
        )
        total_charges = st.number_input(
            "Total Cobrado ($)",
            min_value=float(total_range.get("min", 0.0)),
            max_value=float(total_range.get("max", 10000.0)),
            value=float(tenure * monthly_charges),
        )

    st.markdown("---")

    if st.button("🔮 Fazer Predição", type="primary", width="stretch"):
        customer_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Sim" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner("Processando..."):
            # Tenta API primeiro, depois modelo local
            if check_api_health():
                result = predict_via_api(customer_data)
            else:
                # Usa modelo selecionado do session state
                if "selected_model_path" in st.session_state:
                    package = load_model_by_path(st.session_state.selected_model_path)
                else:
                    package = load_model()

                if package:
                    result = predict_locally(package, customer_data)
                else:
                    result = None

        if result:
            st.markdown("---")
            st.markdown("### 📊 Resultado da Predição")

            col1, col2, col3 = st.columns(3)

            prob = result.get("probability", 0)
            risk = classify_risk(prob)

            with col1:
                if risk == "HIGH":
                    st.markdown(
                        '<div class="prediction-high">⚠️ CHURN PREVISTO</div>',
                        unsafe_allow_html=True,
                    )
                elif risk == "MEDIUM":
                    st.markdown(
                        '<div class="prediction-high">⚠️ RISCO MODERADO</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="prediction-low">✅ NÃO CHURN</div>', unsafe_allow_html=True
                    )

            with col2:
                st.metric("Probabilidade de Churn", f"{prob:.1%}")

            with col3:
                st.metric("Nível de Risco", risk)

            # Gauge de risco
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Probabilidade de Churn (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkred" if risk == "HIGH" else "green"},
                        "steps": [
                            {"range": [0, RISK_THRESHOLD_LOW * 100], "color": "#c8e6c9"},
                            {
                                "range": [RISK_THRESHOLD_LOW * 100, RISK_THRESHOLD_HIGH * 100],
                                "color": "#fff9c4",
                            },
                            {"range": [RISK_THRESHOLD_HIGH * 100, 100], "color": "#ffcdd2"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": RISK_THRESHOLD_HIGH * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

            # Recomendações
            if risk == "HIGH":
                st.markdown("### 💡 Recomendações")
                st.warning(
                    """
                **Ações sugeridas para retenção:**
                - 📞 Contato proativo do time de retenção
                - 🎁 Oferecer desconto ou upgrade de plano
                - 📋 Migrar para contrato anual com benefícios
                - 🛡️ Adicionar serviços de proteção/suporte
                """
                )

            # Explicação SHAP para esta predição específica
            st.markdown("---")
            st.markdown("### 🧠 Explicação da Decisão (SHAP)")
            st.markdown(
                "Entenda como cada variável influenciou a predição para este cliente específico."
            )

            render_shap_explanation(customer_data)

        else:
            st.error(
                "Erro ao fazer predição. Verifique se a API está online ou se o modelo está carregado."
            )


def render_performance():
    """Renderiza página de visualização de performance."""
    st.markdown("## 📈 Performance do Modelo")

    model_name = _get_active_model_name()
    st.caption(f"📦 Modelo ativo: **{model_name}**")

    df = load_sample_data()
    if df is None:
        st.error("Dados não encontrados.")
        return

    # Abas para diferentes visualizações
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Métricas", "📈 Distribuições", "🔥 Correlações", "📋 Análise por Segmento"]
    )

    with tab1:
        st.markdown("### Métricas de Performance")

        metrics = _load_metrics_from_report()

        f1 = metrics.get("f1_score")
        auc = metrics.get("roc_auc")
        prec = metrics.get("precision")
        rec = metrics.get("recall")
        acc = metrics.get("accuracy")
        auprc = metrics.get("auprc")
        threshold = metrics.get("threshold")

        if not any([f1, auc, prec, rec, acc]):
            st.info(
                "Métricas de performance não disponíveis. "
                "Execute o pipeline para gerar reports/metrics.json."
            )

        # Cards de métricas
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("F1-Score", f"{f1:.4f}" if f1 else "N/A")
        with col2:
            st.metric("Precision", f"{prec:.4f}" if prec else "N/A")
        with col3:
            st.metric("Recall", f"{rec:.4f}" if rec else "N/A")
        with col4:
            st.metric("AUC-ROC", f"{auc:.4f}" if auc else "N/A")
        with col5:
            st.metric("Accuracy", f"{acc:.4f}" if acc else "N/A")
        with col6:
            st.metric("AUPRC", f"{auprc:.4f}" if auprc else "N/A")

        if threshold is not None:
            st.caption(f"Threshold de classificação: {threshold}")

        # Tabela detalhada
        metrics_df = pd.DataFrame(
            {
                "Métrica": ["F1-Score", "Precision", "Recall", "AUC-ROC", "Accuracy", "AUPRC"],
                "Valor": [
                    f"{f1:.4f}" if f1 else "N/A",
                    f"{prec:.4f}" if prec else "N/A",
                    f"{rec:.4f}" if rec else "N/A",
                    f"{auc:.4f}" if auc else "N/A",
                    f"{acc:.4f}" if acc else "N/A",
                    f"{auprc:.4f}" if auprc else "N/A",
                ],
                "Descrição": [
                    "Média harmônica de Precision e Recall",
                    "Dos preditos como churn, quantos realmente são",
                    "Dos churns reais, quantos foram identificados",
                    "Área sob a curva ROC",
                    "Total de acertos / Total de predições",
                    "Área sob a curva Precision-Recall",
                ],
            }
        )
        st.dataframe(metrics_df, width="stretch", hide_index=True)

        # Métricas de cross-validation (se disponíveis)
        import json

        metrics_json_path = REPORTS_DIR / "metrics.json"
        raw_metrics = {}
        if metrics_json_path.exists():
            with open(metrics_json_path) as f:
                raw_metrics = json.load(f)
        cv_f1_std = raw_metrics.get("cv_f1_std")
        cv_roc_std = raw_metrics.get("cv_roc_auc_std")
        cv_f1_m = raw_metrics.get("cv_f1_mean")
        cv_roc_m = raw_metrics.get("cv_roc_auc_mean")

        if cv_f1_m is not None or cv_roc_m is not None:
            st.markdown("---")
            st.markdown("#### Validação Cruzada")
            cv_col1, cv_col2 = st.columns(2)
            with cv_col1:
                if cv_f1_m is not None:
                    st.metric(
                        "CV F1-Score (média ± std)",
                        f"{cv_f1_m:.4f} ± {cv_f1_std:.4f}" if cv_f1_std else f"{cv_f1_m:.4f}",
                    )
            with cv_col2:
                if cv_roc_m is not None:
                    st.metric(
                        "CV ROC-AUC (média ± std)",
                        f"{cv_roc_m:.4f} ± {cv_roc_std:.4f}" if cv_roc_std else f"{cv_roc_m:.4f}",
                    )

    with tab2:
        st.markdown("### Distribuições de Features")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df,
                x="MonthlyCharges",
                color=df["Churn"].map({0: "Não Churn", 1: "Churn"}),
                title="Distribuição de Cobrança Mensal",
                barmode="overlay",
                opacity=0.7,
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            fig = px.box(
                df,
                x=df["Churn"].map({0: "Não Churn", 1: "Churn"}),
                y="tenure",
                title="Tenure por Status de Churn",
                color=df["Churn"].map({0: "Não Churn", 1: "Churn"}),
            )
            st.plotly_chart(fig, width="stretch")

        # Distribuição por contrato
        fig = px.histogram(
            df,
            x="Contract",
            color=df["Churn"].map({0: "Não Churn", 1: "Churn"}),
            title="Distribuição por Tipo de Contrato",
            barmode="group",
        )
        st.plotly_chart(fig, width="stretch")

    with tab3:
        st.markdown("### Mapa de Correlação")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            title="Correlação entre Features Numéricas",
            color_continuous_scale="RdBu_r",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")

    with tab4:
        st.markdown("### Análise por Segmento")

        # Taxa de churn por contrato
        churn_by_contract = df.groupby("Contract")["Churn"].mean().reset_index()
        churn_by_contract.columns = ["Contrato", "Taxa de Churn"]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                churn_by_contract,
                x="Contrato",
                y="Taxa de Churn",
                title="Taxa de Churn por Tipo de Contrato",
                color="Taxa de Churn",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            churn_by_internet = df.groupby("InternetService")["Churn"].mean().reset_index()
            churn_by_internet.columns = ["Serviço Internet", "Taxa de Churn"]

            fig = px.bar(
                churn_by_internet,
                x="Serviço Internet",
                y="Taxa de Churn",
                title="Taxa de Churn por Serviço de Internet",
                color="Taxa de Churn",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, width="stretch")


def render_interpretability():
    """Renderiza página de interpretabilidade SHAP."""
    st.markdown("## 🧠 Interpretabilidade do Modelo")

    model_name = _get_active_model_name()
    st.caption(f"📦 Modelo ativo: **{model_name}**")

    st.markdown(
        """
    Análise de interpretabilidade usando **SHAP (SHapley Additive exPlanations)**
    para entender como o modelo toma decisões.
    """
    )

    api_online = check_api_health()

    # =========================================================================
    # Feature Importance
    # =========================================================================
    st.markdown("### 📊 Feature Importance")

    if api_online:
        importances = get_feature_importance_via_api(top_n=20)
        if importances:
            importance_df = pd.DataFrame(
                [
                    {"Feature": fi["feature_name"], "Importance": fi["importance"]}
                    for fi in importances
                ]
            ).sort_values("Importance", ascending=True)

            fig = px.bar(
                importance_df,
                y="Feature",
                x="Importance",
                orientation="h",
                title="Top 20 Features mais Importantes",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Não foi possível obter importância de features da API.")
    else:
        _render_local_feature_importance()

    # =========================================================================
    # Análise SHAP Global
    # =========================================================================
    st.markdown("### 🔍 Análise SHAP")

    if api_online:
        with st.spinner("Calculando SHAP values via API... Isso pode levar alguns segundos."):
            shap_data = get_global_shap_via_api(sample_size=SHAP_SAMPLE_SIZE, top_n=15)

        if shap_data:
            mean_abs_list = shap_data.get("mean_abs_shap", [])
            sample_sz = shap_data.get("sample_size", 0)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Mean |SHAP Value|")

                shap_df = pd.DataFrame(
                    [
                        {"Feature": item["feature_name"], "Mean |SHAP|": item["importance"]}
                        for item in mean_abs_list
                    ]
                ).sort_values("Mean |SHAP|", ascending=True)

                fig = px.bar(
                    shap_df,
                    y="Feature",
                    x="Mean |SHAP|",
                    orientation="h",
                    title=f"Impacto Médio das Features (amostra: {sample_sz})",
                    color="Mean |SHAP|",
                    color_continuous_scale="Reds",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width="stretch")

            with col2:
                st.markdown("#### Distribuição SHAP")
                st.info(
                    """
                **Como interpretar:**
                - Valores SHAP positivos → aumentam probabilidade de churn
                - Valores SHAP negativos → diminuem probabilidade de churn
                - Maior magnitude → maior impacto na predição
                """
                )

                st.markdown("**Top Features:**")
                for i, item in enumerate(mean_abs_list[:5]):
                    st.write(f"{i+1}. **{item['feature_name']}**: {item['importance']:.4f}")

            st.success("✅ Análise SHAP concluída!")
        else:
            st.warning("Não foi possível calcular SHAP via API.")
    else:
        _render_local_shap_analysis()


def _render_local_feature_importance() -> None:
    """Renderiza feature importance localmente."""
    package = load_model()
    if package is None:
        st.error("Modelo não carregado. Execute o treinamento primeiro.")
        return

    if isinstance(package, dict):
        model = package.get("model")
        pkg_preprocessor = package.get("scaler")
    else:
        model = package
        pkg_preprocessor = None

    tree_model, preprocessor, _ = extract_model_components(model)
    if preprocessor is None and pkg_preprocessor is not None:
        preprocessor = pkg_preprocessor
    if tree_model is None:
        st.error("Não foi possível extrair o modelo.")
        return

    if hasattr(tree_model, "feature_importances_"):
        importances = tree_model.feature_importances_
        feature_names = (
            _get_preprocessor_feature_names(preprocessor, len(importances))
            if preprocessor
            else [f"Feature {i}" for i in range(len(importances))]
        )

        importance_df = (
            pd.DataFrame({"Feature": feature_names[: len(importances)], "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(20)
        )

        fig = px.bar(
            importance_df,
            y="Feature",
            x="Importance",
            orientation="h",
            title="Top 20 Features mais Importantes",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width="stretch")
    elif hasattr(tree_model, "coef_"):
        importances = np.abs(np.ravel(tree_model.coef_))
        feature_names = (
            _get_preprocessor_feature_names(preprocessor, len(importances))
            if preprocessor
            else [f"Feature {i}" for i in range(len(importances))]
        )

        importance_df = (
            pd.DataFrame({"Feature": feature_names[: len(importances)], "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(20)
        )

        fig = px.bar(
            importance_df,
            y="Feature",
            x="Importance",
            orientation="h",
            title="Top 20 Features mais Importantes (|coeficientes|)",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Feature importance não disponível para este tipo de modelo.")


def _render_local_shap_analysis() -> None:
    """Renderiza análise SHAP global localmente."""
    package = load_model()
    df = load_sample_data()

    if package is None:
        st.error("Modelo não carregado. Execute o treinamento primeiro.")
        return
    if df is None:
        st.error("Dados não encontrados.")
        return

    if isinstance(package, dict):
        model = package.get("model")
        pkg_feature_engineer = package.get("feature_engineer")
        pkg_preprocessor = package.get("scaler")
    else:
        model = package
        pkg_feature_engineer = None
        pkg_preprocessor = None

    try:
        import shap

        tree_model, preprocessor, feature_engineering = extract_model_components(model)
        # Fallback para componentes armazenados no pacote
        if feature_engineering is None and pkg_feature_engineer is not None:
            feature_engineering = pkg_feature_engineer
        if preprocessor is None and pkg_preprocessor is not None:
            preprocessor = pkg_preprocessor
        if tree_model is None:
            st.error("Não foi possível extrair o modelo para análise SHAP.")
            return

        with st.spinner("Calculando SHAP values localmente... Isso pode levar alguns segundos."):
            X = df.drop(columns=["Churn", "customerID"], errors="ignore")
            X = X.replace(r"^\s*$", np.nan, regex=True)
            if "TotalCharges" in X.columns:
                X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
            X = X.dropna().head(SHAP_SAMPLE_SIZE)

            if feature_engineering is not None:
                try:
                    X = feature_engineering.transform(X)
                except Exception:
                    pass

            if preprocessor is not None:
                try:
                    X_transformed = preprocessor.transform(X)
                    feature_names = _get_preprocessor_feature_names(
                        preprocessor, X_transformed.shape[1]
                    )
                except Exception:
                    X_transformed = _encode_categoricals(X)
                    feature_names = list(X.columns)
            else:
                X_transformed = _encode_categoricals(X)
                feature_names = list(X.columns)

            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Mean |SHAP Value|")
                mean_shap = np.ravel(np.abs(shap_values).mean(axis=0))
                feat_names = (
                    feature_names[: len(mean_shap)]
                    if preprocessor
                    else [f"Feature {i}" for i in range(len(mean_shap))]
                )

                shap_df = (
                    pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_shap})
                    .sort_values("Mean |SHAP|", ascending=True)
                    .tail(15)
                )

                fig = px.bar(
                    shap_df,
                    y="Feature",
                    x="Mean |SHAP|",
                    orientation="h",
                    title="Impacto Médio das Features",
                    color="Mean |SHAP|",
                    color_continuous_scale="Reds",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width="stretch")

            with col2:
                st.markdown("#### Distribuição SHAP")
                st.info(
                    """
                **Como interpretar:**
                - Valores SHAP positivos → aumentam probabilidade de churn
                - Valores SHAP negativos → diminuem probabilidade de churn
                - Maior magnitude → maior impacto na predição
                """
                )

                st.markdown("**Top Features:**")
                for i, (feat, val) in enumerate(
                    zip(
                        shap_df["Feature"].values[-5:],
                        shap_df["Mean |SHAP|"].values[-5:],
                        strict=False,
                    )
                ):
                    st.write(f"{i+1}. **{feat}**: {val:.4f}")

        st.success("✅ Análise SHAP concluída!")

    except ImportError:
        st.error("SHAP não está instalado. Execute: `pip install shap`")
    except Exception as e:
        st.error(f"Erro na análise SHAP: {e}")
        st.info("Tente treinar um novo modelo usando `poetry run python scripts/train_pipeline.py`")


def render_mlflow():
    """Renderiza página de rastreamento MLflow."""
    st.markdown("## 📋 MLflow Experiment Tracking")

    st.markdown(
        """
    Visualize experimentos e métricas registrados no MLflow.
    """
    )

    mlruns_path = Path("mlruns")

    if not mlruns_path.exists():
        st.warning("Diretório mlruns não encontrado. Execute um treinamento primeiro.")
        return

    # Lista experimentos
    experiments = []
    for exp_dir in mlruns_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name.isdigit():
            meta_path = exp_dir / "meta.yaml"
            if meta_path.exists():
                experiments.append(exp_dir.name)

    if not experiments:
        st.info("Nenhum experimento encontrado.")
        return

    st.markdown(f"### 📁 Experimentos Encontrados: {len(experiments)}")

    # Exibe runs do primeiro experimento
    for exp_id in experiments[:1]:
        exp_path = mlruns_path / exp_id

        runs = []
        for run_dir in exp_path.iterdir():
            if run_dir.is_dir() and run_dir.name != "meta.yaml":
                metrics_path = run_dir / "metrics"

                run_data = {"run_id": run_dir.name[:8]}

                # Lê métricas
                if metrics_path.exists():
                    for metric_file in metrics_path.iterdir():
                        try:
                            with open(metric_file) as f:
                                content = f.read().strip()
                                if content:
                                    value = float(content.split()[1])
                                    run_data[metric_file.name] = value
                        except (OSError, ValueError, IndexError):
                            pass

                if run_data:
                    runs.append(run_data)

        if runs:
            st.markdown("### 📊 Runs Recentes")
            runs_df = pd.DataFrame(runs)

            # Exibe métricas
            st.dataframe(runs_df.head(10), width="stretch")

            # Gráfico de métricas ao longo do tempo
            if "f1_score" in runs_df.columns:
                fig = px.line(
                    runs_df.reset_index(),
                    x="index",
                    y="f1_score",
                    title="F1-Score ao Longo dos Experimentos",
                    markers=True,
                )
                st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### 🚀 Iniciar MLflow UI")
    st.code("poetry run mlflow ui --port 5000", language="bash")
    st.info("Execute o comando acima no terminal e acesse http://localhost:5000")


def render_model_comparison():
    """Renderiza página de comparação de modelos."""
    from joblib import load as joblib_load

    from src.utils import parse_model_filename

    st.markdown('<h1 class="main-header">⚖️ Comparação de Modelos</h1>', unsafe_allow_html=True)

    st.markdown(
        """
    Esta página permite comparar a performance de diferentes modelos treinados.
    Os modelos são carregados do diretório `models/` e suas métricas são extraídas
    dos metadados salvos ou do nome do arquivo.
    """
    )

    # Obtém todos os modelos disponíveis
    available_models = get_available_models()

    if not available_models:
        st.warning("⚠️ Nenhum modelo encontrado no diretório models/")
        return

    st.markdown("---")
    st.markdown("### 📊 Modelos Disponíveis")

    # Coleta métricas dos modelos
    model_data = []

    for model_info in available_models:
        try:
            # Carrega modelo diretamente com joblib para obter pacote bruto
            package = joblib_load(model_info["path"])

            metrics = {}
            algorithm = "Desconhecido"

            if isinstance(package, dict):
                # Obtém métricas do pacote
                pkg_metrics = package.get("metrics", {})
                if isinstance(pkg_metrics, dict):
                    metrics = normalize_metrics_keys(pkg_metrics)

                # Obtém algoritmo de várias fontes
                algorithm = package.get("algorithm", None)
                if not algorithm:
                    algorithm = package.get("model_name", None)
                if not algorithm:
                    # Tenta inferir pelo tipo do modelo
                    model_obj = package.get("model") or package.get("pipeline")
                    if model_obj is not None:
                        # Se for um sklearn Pipeline, inspeciona o último step
                        estimator = model_obj
                        if hasattr(model_obj, "steps"):
                            estimator = model_obj.steps[-1][1]
                        model_type = type(estimator).__name__.lower()
                        if "lightgbm" in model_type or "lgbm" in model_type:
                            algorithm = "LightGBM"
                        elif "xgb" in model_type:
                            algorithm = "XGBoost"
                        elif "catboost" in model_type:
                            algorithm = "CatBoost"
                        elif "pipeline" in model_type:
                            algorithm = "Pipeline"
            else:
                # Modelo não é dict, tenta obter info do tipo
                model_type = type(package).__name__
                algorithm = model_type

            # Tenta parsear nome do arquivo para info adicional
            parsed = parse_model_filename(model_info["name"] + ".joblib")
            if parsed and (not algorithm or algorithm == "Desconhecido"):
                parsed_algo = parsed.get("algorithm", "")
                if parsed_algo and parsed_algo != "unknown":
                    algorithm = parsed_algo.upper()

            # Extrai métricas chave (já normalizadas)
            f1 = metrics.get("f1_score")
            auc = metrics.get("roc_auc")
            precision = metrics.get("precision")
            recall = metrics.get("recall")
            accuracy = metrics.get("accuracy")
            auprc = metrics.get("auprc")

            model_data.append(
                {
                    "Nome": model_info["name"],
                    "Algoritmo": algorithm.upper() if algorithm else "N/A",
                    "F1-Score": f1,
                    "AUC-ROC": auc,
                    "Precision": precision,
                    "Recall": recall,
                    "Accuracy": accuracy,
                    "AUPRC": auprc,
                    "Tamanho (MB)": round(model_info["size_mb"], 2),
                    "Última Modificação": model_info["modified"].strftime("%Y-%m-%d %H:%M"),
                    "path": model_info["path"],
                }
            )

        except Exception as e:
            st.warning(f"Erro ao carregar {model_info['name']}: {e}")
            continue

    if not model_data:
        st.error("Não foi possível carregar métricas de nenhum modelo.")
        return

    # Cria DataFrame
    df = pd.DataFrame(model_data)

    # Formata métricas como porcentagens para exibição
    display_df = df.copy()
    metric_cols = ["F1-Score", "AUC-ROC", "Precision", "Recall", "Accuracy", "AUPRC"]

    for col in metric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) and x is not None else "N/A"
            )

    # Remove coluna de caminho para exibição
    display_df = display_df.drop(columns=["path"], errors="ignore")

    # Mostra tabela
    st.dataframe(display_df, width="stretch", hide_index=True)

    # Gráficos de comparação de modelos
    st.markdown("---")
    st.markdown("### 📈 Comparação Visual")

    # Filtra modelos com métricas válidas
    df_with_metrics = df.dropna(subset=["F1-Score", "AUC-ROC"], how="all")

    if len(df_with_metrics) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            # Comparação F1-Score
            df_f1 = df_with_metrics.dropna(subset=["F1-Score"])
            if len(df_f1) > 0:
                fig = px.bar(
                    df_f1,
                    x="Nome",
                    y="F1-Score",
                    color="Algoritmo",
                    title="Comparação de F1-Score",
                    text_auto=".2%",
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")

        with col2:
            # Comparação AUC-ROC
            df_auc = df_with_metrics.dropna(subset=["AUC-ROC"])
            if len(df_auc) > 0:
                fig = px.bar(
                    df_auc,
                    x="Nome",
                    y="AUC-ROC",
                    color="Algoritmo",
                    title="Comparação de AUC-ROC",
                    text_auto=".2%",
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")

        with col3:
            # Comparação AUPRC
            df_auprc = df_with_metrics.dropna(subset=["AUPRC"])
            if len(df_auprc) > 0:
                fig = px.bar(
                    df_auprc,
                    x="Nome",
                    y="AUPRC",
                    color="Algoritmo",
                    title="Comparação de AUPRC",
                    text_auto=".2%",
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")

        # Gráfico radar para comparação multi-métrica
        st.markdown("### 🎯 Radar de Métricas")

        df_radar = df_with_metrics.dropna(subset=metric_cols, how="all")

        if len(df_radar) > 0:
            # Prepara dados para gráfico radar
            fig = go.Figure()

            for _, row in df_radar.iterrows():
                values = []
                for col in metric_cols:
                    val = row.get(col)
                    values.append(val if pd.notna(val) and val is not None else 0)

                # Fecha o radar
                values.append(values[0])
                categories = metric_cols + [metric_cols[0]]

                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill="toself",
                        name=row["Nome"][:20],  # Truncate long names
                    )
                )

            fig.update_layout(
                polar={"radialaxis": {"visible": True, "range": [0, 1], "tickformat": ".0%"}},
                showlegend=True,
                title="Comparação Multi-Métrica",
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "Métricas não disponíveis para comparação visual. Treine modelos com nomes descritivos para melhor visualização."
        )

    # Recomendação do melhor modelo
    st.markdown("---")
    st.markdown("### 🏆 Recomendação")

    df_ranked = df.dropna(subset=["F1-Score"])
    if len(df_ranked) > 0:
        best_model = df_ranked.loc[df_ranked["F1-Score"].idxmax()]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.success(f"**Melhor Modelo (F1):** {best_model['Nome']}")
        with col2:
            if pd.notna(best_model.get("F1-Score")):
                st.metric("F1-Score", f"{best_model['F1-Score']:.2%}")
        with col3:
            if pd.notna(best_model.get("AUC-ROC")):
                st.metric("AUC-ROC", f"{best_model['AUC-ROC']:.2%}")

        # Botão para selecionar melhor modelo
        if st.button("🎯 Usar este modelo"):
            st.session_state.selected_model_path = best_model["path"]
            st.session_state.selected_model_name = best_model["Nome"]
            st.success(f"Modelo '{best_model['Nome']}' selecionado!")
            st.rerun()
    else:
        st.info("Treine modelos com métricas para ver recomendações.")


def _simulate_production_data(
    reference_df: pd.DataFrame,
    drift_intensity: float,
    sample_size: int,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """
    Simula dados de produção com drift a partir do dataset de referência.

    Args:
        reference_df: Dataset de referência original
        drift_intensity: Intensidade do drift a ser injetado (0.0 = nenhum, 1.0 = extremo)
        sample_size: úmero de linhas a serem geradas
        numeric_features: Lista de nomes de features numéricas
        categorical_features: Lista de nomes de features categóricas

    Returns:
         DataFrame de dados simulados com drift aplicado
    """
    rng = np.random.default_rng()
    simulated = reference_df.sample(n=min(sample_size, len(reference_df)), replace=True).copy()

    # Aplica drift nas features numéricas
    for col in numeric_features:
        if col in simulated.columns:
            simulated[col] = pd.to_numeric(simulated[col], errors="coerce")
            std = simulated[col].std()
            if std == 0 or pd.isna(std):
                continue
            shift = drift_intensity * std * rng.choice([-1, 1])
            noise = rng.normal(0, std * drift_intensity * 0.3, size=len(simulated))
            simulated[col] = simulated[col] + shift + noise

    # Aplica drift nas features categóricas (desloca distribuição)
    if drift_intensity > 0.3:
        for col in categorical_features:
            if col in simulated.columns:
                n_swap = int(len(simulated) * drift_intensity * 0.3)
                if n_swap > 0:
                    unique_vals = simulated[col].unique()
                    if len(unique_vals) > 1:
                        idx = rng.choice(len(simulated), size=n_swap, replace=False)
                        simulated.iloc[idx, simulated.columns.get_loc(col)] = rng.choice(
                            unique_vals, size=n_swap
                        )

    return simulated.reset_index(drop=True)


def _severity_color(severity: str) -> str:
    """Retorna cor para o nível de severidade do drift."""
    return {
        "none": "#00cc66",
        "low": "#ffcc00",
        "moderate": "#ff8800",
        "high": "#ff4b4b",
    }.get(severity, "#888888")


def _severity_icon(severity: str) -> str:
    """Retorna ícone para o nível de severidade do drift."""
    return {
        "none": "\u2705",
        "low": "\U0001f7e1",
        "moderate": "\U0001f7e0",
        "high": "\U0001f534",
    }.get(severity, "\u2753")


def render_data_drift():
    """Renderiza a página de monitoramento de data drift."""
    from src.config import get_config
    from src.monitoring import detect_drift

    st.markdown("## \U0001f4e1 Monitoramento de Data Drift")

    st.markdown(
        "Detecte **desvios nos dados de produ\u00e7\u00e3o** em rela\u00e7\u00e3o aos dados "
        "de refer\u00eancia (treino). O monitoramento utiliza **PSI** (Population Stability "
        "Index) e **teste KS** (Kolmogorov-Smirnov) para identificar mudan\u00e7as na "
        "distribui\u00e7\u00e3o das features."
    )

    # Botão para forçar atualização dos dados de referência
    if st.button("\U0001f504 Atualizar Dados de Referência"):
        load_sample_data.clear()
        if "drift_production_df" in st.session_state:
            del st.session_state["drift_production_df"]
        if "drift_params" in st.session_state:
            del st.session_state["drift_params"]
        st.rerun()

    # Carrega dados de referência
    reference_df = load_sample_data()
    if reference_df is None:
        st.error("Dados de referência não encontrados. " "Verifique o dataset em src/data/raw/.")
        return

    # Obtém listas de features do config
    config = get_config()
    numeric_features = config.get("features", {}).get("numeric", [])
    categorical_features = config.get("features", {}).get("categorical", [])

    # Fallback: auto-detecta dos dados
    if not numeric_features:
        numeric_features = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [c for c in numeric_features if c not in ("Churn", "SeniorCitizen")]
    if not categorical_features:
        categorical_features = reference_df.select_dtypes(include=["object"]).columns.tolist()
        categorical_features = [c for c in categorical_features if c != "customerID"]

    all_features = numeric_features + categorical_features

    st.markdown("---")

    # =========================================================================
    # Seleção de fonte de dados
    # =========================================================================
    st.markdown("### Dados de Produção")

    data_source = st.radio(
        "Selecione a fonte dos dados de produção:",
        ["Upload de CSV", "Simular dados com drift"],
        horizontal=True,
    )

    production_df = None

    if data_source == "Upload de CSV":
        uploaded_file = st.file_uploader(
            "Faça upload de um arquivo CSV com dados de produção",
            type=["csv"],
            help="O CSV deve conter as mesmas colunas do dataset de treino.",
        )
        if uploaded_file is not None:
            try:
                production_df = pd.read_csv(uploaded_file)
                st.success(
                    f"Arquivo carregado: {len(production_df)} registros, "
                    f"{len(production_df.columns)} colunas"
                )

                # Valida colunas
                missing_cols = set(all_features) - set(production_df.columns)
                if missing_cols:
                    st.warning(
                        "Colunas ausentes nos dados de produ\u00e7\u00e3o: "
                        f"{', '.join(missing_cols)}"
                    )
                    all_features = [f for f in all_features if f in production_df.columns]
            except Exception as e:
                st.error(f"Erro ao ler o arquivo: {e}")

    else:  # Simular dados
        st.markdown("#### Configura\u00e7\u00f5es da Simula\u00e7\u00e3o")

        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            drift_intensity = st.slider(
                "Intensidade do drift",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="0.0 = sem drift, 1.0 = drift extremo",
            )
        with col_sim2:
            sample_size = st.slider(
                "Tamanho da amostra",
                min_value=100,
                max_value=min(5000, len(reference_df)),
                value=min(1000, len(reference_df)),
                step=100,
            )

        # Mostra o que a intensidade significa
        if drift_intensity < 0.1:
            st.info(
                "**Drift m\u00ednimo**: Os dados simulados ser\u00e3o muito similares ao treino."
            )
        elif drift_intensity < 0.3:
            st.info("**Drift leve**: Pequenas altera\u00e7\u00f5es nas distribui\u00e7\u00f5es.")
        elif drift_intensity < 0.6:
            st.warning(
                "**Drift moderado**: Mudan\u00e7as percept\u00edveis nas distribui\u00e7\u00f5es."
            )
        else:
            st.error("**Drift severo**: Distribui\u00e7\u00f5es significativamente diferentes.")

        if st.button("Gerar Dados Simulados", type="primary"):
            production_df = _simulate_production_data(
                reference_df,
                drift_intensity,
                sample_size,
                numeric_features,
                categorical_features,
            )
            st.session_state["drift_production_df"] = production_df
            st.session_state["drift_params"] = (drift_intensity, sample_size)
            st.rerun()

        # Recupera do session state se gerado anteriormente
        if "drift_production_df" in st.session_state and production_df is None:
            production_df = st.session_state["drift_production_df"]

    if production_df is None:
        st.info(
            "Selecione ou gere dados de produ\u00e7\u00e3o para iniciar " "a an\u00e1lise de drift."
        )
        return

    # =========================================================================
    # Executa detecção de drift
    # =========================================================================
    st.markdown("---")

    # Converte TotalCharges para numérico (contém ' ' no dataset original)
    for df_temp in [reference_df, production_df]:
        if "TotalCharges" in df_temp.columns:
            df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")

    valid_features = [
        f for f in all_features if f in reference_df.columns and f in production_df.columns
    ]

    api_online = check_api_health()

    with st.spinner("Executando detec\u00e7\u00e3o de drift..."):
        api_report = None
        if api_online:
            api_report = detect_drift_via_api(production_df)

        if api_report and api_report.get("success"):
            # Usa resultado da API
            report_dict = api_report
            severity = api_report["overall_severity"]
            features_checked = api_report["features_checked"]
            features_with_drift = api_report["features_with_drift"]
            drift_pct = api_report["drift_percentage"]
            sev_counts = api_report.get("severity_counts", {})
            feature_results_raw = api_report.get("features", [])
            st.caption("Fonte: API")
        else:
            # Fallback local — detect_drift lida com categóricas automaticamente
            drift_report = detect_drift(
                reference_df=reference_df,
                current_df=production_df,
                features=valid_features,
                categorical_features=categorical_features,
            )
            report_dict = drift_report.to_dict()
            severity = drift_report.overall_severity.value
            features_checked = drift_report.features_checked
            features_with_drift = drift_report.features_with_drift
            drift_pct = (
                drift_report.features_with_drift / drift_report.features_checked * 100
                if drift_report.features_checked > 0
                else 0
            )
            sev_counts = drift_report.severity_counts
            feature_results_raw = report_dict.get("features", [])
            st.caption("Fonte: an\u00e1lise local")

    # =========================================================================
    # Resumo Geral
    # =========================================================================
    st.markdown("### Resumo Geral")

    sev_icon = _severity_icon(severity)
    sev_color = _severity_color(severity)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Features Analisadas", features_checked)
    with col2:
        st.metric("Features com Drift", features_with_drift)
    with col3:
        st.metric("% com Drift", f"{drift_pct:.1f}%")
    with col4:
        st.markdown(
            f'<div style="background-color: {sev_color}; color: white; padding: 1rem; '
            f'border-radius: 10px; text-align: center;">'
            f'<div style="font-size: 0.85rem; opacity: 0.9;">Severidade Geral</div>'
            f'<div style="font-size: 1.5rem; font-weight: bold;">'
            f"{sev_icon} {severity.upper()}</div></div>",
            unsafe_allow_html=True,
        )

    # Distribuição de severidade
    st.markdown("---")

    col_sev1, col_sev2 = st.columns([2, 1])

    with col_sev1:
        sev_df = pd.DataFrame(
            {
                "Severidade": [s.upper() for s in sev_counts.keys()],
                "Quantidade": list(sev_counts.values()),
            }
        )
        sev_df = sev_df[sev_df["Quantidade"] > 0]

        if not sev_df.empty:
            fig = px.bar(
                sev_df,
                x="Severidade",
                y="Quantidade",
                color="Severidade",
                color_discrete_map={s.upper(): _severity_color(s) for s in sev_counts.keys()},
                title="Distribui\u00e7\u00e3o de Severidade do Drift",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_sev2:
        st.markdown("#### Legenda de Severidade")
        st.markdown(
            "- **NONE** (PSI < 0.10): Sem mudan\u00e7a significativa\n"
            "- **LOW** (0.10 \u2264 PSI < 0.20): Mudan\u00e7a leve\n"
            "- **MODERATE** (0.20 \u2264 PSI < 0.25): Investigar\n"
            "- **HIGH** (PSI \u2265 0.25): A\u00e7\u00e3o necess\u00e1ria"
        )
        st.markdown("---")
        st.markdown(f"**Timestamp:** {report_dict['timestamp'][:19]}")

    # =========================================================================
    # Gráfico PSI por feature
    # =========================================================================
    st.markdown("---")
    st.markdown("### PSI por Feature")

    if feature_results_raw:
        psi_data = pd.DataFrame(
            [
                {
                    "Feature": r["name"],
                    "PSI": r["psi"],
                    "KS Statistic": r["ks_statistic"],
                    "KS p-value": r["ks_pvalue"],
                    "Severidade": r["severity"].upper(),
                }
                for r in feature_results_raw
            ]
        )

        # Gráfico de barras - PSI
        fig = go.Figure()
        colors = [_severity_color(r["severity"]) for r in feature_results_raw]

        fig.add_trace(
            go.Bar(
                x=psi_data["Feature"],
                y=psi_data["PSI"],
                marker_color=colors,
                text=[f"{v:.4f}" for v in psi_data["PSI"]],
                textposition="outside",
                name="PSI",
            )
        )

        # Adiciona linhas de threshold
        fig.add_hline(
            y=0.10,
            line_dash="dot",
            line_color="#ffcc00",
            annotation_text="Low (0.10)",
        )
        fig.add_hline(
            y=0.20,
            line_dash="dash",
            line_color="#ff8800",
            annotation_text="Moderate (0.20)",
        )
        fig.add_hline(
            y=0.25,
            line_dash="solid",
            line_color="#ff4b4b",
            annotation_text="High (0.25)",
        )

        fig.update_layout(
            title="Population Stability Index (PSI) por Feature",
            xaxis_title="Feature",
            yaxis_title="PSI",
            height=500,
            showlegend=False,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

        # =================================================================
        # Gráfico Estatística KS
        # =================================================================
        st.markdown("### Teste Kolmogorov-Smirnov")

        fig_ks = go.Figure()
        ks_colors = ["#ff4b4b" if p < 0.05 else "#00cc66" for p in psi_data["KS p-value"]]

        fig_ks.add_trace(
            go.Bar(
                x=psi_data["Feature"],
                y=psi_data["KS Statistic"],
                marker_color=ks_colors,
                text=[f"{v:.4f}" for v in psi_data["KS Statistic"]],
                textposition="outside",
                name="KS Statistic",
            )
        )

        fig_ks.update_layout(
            title="KS Statistic por Feature (vermelho = p < 0.05, significativo)",
            xaxis_title="Feature",
            yaxis_title="KS Statistic",
            height=450,
            showlegend=False,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_ks, use_container_width=True)

        # =================================================================
        # Comparação de distribuição para features com maior drift
        # =================================================================
        st.markdown("---")
        st.markdown("### Compara\u00e7\u00e3o de Distribui\u00e7\u00f5es")

        features_to_plot = [r["name"] for r in feature_results_raw[:6]]
        numeric_to_plot = [f for f in features_to_plot if f in numeric_features]
        categorical_to_plot = [f for f in features_to_plot if f in categorical_features]

        if numeric_to_plot:
            st.markdown("#### Features Num\u00e9ricas")
            n_cols = min(3, len(numeric_to_plot))
            for i in range(0, len(numeric_to_plot), n_cols):
                cols = st.columns(n_cols)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(numeric_to_plot):
                        feat = numeric_to_plot[idx]
                        with col:
                            result = next(
                                (r for r in feature_results_raw if r["name"] == feat),
                                None,
                            )
                            title_suffix = f" (PSI={result['psi']:.4f})" if result else ""

                            fig = go.Figure()
                            fig.add_trace(
                                go.Histogram(
                                    x=reference_df[feat].dropna(),
                                    name="Refer\u00eancia",
                                    opacity=0.6,
                                    marker_color="#667eea",
                                    histnorm="probability density",
                                )
                            )
                            fig.add_trace(
                                go.Histogram(
                                    x=(
                                        production_df[feat].dropna()
                                        if feat in production_df.columns
                                        else []
                                    ),
                                    name="Produ\u00e7\u00e3o",
                                    opacity=0.6,
                                    marker_color="#ff4b4b",
                                    histnorm="probability density",
                                )
                            )
                            fig.update_layout(
                                title=f"{feat}{title_suffix}",
                                barmode="overlay",
                                height=300,
                                showlegend=True,
                                legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
                                margin={"l": 40, "r": 20, "t": 40, "b": 40},
                            )
                            st.plotly_chart(fig, use_container_width=True)

        if categorical_to_plot:
            st.markdown("#### Features Categ\u00f3ricas")
            n_cols = min(3, len(categorical_to_plot))
            for i in range(0, len(categorical_to_plot), n_cols):
                cols = st.columns(n_cols)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(categorical_to_plot):
                        feat = categorical_to_plot[idx]
                        with col:
                            result = next(
                                (r for r in feature_results_raw if r["name"] == feat),
                                None,
                            )
                            title_suffix = f" (PSI={result['psi']:.4f})" if result else ""

                            ref_counts = (
                                reference_df[feat].value_counts(normalize=True).reset_index()
                            )
                            ref_counts.columns = ["Valor", "Propor\u00e7\u00e3o"]
                            ref_counts["Fonte"] = "Refer\u00eancia"

                            prod_counts = pd.DataFrame(
                                columns=["Valor", "Propor\u00e7\u00e3o", "Fonte"]
                            )
                            if feat in production_df.columns:
                                prod_counts = (
                                    production_df[feat].value_counts(normalize=True).reset_index()
                                )
                                prod_counts.columns = ["Valor", "Propor\u00e7\u00e3o"]
                                prod_counts["Fonte"] = "Produ\u00e7\u00e3o"

                            combined = pd.concat([ref_counts, prod_counts], ignore_index=True)

                            fig = px.bar(
                                combined,
                                x="Valor",
                                y="Propor\u00e7\u00e3o",
                                color="Fonte",
                                barmode="group",
                                title=f"{feat}{title_suffix}",
                                color_discrete_map={
                                    "Refer\u00eancia": "#667eea",
                                    "Produ\u00e7\u00e3o": "#ff4b4b",
                                },
                            )
                            fig.update_layout(
                                height=300,
                                margin={"l": 40, "r": 20, "t": 40, "b": 40},
                                legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
                            )
                            st.plotly_chart(fig, use_container_width=True)

        # =================================================================
        # Tabela detalhada
        # =================================================================
        st.markdown("---")
        st.markdown("### Tabela Detalhada")

        detail_df = psi_data.copy()
        detail_df["Drift?"] = ["Sim" if r["has_drift"] else "N\u00e3o" for r in feature_results_raw]

        def _highlight_severity(row):
            color = _severity_color(row["Severidade"].lower())
            return [f"background-color: {color}20" for _ in row]

        styled = detail_df.style.apply(_highlight_severity, axis=1).format(
            {"PSI": "{:.4f}", "KS Statistic": "{:.4f}", "KS p-value": "{:.6f}"}
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # =================================================================
        # Actionable recommendations
        # =================================================================
        st.markdown("---")
        st.markdown("### Recomenda\u00e7\u00f5es")

        if severity == "none":
            st.success(
                "**Nenhum drift significativo detectado.**\n\n"
                "Os dados de produ\u00e7\u00e3o est\u00e3o alinhados com os dados de treino. "
                "Continue monitorando periodicamente."
            )
        elif severity == "low":
            st.info(
                "**Drift leve detectado.**\n\n"
                "- Monitore as features afetadas com maior frequ\u00eancia\n"
                "- Avalie se as mudan\u00e7as refletem tend\u00eancias reais do neg\u00f3cio\n"
                "- Retreinamento ainda n\u00e3o \u00e9 necess\u00e1rio"
            )
        elif severity == "moderate":
            drifted = [r["name"] for r in feature_results_raw if r["has_drift"]]
            st.warning(
                f"**Drift moderado detectado em {len(drifted)} feature(s).**\n\n"
                f"**Features afetadas:** {', '.join(drifted)}\n\n"
                "**A\u00e7\u00f5es recomendadas:**\n"
                "- Investigar a causa raiz das mudan\u00e7as\n"
                "- Avaliar impacto na performance do modelo com dados recentes\n"
                "- Considerar retreinamento parcial ou completo\n"
                "- Verificar se houve mudan\u00e7as no processo de coleta de dados"
            )
        else:
            drifted = [r["name"] for r in feature_results_raw if r["has_drift"]]
            st.error(
                f"**Drift severo detectado! A\u00e7\u00e3o imediata necess\u00e1ria.**\n\n"
                f"**Features com drift cr\u00edtico:** {', '.join(drifted)}\n\n"
                "**A\u00e7\u00f5es urgentes:**\n"
                "- **Retreinar o modelo** com dados recentes\n"
                "- Investigar mudan\u00e7as na fonte de dados\n"
                "- Validar performance do modelo atual com m\u00e9tricas de neg\u00f3cio\n"
                "- Documentar as mudan\u00e7as observadas\n"
                "- Considerar ajuste no pipeline de feature engineering"
            )

        # =================================================================
        # Export report
        # =================================================================
        st.markdown("---")
        st.markdown("### Exportar Relat\u00f3rio")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            import json

            report_json = json.dumps(report_dict, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=report_json,
                file_name=f"drift_report_{report_dict['timestamp'][:10]}.json",
                mime="application/json",
            )

        with col_exp2:
            csv_data = detail_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"drift_report_{report_dict['timestamp'][:10]}.csv",
                mime="text/csv",
            )


def main():
    """Ponto de entrada principal para o dashboard Streamlit. Renderiza a barra lateral e a página selecionada."""
    page = render_sidebar()

    if page == "🏠 Home":
        render_home()
    elif page == "🔮 Predição":
        render_prediction()
    elif page == "📈 Performance":
        render_performance()
    elif page == "⚖️ Comparação":
        render_model_comparison()
    elif page == "🧠 Interpretabilidade":
        render_interpretability()
    elif page == "📡 Data Drift":
        render_data_drift()
    elif page == "📋 MLflow":
        render_mlflow()


if __name__ == "__main__":
    main()
