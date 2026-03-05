"""
Dashboard interativo de monitoramento de performance de ML.

Visualiza:
- Último ROC-AUC e F1-Score
- Evolução ao longo do tempo
- Histórico de runs
- Detecção de degradação

Executar com:
    streamlit run scripts/dashboard.py
"""

import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime, timedelta
import numpy as np
from src.config import MLFLOW_TRACKING_URI

# Configurar Streamlit
st.set_page_config(
    page_title="ML Performance Monitor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customizar tema
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Título e descrição
st.title("🏆 ML Performance Monitor")
st.markdown("Dashboard de monitoramento em tempo real do modelo XGBoost")

# Conectar ao MLflow
@st.cache_resource
def load_mlflow_data():
    """Carrega dados do MLflow com cache."""
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    client = mlflow.tracking.MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name("mlflow_test_experiments")
        if experiment is None:
            st.warning("⚠️ Nenhum experimento encontrado. Execute o pipeline primeiro!")
            return None, None
        
        experiment_id = experiment.experiment_id
        runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
        return runs, experiment_id
    except Exception as e:
        st.error(f"Erro ao conectar ao MLflow: {e}")
        return None, None


# Carregar dados
runs, experiment_id = load_mlflow_data()

if runs is None or len(runs) == 0:
    st.error("❌ Nenhuma execução encontrada no MLflow. Por favor, execute o pipeline.")
    st.stop()

# Processar dados das runs
data = []
for run in runs:
    metrics = run.data.metrics
    params = run.data.params
    tags = run.data.tags
    
    data.append({
        'timestamp': datetime.fromtimestamp(run.info.start_time / 1000),
        'run_id': run.info.run_id[:8],
        'roc_auc': metrics.get('roc_auc', None),
        'f1_score': metrics.get('weighted_avg_f1-score', None),
        'precision': metrics.get('weighted_avg_precision', None),
        'recall': metrics.get('weighted_avg_recall', None),
        'status': 'Degradado' if metrics.get('roc_auc', 0) < 0.5 else 'Normal'
    })

df = pd.DataFrame(data).sort_values('timestamp')

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📊 Filtros")
    
    # Período
    days_back = st.slider(
        "Últimos N dias",
        min_value=1,
        max_value=30,
        value=7,
        help="Filtrar execuções dos últimos N dias"
    )
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    df_filtered = df[df['timestamp'] >= cutoff_date]
    
    # Métrica
    metric_selected = st.selectbox(
        "Métrica para análise",
        ["ROC-AUC", "F1-Score", "Precision", "Recall"]
    )
    
    st.divider()
    
    # Estatísticas
    st.subheader("📈 Resumo")
    if len(df_filtered) > 0:
        st.metric(
            "Total de Execuções",
            len(df_filtered),
            delta=f"+{len(df_filtered)}" if len(df_filtered) > 1 else "Primeira"
        )

# ==================== MÉTRICAS PRINCIPAIS ====================
st.header("🎯 Métricas Atuais")

if len(df_filtered) > 0:
    latest = df_filtered.iloc[-1]
    previous = df_filtered.iloc[-2] if len(df_filtered) > 1 else None
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_roc = None
        if previous is not None and previous['roc_auc'] is not None:
            delta_roc = latest['roc_auc'] - previous['roc_auc']
        st.metric(
            "ROC-AUC",
            f"{latest['roc_auc']:.4f}" if latest['roc_auc'] else "N/A",
            delta=f"{delta_roc:+.4f}" if delta_roc else None,
            help="Métrica principal de performance"
        )
    
    with col2:
        delta_f1 = None
        if previous is not None and previous['f1_score'] is not None:
            delta_f1 = latest['f1_score'] - previous['f1_score']
        st.metric(
            "F1-Score",
            f"{latest['f1_score']:.4f}" if latest['f1_score'] else "N/A",
            delta=f"{delta_f1:+.4f}" if delta_f1 else None,
            help="Balanço entre Precision e Recall"
        )
    
    with col3:
        st.metric(
            "Precision",
            f"{latest['precision']:.4f}" if latest['precision'] else "N/A",
            help="Proporção de positivos corretos"
        )
    
    with col4:
        st.metric(
            "Recall",
            f"{latest['recall']:.4f}" if latest['recall'] else "N/A",
            help="Cobertura de positivos reais"
        )

# ==================== GRÁFICOS ====================
st.header("📊 Análise Temporal")

if len(df_filtered) > 0:
    # Gráfico de evolução
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # ROC-AUC
    ax1 = axes[0]
    ax1.plot(df_filtered['timestamp'], df_filtered['roc_auc'], 
             marker='o', linewidth=2, markersize=6, color='#1f77b4', label='ROC-AUC')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Baseline (0.5)')
    ax1.fill_between(df_filtered['timestamp'], 0.5, df_filtered['roc_auc'], 
                     alpha=0.2, color='#1f77b4')
    ax1.set_xlabel('Data', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('ROC-AUC ao Longo do Tempo', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.3, 1.0])
    
    # F1-Score
    ax2 = axes[1]
    ax2.plot(df_filtered['timestamp'], df_filtered['f1_score'], 
             marker='s', linewidth=2, markersize=6, color='#ff7f0e', label='F1-Score')
    ax2.fill_between(df_filtered['timestamp'], 0, df_filtered['f1_score'], 
                     alpha=0.2, color='#ff7f0e')
    ax2.set_xlabel('Data', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('F1-Score ao Longo do Tempo', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    st.pyplot(fig)

# ==================== HISTÓRICO ====================
st.header("📋 Histórico de Execuções")

if len(df_filtered) > 0:
    # Criar DataFrame para display
    display_df = df_filtered[['timestamp', 'run_id', 'roc_auc', 'f1_score', 'precision', 'recall', 'status']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df.columns = ['Data/Hora', 'Run ID', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Status']
    
    # Formatar números
    for col in ['ROC-AUC', 'F1-Score', 'Precision', 'Recall']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    # Colorir status
    def color_status(status):
        return '🔴' if status == 'Degradado' else '🟢'
    
    display_df['Status'] = display_df['Status'].apply(color_status)
    
    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        height=400
    )

# ==================== ALERTAS ====================
st.header("⚠️ Alertas de Degradação")

if len(df_filtered) > 1:
    latest_roc = df_filtered.iloc[-1]['roc_auc']
    previous_roc = df_filtered.iloc[-2]['roc_auc']
    
    if latest_roc is not None and previous_roc is not None:
        degradation = latest_roc - previous_roc
        
        if degradation < -0.05:
            st.error(f"🔴 **ALERTA CRÍTICO**: Degradação de ROC-AUC ({degradation:+.4f})")
            st.warning("Considere retreinar o modelo com dados mais recentes.")
        elif degradation < 0:
            st.warning(f"🟠 **ALERTA**: Ligeira degradação de ROC-AUC ({degradation:+.4f})")
        else:
            st.success(f"🟢 **Melhoria detectada**: ROC-AUC {degradation:+.4f}")
else:
    st.info("ℹ️ Necessária mais de uma execução para detectar degradação.")

# ==================== RODAPÉ ====================
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"📊 Total de execuções: {len(df_filtered)}")
with col2:
    st.caption(f"🕐 Última atualização: {df_filtered.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}" if len(df_filtered) > 0 else "N/A")
with col3:
    st.caption("Powered by MLflow + Streamlit")
