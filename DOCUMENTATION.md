# DOCUMENTATION.md - Churn Prediction Project

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Projeto](#arquitetura-do-projeto)
3. [Processo Completo](#processo-completo)
4. [Instalação e Configuração](#instalação-e-configuração)
5. [Execução do Pipeline](#execução-do-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Modelos e Otimização](#modelos-e-otimização)
8. [Parâmetros de Configuração](#parâmetros-de-configuração)
9. [API REST (FastAPI)](#api-rest-fastapi)
10. [Dashboard (Streamlit)](#dashboard-streamlit)
11. [Interpretabilidade (SHAP)](#interpretabilidade-shap)
    - [Impactos do Projeto](#impactos-do-projeto)
12. [Inferência](#inferência)
13. [Monitoramento](#monitoramento)
14. [Testes](#testes)
15. [CI/CD](#cicd)
16. [Changelog](#changelog)
17. [Troubleshooting](#troubleshooting)

---

## Visão Geral

### Objetivo
Sistema de predição de churn (cancelamento) de clientes para empresas de telecomunicações, utilizando machine learning para identificar clientes com alto risco de cancelamento.

### Métricas Alcançadas (CatBoost — Otimizado)
- **F1-Score**: 63.5%
- **AUC-ROC**: 84.2%
- **AUPRC**: 65.4%
- **Accuracy**: 76.9%
- **Precision**: 54.6%
- **Recall**: 75.7%
- **Threshold otimizado**: 0.41
- **Testes**: 256 passando (cobertura 70%+)

> Baseline LightGBM (train_pipeline.py): F1=57.4%, AUC=83.3%.

### Stack Tecnológica
- **Python 3.12+**
- **XGBoost / LightGBM / CatBoost**: Modelos principais
- **Optuna 4.7+**: Otimização bayesiana de hiperparâmetros
- **imbalanced-learn**: Balanceamento de classes (SMOTETomek)
- **scikit-learn**: Pipeline e métricas
- **MLflow**: Tracking de experimentos
- **SHAP**: Interpretabilidade
- **FastAPI**: API REST
- **Streamlit**: Dashboard interativo
- **Poetry**: Gerenciamento de dependências

---

## Arquitetura do Projeto

```
churn_clientes/
├── src/                          # Código fonte principal
│   ├── __init__.py
│   ├── config.py                 # Configurações centralizadas (paths, constantes)
│   ├── enums.py                  # Enumerações do framework (ModelType, DriftSeverity)
│   ├── preprocessing.py          # Pré-processamento
│   ├── feature_engineering.py    # FeatureEngineer e AdvancedFeatureEngineer
│   ├── pipeline.py               # ChurnPipeline principal
│   ├── optimization.py           # HyperparameterOptimizer, quick_optimize, optimize_for_f1
│   ├── evaluate.py               # Métricas de avaliação (F1, AUC-ROC, AUPRC, Accuracy, etc.)
│   ├── inference.py              # Inferência em produção
│   ├── interpret.py              # Interpretabilidade (SHAP, model-agnostic)
│   ├── monitoring.py             # Monitoramento de drift (PSI)
│   ├── logger.py                 # Sistema de logging JSON
│   ├── utils.py                  # Utilitários centralizados
│   └── notebook_utils.py         # Utilitários para notebooks
│
├── scripts/                      # Scripts e Dashboard
│   ├── dashboard.py              # Dashboard Streamlit interativo
│   ├── train_pipeline.py         # Pipeline de treinamento padrão
│   ├── run_pipeline.py           # Pipeline completo (treino + avaliação + salvar)
│   ├── optimize_model.py         # Otimização multi-modelo (LightGBM, XGBoost, CatBoost)
│   └── monitoring_pipeline.py    # Pipeline de monitoramento
│
├── api/                          # API REST (FastAPI)
│   ├── main.py                   # Aplicação FastAPI
│   ├── core/                     # Configuração e logging
│   │   ├── config.py             # Settings Pydantic
│   │   └── logging.py            # Logging estruturado
│   ├── routes/                   # Endpoints (predict, models, health, interpret, drift)
│   ├── schemas/                  # Modelos Pydantic (gerados dinamicamente do YAML)
│   └── services/                 # ModelService singleton
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb              # Análise exploratória
│   ├── 02_preprocess_feature_engineering.ipynb
│   ├── 03_modeling.ipynb         # Modelagem otimizada (FASE 5)
│   └── 04_shap_interpretability.ipynb  # Interpretabilidade SHAP
│
├── tests/                        # Testes unitários
│   ├── conftest.py               # Fixtures pytest
│   └── test_*.py                 # Testes por módulo
│
├── models/                       # Modelos treinados
├── reports/                      # Relatórios gerados
├── mlruns/                       # Tracking MLflow
├── optuna_studies/               # Cache de estudos Optuna
├── logs/                         # Logs do pipeline
│
├── Dockerfile                    # Multi-stage build (builder → production → dashboard → dev)
├── docker-compose.yml            # Orquestração (api, dashboard, api-dev, mlflow)
├── Makefile                      # Comandos de automação (make test, make api, make dashboard)
├── pyproject.toml                # Configuração Poetry
├── requirements.txt              # Dependências (pip) - sincronizado
├── QUICKSTART.md                 # Guia de início rápido
└── DOCUMENTATION.md              # Esta documentação
```

### Componentes Principais

#### 1. ChurnPipeline (`src/pipeline.py`)
Pipeline principal de treinamento com sklearn Pipeline.

```python
from src.pipeline import ChurnPipeline

from src.config import ModelConfig
from src.enums import ModelType

config = ModelConfig(n_estimators=100, max_depth=5, model_type=ModelType.XGBOOST)
pipeline = ChurnPipeline(config)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

#### 2. HyperparameterOptimizer (`src/optimization.py`)
Otimização bayesiana de hiperparâmetros.

```python
from src.optimization import HyperparameterOptimizer, OptimizationConfig

config = OptimizationConfig(n_trials=100, metric="f1")
optimizer = HyperparameterOptimizer(X_train, y_train, config)
result = optimizer.optimize()
```

#### 3. Funções Auxiliares de Otimização (`src/optimization.py`)

```python
from src.optimization import quick_optimize, optimize_for_f1

# Otimização rápida (default: n_trials=30, sem SMOTE, sem threshold)
result = quick_optimize(X_train, y_train, n_trials=30)

# Otimização focada em F1 (default: n_trials=50, com SMOTE + threshold)
result = optimize_for_f1(X_train, y_train, n_trials=100)
```

#### 4. FeatureEngineer / AdvancedFeatureEngineer (`src/feature_engineering.py`)
Transformadores de features.

```python
from src.feature_engineering import AdvancedFeatureEngineer

fe = AdvancedFeatureEngineer()
X_train_fe = fe.fit_transform(X_train, y_train)
X_test_fe = fe.transform(X_test)
```

---

## Processo Completo

### Fluxo de Treinamento

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Dados Raw     │───▶│  Feature Eng.    │───▶│  Preprocessor    │
│  (CSV/Parquet)  │    │  (67 features)   │    │  (Scale/Encode)  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Evaluation    │◀───│    Training      │◀───│   SMOTE/Tomek   │
│ (F1, AUC, etc)  │    │(XGBoost/LightGBM)│    │  Balanceamento   │
└─────────────────┘    └──────────────────┘    └──────────────────┘
        │
        ▼
┌─────────────────┐    ┌──────────────────┐
│  Threshold Opt  │───▶│   Save Model     │
│ (Max F1-Score)  │    │   (.joblib)      │
└─────────────────┘    └──────────────────┘
```

### Etapas Detalhadas

1. **Carregamento dos Dados**
   - Leitura do CSV/Parquet
   - Remoção de ID
   - Conversão do target (Yes/No → 1/0)

2. **Divisão Train/Test**
   - Estratificado por target
   - 80% train, 20% test (configurável)

3. **Feature Engineering**
   - Features básicas (19): tenure, MonthlyCharges, etc.
   - Features derivadas: bins, ratios, flags
   - Features avançadas: risk scores, interactions
   - **Total: 67 features (48 novas + 19 originais)**

4. **Pré-processamento**
   - StandardScaler ou RobustScaler para numéricas
   - OneHotEncoder para categóricas

5. **Balanceamento (opcional)**
   - SMOTE: Synthetic Minority Oversampling
   - SMOTETomek: SMOTE + limpeza Tomek Links

6. **Otimização (Optuna)**
   - TPE Sampler (Tree-structured Parzen Estimators)
   - Cross-validation estratificado (5-fold)

7. **Threshold Optimization**
   - Curva Precision-Recall
   - Threshold que maximiza F1

8. **Avaliação Final**
   - F1-Score, Precision, Recall
   - AUC-ROC, AUPRC, Accuracy
   - Matriz de confusão

---

## Instalação e Configuração

### Pré-requisitos
- Python 3.12+
- Poetry (recomendado) ou pip
- Docker (opcional, para deploy containerizado)

### Instalação com Poetry

```bash
# Clonar repositório
git clone <repo-url>
cd churn_clientes

# Instalar dependências
poetry install

# Ativar ambiente virtual
poetry shell
```

### Instalação com pip

```bash
pip install -r requirements.txt
```

### Verificar Instalação

```bash
# Executar testes
poetry run pytest tests/ -v

# Verificar imports
poetry run python -c "from src.pipeline import ChurnPipeline; print('OK')"
```

---

## Execução do Pipeline

### 1. Treinamento Padrão

```bash
# Treinamento com LightGBM (parâmetros definidos no config YAML)
poetry run python scripts/train_pipeline.py
```

### 2. Otimização Multi-Modelo

```bash
# Otimização completa (20 trials por modelo, 5-fold CV, SMOTE)
poetry run python scripts/optimize_model.py

# Otimização rápida (10 trials, 3-fold CV)
poetry run python scripts/optimize_model.py --quick

# Mais trials para resultado mais refinado
poetry run python scripts/optimize_model.py --trials 50

# Modelos específicos
poetry run python scripts/optimize_model.py --models lgb xgb
```

O script compara LightGBM, XGBoost e CatBoost, salva cada modelo
individualmente (`model_lightgbm.joblib`, etc.) e copia o melhor
como `model.joblib` (padrão da API e dashboard).

### 3. Via Makefile

```bash
# Treinamento padrão
make train

# Otimização multi-modelo
make train-optimized

# Otimização rápida
make train-quick

# Ou via Docker
docker compose up api  # O modelo é carregado do diretório models/
```

### 4. Apenas Feature Engineering

```python
from src.feature_engineering import AdvancedFeatureEngineer
import pandas as pd

df = pd.read_csv("src/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

fe = AdvancedFeatureEngineer()
X_transformed = fe.fit_transform(df.drop("Churn", axis=1), df["Churn"])
print(f"Features: {X_transformed.shape[1]}")  # 67 features
```

---

## Feature Engineering

### Features Originais (19)
| Feature | Tipo | Descrição |
|---------|------|-----------|
| gender | Categórica | Gênero do cliente |
| SeniorCitizen | Binária | Se é idoso (65+) |
| Partner | Binária | Se tem parceiro |
| Dependents | Binária | Se tem dependentes |
| tenure | Numérica | Meses como cliente |
| PhoneService | Binária | Serviço telefônico |
| MultipleLines | Categórica | Linhas múltiplas |
| InternetService | Categórica | Tipo de internet |
| OnlineSecurity | Categórica | Segurança online |
| OnlineBackup | Categórica | Backup online |
| DeviceProtection | Categórica | Proteção de dispositivo |
| TechSupport | Categórica | Suporte técnico |
| StreamingTV | Categórica | Streaming TV |
| StreamingMovies | Categórica | Streaming filmes |
| Contract | Categórica | Tipo de contrato |
| PaperlessBilling | Binária | Fatura digital |
| PaymentMethod | Categórica | Método de pagamento |
| MonthlyCharges | Numérica | Cobrança mensal |
| TotalCharges | Numérica | Total cobrado |

### Features Derivadas (FeatureEngineer) [19]
| Feature | Fórmula | Descrição |
|---------|---------|-----------|
| new_customer_flag | tenure == 0 | Flag de cliente novo |
| tenure_group | pd.cut(tenure, bins) | Faixa de tenure (0-6m, 7-12m, etc.) |
| avg_monthly_charge | TotalCharges / tenure | Cobrança média mensal |
| charges_mismatch_flag | divergência > 0.3 | Divergência entre charges esperado e real |
| is_monthly_contract | Contract == "Month-to-month" | Contrato mensal |
| long_term_contract | Contract ∈ {"One year", "Two year"} | Contrato de longo prazo |
| MonthlyCharges_binned | Quartis de MonthlyCharges | Faixa de cobrança mensal |
| is_high_MonthlyCharges | avg_monthly > mediana | Cobrança acima da mediana |
| price_sensitive_contract | Mensal + MonthlyCharges alto | Sensibilidade a preço |
| early_churn_risk | Mensal + tenure < 12 | Risco de churn precoce |
| tenure_vs_contract_avg | tenure / avg_tenure_contrato | Comparativo com segmento |
| contract_risk_ordinal | Ordinal por churn rate | Contrato ordenado por risco |
| contract_churn_rate | Taxa churn por contrato | Risco por contrato (supervisionado) |
| MonthlyCharges_log | log1p(MonthlyCharges) | Transformação log |
| MonthlyCharges_squared | MonthlyCharges² | Transformação quadrática |
| TotalCharges_log | log1p(TotalCharges) | Transformação log |
| TotalCharges_squared | TotalCharges² | Transformação quadrática |
| avg_monthly_charge_log | log1p(avg_monthly_charge) | Transformação log |
| avg_monthly_charge_squared | avg_monthly_charge² | Transformação quadrática |

### Features Avançadas (AdvancedFeatureEngineer) [29]
| Categoria | Features | Descrição |
|-----------|----------|-----------|
| **Serviços** | total_services, service_density, streaming_services, has_streaming, security_services, has_security | Contagem e proporção de serviços |
| **Engajamento** | low_engagement, high_engagement, digital_only, family_account, family_stability | Padrões de uso e estabilidade |
| **Interações** | fiber_no_security, streamer_unprotected, senior_monthly, new_high_value | Combinações de risco |
| **Risk Scores** | churn_risk_score, high_risk_flag, very_high_risk_flag, internet_risk, payment_risk | Scores compostos e por categoria |
| **Lifecycle** | lifecycle_new, lifecycle_early, lifecycle_mid, lifecycle_mature, tenure_years, tenure_sqrt | Fases do ciclo de vida |
| **Valor** | estimated_clv, value_per_month, cost_per_service | Valor e custo do cliente |

---

## Modelos e Otimização

### Modelos Disponíveis

#### XGBoost / LightGBM (compartilhado)
```python
# Espaço de busca definido em config/project.yaml
{
    'n_estimators': [100, 500],
    'max_depth': [3, 10],
    'learning_rate': [0.01, 0.3],
    'num_leaves': [15, 127],       # LightGBM específico
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'min_child_samples': [5, 100], # LightGBM específico
    'reg_alpha': [0.0, 1.0],
    'reg_lambda': [0.0, 1.0]
}
```

### Ensemble

Configurado no `config/project.yaml` (desabilitado por padrão):

```yaml
# config/project.yaml
model:
  ensemble:
    enabled: false
    models:
      - algorithm: "xgboost"
        weight: 0.5
      - algorithm: "lightgbm"
        weight: 0.5
    method: "voting"  # voting, stacking
```

### Otimização Optuna

```python
# Configuração padrão (n_trials vem do config/project.yaml, default=50)
OptimizationConfig(
    n_trials=50,
    n_cv_splits=5,
    metric="f1",
    random_state=42
)
```

**Sampler**: TPESampler (Tree-structured Parzen Estimators)
- Bayesian optimization
- Explora regiões promissoras do espaço de busca

---

## Parâmetros de Configuração

### config.py

```python
# Paths
DATA_DIR_RAW = Path("src/data/raw")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Data
FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COL = "customerID"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

# Features
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [...]

```

### Variáveis de Ambiente

```bash
# MLflow (padrão: "mlruns" — file-based)
export MLFLOW_TRACKING_URI="mlruns"

# API
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

---

## Inferência

### Carregar e Usar Modelo

```python
from src.inference import load_model_package, predict_with_package
import pandas as pd

# Carregar modelo salvo
package = load_model_package("models/model.joblib")

# Predição única
customer_df = pd.DataFrame([{
    "tenure": 24,
    "MonthlyCharges": 75.50,
    "Contract": "Month-to-month",
    # ... demais features
}])

# Retorna np.ndarray com predições (0 ou 1)
predictions = predict_with_package(package, customer_df)
print(f"Churn: {predictions[0]}")  # 0 ou 1

# Com probabilidades: retorna tupla (predictions, probabilities)
predictions, probas = predict_with_package(package, customer_df, return_proba=True)
print(f"Churn: {predictions[0]}, Prob: {probas[0][1]:.2%}")
```

### API de Predição (exemplo)

```python
# Usando o módulo de inferência diretamente
from src.inference import load_model_package, predict_with_package

package = load_model_package("models/model.joblib")
predictions, probas = predict_with_package(package, customer_data, return_proba=True)
print(f"Probabilidade de churn: {probas[0][1]:.2%}")
```

### Formato de Entrada

```json
{
    "customerID": "optional",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 2290.0
}
```

### Formato de Saída

#### Via `predict_with_package()` (módulo Python)
```python
# Sem probabilidades:
predictions = predict_with_package(package, df)
# → np.ndarray([0, 1, 0, 1, ...])

# Com probabilidades:
predictions, probas = predict_with_package(package, df, return_proba=True)
# predictions → np.ndarray([0, 1, 0, 1, ...])
# probas → np.ndarray([[0.85, 0.15], [0.22, 0.78], ...])  # [prob_no_churn, prob_churn]
```

#### Via API REST (`POST /predict/`)
```json
{
    "success": true,
    "prediction": {
        "customer_id": "123",
        "churn_prediction": 1,
        "churn_probability": 0.78,
        "churn_risk": "HIGH",
        "confidence": 0.56
    },
    "model_version": "1.3.0",
    "timestamp": "2026-03-06T12:00:00Z"
}
```

---

## Monitoramento

### Drift Detection

```python
from src.monitoring import detect_drift, population_stability_index

# Detectar drift entre dados de referência e novos dados
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]
all_features = numeric_features + categorical_features

drift_report = detect_drift(
    X_train, X_new,
    features=all_features,
    categorical_features=categorical_features  # PSI categórico por proporções
)

if drift_report.features_with_drift > 0:
    print("ALERTA: Drift detectado!")
    for result in drift_report.feature_results:
        if result.has_drift:
            print(f"  {result.feature_name}: PSI={result.psi:.4f} ({result.severity})")
```

> **Nota**: Para features categóricas, `detect_drift()` usa `categorical_psi()` (comparação por proporções de cada categoria) em vez de histogram binning, evitando PSI artificialmente alto em variáveis com poucas categorias.

### PSI (Population Stability Index)

```python
from src.monitoring import population_stability_index, classify_drift_severity

# Calcular PSI para uma feature numérica
psi_value = population_stability_index(reference_values, current_values)
severity = classify_drift_severity(psi_value)
print(f"PSI: {psi_value:.4f} - Severity: {severity}")
```

### Lógica de Escalação de Severidade

A severidade geral do drift é determinada pela contagem de features afetadas:

| Condição | Severidade Geral |
|----------|------------------|
| ≥2 features com drift HIGH | **HIGH** |
| 1 feature HIGH ou ≥2 features MODERATE | **MODERATE** |
| Qualquer feature com drift LOW ou MODERATE | **LOW** |
| Nenhum drift detectado | **NONE** |

### MLflow Tracking

```python
import mlflow

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("churn_prediction")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "f1_score": f1,
        "auc_roc": auc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    })
    mlflow.sklearn.log_model(model, "model")
```

### Monitoramento via Dashboard

```bash
# Executar dashboard (inclui aba Data Drift)
poetry run streamlit run scripts/dashboard.py
```

---

## Testes

### Executar Todos os Testes

```bash
poetry run pytest tests/ -v
```

### Executar Testes Específicos

```bash
# Por módulo
poetry run pytest tests/test_optimization.py -v

# Por função
poetry run pytest tests/test_evaluate.py::test_evaluate_basic -v

# Com coverage
poetry run pytest tests/ --cov=src --cov-report=html
```

### Estrutura de Testes

```python
# tests/test_optimization.py
def test_default_values():
    config = OptimizationConfig()
    assert config.n_trials == 50  # Valor do config/project.yaml
    assert config.use_smote is True

```

### Fixtures Disponíveis

```python
# conftest.py
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"age": [...], "income": [...], "city": [...], "target": [...]})

@pytest.fixture
def sample_trained_model():
    # XGBClassifier treinado com make_classification
    model = XGBClassifier(n_estimators=10, max_depth=2)
    model.fit(X, y)
    return model

@pytest.fixture
def sample_X_y():
    # 100 amostras com 3 features
    return X, y
```

---

## CI/CD

### GitHub Actions Workflows

O projeto utiliza 3 workflows para CI/CD:

```yaml
# .github/workflows/ci.yml - Pipeline principal
# Trigger: push/PR para main, develop, feature/*
# Jobs: lint → test (coverage ≥70%) → build Docker

# .github/workflows/code-quality.yml - Qualidade de código
# Trigger: push/PR
# Jobs: ruff check, black --check, isort --check

# .github/workflows/pr-analysis.yml - Análise de PR
# Trigger: pull_request
# Jobs: análise de código e métricas
```

### Práticas de CI/CD

1. **Lint**: Ruff, Black, isort
2. **Tests**: pytest com coverage mínimo 70%
3. **Security**: bandit, safety
4. **Build**: Docker multi-stage
5. **Deploy**: GitHub Releases

### Workflows Disponíveis

| Workflow | Arquivo | Trigger | Descrição |
|----------|---------|---------|----------|
| CI | `ci.yml` | push/PR | Testes, coverage (≥70%), lint |
| Code Quality | `code-quality.yml` | push/PR | Ruff, Black, isort, type checking |
| PR Analysis | `pr-analysis.yml` | PR | Análise de código e métricas |

### Docker

```bash
# Build e execução com docker-compose
docker compose up api

# Build manual
docker build --target production -t churn-api .

# Executar container
docker run -p 8000:8000 -v ./models:/app/models:ro churn-api
```

O Dockerfile usa multi-stage build:
- **builder**: Instala Poetry e dependências
- **production**: Imagem slim com `libgomp1` (requerido por LightGBM/CatBoost)
- **dashboard**: Streamlit em modo headless na porta 8501
- **development**: Adiciona ferramentas de dev (pytest, ruff, etc.)

### Docker Compose — Serviços

| Serviço | Container | Porta | Profile | Descrição |
|---------|-----------|:-----:|---------|----------|
| `api` | churn-api | 8000 | — | API REST com healthcheck |
| `dashboard` | churn-dashboard | 8501 | — | Dashboard Streamlit (depende da API) |
| `api-dev` | churn-api-dev | 8000 | `dev` | API com hot reload |
| `mlflow` | mlflow-server | 5000 | `mlflow` | MLflow Tracking Server |

### Comandos Docker

```bash
# Subir API + Dashboard
docker compose up --build

# Subir em background
docker compose up -d

# Subir apenas a API
docker compose up api

# Subir apenas o Dashboard (inicia a API automaticamente)
docker compose up dashboard

# Subir com MLflow
docker compose --profile mlflow up

# API em modo desenvolvimento (hot reload + volumes)
docker compose --profile dev up api-dev

# Ver logs
docker compose logs -f
docker compose logs -f dashboard

# Rebuild após mudanças
docker compose up --build --force-recreate

# Build manual de um stage
docker build --target production -t churn-api .
docker build --target dashboard -t churn-dashboard .

# Parar tudo
docker compose down
```

### Comunicação entre Containers

O Dashboard se comunica com a API via rede interna do Docker:
- Variável `API_HOST=api` aponta para o nome do serviço (resolvido pelo DNS do Docker)
- Variável `API_PORT=8000` define a porta interna
- O Dashboard só inicia após o healthcheck da API passar (`depends_on` com `condition: service_healthy`)

### Volumes

| Volume | Container | Modo | Conteúdo |
|--------|-----------|------|--------|
| `./models` | api, dashboard | `ro` | Modelos .joblib |
| `./reports` | api, dashboard | `ro` | Métricas e relatórios |
| `./config` | dashboard | `ro` | Configurações YAML |
| `mlflow-data` | mlflow | `rw` | Artefatos MLflow |

### URLs de Acesso

| Serviço | URL |
|---------|-----|
| API (Swagger) | http://localhost:8000/docs |
| API (ReDoc) | http://localhost:8000/redoc |
| Dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |

---

## API REST (FastAPI)

### Iniciar API

```bash
# Desenvolvimento
poetry run uvicorn api.main:app --reload --port 8000

# Produção
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Endpoints Disponíveis

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Informações da API |
| `/health` | GET | Health check |
| `/health/live` | GET | Liveness probe (Kubernetes) |
| `/health/ready` | GET | Readiness probe (Kubernetes) |
| `/predict/` | POST | Predição single |
| `/predict/batch` | POST | Predição em batch |
| `/models/` | GET | Listar modelos disponíveis |
| `/models/switch` | POST | Trocar modelo ativo (hot-swap) |
| `/model/info` | GET | Informações do modelo |
| `/interpret/feature-importance` | GET | Feature importance |
| `/interpret/shap/explain` | POST | SHAP explanation |
| `/interpret/shap/global` | GET | SHAP global summary |
| `/drift/detect` | POST | Detecção de data drift (upload CSV) |
| `/drift/status` | GET | Status do monitoramento de drift |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc docs |

### Exemplo de Predição

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "123",
    "features": {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 24,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "Yes",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 95.50,
      "TotalCharges": 2290.0
    }
  }'
```

### Resposta

```json
{
  "success": true,
  "prediction": {
    "customer_id": "123",
    "churn_prediction": 1,
    "churn_probability": 0.78,
    "churn_risk": "HIGH",
    "confidence": 0.56
  },
  "model_version": "1.3.0"
}
```

### Gerenciamento de Modelos

A API permite listar e trocar o modelo ativo em tempo real (hot-swap):

```bash
# Listar modelos disponíveis
curl http://localhost:8000/models/

# Resposta:
# {
#   "models": [
#     {"name": "model", "filename": "model.joblib", "size_mb": 1.23, "is_default": true},
#     {"name": "model_xgboost", "filename": "model_xgboost.joblib", "size_mb": 0.85, "is_default": false},
#     ...
#   ],
#   "active_model": "model_catboost",
#   "total": 5
# }

# Trocar modelo ativo
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "model_xgboost"}'

# Resposta:
# {"success": true, "active_model": "model_xgboost", "estimator_name": "XGBClassifier", "message": "Modelo trocado para 'model_xgboost' com sucesso."}
```

A troca é atômica com rollback automático: se o novo modelo falhar ao carregar, o modelo anterior é restaurado.

---

## Dashboard (Streamlit)

### Dashboard Streamlit

#### Iniciar Dashboard

```bash
# Local
poetry run streamlit run scripts/dashboard.py

# Via Docker
docker compose up dashboard

# Via Makefile
make dashboard
```

Acesse: http://localhost:8501

#### Funcionalidades

1. **Home**: Visão geral do projeto, métricas e modelo ativo
2. **Predição**: Interface interativa para fazer predições (API-first com fallback local)
3. **Performance**: Métricas detalhadas (F1, AUC-ROC, AUPRC, Accuracy, Precision, Recall) + Validação Cruzada
4. **Comparação de Modelos**: Tabela comparativa, gráficos de barras e radar multi-métrica
5. **Interpretabilidade**: Análise SHAP interativa (model-agnostic) + Feature Importance
6. **Data Drift**: Monitoramento de drift com PSI (upload CSV ou dados de referência)
7. **MLflow**: Tracking de experimentos

### Identificação do Modelo

O dashboard mostra o nome do modelo ativo em cada página. É possível trocar o modelo na sidebar, e todas as páginas refletem o modelo selecionado.

### Seleção de Modelo via API

Quando a API está online, o dashboard sincroniza a seleção de modelo diretamente com a API:
- A lista de modelos é obtida via `GET /models/`
- Ao trocar o modelo no sidebar, o dashboard chama `POST /models/switch` para aplicar a mudança na API
- As predições e interpretações subsequentes usam o modelo recém-ativado
- Se a API estiver offline, o dashboard faz fallback para carregamento local de modelos

---

## Interpretabilidade (SHAP)

### O que é SHAP?

SHAP (SHapley Additive exPlanations) é uma técnica de interpretabilidade que:
- Explica predições individuais
- Mostra a contribuição de cada feature
- Baseado em teoria de jogos cooperativos

### Notebook de Análise

```bash
poetry run jupyter notebook notebooks/04_shap_interpretability.ipynb
```

### Usando SHAP Programaticamente

```python
import shap
from joblib import load

# Carregar modelo
model = load("models/model.joblib")
xgb_model = model.named_steps['model']

# Criar explainer
explainer = shap.TreeExplainer(xgb_model)

# Calcular SHAP values
shap_values = explainer.shap_values(X_test)

# Visualizar
shap.summary_plot(shap_values, X_test)
```

### Via API

```bash
# Feature Importance
curl "http://localhost:8000/interpret/feature-importance?top_n=10"

# SHAP Global
curl "http://localhost:8000/interpret/shap/global?sample_size=100"

# SHAP Individual (ShapExplainRequest)
curl -X POST "http://localhost:8000/interpret/shap/explain" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "cust-001", "features": {"tenure": 24, "MonthlyCharges": 95.50, ...}}'
```

### Tipos de Visualização

1. **Summary Plot (Bar)**: Importância média por feature
2. **Summary Plot (Beeswarm)**: Distribuição de impacto
3. **Waterfall Plot**: Explicação de predição individual
4. **Force Plot**: Contribuição de cada feature
5. **Dependence Plot**: Relação feature-impacto

### Impactos do Projeto

#### Insights de Negócio Identificados

Com base nas análises SHAP do modelo de churn, foram identificados os seguintes padrões críticos:

| Feature | Impacto no Churn | Insight de Negócio |
|---------|------------------|-------------------|
| **tenure** | ↓ Alta permanência = Baixo churn | Clientes nos primeiros 12 meses são mais vulneráveis |
| **Contract** | ↑ Mensal = Alto churn | Contratos mensais têm 3x mais churn que anuais |
| **MonthlyCharges** | ↑ Valor alto + tenure baixo = Risco | Desproporção preço/tempo é fator crítico |
| **TechSupport** | ↓ Com suporte = Baixo churn | Suporte técnico reduz churn em até 40% |
| **OnlineSecurity** | ↓ Com segurança = Baixo churn | Serviços de segurança aumentam retenção |
| **InternetService** | ↑ Fiber = Maior churn | Clientes de fibra têm expectativas mais altas |
| **PaymentMethod** | ↑ Electronic check = Alto risco | Pagamento automático reduz churn |

#### Recomendações Estratégicas

**1. Programa de Onboarding Intensivo (Primeiros 90 dias)**
```
- Contato proativo no 1º, 30º e 60º dia
- Ofertas de upgrade para contratos anuais
- Tutorial de uso dos serviços contratados
```

**2. Migração de Contratos Mensais**
```
- Desconto de 15-20% para migração anual
- Benefícios exclusivos (ex: velocidade extra)
- Campanha nos meses 3, 6 e 9 de tenure
```

**3. Bundle de Serviços de Retenção**
```
- TechSupport + OnlineSecurity como pacote
- Redução esperada de churn: 35-45%
- Target: clientes sem estes serviços
```

**4. Alertas de Risco em Tempo Real**
```python
# Exemplo de integração com CRM
if prediction['churn_probability'] > 0.7:
    shap_top_factors = get_shap_explanation(customer)
    create_retention_ticket(
        customer_id=customer.id,
        risk_level='high',
        main_factors=shap_top_factors[:3],
        suggested_actions=get_retention_playbook(shap_top_factors)
    )
```

#### Métricas de Impacto

| Métrica | Antes do Modelo | Com o Modelo | Melhoria |
|---------|-----------------|--------------|----------|
| Taxa de Churn Mensal | 26.5% | ~18-20%* | -25% |
| Custo de Retenção | Reativo | Proativo | -40%* |
| LTV Médio | Baseline | +15%* | Significativo |
| NPS | Baseline | +8 pts* | Positivo |

*Valores projetados baseados em benchmarks de mercado

#### Caso de Uso: Explicação SHAP Individual

```python
# Cliente de alto risco identificado
cliente_exemplo = {
    "tenure": 3,
    "Contract": "Month-to-month", 
    "MonthlyCharges": 95.50,
    "TechSupport": "No",
    "OnlineSecurity": "No"
}

# SHAP revela:
# - tenure=3: +0.28 (muito curto)
# - Contract=Monthly: +0.22 (sem compromisso)
# - TechSupport=No: +0.15 (sem suporte)

# Ação recomendada:
# 1. Oferecer desconto para contrato anual
# 2. Trial gratuito de TechSupport por 3 meses
# 3. Acompanhamento do Customer Success
```

#### Integração com Decisões de Negócio

```
┌─────────────────────────────────────────────────────────────┐
│                    CICLO DE RETENÇÃO                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
│   │ Modelo   │───►│  SHAP    │───►│ Ação Automática  │     │
│   │ Predição │    │ Explain  │    │ (CRM/Marketing)  │     │
│   └──────────┘    └──────────┘    └──────────────────┘     │
│        │                                    │               │
│        ▼                                    ▼               │
│   ┌──────────┐                      ┌──────────────┐       │
│   │ Dashboard│◄─────────────────────│   Feedback   │       │
│   │ Métricas │                      │   Loop       │       │
│   └──────────┘                      └──────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Erros Comuns

#### 1. ModuleNotFoundError
```bash
# Solução: Adicionar src ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Ou usar poetry run
poetry run python script.py
```

#### 2. MemoryError durante SMOTE
```python
# Reduzir sampling_strategy
smote = SMOTETomek(
    sampling_strategy=0.5,  # Em vez de 1.0
    random_state=42
)
```

#### 3. F1-Score Baixo
```bash
# Retreinar o modelo com configurações ajustadas no config/project.yaml
poetry run python scripts/train_pipeline.py
```

#### 4. Optuna Study não salva
```python
# Usar storage persistente (opcional)
storage = optuna.storages.RDBStorage(
    url="sqlite:///optuna_study.db"
)
study = optuna.create_study(storage=storage)
```

### Logs

```python
# Ver logs detalhados
import logging
logging.basicConfig(level=logging.DEBUG)

# Ou configurar em config.py
LOGGING_LEVEL = "DEBUG"
```

### Performance

```bash
# Profiling
poetry run python -m cProfile -s tottime scripts/train_pipeline.py

# Memory profiling
poetry run python -m memory_profiler scripts/train_pipeline.py
```

---

## Changelog

### v1.8.1 (Março 2026)

#### Correções e Consistência
- **`categorical_psi()` (monitoring.py)**: Corrigido cálculo de PSI categórico — `dropna()` agora aplicado antes de `value_counts()` e `len()`, não apenas em `unique()`. Proporções eram distorcidas quando a série continha NaN.
- **Dashboard model switch (dashboard.py)**: Corrigido auto-trigger de `POST /models/switch` no primeiro render — `selected_model_name` agora inicializado a partir de `active_model` da API, e switch só dispara quando `prev_selected is not None`.
- **`POST /models/switch` response**: `active_model` agora retorna o stem do arquivo (consistente com `GET /models/`). Adicionado campo `estimator_name` para nome legível do estimador.
- **`ModelService.reload_model()` atômico**: Refatorado para criar instância temporária isolada, carregar modelo nela, e só então fazer swap atômico das referências — elimina race condition onde requests concorrentes podiam ver estado parcialmente inicializado.
- **`features_with_drift` consistência (monitoring.py)**: Adicionada property `has_any_drift` (severity != NONE); `features_with_drift` agora usa `has_any_drift` para consistência com `overall_severity` (antes podia reportar 0 features com drift mas severity LOW).
- **Dashboard métricas por modelo**: `_load_metrics_from_report()` agora retorna métricas específicas do modelo selecionado via `individual_results`, em vez de sempre retornar as métricas top-level (CatBoost).
- **Dashboard stale model cache**: Removido `@st.cache_resource` de `load_model()` que causava retorno do modelo da primeira chamada ignorando seleção posterior. Simplificados 3 call sites que duplicavam lógica de seleção de modelo.

#### Testes
- **256 testes passando** (era 196)
- Novos testes em `test_api.py`: `TestModelsManagementEndpoints` (6 testes cobrindo `GET /models/` e `POST /models/switch`)
- Novo `test_dashboard.py`: 60 testes cobrindo funções utilitárias, predição local, métricas por modelo, helpers de severity/drift, API wrappers e routing de `load_model()`

### v1.8.0 (Março 2026)

#### Revisão Completa e Correções de Bugs
- **Feature Engineering**: Corrigido alinhamento de índices em `AdvancedFeatureEngineer.fit()` e `_fit_supervised_encodings()` — `y_binary`/`y` podia ter `RangeIndex` enquanto `df` mantinha índices originais do `train_test_split`, causando `KeyError` em `groupby + .loc`
- **Monitoramento**: Corrigido operador assimétrico na escalação de severidade — `mod_count` agora usa `>=` consistente com `high_count`
- **API Batch Prediction**: Alterado `zip(..., strict=False)` para `strict=True` em `predict_batch()` — evita descarte silencioso de resultados quando arrays têm tamanhos diferentes
- **API SHAP**: Alterado `zip(..., strict=False)` para `strict=True` nos endpoints `/shap/explain` e `/shap/global` — evita mismatch silencioso entre feature names e SHAP values
- **Utils**: Adicionado warning em `convert_target_to_binary()` quando target possui mais de 2 classes
- **Cleanup**: Removido import não utilizado `DRIFT_ESCALATION_LOW_FRACTION` e variável `n` não utilizada em `monitoring.py`

### v1.7.0 (Março 2026)

#### Gerenciamento de Modelos via API
- **Novos endpoints**: `GET /models/` (listar modelos) e `POST /models/switch` (trocar modelo ativo)
- **Hot-swap de modelos**: Troca atômica com rollback automático em caso de falha
- **`ModelService.reload_model()`**: Novo método para recarregar modelo em tempo real
- **Dashboard sincronizado com API**: Quando a API está online, a seleção de modelo no sidebar do Dashboard troca o modelo diretamente na API via `POST /models/switch`
- **Fallback local**: Se a API estiver offline, o Dashboard continua funcionando com seleção local de modelos
- **Novo arquivo**: `api/routes/models.py` com rotas de gerenciamento de modelos

### v1.6.0 (Março 2026)

#### Otimização Multi-Modelo
- **optimize_model.py**: Novo script de otimização multi-modelo (LightGBM, XGBoost, CatBoost) com Optuna
  - Flags: `--quick` (10 trials, 3-fold CV), `--trials N`, `--models lgb xgb cat`
  - Salva cada modelo individualmente e copia o melhor como `model.joblib`
- **CatBoost como melhor modelo**: F1=0.635, AUC=0.842, AUPRC=0.654 (threshold=0.41)
- **Makefile atualizado**: `make train-optimized` e `make train-quick` usam `optimize_model.py`
- **Documentação**: Todas as docs atualizadas com métricas do CatBoost e referências ao optimize_model.py

### v1.5.2 (Março 2026)

#### Correções
- **Feature Engineering**: Corrigido `_calculate_monthly_bins()` — bins agora usam `-np.inf`/`+np.inf` nas bordas para evitar NaN em dados de teste com valores fora do intervalo de treino
- **Feature Engineering**: Adicionado `include_groups=False` em chamadas `groupby.apply()` do `AdvancedFeatureEngineer.fit()` para suprimir FutureWarning do pandas
- **requirements.txt (SCC)**: Fixado `altair>=5.0.0,<6.0.0` (Altair 6 conflita com Streamlit no SCC), relaxado versões de `plotly`, `mlflow`, `streamlit` e `watchdog` para compatibilidade
- **Documentação**: DOCUMENTATION.md e QUICK_REFERENCE.md atualizados com contagem de features correta (67), formato de retorno atualizado do `predict_with_package()`, e CLI do `train_pipeline.py` corrigido

### v1.5.1 (Março 2026)

#### Docker & Dashboard
- **Dashboard no Docker**: Novo stage `dashboard` no Dockerfile (Streamlit headless na porta 8501)
- **docker-compose**: Novo serviço `dashboard` com `depends_on` API healthcheck
- **Comunicação entre containers**: Dashboard usa rede interna Docker (`API_HOST=api`)
- **Documentação atualizada**: README, DOCUMENTATION, QUICKSTART e QUICK_REFERENCE com instruções Docker completas

### v1.5.0 (Março 2026)

#### Qualidade & Testes
- **256 testes passando** (cobertura 70%+, threshold 70%)
- Novos testes: `test_utils.py` (38), `test_feature_engineering.py` (25), `test_notebook_utils.py` (24)
- Coverage mínimo atualizado de 40% para 70% (pyproject.toml + CI)

#### Docker & DevOps
- **Dockerfile**: Adicionado `libgomp1` para suporte a LightGBM/CatBoost em produção
- **docker-compose.yml**: Container `churn-api` com healthcheck e volumes
- **Makefile**: Targets para build, test, lint, api, dashboard
- **.gitignore**: Adicionados `models/`, `src/data/`, `catboost_info/`, `optuna_studies/`, `logs/`, `reports/*.png`

#### CI/CD
- 3 workflows GitHub Actions: CI, Code Quality, PR Analysis
- Coverage threshold atualizado para 70% no CI

### v1.4.0 (Março 2026)

#### Novas Funcionalidades
- **Métrica AUPRC**: Adicionada Área sob a Curva Precision-Recall em `pipeline.evaluate()`, `evaluate.py` e dashboard
- **Identificação do modelo ativo**: Cada página do dashboard exibe o nome do modelo selecionado
- **Comparação de Modelos**: Gráfico de barras AUPRC e spoke no radar multi-métrica
- **Dashboard API-first**: Padrão API-first com fallback local para predição, SHAP, feature importance e drift
- **Interpretabilidade model-agnostic**: SHAP funciona com qualquer modelo (LightGBM, CatBoost, XGBoost, LogisticRegression, etc.)
- **Data Drift no dashboard**: Aba dedicada com upload CSV e detecção via PSI
- **Seleção de modelo no sidebar**: Troca dinâmica de modelo pelo dashboard

#### Melhorias
- **Performance tab**: 6 métricas (F1, Precision, Recall, AUC-ROC, Accuracy, AUPRC) + tabela detalhada + CV
- **Home page**: Carrega métricas de `reports/metrics.json` com fallback para pacote do modelo
- **`src/utils.py`**: Normalização de chaves de métricas (AUPRC, average_precision, pr_auc)
- **`reports/metrics.json`**: Inclui AUPRC nas métricas de teste persistidas

### v1.3.0 (Março 2026)

#### Correções Críticas
- **`_cross_validate` (pipeline.py)**: Duas chamadas `cross_val_score` substituídas por `cross_validate` com multi-metric scoring
- **`get_feature_importance()` (model_service.py)**: Nomes pós-OHE (~45) em vez de features raw (19)
- **Endpoint SHAP**: `dict` genérico substituído por `ShapExplainRequest` (Pydantic)
- **Aliases removidos**: `CATEGORICAL_COLS`/`NUMERIC_COLS` removidos (usar `CATEGORICAL_FEATURES`/`NUMERIC_FEATURES`)
- 7 bugs críticos corrigidos (SMOTE leakage, ModelConfig, test-set snooping, etc.)

#### Testes
- **256 testes passando** (cobertura 70%+)
- Novos testes de integração (`test_integration.py`)
- Novos testes: `test_utils.py`, `test_feature_engineering.py`, `test_notebook_utils.py`

Veja a seção de Changelog abaixo para detalhes completos.

### v1.2.0 (Março 2026)

#### Melhorias de Código
- **Schema dinâmico**: `CustomerFeatures` gerado automaticamente do YAML via `create_model()`
- **Configuração centralizada**: Todos os valores categóricos e ranges definidos em `config/project.yaml`
- **Dashboard otimizado**: Usa `src.config` e `src.inference` centralizados

#### Notebooks
- **SMOTE Data Leakage corrigido**: SMOTE dentro de cada fold do CV via `ImbPipeline`
- **Threshold optimization**: `cross_val_predict` + `precision_recall_curve` por trial
- **Feature Selection**: Features com IV=0 removidas antes da otimização
- **Preprocessing alinhado**: `build_preprocessor` com imputação
- **Cache de estudos Optuna**: Persistência em `optuna_studies/`

#### Dependências
- **requirements.txt sincronizado**: Adicionados optuna, imbalanced-learn, catboost, matplotlib, seaborn, shap

#### Testes
- **82→256 testes passando**
- Novos testes para `optimization.py` (16 testes)

### v1.2.1 (Março 2026)

#### Qualidade de Código
- **PEP 8 compliance**: Ruff scan + auto-fix (import sorting, whitespace, unused imports)
- **Typing moderno**: Migração de `typing.Dict/List/Optional/Tuple/Union` para sintaxe nativa Python 3.12
- **Imports limpos**: Removidos imports não utilizados em 6 arquivos
- **Variáveis não utilizadas**: Removidas 3 atribuições sem uso
- **Testes**: 82→101 passando (v1.2.1)

### v1.1.0
- AdvancedFeatureEngineer com 67 features
- Ensemble XGBoost + LightGBM
- SMOTETomek para balanceamento
- Threshold optimization
- F1-Score: 87.9% (CV)

### v1.0.0
- Pipeline básico com XGBoost
- API FastAPI
- Dashboard Streamlit

---

## Referências

- [QUICKSTART.md](QUICKSTART.md) - Guia de início rápido e testes
- [README.md](README.md) - Visão geral do projeto
- [Adaptation Guide](ADAPTATION_GUIDE.md) - Guia de adaptação do modelo ao seu contexto
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*Última atualização: Março 2026*
