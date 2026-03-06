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

### Métricas Alcançadas
- **F1-Score**: 57.4%
- **AUC-ROC**: 83.3%
- **AUPRC**: 79.1%
- **Accuracy**: 79.7%
- **Precision**: 64.8%
- **Recall**: 51.6%
- **Testes**: 190 passando (cobertura 74%)

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
│   ├── train_pipeline.py         # Pipeline de treinamento (quick/optimized/ensemble)
│   ├── run_pipeline.py           # Pipeline de inferência
│   └── monitoring_pipeline.py    # Pipeline de monitoramento
│
├── api/                          # API REST (FastAPI)
│   ├── main.py                   # Aplicação FastAPI
│   ├── core/                     # Configuração e logging
│   │   ├── config.py             # Settings Pydantic
│   │   └── logging.py            # Logging estruturado
│   ├── routes/                   # Endpoints (predict, health, interpret, drift)
│   ├── schemas/                  # Modelos Pydantic (gerados dinamicamente do YAML)
│   └── services/                 # ModelService singleton
│
├── scripts/                      # Scripts executáveis
│   ├── dashboard.py              # Dashboard Streamlit interativo
│   ├── train_pipeline.py         # Pipeline de treinamento (quick/optimized/ensemble)
│   ├── run_pipeline.py           # Pipeline de inferência
│   └── monitoring_pipeline.py    # Pipeline de monitoramento
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

# Otimização rápida
result = quick_optimize(X_train, y_train, n_trials=50)

# Otimização focada em F1
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
│  (CSV/Parquet)  │    │  (63 features)   │    │  (Scale/Encode)  │
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
   - **Total: 63 features (44 novas + 19 originais)**

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
# Treinamento básico (modo optimized por default)
poetry run python scripts/train_pipeline.py

# Com opções
poetry run python scripts/train_pipeline.py \
    --mode optimized \
    --trials 50
```

### 2. Treinamento Rápido

```bash
# Treino rápido com parâmetros default (~2 min)
poetry run python scripts/train_pipeline.py --quick
```

### 3. Treinamento com Ensemble

```bash
# Ensemble XGBoost + LightGBM com otimização Optuna
poetry run python scripts/train_pipeline.py \
    --mode ensemble \
    --trials 200
```

**Parâmetros:**
- `--mode`: Modo de treinamento (`quick`, `optimized`, `ensemble`; default: `optimized`)
- `--quick`: Atalho para `--mode quick`
- `--trials`: Número de trials Optuna (default: 100)
- `--no-smote`: Desabilitar SMOTE

### Inferir via CLI

```bash
# Modo interativo
poetry run python scripts/run_pipeline.py
```

### 4. Apenas Feature Engineering

```python
from src.feature_engineering import AdvancedFeatureEngineer
import pandas as pd

df = pd.read_csv("src/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

fe = AdvancedFeatureEngineer()
X_transformed = fe.fit_transform(df.drop("Churn", axis=1), df["Churn"])
print(f"Features: {X_transformed.shape[1]}")  # 63 features
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
| has_streaming | StreamingTV + StreamingMovies | Tem serviço de streaming |
| has_support | TechSupport + OnlineSecurity | Tem suporte/segurança |
| avg_monthly_charge | TotalCharges / tenure | Cobrança média mensal |
| charge_percentile | Quartil de MonthlyCharges | Faixa de cobrança |
| is_senior_alone | SeniorCitizen & !Partner | Idoso sozinho |
| tenure_contract_ratio | tenure / avg_tenure_contract | Razão tenure vs média |
| contract_ordinal | 0, 1, 2 | Contrato ordenado |
| contract_risk | Taxa churn por contrato | Risco por contrato |
| is_new_customer | tenure < 12 | Cliente novo |
| payment_risk | Taxa por método pagamento | Risco por pagamento |
| is_high_value | MonthlyCharges > mediana | Cliente alto valor |
| no_services | Sem serviços adicionais | Sem extras |
| full_protection | Todos serviços proteção | Proteção completa |
| tenure_charge_interaction | tenure * MonthlyCharges | Interação |
| high_charge_new_customer | Alta cobrança + novo | Flag de risco |
| fiber_no_protection | Fibra sem proteção | Flag de risco |
| tenure_years | tenure / 12 | Tenure em anos |
| tenure_sqrt | √tenure | Sqrt tenure |
| tenure_log | log(tenure + 1) | Log tenure |

### Features Avançadas (AdvancedFeatureEngineer) [29]
| Categoria | Features | Descrição |
|-----------|----------|-----------|
| **Serviços** | total_services, streaming_services, security_services | Contagem de serviços |
| **Engajamento** | low_engagement, digital_only, family_account | Padrões de uso |
| **Interações** | fiber_no_security, senior_monthly, new_high_value | Combinações de risco |
| **Risk Scores** | churn_risk_score, high_risk_flag | Scores compostos |
| **Lifecycle** | lifecycle_phase (new/growing/mature/loyal) | Fase do ciclo |
| **Valor** | estimated_clv, cost_per_service | Valor do cliente |

---

## Modelos e Otimização

### Modelos Disponíveis

#### XGBoost (Padrão)
```python
# Parâmetros otimizados
{
    'n_estimators': [100, 600],
    'max_depth': [3, 12],
    'learning_rate': [0.005, 0.3],
    'subsample': [0.5, 1.0],
    'colsample_bytree': [0.5, 1.0],
    'min_child_weight': [1, 20],
    'gamma': [0.0, 2.0],
    'reg_alpha': [0.0, 3.0],
    'reg_lambda': [0.0, 3.0],
    'scale_pos_weight': [1.0, 10.0]
}
```

#### LightGBM
```python
{
    'n_estimators': [100, 600],
    'max_depth': [3, 15],
    'learning_rate': [0.005, 0.3],
    'num_leaves': [10, 200],
    'subsample': [0.5, 1.0],
    'colsample_bytree': [0.5, 1.0],
    'min_child_samples': [5, 100],
    'reg_alpha': [0.0, 3.0],
    'reg_lambda': [0.0, 3.0],
    'scale_pos_weight': [1.0, 10.0]
}
```

### Ensemble

```python
VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(**best_xgb_params)),
        ('lgbm', LGBMClassifier(**best_lgbm_params))
    ],
    voting='soft',
    weights=[xgb_f1_score, lgbm_f1_score]
)
```

### Otimização Optuna

```python
# Configuração padrão
OptimizationConfig(
    n_trials=100,
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
# MLflow
export MLFLOW_TRACKING_URI="sqlite:///mlruns/mlflow.db"

# Optuna (opcional - armazenamento em memória por padrão)
# export OPTUNA_STORAGE="sqlite:///optuna_study.db"
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
    # ...
}])
result = predict_with_package(package, customer_df)
print(f"Churn: {result['prediction']}, Prob: {result['probability']:.2%}")
```

### API de Predição (exemplo)

```python
# Usando o módulo de inferência
from src.inference import load_model_package, predict_with_package

package = load_model_package("models/model.joblib")
prediction = predict_with_package(package, customer_data)
print(f"Probabilidade de churn: {prediction['probability']:.2%}")
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

```json
{
    "prediction": 1,
    "probability": 0.78,
    "risk_level": "high",
    "threshold_used": 0.42,
    "features_used": 67
}
```

---

## Monitoramento

### Drift Detection

```python
from src.monitoring import detect_drift, population_stability_index

# Detectar drift entre dados de referência e novos dados
features = X_train.columns.tolist()
drift_report = detect_drift(X_train, X_new, features=features)

if drift_report.features_with_drift > 0:
    print("ALERTA: Drift detectado!")
    for result in drift_report.feature_results:
        if result.has_drift:
            print(f"  {result.feature_name}: PSI={result.psi:.4f} ({result.severity})")
```

### PSI (Population Stability Index)

```python
from src.monitoring import population_stability_index, classify_drift_severity

# Calcular PSI para uma feature
psi_value = population_stability_index(reference_values, current_values)
severity = classify_drift_severity(psi_value)
print(f"PSI: {psi_value:.4f} - Severity: {severity}")
```

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
    assert config.n_trials == 100
    assert config.use_smote is True

```

### Fixtures Disponíveis

```python
# conftest.py
@pytest.fixture
def sample_data():
    return pd.read_csv(DATA_PATH).head(100)

@pytest.fixture
def trained_pipeline(sample_data):
    pipeline = ChurnPipeline()
    pipeline.fit(X, y)
    return pipeline
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
# Usar treinamento otimizado
poetry run python scripts/train_pipeline.py --mode ensemble --trials 200
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

### v1.5.1 (Março 2026)

#### Docker & Dashboard
- **Dashboard no Docker**: Novo stage `dashboard` no Dockerfile (Streamlit headless na porta 8501)
- **docker-compose**: Novo serviço `dashboard` com `depends_on` API healthcheck
- **Comunicação entre containers**: Dashboard usa rede interna Docker (`API_HOST=api`)
- **Documentação atualizada**: README, DOCUMENTATION, QUICKSTART e QUICK_REFERENCE com instruções Docker completas

### v1.5.0 (Março 2026)

#### Qualidade & Testes
- **190 testes passando** (cobertura 74%, threshold 70%)
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
- **190 testes passando** (cobertura 74%)
- Novos testes de integração (`test_integration.py`)
- Novos testes: `test_utils.py`, `test_feature_engineering.py`, `test_notebook_utils.py`

Veja [CHANGELOG.md](CHANGELOG.md) para detalhes completos.

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
- **82→190 testes passando**
- Novos testes para `optimization.py` (16 testes)

### v1.2.1 (Março 2026)

#### Qualidade de Código
- **PEP 8 compliance**: Ruff scan + auto-fix (import sorting, whitespace, unused imports)
- **Typing moderno**: Migração de `typing.Dict/List/Optional/Tuple/Union` para sintaxe nativa Python 3.12
- **Imports limpos**: Removidos imports não utilizados em 6 arquivos
- **Variáveis não utilizadas**: Removidas 3 atribuições sem uso
- **Testes**: 82→101 passando (v1.2.1)

### v1.1.0
- AdvancedFeatureEngineer com 63 features
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
- [Guia de Adaptação](docs/ADAPTATION_GUIDE.md) - Como adaptar para outro problema de classificação
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*Última atualização: Março 2026*
