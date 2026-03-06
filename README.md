<div align="center">

# Customer Churn Prediction

### Pipeline Completo de Machine Learning: Do Notebook à Produção

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow.svg)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-4.7-blueviolet.svg)](https://optuna.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.22-0194E2.svg)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/tests-190%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-74%25-green.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Pipeline end-to-end de Data Science para predição de churn em telecomunicações**

[Início Rápido](#início-rápido) |
[Arquitetura](#arquitetura) |
[Skills Demonstrados](#skills-demonstrados) |
[API](#-api) |
[Documentação](DOCUMENTATION.md) |
[Documentação](DOCUMENTATION.md)

</div>

---

## Sobre o Projeto

Este projeto demonstra um **pipeline completo de Data Science e MLOps** para prever quais clientes têm maior probabilidade de cancelar seus serviços (churn). É um case real que abrange todo o ciclo de vida de um projeto de ML:

```
EDA → Feature Engineering → Modelagem → Otimização → Deploy → Monitoramento
```

### Impacto de Negócio

| Métrica de Negócio | Valor |
|-------------------|-------|
| **Base Real** | 7.043 clientes, churn rate 26,5% |
| **Clientes em Risco Identificados** | 52% dos churners (recall) |
| **Precisão na Identificação** | 65% (precision) |
| **Receita em Risco Detectada** | ~R$ 40k/mês* |

*Calculado sobre os verdadeiros positivos do modelo (ticket médio R$64,76/mês × ~625 clientes corretamente identificados)

### Métricas do Modelo (LightGBM)

| Métrica | Valor | Significado |
|---------|:-----:|-------------|
| **F1-Score** | **0.574** | Equilíbrio entre precisão e recall |
| **ROC-AUC** | 0.833 | Excelente capacidade discriminativa |
| **AUPRC** | 0.791 | Área sob a curva Precision-Recall |
| **Accuracy** | 0.797 | Acurácia geral do modelo |
| **Precision** | 0.648 | Dos preditos como churn, quantos realmente são |
| **Recall** | 0.516 | Dos churns reais, quantos foram identificados |

### Diferenciais Técnicos

| Feature | Descrição |
|---------|-----------|
| **44 Features Engenheiradas** | A partir de 19 features originais (63 total) |
| **Multi-modelo (LightGBM, CatBoost, XGBoost)** | Comparação e seleção do melhor modelo |
| **SMOTETomek** | Tratamento de desbalanceamento |
| **Threshold Optimization** | Otimização via Optuna para F1 máximo |
| **SHAP Interpretability** | Explicabilidade das predições |
| **Drift Detection** | Monitoramento em produção |

---

## Skills Demonstrados

Este projeto demonstra competências em todo o ciclo de Data Science:

### Data Engineering & Analysis
- **Análise Exploratória** (EDA) com Pandas, Matplotlib, Seaborn
- **ETL Pipeline** estruturado e reprodutível
- **Information Value (IV)** para seleção de features

### Machine Learning
- **Modelagem Avançada**: XGBoost, LightGBM, CatBoost, ensemble methods
- **Hyperparameter Tuning**: Optuna com pruning bayesiano
- **Imbalanced Learning**: SMOTE, ADASYN, SMOTETomek
- **Threshold Optimization**: maximização de F1-Score

### MLOps & Production
- **Tracking**: MLflow para experimentos
- **API REST**: FastAPI com validação Pydantic
- **Testing**: 190 testes pytest com 74% coverage
- **Containerização**: Docker multi-stage com docker-compose
- **CI/CD**: 3 workflows GitHub Actions (CI, Code Quality, PR Analysis)

### Software Engineering
- **Clean Architecture**: separação de responsabilidades
- **Design Patterns**: Factory, Strategy, Singleton
- **Docker**: Multi-stage build com 4 stages e docker-compose
- **Documentation**: docstrings, type hints, markdown
- **Code Quality**: Ruff, Black, pre-commit

### Data Visualization & Communication
- **Dashboard**: Streamlit interativo
- **Interpretabilidade**: SHAP values
- **Reporting**: apresentações automáticas

---

## Início Rápido

### Pré-requisitos

- Python 3.12+
- Poetry (gerenciador de dependências)
- Docker (opcional, para deploy containerizado)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/gabriela/churn_clientes.git
cd churn_clientes

# Instale as dependências
poetry install
```

### Docker (Produção)

```bash
# API + Dashboard (sobe os dois serviços juntos)
docker compose up --build

# Apenas a API
docker compose up api

# Apenas o Dashboard (depende da API)
docker compose up dashboard

# Com MLflow Tracking Server
docker compose --profile mlflow up

# API em modo desenvolvimento (hot reload)
docker compose --profile dev up api-dev
```

| Serviço | URL | Descrição |
|---------|-----|----------|
| API (Swagger) | http://localhost:8000/docs | Documentação interativa da API |
| Dashboard | http://localhost:8501 | Dashboard Streamlit interativo |
| MLflow | http://localhost:5000 | Tracking de experimentos (profile `mlflow`) |

### Treinar o Modelo

```bash
# Treino rápido com parâmetros default
poetry run python scripts/train_pipeline.py --quick

# Treino com otimização Optuna (recomendado)
poetry run python scripts/train_pipeline.py --mode optimized --trials 100

# Treino com ensemble XGBoost + LightGBM
poetry run python scripts/train_pipeline.py --mode ensemble --trials 200
```

### Fazer Predições

```bash
# Modo interativo
poetry run python scripts/run_pipeline.py

# Via API
poetry run uvicorn api.main:app --reload
# Acesse: http://localhost:8000/docs
```

### Dashboard Interativo

```bash
# Iniciar dashboard Streamlit
poetry run streamlit run scripts/dashboard.py
# Acesse: http://localhost:8501
```

### Análise SHAP

```bash
# Notebook de interpretabilidade
poetry run jupyter notebook notebooks/04_shap_interpretability.ipynb
```

---

## Notebooks do Projeto

O projeto inclui 4 notebooks completos e documentados, ideais para aprendizado:

| # | Notebook | Descrição | Conceitos |
|---|----------|-----------|-----------|
| 1 | [01_eda.ipynb](notebooks/01_eda.ipynb) | **Análise Exploratória** | Distribuições, correlações, missing values |
| 2 | [02_preprocess_feature_engineering.ipynb](notebooks/02_preprocess_feature_engineering.ipynb) | **Feature Engineering** | Encoding, scaling, criação de features |
| 3 | [03_modeling.ipynb](notebooks/03_modeling.ipynb) | **Modelagem** | XGBoost, LightGBM, Optuna, ensemble |
| 4 | [04_shap_interpretability.ipynb](notebooks/04_shap_interpretability.ipynb) | **Interpretabilidade** | SHAP values, feature importance |

### Ordem de Execução Recomendada

```bash
# Executar notebooks em ordem
poetry run jupyter notebook notebooks/
```

---

## Arquitetura

```
churn_clientes/
│
├── api/                          # API REST (FastAPI)
│   ├── main.py                   # Aplicação FastAPI
│   ├── routes/                   # Endpoints (predict, health, interpret, drift)
│   ├── schemas/                  # Modelos Pydantic
│   └── services/                 # Lógica de negócio (ModelService)
│
├── src/                          # Módulos principais
│   ├── config.py                 # Configurações centralizadas
│   ├── enums.py                  # Enumerações (ModelType, DriftSeverity)
│   ├── pipeline.py               # ChurnPipeline - pipeline completo
│   ├── optimization.py           # HyperparameterOptimizer - otimização F1
│   ├── preprocessing.py          # Transformações de dados
│   ├── feature_engineering.py    # FeatureEngineer e AdvancedFeatureEngineer
│   ├── evaluate.py               # Avaliação (F1, AUC-ROC, AUPRC, etc.)
│   ├── inference.py              # Lógica de predição
│   ├── interpret.py              # SHAP e Feature Importance (model-agnostic)
│   ├── utils.py                  # Utilitários centralizados
│   ├── monitoring.py             # Detecção de drift (PSI)
│   ├── notebook_utils.py         # Utilitários para notebooks
│   └── logger.py                 # Sistema de logging JSON
│
├── scripts/                      # Scripts de produção
│   ├── dashboard.py              # Dashboard Streamlit interativo
│   ├── train_pipeline.py         # Pipeline de treino (quick/optimized/ensemble)
│   ├── run_pipeline.py           # Pipeline de inferência
│   └── monitoring_pipeline.py    # Pipeline de monitoramento
│
├── notebooks/                    # Análise e experimentos
│   ├── 01_eda.ipynb              # Análise exploratória
│   ├── 02_preprocess_*.ipynb     # Feature engineering
│   ├── 03_modeling.ipynb         # Modelagem otimizada
│   └── 04_shap_interpretability.ipynb  # Interpretabilidade SHAP
│
├── tests/                        # Suite de testes (190 testes, 74% coverage)
├── models/                       # Modelos treinados (.joblib)
├── reports/                      # Relatórios (metrics.json, drift.json)
├── config/                       # Configurações YAML (project.yaml)
├── Dockerfile                    # Multi-stage build (builder → production → dashboard → dev)
├── docker-compose.yml            # Orquestração (api, dashboard, api-dev, mlflow)
└── .github/workflows/            # CI/CD (ci.yml, code-quality.yml, pr-analysis.yml)
```

### Fluxo de Dados

```
Dados CSV → Preprocessing → Feature Engineering → ChurnPipeline → Predição
                                                        ↓
                                               HyperparameterOptimizer
                                               (otimização F1)
```

### Consistência entre Componentes

O projeto usa as **mesmas classes** em todos os componentes:

| Componente | Classes Utilizadas |
|------------|-------------------|
| Notebooks | `ChurnPipeline`, `HyperparameterOptimizer` |
| Scripts CLI | `ChurnPipeline`, `evaluate()` |
| API REST | `ModelService` (usa `load_model_package`) |
| Dashboard | `load_model()`, `predict_via_api()`, `_load_metrics_from_report()` |

---

## API

### Endpoints Principais

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/predict/` | Predição individual |
| `POST` | `/predict/batch` | Predição em lote |
| `GET` | `/health` | Status da API |
| `GET` | `/health/live` | Liveness probe (Kubernetes) |
| `GET` | `/health/ready` | Readiness probe (Kubernetes) |
| `GET` | `/model/info` | Informações do modelo |
| `GET` | `/interpret/feature-importance` | Feature importance |
| `POST` | `/interpret/shap/explain` | Explicação SHAP individual |
| `GET` | `/interpret/shap/global` | Resumo SHAP global |
| `POST` | `/drift/detect` | Detecção de data drift (upload CSV) |
| `GET` | `/drift/status` | Status do monitoramento de drift |

### Exemplo de Uso

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/",
    json={
        "features": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 45.50,
            "TotalCharges": 546.00
        }
    }
)

print(response.json())
# {"churn": true, "probability": 0.68, "risk_level": "HIGH"}
```

---

## Testes

```bash
# Executar todos os testes (190 testes)
poetry run pytest

# Com relatório de cobertura (mínimo 70%)
poetry run pytest --cov=src --cov=api --cov-report=html

# Executar com Make
make test
```

**190 testes** (74% coverage) cobrindo:
- Endpoints da API (predict, health, interpret, drift)
- Feature Engineering (FeatureEngineer + AdvancedFeatureEngineer)
- Utilitários e Notebook Utils (IV, outliers, métricas)
- Preprocessamento e avaliação (incluindo AUPRC)
- Inferência e Otimização (Optuna)
- Monitoramento e Interpretabilidade (SHAP model-agnostic)
- Testes de integração end-to-end

> Veja o [Guia de Testes](QUICKSTART.md) para instruções detalhadas.

---

## Docker

O projeto usa **Docker multi-stage build** com 4 stages e **docker-compose** para orquestrar os serviços.

### Arquitetura dos Containers

```
┌─────────────────────────────────────────────────────────────┐
│                    docker-compose                           │
├─────────────────┬──────────────────┬────────────────────────┤
│   api (8000)    │ dashboard (8501) │  mlflow (5000)         │
│   FastAPI +     │  Streamlit       │  MLflow Tracking       │
│   Uvicorn       │  (depende da     │  (profile: mlflow)     │
│                 │   API)           │                        │
├─────────────────┴──────────────────┴────────────────────────┤
│                 Rede interna Docker                         │
│          dashboard → http://api:8000 (interno)              │
└─────────────────────────────────────────────────────────────┘
```

### Dockerfile Stages

| Stage | Base | Porta | Descrição |
|-------|------|:-----:|----------|
| `builder` | python:3.12-slim | — | Instala Poetry e dependências |
| `production` | python:3.12-slim | 8000 | API FastAPI com Uvicorn |
| `dashboard` | production | 8501 | Dashboard Streamlit (headless) |
| `development` | production | 8000 | API + ferramentas de dev (pytest, ruff) |

### Serviços docker-compose

| Serviço | Container | Porta | Profile | Descrição |
|---------|-----------|:-----:|---------|----------|
| `api` | churn-api | 8000 | — | API REST (produção) com healthcheck |
| `dashboard` | churn-dashboard | 8501 | — | Dashboard Streamlit, depende da API |
| `api-dev` | churn-api-dev | 8000 | `dev` | API com hot reload e volumes montados |
| `mlflow` | mlflow-server | 5000 | `mlflow` | MLflow Tracking Server com SQLite |

### Comandos Docker

```bash
# Subir API + Dashboard
docker compose up --build

# Subir em background
docker compose up -d

# Subir com MLflow
docker compose --profile mlflow up -d

# Ver logs
docker compose logs -f
docker compose logs -f dashboard

# Parar tudo
docker compose down

# Rebuild após alterações
docker compose up --build --force-recreate

# Build manual de um stage específico
docker build --target production -t churn-api .
docker build --target dashboard -t churn-dashboard .
```

### Variáveis de Ambiente (Docker)

| Variável | Serviço | Default | Descrição |
|----------|---------|---------|----------|
| `HOST` | api | `0.0.0.0` | Host de bind |
| `PORT` | api | `8000` | Porta da API |
| `DEBUG` | api | `false` | Modo debug |
| `LOG_LEVEL` | api | `INFO` | Nível de log |
| `MODEL_PATH` | api | `/app/models/model.joblib` | Caminho do modelo |
| `API_HOST` | dashboard | `api` | Host da API (rede interna Docker) |
| `API_PORT` | dashboard | `8000` | Porta da API |

---

## Customização para Novos Projetos

Este projeto foi estruturado como **template** para novos projetos de ML:

### 1. Configuração Centralizada

Todas as configurações estão em `src/config.py`:

```python
from src.config import get_config, ModelConfig

# Carregar configurações
config = get_config()

# Configuração do modelo
model_config = ModelConfig(
    model_type=ModelType.LIGHTGBM,
    n_estimators=500,
    learning_rate=0.01
)
```

### 2. Estrutura Modular

| Módulo | Responsabilidade |
|--------|-----------------|
| `src/config.py` | Configurações centralizadas |
| `src/preprocessing.py` | Transformação de dados |
| `src/feature_engineering.py` | Criação de features |
| `src/pipeline.py` | Pipeline de treinamento (ChurnPipeline) |
| `src/evaluate.py` | Avaliação e métricas (F1, AUC-ROC, AUPRC, Accuracy) |
| `src/inference.py` | Predições em produção |
| `src/monitoring.py` | Detecção de drift |

### 3. Comandos Make

```bash
make help       # Ver todos os comandos
make dev        # Instalar dependências
make test       # Rodar testes
make api        # Iniciar API
make dashboard  # Iniciar dashboard (scripts/dashboard.py)
make train      # Treinar modelo
```

---

## Contributing

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

### Code Style

```bash
# Formatar código
make format

# Verificar linting
make lint

# Verificar tipos
make type-check
```

---

## Documentação

| Documento | Descrição |
|-----------|-----------|
| **[QUICKSTART.md](QUICKSTART.md)** | Guia passo a passo de instalação, testes e execução |
| **[DOCUMENTATION.md](DOCUMENTATION.md)** | Documentação técnica completa |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Referência rápida para adaptação |
| **[API Swagger](http://localhost:8000/docs)** | Documentação interativa da API |
| **[Dashboard](http://localhost:8501)** | Dashboard Streamlit (via Docker ou local) |

---

## Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

---

<div align="center">

**[Documentação Completa](DOCUMENTATION.md)** | **[Notebooks](notebooks/)** | **[API Docs](http://localhost:8000/docs)**

</div>
