# Guia de Início Rápido e Testes

Este guia fornece instruções passo a passo para testar e executar todos os componentes do projeto.

---

## Índice

1. [Pré-requisitos](#1-pré-requisitos)
2. [Instalação](#2-instalação)
3. [Executar Testes](#3-executar-testes)
4. [Treinar Modelo](#4-treinar-modelo)
5. [Executar API](#5-executar-api)
6. [Executar Dashboard](#6-executar-dashboard)
7. [Fazer Predições](#7-fazer-predições)
8. [Notebooks](#8-notebooks)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Pré-requisitos

### Software Necessário

| Software | Versão | Comando de verificação |
|----------|--------|------------------------|
| Python | 3.12+ | `python --version` |
| Poetry | 1.5+ | `poetry --version` |
| Git | 2.0+ | `git --version` |
| Docker | 24+ (opcional) | `docker --version` |
| Docker Compose | v2+ (opcional) | `docker compose version` |

### Instalar Poetry (se necessário)

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Verificar instalação
poetry --version
```

---

## 2. Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/gabriela/churn_clientes.git
cd churn_clientes

# 2. Instale as dependências (cria ambiente virtual automaticamente)
poetry install

# 3. Ative o ambiente virtual
poetry shell

# 4. Verifique se tudo está OK
poetry run python -c "import src; print('Instalação OK')"
```

### Verificar Dependências

```bash
# Listar pacotes instalados
poetry show

# Verificar dependências críticas
poetry run python -c "
import xgboost, lightgbm, optuna, mlflow, fastapi, streamlit
print('Todas as dependências estão instaladas')
"
```

---

## 3. Executar Testes

### 3.1 Testes Completos (190 testes)

```bash
# Executar todos os testes
poetry run pytest tests/ -v

# Saída esperada:
# ======================= 190 passed in ~30s =======================
```

### 3.2 Testes por Módulo

```bash
# Testes da API
poetry run pytest tests/test_api.py -v

# Testes de preprocessing
poetry run pytest tests/test_preprocessing.py -v

# Testes de inferência
poetry run pytest tests/test_inference.py -v

# Testes de otimização
poetry run pytest tests/test_optimization.py -v

# Testes de monitoramento
poetry run pytest tests/test_monitoring.py -v
```

### 3.3 Testes com Cobertura

```bash
# Gerar relatório de cobertura (mínimo 70%)
poetry run pytest tests/ --cov=src --cov=api --cov-report=html

# Abrir relatório (Linux)
xdg-open htmlcov/index.html

# Abrir relatório (macOS)
open htmlcov/index.html
```

### 3.4 Makefile

```bash
# Alternativa: usar Makefile
make test           # Rodar testes
make lint           # Verificar estilo de código
make format         # Formatar código
```

---

## 4. Treinar Modelo

### 4.1 Treino Padrão

```bash
# Treino com LightGBM e parâmetros do config/project.yaml (~2 min)
poetry run python scripts/train_pipeline.py
```

### 4.2 Pipeline Completo (Treino + Avaliação + Salvar)

```bash
# Treina, avalia, salva modelo e métricas (reports/metrics.json)
poetry run python scripts/run_pipeline.py
```

### 4.3 Otimização Multi-Modelo

```bash
# Otimização rápida (~5 min): 10 trials × 3 modelos, 3-fold CV
poetry run python scripts/optimize_model.py --quick

# Otimização completa (~20 min): 20 trials × 3 modelos, 5-fold CV
poetry run python scripts/optimize_model.py

# Ou via Makefile:
make train-quick      # Rápido
make train-optimized  # Completo
```

Compara LightGBM, XGBoost e CatBoost. O melhor modelo é salvo como `model.joblib`.

### 4.4 Verificar Modelo Treinado

```bash
# Listar modelos salvos
ls -la models/

# Saída esperada:
# model.joblib                    # Modelo principal
```

---

## 5. Executar API

### 5.1 Iniciar API

```bash
# Modo desenvolvimento (com hot-reload)
poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Modo produção
poetry run uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### 5.2 Testar Health Check

```bash
# Verificar se API está rodando
curl http://localhost:8000/health

# Saída esperada:
# {"status":"healthy","model_loaded":true,"model_version":"1.3.0",...}
```

### 5.3 Testar Predição

```bash
# Predição individual
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "test-001",
    "features": {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "MultipleLines": "No phone service",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 70.70,
      "TotalCharges": 848.40
    }
  }'

# Saída esperada:
# {"success":true,"prediction":{"customer_id":"test-001","churn_prediction":1,"churn_probability":0.84,"churn_risk":"HIGH"},...}
```

### 5.4 Gerenciamento de Modelos via API

```bash
# Listar modelos disponíveis
curl http://localhost:8000/models/

# Saída esperada:
# {"models":[{"name":"model","filename":"model.joblib","size_mb":1.23,"is_default":true},...],"active_model":"CatBoostClassifier (ChurnPipeline)","total":5}

# Trocar o modelo ativo
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "model_xgboost"}'

# Saída esperada:
# {"success":true,"active_model":"XGBClassifier","message":"Modelo trocado para 'model_xgboost' com sucesso."}
```

> **Nota:** O Dashboard detecta automaticamente se a API está online. Quando online, a seleção de modelo no sidebar troca o modelo diretamente na API, garantindo consistência entre Dashboard e API.

### 5.5 Documentação da API (Swagger)

Acesse: **http://localhost:8000/docs**

A documentação interativa permite testar todos os endpoints diretamente no navegador.

---

## 6. Executar Dashboard

### 6.1 Iniciar Dashboard Streamlit

```bash
# Local
poetry run streamlit run scripts/dashboard.py --server.port 8501

# Via Docker (sobe API + Dashboard juntos)
docker compose up dashboard

# Via Makefile
make dashboard
```

Acesse: **http://localhost:8501**

### 6.2 Funcionalidades do Dashboard

- **Home**: Visão geral do projeto e métricas do modelo ativo
- **Predição**: Interface para fazer predições (API-first com fallback local)
- **Performance**: Métricas detalhadas (F1, AUC-ROC, AUPRC, Accuracy, Precision, Recall) + Validação Cruzada
- **Comparação de Modelos**: Compara todos os modelos disponíveis com tabela, gráficos de barras e radar
- **Interpretabilidade**: SHAP analysis (model-agnostic) e Feature Importance
- **Data Drift**: Monitoramento de drift com PSI (upload CSV)
- **MLflow**: Tracking de experimentos

> **Nota:** O nome do modelo ativo é exibido em cada página do dashboard. Troque o modelo pela sidebar. Quando a API está online, a troca de modelo é sincronizada automaticamente via `POST /models/switch`.

---

## 6.5 Executar com Docker

O Docker é a forma mais simples de executar todos os serviços em produção.

### Subir Todos os Serviços

```bash
# API (porta 8000) + Dashboard (porta 8501)
docker compose up --build

# Em background
docker compose up -d
```

### Subir Serviços Individuais

```bash
# Apenas a API
docker compose up api

# Apenas o Dashboard (inicia a API automaticamente)
docker compose up dashboard

# API em modo desenvolvimento (hot reload)
docker compose --profile dev up api-dev

# Com MLflow Tracking Server
docker compose --profile mlflow up
```

### Acessar Serviços

| Serviço | URL |
|---------|-----|
| API (Swagger) | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |

### Gerenciar Containers

```bash
# Ver status dos containers
docker ps

# Ver logs em tempo real
docker compose logs -f
docker compose logs -f dashboard

# Parar tudo
docker compose down

# Rebuild após alterações no código
docker compose up --build --force-recreate
```

### Build Manual de Imagens

```bash
# Build da API
docker build --target production -t churn-api .

# Build do Dashboard
docker build --target dashboard -t churn-dashboard .

# Executar manualmente
docker run -p 8000:8000 -v ./models:/app/models:ro churn-api
docker run -p 8501:8501 -v ./models:/app/models:ro churn-dashboard
```

---

## 7. Fazer Predições

### 7.1 Via Pipeline (Treino + Avaliação)

```bash
# Executar pipeline completo (treino + avaliação + salvar modelo e métricas)
poetry run python scripts/run_pipeline.py
```

### 7.2 Via Python (Inferência)

```python
from src.inference import load_model_package, predict_with_package
import pandas as pd

# Carregar modelo
package = load_model_package()

# Criar dados de um cliente
customer = pd.DataFrame([{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 140.0
}])

# Fazer predição
result = predict_with_package(package, customer)
print(result)
```

### 7.3 Via API REST

```bash
# Iniciar API
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000

# Fazer predição (em outro terminal)
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 140.0}'
```

### 7.4 Via Dashboard

```bash
poetry run streamlit run scripts/dashboard.py
# Acesse: http://localhost:8501 → Página "Predição"
```

---

## 8. Notebooks

### 8.1 Abrir Jupyter

```bash
# Iniciar Jupyter Lab
poetry run jupyter lab

# Ou Jupyter Notebook
poetry run jupyter notebook
```

### 8.2 Notebooks Disponíveis

| Notebook | Descrição |
|----------|-----------|
| `01_eda.ipynb` | Análise Exploratória de Dados |
| `02_preprocess_feature_engineering.ipynb` | Feature Engineering |
| `03_modeling.ipynb` | Treinamento e Otimização |
| `04_shap_interpretability.ipynb` | Interpretabilidade SHAP |

### 8.3 Executar Notebook via CLI

```bash
# Executar notebook e gerar HTML
poetry run jupyter nbconvert --execute notebooks/03_modeling.ipynb --to html
```

---

## 9. Troubleshooting

### Erro: "Model not found"

```bash
# Solução: Treinar modelo primeiro
poetry run python scripts/train_pipeline.py
```

### Erro: "Port already in use"

```bash
# Solução: Liberar porta
pkill -f "uvicorn api.main:app"
# Ou usar outra porta
poetry run uvicorn api.main:app --port 8001
```

### Erro: "Package not found"

```bash
# Solução: Reinstalar dependências
poetry install --no-root
```

### Erro de Import

```bash
# Solução: Garantir que está no diretório correto
cd /path/to/churn_clientes
poetry shell
```

### Verificar Logs

```bash
# Ver logs do pipeline
tail -f logs/pipeline.log

# Ver logs do MLflow
ls -la mlruns/
```

---

## Checklist de Validação Completa

Execute este checklist para validar que tudo está funcionando:

```bash
# 1. Testes unitários
poetry run pytest tests/ -v
# Esperado: 190 passed

# 2. API - Health
curl http://localhost:8000/health 2>/dev/null | grep -q "healthy" && echo "API OK" || echo "API FAIL"

# 3. Scripts de treino
poetry run python -c "from scripts.train_pipeline import main; print('Train Pipeline OK')"

# 4. Dashboard imports
poetry run python -c "import scripts.dashboard; print('Dashboard OK')"

# 5. Core modules
poetry run python -c "
from src.pipeline import ChurnPipeline
from src.optimization import HyperparameterOptimizer
from src.inference import load_model_package
from src.enums import ModelType, DriftSeverity
print('Core Modules OK')
"
```

---

## Recursos Adicionais

- [DOCUMENTATION.md](DOCUMENTATION.md) - Documentação técnica completa
- [README.md](README.md) - Visão geral do projeto
- [API Swagger](http://localhost:8000/docs) - Documentação interativa da API

---

*Última atualização: Março 2026*
