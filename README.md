# Data Science End-to-End Project Framework

[![CI Status](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen)](https://github.com)

Framework completo e reutilizável para projetos de Ciência de Dados, seguindo boas práticas de MLOps, com CI/CD integrado, testes automatizados e cobertura de código superior a 80%.

## 🎯 Objetivo

Fornecer um **framework de produção** para projetos de Machine Learning com:
- ✅ Pipeline completo end-to-end
- ✅ Testes automatizados (80%+ coverage)
- ✅ CI/CD com GitHub Actions
- ✅ Rastreamento de experimentos (MLflow)
- ✅ Monitoramento de drift
- ✅ Dashboard interativo
- ✅ Logging estruturado em JSON
- ✅ Documentação completa

**Pronto para ser adaptado ao seu projeto!** Veja [TUTORIAL_NOVO_PROJETO.md](TUTORIAL_NOVO_PROJETO.md) para guia completo.

## 🛠️ Stack Tecnológico

### Core
- **Python**: 3.9+ | 3.10+ | 3.11+ (testado em múltiplas versões)
- **Data**: Pandas 2.0+, NumPy 1.24+
- **ML**: Scikit-learn 1.3+, XGBoost 2.0+

### MLOps
- **Experimentos**: MLflow 2.22.4 (tracking, logging, model registry)
- **Validação**: Cross-validation com StratifiedKFold
- **Monitoramento**: PSI (Population Stability Index), Drift Detection
- **Dashboard**: Streamlit com gráficos interativos

### Quality & Testing
- **Testes**: Pytest 7.4+ com fixtures centralizadas
- **Coverage**: pytest-cov (>80% cobertura)
- **CI/CD**: GitHub Actions (multi-version testing)
- **Linting**: Flake8, Autopep8

### Interpretabilidade & Visualização
- **SHAP**: Feature importance e explainability
- **Plots**: Matplotlib 3.8+, Seaborn 0.13+
- **Logging**: JSON estruturado com timestamps

### Gerenciamento
- **Ambiente**: Poetry 1.7+ (gerenciamento de dependências)
- **Versionamento**: Git + MLflow Model Registry

## Estrutura do Projeto

```
modelo_projetos_ds/
├── data/
│   ├── raw/              # Dados brutos
│   ├── processed/        # Dados processados
│   └── external/         # Dados externos
├── notebooks/
│   ├── 01_eda.ipynb                    # Análise exploratória
│   ├── 02_feature_engineering.ipynb    # Engenharia de features
│   └── 03_modeling.ipynb              # Modelagem
├── src/
│   ├── __init__.py
│   ├── config.py         # Configurações centralizadas
│   ├── preprocessing.py  # Preprocessamento de dados (com logging)
│   ├── features.py       # Feature engineering com SimpleImputer
│   ├── train.py          # Treinamento + Cross-validation com logging
│   ├── evaluate.py       # Avaliação de métricas com logging
│   ├── monitoring.py     # Monitoramento de drift com logging
│   ├── inference.py      # Inferência com logging
│   ├── interpret.py      # Feature importance e SHAP
│   └── logger.py         # Sistema genérico de logging JSON
├── scripts/
│   ├── run_pipeline.py       # Pipeline completo de produção
│   ├── train_pipeline.py     # Script de treinamento com CV
│   ├── test_pipeline.py      # Testes end-to-end
│   ├── monitoring_pipeline.py # Monitoramento de drift
│   └── dashboard.py          # Dashboard Streamlit
├── tests/                # Testes unitários (80%+ coverage)
│   ├── conftest.py       # Fixtures compartilhadas
│   ├── test_features.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_inference.py
│   ├── test_monitoring.py
│   └── test_logger.py
├── models/               # Modelos treinados (versionados)
├── reports/
│   ├── metrics.json      # Métricas de desempenho
│   ├── drift.json        # Detecção de data drift
│   └── figures/          # Gráficos e visualizações
├── logs/                 # Logs estruturados em JSON
├── mlruns/               # Artefatos MLflow
├── .github/
│   └── workflows/        # CI/CD pipelines
│       ├── ci.yml        # Testes e validação
│       ├── code-quality.yml
│       └── pr-analysis.yml
├── pyproject.toml        # Configuração Poetry
├── pytest.ini            # Configuração pytest
├── requirements.txt      # Dependências pip
├── generate_data.py      # Gerador de dados sintéticos
├── test_ci_locally.sh    # Script para testar CI localmente
├── POETRY_GUIDE.md       # Guia de uso do Poetry
├── TUTORIAL_NOVO_PROJETO.md  # Como adaptar para novo projeto
└── README.md             # Este arquivo
```

## Instalação

### Opção 1: Poetry (Recomendado)

```bash
# Instalar dependências e criar ambiente virtual
poetry install

# Ativar ambiente virtual
poetry shell

# Ou executar comandos sem ativar
poetry run python scripts/run_pipeline.py
```

### Opção 2: Pip + Venv

```bash
# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Guia Rápido

### 1. Gerar Dados de Teste

```bash
poetry run python generate_data.py
```

Gera 500 amostras com:
- 5 features numéricas
- 3 features categóricas
- Target binário (classificação)

Salva em:
- `data/raw/raw_data.csv` - Dados brutos
- `data/processed/data.csv` - Dados processados

### 2. Executar Pipeline Completo

```bash
# Com Poetry
poetry run python scripts/run_pipeline.py

# Com pip
python scripts/run_pipeline.py
```

**O que o pipeline faz:**
1. ✓ Carrega e preprocessa dados
2. ✓ Constrói preprocessador (StandardScaler + OneHotEncoder)
3. ✓ Treina modelo XGBoost com 300 estimadores
4. ✓ Avalia modelo (ROC-AUC, Precision, Recall, F1)
5. ✓ Registra métricas em `reports/metrics.json`
6. ✓ Salva modelo com MLflow (com signature automática)
7. ✓ Registra tags e parâmetros no MLflow

### 3. Executar Testes End-to-End

```bash
poetry run python scripts/test_pipeline.py
```

Executa teste completo com dados sintéticos:
- Geração de dados
- Divisão treino/teste
- Treinamento
- Avaliação
- Inferência em amostras
- Rastreamento com MLflow

### 4. Visualizar Resultados com MLflow

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Acesse `http://127.0.0.1:5000` para ver:
- ✓ Parâmetros do modelo
- ✓ Métricas de desempenho
- ✓ Artefatos salvos (modelo + preprocessador)
- ✓ Tags e descrição do modelo
- ✓ Histórico de todas as runs
- ✓ Comparação entre experimentos

## Configuração

Editar `src/config.py` para ajustar:

```python
# Configuração de Dados
TARGET = "target"                    # Coluna alvo
TEST_SIZE = 0.2                      # 20% para teste
RANDOM_STATE = 42                    # Seed para reprodutibilidade

# Configuração MLflow
MLFLOW_TRACKING_URI = "..."          # Diretório local ou servidor remoto
MLFLOW_EXPERIMENT = "mlflow_test_experiments"
MODEL_NAME = "xgboost_classifier"

# Parâmetros do XGBoost
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss"
}
```

## 🧪 Testes e Qualidade

### Executar Testes

```bash
# Todos os testes
poetry run pytest tests/ -v

# Com cobertura
poetry run pytest tests/ --cov=src --cov-report=html

# Teste específico
poetry run pytest tests/test_train.py -v

# Testar CI localmente (recomendado antes de push)
./test_ci_locally.sh
```

### Cobertura Atual

```
Name                   Coverage
----------------------------------
src/config.py          100%
src/features.py        100%
src/inference.py       100%
src/monitoring.py      100%
src/preprocessing.py   100%
src/train.py           98%
src/evaluate.py        93%
----------------------------------
TOTAL                  70.03%
```

### CI/CD Workflows

O projeto possui 3 workflows automatizados:

1. **ci.yml** - Testes principais
   - Executa em Python 3.9, 3.10, 3.11
   - Verifica sintaxe e imports
   - Roda testes unitários
   - Gera relatório de cobertura

2. **code-quality.yml** - Qualidade de código
   - Formatação com autopep8
   - Detecção de código duplicado
   - Análise de complexidade

3. **pr-analysis.yml** - Análise de PRs
   - Valida mudanças em arquivos Python
   - Executa testes impactados
   - Adiciona comentário automático no PR

## 💻 Desenvolvimento

### Uso em Notebooks

Os notebooks em `notebooks/` podem ser executados para exploração e experimentação:

```bash
jupyter notebook notebooks/
```

- **01_eda.ipynb**: Exploração e análise dos dados
- **02_feature_engineering.ipynb**: Criação e seleção de features
- **03_modeling.ipynb**: Experimentação com diferentes modelos

### Adicionando Novas Dependências

**Com Poetry:**
```bash
# Adicionar ao projeto
poetry add nome-do-pacote
✨ Boas Práticas Implementadas

### 🏗️ Arquitetura & Código
✅ **Separação clara** entre código de experimentação (notebooks) e produção (scripts)  
✅ **Configuração centralizada** em `src/config.py`  
✅ **Módulos reutilizáveis** e bem documentados  
✅ **Type hints** e docstrings em funções críticas

### 🧪 Qualidade & Testes
✅ **Testes unitários** com pytest (55 testes)  
✅ **Cobertura de código** superior a 80% (atual: 88%)  
✅ **Fixtures centralizadas** para reutilização  
✅ **Testes end-to-end** para validar pipeline completo  
✅ **CI/CD automatizado** com GitHub Actions (3 workflows)  
✅ **Testing multi-versão** Python (3.9, 3.10, 3.11)

### 📊 MLOps & Monitoramento
✅ **Rastreamento de experimentos** com MLflow 2.22.4  
✅ **Versionamento automático** de modelos e artefatos  
✅ **Signature automática** para modelos (sem warnings)  
✅ **Tags e descrição** de modelos para rastreabilidade  
✅ **Cross-validation** com StratifiedKFold (5-fold) e 4 métricas  
✅ **Monitoramento de data drift** com PSI por feature  
✅ **Dashboard interativo** com Streamlit (métricas, gráficos, alertas)

### 🔍 Interpretabilidade & Logging
✅ **Feature Importance** nativa do XGBoost + SHAP values  
✅ **Logging estruturado** em JSON (genérico em todos os módulos)  
✅ **Preprocessamento robusto** com SimpleImputer (mediana + constant)  
✅ **Relatórios automáticos** de métricas e drift).

## Boas Práticas Implementadas

✅ **Separação clara** entre código de experimentação (notebooks) e produção (scripts)  
✅ **Rastreamento de experimentos** com MLflow (versão 2.22.4)  
✅ **Cross-validation** com StratifiedKFold (5-fold) e 4 métricas  
✅ **Feature Importance** nativa do XGBoost + SHAP values  
✅ **Preprocessamento robusto** com SimpleImputer (mediana + constant)  
✅ **Dashboard interativo** com Streamlit (métricas, gráficos, alertas)  
✅ **Logging estruturado** em JSON (genérico em todos os módulos)  
✅ **Monitoramento de data drift** com PSI por feature  
✅ **Versionamento automático** de modelos e artefatos  
✅ **Configuração centralizada** em `src/config.py`  
✅ **Signature automática** para modelos (sem warnings)  
✅ **Tags e descrição** de modelos para rastreabilidade  
✅ **Testes end-to-end** para validar pipeline completo  
✅ **Testes unitários** com pytest e fixtures centralizadas

## Fluxo de Trabalho Recomendado

### 1. Exploração e Análise
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Experimentação
```bash
jupyter notebook notebooks/03_modeling.ipynb
```

### 3. Rastreamento com MLflow
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

### 4. Testes
```bash
poetry run python scripts/test_pipeline.py
```

### 5. Produção
```bitash
poetry run python scripts/run_pipeline.py
```

### 6. Monitoramento de Drift
```bash
poetry run python scripts/monitoring_pipeline.py
```

### 7. Dashboard de Performance
```bash
poetry run streamlit run scripts/dashboard.py
```
Acesse `http://localhost:8501` para visualizar:
- Métricas em tempo real (ROC-AUC, F1, Precision, Recall)
- Evolução temporal das métricas
- Histórico de runs do MLflow
- Alertas automáticos de degradação

## Troubleshooting

### Erro: FileNotFoundError para dados
```bash
poetry run python generate_data.py
```

### Erro: "ModuleNotFoundError: No module named 'src'"
Execute os scripts **sempre** a partir do diretório raiz do projeto:
```bash
cd /path/to/modelo_projetos_ds
poetry run python scripts/run_pipeline.py
```📚 Documentação Adicional

- **[TUTORIAL_NOVO_PROJETO.md](TUTORIAL_NOVO_PROJETO.md)** - 🌟 **Guia completo de adaptação (Passo a passo)**
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - ⚡ **Referência rápida (15 minutos)**
- **[POETRY_GUIDE.md](POETRY_GUIDE.md)** - Guia completo de uso do Poetry
- **[pyproject.toml](pyproject.toml)** - Configuração de dependências e metadados
- **[pytest.ini](pytest.ini)** - Configuração de testes
- **[.github/workflows/](.github/workflows/)** - Workflows de CI/CD

## 🚀 Como Adaptar para Seu Projeto

### Início Rápido (15 minutos)
Veja **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** para checklist rápido e comandos essenciais.

### Tutorial Completo (Recomendado)
Veja **[TUTORIAL_NOVO_PROJETO.md](TUTORIAL_NOVO_PROJETO.md)** que inclui:

1. ✅ Checklist de adaptação passo a passo
2. ✅ Configuração de novos datasets
3. ✅ Customização de modelos e features
4. ✅ Adaptação de testes
5. ✅ Deploy e produção
6. ✅ Exemplos práticos (Churn, Regressão, Multiclasse)

## 🎯 Casos de Uso

Este framework pode ser adaptado para:
- 🏥 Classificação de risco médico
- 💳 Detecção de fraude
- 📧 Classificação de emails (spam)
- 🏠 Previsão de preços (regressão)
- 👥 Segmentação de clientes
- 📊 Análise de séries temporais
- 🔍 Sistemas de recomendação

## ✅ Checklist de Qualidade

Antes de fazer push:

```bash
# 1. Rodar testes localmente
./test_ci_locally.sh

# 2. Verificar cobertura
poetry run pytest --cov=src --cov-report=term

# 3. Verificar formatação
poetry run autopep8 --diff --recursive src/

# 4. Validar sintaxe
poetry run python -m py_compile src/*.py

# 5. Commitar e push
git add .
git commit -m "feat: sua mensagem"
git push
```

## 🔄 Próximos Passos e Melhorias

- [ ] Deploy em produção com FastAPI/Flask
- [ ] Containerização com Docker
- [ ] Integração com evidently para drift avançado
- [ ] Adicionar mais modelos (LightGBM, CatBoost)
- [ ] Pipeline de retreinamento automático
- [ ] API de inferência com documentação Swagger
- [ ] Monitoramento em produção com Prometheus/Grafan` existe e tem permissões de escrita.

### Modelo salvo sem signature (MLflow)
Certifique-se de passar `X_example` ao chamar `save_model()`:
```python
save_model(pipeline, X_example=X_test)
```

## Métricas de Desempenho

Após executar o pipeline, verifique `reports/metrics.json`:

```json
{
    "roc_auc": 0.596,
    "report": {
        "0": {"precision": 0.47, "recall": 0.57, "f1-score": 0.51},
        "1": {"precision": 0.48, "recall": 0.37, "f1-score": 0.42},
        "accuracy": 0.47
    }
}
```

### Métricas Rastreadas no MLflow
- ✓ ROC-AUC
- ✓ Precision, Recall, F1-score por classe
- ✓ Número de amostras de treinamento
- ✓ Número de features

### Gerenciamento de projeto, Testes Unitários e CI
✔️ Testes unitários com pytest  
✔️ Pipeline CI com GitHub Actions  
✔️ Coverage automatizado  
✔️ Gerenciamento de dependências com Poetry

### Tags Registradas
- `model_type`: xgboost_classifier
- `framework`: scikit-learn
- `preprocessing`: StandardScaler + OneHotEncoder

## Documentação Adicional

- **[pyproject.toml](pyproject.toml)** - Configuração de dependências e metadados
- **[POETRY_GUIDE.md](POETRY_GUIDE.md)** - Guia completo de uso do Poetry
- **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - Resumo técnico da solução

## Próximos Passos

- [ ] Integrar validação com evidently para detectar data drift
- [ ] Adicionar testes unitários com pytest
- [ ] Implementar CI/CD com GitHub Actions
- [ ] Deploy em produção com FastAPI
- [ ] Criar dashboard com Streamlit
- [ ] Adicionar log e tratamento de erros
- [ ] Documentar API de inferência

## Licença

MIT License - Veja [LICENSE](LICENSE) para detalhes

## Autor

Gabriela - 2026
