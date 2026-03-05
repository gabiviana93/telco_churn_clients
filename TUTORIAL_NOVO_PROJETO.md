# 🚀 Tutorial: Adaptando o Framework para Seu Novo Projeto

Este guia passo a passo mostra como adaptar este framework de Data Science para seu próprio projeto, seja de classificação, regressão ou outro tipo de problema de Machine Learning.

## 📋 Índice

1. [Visão Geral do Framework](#visão-geral-do-framework)
2. [Preparação Inicial](#preparação-inicial)
3. [Adaptação Passo a Passo](#adaptação-passo-a-passo)
4. [Exemplos Práticos](#exemplos-práticos)
5. [Checklist Final](#checklist-final)

---

## 📊 Visão Geral do Framework

### Estrutura Modular

O framework está organizado em módulos independentes e reutilizáveis:

```
Entrada de Dados → Preprocessamento → Feature Engineering → 
Treinamento → Avaliação → Monitoramento → Inferência
```

### Componentes Principais

| Componente | Arquivo | Função |
|------------|---------|--------|
| **Configuração** | `src/config.py` | Parâmetros centralizados |
| **Preprocessamento** | `src/preprocessing.py` | Limpeza e divisão de dados |
| **Features** | `src/features.py` | Transformações e encoding |
| **Treinamento** | `src/train.py` | Treinamento e CV |
| **Avaliação** | `src/evaluate.py` | Métricas e relatórios |
| **Inferência** | `src/inference.py` | Predições em produção |
| **Monitoramento** | `src/monitoring.py` | Detecção de drift |
| **Logging** | `src/logger.py` | Logs estruturados |

---

## 🎯 Preparação Inicial

### 1. Clone ou Fork do Repositório

```bash
# Opção 1: Clone direto
git clone <seu-repo>
cd modelo_projetos_ds

# Opção 2: Use como template
# No GitHub: Use this template → Create a new repository

# Renomeie para seu projeto
mv modelo_projetos_ds meu_projeto_ml
cd meu_projeto_ml
```

### 2. Configure o Ambiente

```bash
# Instale Poetry (se não tiver)
curl -sSL https://install.python-poetry.org | python3 -

# Instale dependências
poetry install

# Ative o ambiente
poetry shell
```

### 3. Limpe Dados de Exemplo

```bash
# Remova dados de exemplo (mantenha estrutura)
rm -f data/raw/*.csv
rm -f data/processed/*.csv
rm -rf mlruns/*

# Limpe notebooks de exemplo (ou adapte-os)
# (Opcional) Mantenha-os como referência
```

---

## 🔧 Adaptação Passo a Passo

### PASSO 1: Configuração Base (`src/config.py`)

**O que mudar:**

```python
# ==================== CONFIGURAÇÃO DO PROJETO ====================
PROJECT_NAME = "meu_projeto"  # ← MUDE AQUI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== CONFIGURAÇÃO DE DADOS ====================
TARGET = "sua_coluna_target"  # ← Nome da coluna alvo
TEST_SIZE = 0.2              # ← % para teste (0.2 = 20%)
RANDOM_STATE = 42            # ← Seed para reprodutibilidade

# ==================== CAMINHOS ====================
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "seus_dados.csv")  # ← MUDE
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# ==================== MLFLOW ====================
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
MLFLOW_EXPERIMENT = "meu_experimento"  # ← Nome do experimento
MODEL_NAME = "seu_modelo"              # ← Nome do modelo
```

**Exemplo para classificação de clientes:**

```python
PROJECT_NAME = "customer_churn"
TARGET = "churn"
MLFLOW_EXPERIMENT = "churn_prediction"
MODEL_NAME = "churn_classifier"
```

### PASSO 2: Parâmetros do Modelo

**Para Classificação (XGBoost):**

```python
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss"  # ou "auc", "error"
}
```

**Para Regressão:**

```python
from xgboost import XGBRegressor  # Em src/train.py

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "rmse"  # ou "mae"
}
```

**Para usar outro algoritmo (exemplo: Random Forest):**

```python
# Em src/train.py, substitua:
from sklearn.ensemble import RandomForestClassifier

# E em train_model():
model = RandomForestClassifier(
    n_estimators=params.get("n_estimators", 100),
    max_depth=params.get("max_depth", 10),
    random_state=params.get("random_state", 42)
)
```

### PASSO 3: Features e Preprocessamento (`src/features.py`)

**Identifique suas features:**

```python
# No seu script de análise ou notebook
import pandas as pd

df = pd.read_csv("data/raw/seus_dados.csv")

# Separe por tipo
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remova o target
numeric_features.remove('seu_target')  # se for numérico

print("Features numéricas:", numeric_features)
print("Features categóricas:", categorical_features)
```

**Adapte o preprocessador:**

```python
# src/features.py - função build_preprocessor()

def build_preprocessor(numeric_features, categorical_features):
    """
    CUSTOMIZE AQUI:
    - Adicione transformações específicas
    - Mude estratégias de imputação
    - Adicione feature scaling diferente
    """
    
    # Pipeline numérico
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # ou 'mean', 'most_frequent'
        ('scaler', StandardScaler())  # ou MinMaxScaler(), RobustScaler()
    ])
    
    # Pipeline categórico
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        # Alternativa: LabelEncoder, TargetEncoder, etc.
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
```

### PASSO 4: Métricas de Avaliação (`src/evaluate.py`)

**Para Classificação Binária (já implementado):**

```python
# Mantém como está
- ROC-AUC
- Precision, Recall, F1-Score
- Classification Report
```

**Para Classificação Multiclasse:**

```python
# src/evaluate.py - adapte a função evaluate()

# Adicione average='weighted' ou 'macro'
from sklearn.metrics import roc_auc_score, classification_report

# Para multiclasse com probabilidades
roc_auc = roc_auc_score(y_test, proba, multi_class='ovr', average='weighted')
```

**Para Regressão:**

```python
# src/evaluate.py - SUBSTITUA a função evaluate()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate(model, X_test, y_test):
    """Avalia modelo de regressão."""
    logger.info(f"Iniciando avaliação com {X_test.shape[0]} amostras")
    
    preds = model.predict(X_test)
    
    # Métricas de regressão
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Log no MLflow
    mlflow.log_metrics({
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    })
    
    logger.info("Avaliação concluída", extra={
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    }
```

### PASSO 5: Carregamento de Dados

**Opção A: CSV local**

```python
# scripts/run_pipeline.py ou seu script principal

import pandas as pd
from src.config import RAW_DATA_PATH, TARGET

# Carregue seus dados
df = pd.read_csv(RAW_DATA_PATH)

# Separe features e target
X = df.drop(columns=[TARGET])
y = df[TARGET]
```

**Opção B: Banco de dados**

```python
# Adicione ao pyproject.toml:
# sqlalchemy = "^2.0"
# psycopg2-binary = "^2.9"  # para PostgreSQL

import sqlalchemy as sa

# Crie conexão
engine = sa.create_engine('postgresql://user:pass@localhost/dbname')

# Carregue dados
query = "SELECT * FROM sua_tabela"
df = pd.read_sql(query, engine)
```

**Opção C: API**

```python
import requests
import pandas as pd

response = requests.get('https://api.example.com/data')
data = response.json()
df = pd.DataFrame(data)
```

### PASSO 6: Scripts de Pipeline

**Adapte `scripts/run_pipeline.py`:**

```python
from src.config import *
from src.preprocessing import split_data
from src.features import build_preprocessor
from src.train import train_model, save_model
from src.evaluate import evaluate
import pandas as pd
import mlflow

def main():
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    # 1. CARREGUE SEUS DADOS (customize aqui)
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 2. IDENTIFIQUE FEATURES (customize)
    numeric_features = ['idade', 'renda', 'score']  # ← SEUS DADOS
    categorical_features = ['estado', 'categoria']  # ← SEUS DADOS
    
    # 3. SPLIT (usa sua config)
    X_train, X_test, y_train, y_test = split_data(
        df, target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 4. PREPROCESSAMENTO
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # 5. TREINAMENTO
    with mlflow.start_run(run_name="production_run"):
        pipeline = train_model(preprocessor, X_train, y_train)
        
        # 6. AVALIAÇÃO
        metrics = evaluate(pipeline, X_test, y_test)
        
        # 7. SALVAR MODELO
        save_model(pipeline, X_example=X_test)
        
        print(f"✅ Pipeline completo! Métricas: {metrics}")

if __name__ == "__main__":
    main()
```

### PASSO 7: Testes

**Adapte fixtures em `tests/conftest.py`:**

```python
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Crie dados de exemplo do SEU domínio."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)  # Para classificação
        # ou: 'target': np.random.randn(100) * 10 + 50  # Para regressão
    })

@pytest.fixture
def preprocessor_features():
    """Features do seu projeto."""
    return {
        "numeric_features": ['feature1', 'feature2'],
        "categorical_features": ['category']
    }
```

**Execute testes:**

```bash
# Teste individual
poetry run pytest tests/test_preprocessing.py -v

# Todos os testes
poetry run pytest tests/ -v

# Com cobertura
poetry run pytest --cov=src --cov-report=html
```

---

## 💡 Exemplos Práticos

### Exemplo 1: Classificação de Churn

```python
# config.py
PROJECT_NAME = "customer_churn"
TARGET = "churn"
MLFLOW_EXPERIMENT = "churn_prediction"

# Suas features
NUMERIC_FEATURES = [
    'tenure', 'monthly_charges', 'total_charges',
    'num_services', 'contract_months'
]

CATEGORICAL_FEATURES = [
    'gender', 'senior_citizen', 'partner', 'dependents',
    'phone_service', 'internet_service', 'contract_type'
]

# Modelo
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "scale_pos_weight": 3,  # Para dados desbalanceados
    "eval_metric": "logloss"
}
```

### Exemplo 2: Previsão de Preços (Regressão)

```python
# config.py
PROJECT_NAME = "house_prices"
TARGET = "price"
MLFLOW_EXPERIMENT = "price_prediction"

# Suas features
NUMERIC_FEATURES = [
    'area', 'bedrooms', 'bathrooms', 'age',
    'distance_center', 'floor'
]

CATEGORICAL_FEATURES = [
    'neighborhood', 'type', 'condition'
]

# Modelo (regressão)
from xgboost import XGBRegressor

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.01,
    "eval_metric": "rmse"
}
```

### Exemplo 3: Classificação Multiclasse

```python
# config.py
PROJECT_NAME = "product_category"
TARGET = "category"
MLFLOW_EXPERIMENT = "category_classification"

# Modelo (multiclasse)
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "objective": "multi:softprob",  # ← Para multiclasse
    "num_class": 5,  # ← Número de classes
    "eval_metric": "mlogloss"
}
```

---

## ✅ Checklist Final

### Antes de Treinar

- [ ] **Config atualizada**: `src/config.py` com seus parâmetros
- [ ] **Dados carregados**: CSV, DB ou API funcionando
- [ ] **Features identificadas**: Listas de numeric/categorical
- [ ] **Target correto**: Nome da coluna alvo no config
- [ ] **Preprocessador adaptado**: Transformações adequadas ao seu caso
- [ ] **Modelo escolhido**: XGBoost, RF, ou outro
- [ ] **Métricas corretas**: Classificação vs Regressão

### Validação

```bash
# 1. Teste preprocessamento
poetry run python -c "
from src.features import build_preprocessor
from src.config import *
import pandas as pd

df = pd.read_csv(RAW_DATA_PATH)
# ... teste suas features
print('✅ Preprocessamento OK')
"

# 2. Teste pipeline completo
poetry run python scripts/test_pipeline.py

# 3. Rode CI local
./test_ci_locally.sh
```

### Deploy e Produção

- [ ] **Testes passando**: >80% cobertura
- [ ] **MLflow configurado**: Experimentos rodando
- [ ] **Dashboard funcionando**: `streamlit run scripts/dashboard.py`
- [ ] **Monitoramento ativo**: Drift detection configurado
- [ ] **Documentação atualizada**: README do seu projeto
- [ ] **CI/CD configurado**: GitHub Actions funcionando

---

## 🎨 Customizações Avançadas

### Adicionar Feature Engineering

```python
# src/features.py - adicione transformadores customizados

from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extrai features específicas do seu domínio."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Suas transformações
        X['feature_ratio'] = X['num1'] / (X['num2'] + 1e-10)
        X['feature_interaction'] = X['cat1'] + '_' + X['cat2']
        return X

# Use no preprocessador
preprocessor = ColumnTransformer([
    ('custom', CustomFeatureExtractor(), slice(None)),
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

### Adicionar Validação de Schema

```python
# Adicione pandera para validação
# poetry add pandera

import pandera as pa

# Defina schema
schema = pa.DataFrameSchema({
    "idade": pa.Column(int, pa.Check.in_range(0, 120)),
    "renda": pa.Column(float, pa.Check.greater_than(0)),
    "categoria": pa.Column(str, pa.Check.isin(['A', 'B', 'C']))
})

# Valide dados
df = schema.validate(df)
```

### Adicionar Seleção de Features

```python
# src/features.py

from sklearn.feature_selection import SelectKBest, f_classif

# No preprocessador
preprocessor = Pipeline([
    ('preprocessing', column_transformer),
    ('feature_selection', SelectKBest(f_classif, k=20))  # Top 20 features
])
```

---

## 🚑 Troubleshooting Comum

### Problema: Erro de dimensão de features

**Causa:** Features do treino ≠ features da inferência

**Solução:**
```python
# Salve a lista de features junto com o modelo
import joblib

joblib.dump({
    'model': pipeline,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features
}, 'models/model_with_features.pkl')
```

### Problema: Dados desbalanceados

**Solução:**
```python
# Opção 1: Use class_weight
MODEL_PARAMS = {
    # ... outros params
    "scale_pos_weight": sum(y == 0) / sum(y == 1)  # Para XGBoost
}

# Opção 2: Use SMOTE (poetry add imbalanced-learn)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

### Problema: Overfitting

**Solução:**
```python
# Regularização mais forte
MODEL_PARAMS = {
    "n_estimators": 100,  # Menos árvores
    "max_depth": 4,       # Árvores mais rasas
    "learning_rate": 0.01, # Learning rate menor
    "subsample": 0.7,     # Mais agressivo
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,     # L1 regularization
    "reg_lambda": 1.0     # L2 regularization
}
```

---

## 📞 Suporte e Recursos

### Recursos Adicionais

- **Documentação Scikit-learn**: https://scikit-learn.org/
- **Documentação XGBoost**: https://xgboost.readthedocs.io/
- **MLflow Guide**: https://mlflow.org/docs/latest/
- **Pytest Documentation**: https://docs.pytest.org/

### Próximos Passos Recomendados

1. **Comece simples**: Use o exemplo de classificação binária
2. **Valide incrementalmente**: Teste cada módulo isoladamente
3. **Use notebooks**: Explore seus dados primeiro
4. **Versione tudo**: Git + MLflow para rastreabilidade
5. **Documente mudanças**: Mantenha README atualizado

### Estrutura de Commits

```bash
# Use conventional commits
git commit -m "feat: adiciona suporte para regressão"
git commit -m "fix: corrige encoding de features categóricas"
git commit -m "docs: atualiza README com novo dataset"
git commit -m "test: adiciona testes para novo preprocessador"
```

---

## 🎉 Conclusão

Você agora tem um framework completo e pronto para produção! 

**Lembre-se:**
- ✅ Comece adaptando `config.py`
- ✅ Identifique suas features
- ✅ Adapte preprocessamento e métricas
- ✅ Teste localmente antes de push
- ✅ Use MLflow para rastrear tudo
- ✅ Mantenha >80% de cobertura de testes

**Boa sorte com seu projeto! 🚀**

---

**Dúvidas?** Abra uma issue ou consulte a documentação em [README.md](README.md)
