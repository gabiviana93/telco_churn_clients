# ⚡ Quick Reference - Adaptação Rápida

Guia de referência rápida para adaptar o framework. Para detalhes completos, veja [TUTORIAL_NOVO_PROJETO.md](TUTORIAL_NOVO_PROJETO.md).

## 📝 Checklist de 10 Minutos

### 1. Configuração Base (2 min)

```python
# src/config.py

PROJECT_NAME = "seu_projeto"
TARGET = "sua_coluna_target"
MLFLOW_EXPERIMENT = "seu_experimento"
MODEL_NAME = "seu_modelo"
```

### 2. Identifique Features (3 min)

```python
import pandas as pd

df = pd.read_csv("seus_dados.csv")

# Automático
numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical = df.select_dtypes(include=['object']).columns.tolist()

# Remove target
numeric.remove('target') if 'target' in numeric else None
```

### 3. Adapte Pipeline (3 min)

```python
# scripts/run_pipeline.py

# Apenas mude estas linhas:
df = pd.read_csv("SEUS_DADOS.csv")
numeric_features = ['feat1', 'feat2', ...]  # ← Suas features numéricas
categorical_features = ['cat1', 'cat2', ...]  # ← Suas categóricas
```

### 4. Execute (2 min)

```bash
# Teste
poetry run python scripts/test_pipeline.py

# Produção
poetry run python scripts/run_pipeline.py

# Visualize
mlflow ui
```

---

## 🎯 Casos de Uso Rápidos

### Classificação Binária (Padrão)

```python
# Já está pronto! Só mude:
# - config.py (TARGET, features)
# - Carregue seus dados
```

### Regressão

```python
# 1. config.py
from xgboost import XGBRegressor

MODEL_PARAMS = {
    # ... params
    "eval_metric": "rmse"
}

# 2. src/evaluate.py - substitua função evaluate()
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    return {"rmse": rmse, "r2": r2}

# 3. src/train.py - substitua XGBClassifier por XGBRegressor
model = XGBRegressor(**params)
```

### Multiclasse

```python
# config.py
MODEL_PARAMS = {
    # ... outros
    "objective": "multi:softprob",
    "num_class": 5,  # ← Número de classes
    "eval_metric": "mlogloss"
}
```

---

## 🔍 Comandos Essenciais

```bash
# Setup inicial
poetry install
poetry shell

# Testes
./test_ci_locally.sh                    # CI completo
poetry run pytest tests/ -v             # Todos os testes
poetry run pytest --cov=src             # Com cobertura

# Pipeline
poetry run python generate_data.py      # Gerar dados (exemplo)
poetry run python scripts/run_pipeline.py  # Rodar pipeline
poetry run python scripts/test_pipeline.py # Testar end-to-end

# MLflow
mlflow ui --port 5000                   # Dashboard MLflow

# Dashboard
poetry run streamlit run scripts/dashboard.py  # Monitoramento

# Qualidade
poetry run autopep8 --diff src/         # Ver formatação
poetry run flake8 src/                  # Linting
```

---

## 📊 Estrutura de Arquivos Importantes

```
src/config.py           ← Mude TUDO aqui primeiro
scripts/run_pipeline.py ← Adapte loading de dados + features
src/features.py         ← Customize preprocessamento
src/evaluate.py         ← Mude métricas (regressão/multiclasse)
tests/conftest.py       ← Adapte fixtures com seus dados
```

---

## 🚨 Erros Comuns

### "Feature X not found"
→ Verifique nomes de colunas em `numeric_features` e `categorical_features`

### "Target dimension mismatch"
→ Confira se `TARGET` está correto no `config.py`

### "Module not found"
→ Execute sempre do diretório raiz: `cd /path/to/projeto`

### Testes falhando
→ Adapte fixtures em `tests/conftest.py` com dados do seu domínio

---

## 🎨 Customizações Rápidas

### Mudar Algoritmo

```python
# src/train.py

# De:
from xgboost import XGBClassifier
model = XGBClassifier(**params)

# Para:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(**params)
```

### Adicionar Transformação

```python
# src/features.py - no build_preprocessor()

from sklearn.preprocessing import PolynomialFeatures

# Adicione ao pipeline numérico
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2)),  # ← NOVO
    ('scaler', StandardScaler())
])
```

### Dados Desbalanceados

```python
# config.py

MODEL_PARAMS = {
    # ... outros
    "scale_pos_weight": 5,  # ← Ajuste o peso da classe positiva
}
```

---

## 📦 Dependências Extras

```bash
# Para bases SQL
poetry add sqlalchemy psycopg2-binary

# Para balanceamento
poetry add imbalanced-learn

# Para validação de schema
poetry add pandera

# Para feature selection
# (já incluído no scikit-learn)
```

---

## ✅ Validação Antes do Push

```bash
# 1. Testes
./test_ci_locally.sh

# 2. Cobertura
poetry run pytest --cov=src --cov-report=term

# 3. Formatação
poetry run autopep8 --diff --recursive src/

# 4. Git
git add .
git commit -m "feat: adapta framework para [seu projeto]"
git push
```

---

## 📚 Mais Informações

- Tutorial completo: [TUTORIAL_NOVO_PROJETO.md](TUTORIAL_NOVO_PROJETO.md)
- README principal: [README.md](README.md)
- Guia Poetry: [POETRY_GUIDE.md](POETRY_GUIDE.md)

---

**⏱️ Tempo estimado de adaptação: 15-30 minutos**

**🎯 Objetivo: Framework pronto para seu projeto em menos de 1 hora!**
