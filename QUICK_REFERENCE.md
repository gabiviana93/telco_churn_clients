# ⚡ Quick Reference - Adaptação Rápida

Guia de referência rápida para adaptar o framework. Para detalhes completos, veja [DOCUMENTATION.md](DOCUMENTATION.md).

## Checklist de 10 Minutos

### 1. Configuração Base (2 min)

As configurações são centralizadas no arquivo YAML:

```yaml
# config/project.yaml

project:
  name: "seu_projeto"
  version: "1.0.0"

data:
  target_column: "sua_coluna_target"
  id_column: "seu_id"
  filename: "seus_dados.csv"

mlflow:
  experiment_name: "seu_experimento"
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
# scripts/train_pipeline.py

# O pipeline carrega dados automaticamente do YAML.
# Para customizar, edite config/project.yaml:
features:
  numeric:
    - feat1
    - feat2
  categorical:
    - cat1
    - cat2
```

### 4. Execute (2 min)

```bash
# Teste
poetry run pytest tests/ -q

# Produção
poetry run python scripts/run_pipeline.py

# Visualize
mlflow ui
```

---

## Casos de Uso Rápidos

### Classificação Binária (Padrão)

```python
# Já está pronto! Só mude:
# - config/project.yaml (target_column, features)
# - Coloque seus dados em src/data/raw/
```

### Regressão

```yaml
# 1. config/project.yaml - mude o modelo e métricas
model:
  model_type: "xgboost"
  eval_metric: "rmse"

# 2. src/evaluate.py - substitua função evaluate()
```

```python
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    return {"rmse": rmse, "r2": r2}
```

### Multiclasse

```yaml
# config/project.yaml
model:
  model_type: "xgboost"
  objective: "multi:softprob"
  num_class: 5   # ← Número de classes
  eval_metric: "mlogloss"
```

---

## Comandos Essenciais

```bash
# Setup inicial
poetry install
poetry shell

# Testes (190 testes, 70%+ coverage)
bash scripts/test_ci_locally.sh            # CI completo
poetry run pytest tests/ -v             # Todos os testes
poetry run pytest --cov=src --cov=api   # Com cobertura

# Pipeline
poetry run python generate_data.py      # Gerar dados (exemplo)
poetry run python scripts/run_pipeline.py  # Rodar pipeline
poetry run python scripts/optimize_model.py --quick  # Otimização rápida
poetry run python scripts/optimize_model.py  # Otimização completa
poetry run pytest tests/ -q              # Testar end-to-end

# MLflow
mlflow ui --port 5000                   # Dashboard MLflow

# Docker - Serviços individuais
docker compose up api                   # Apenas API (porta 8000)
docker compose up dashboard             # Dashboard + API (porta 8501)

# Docker - Todos os serviços
docker compose up --build               # API + Dashboard
docker compose --profile mlflow up      # + MLflow (porta 5000)
docker compose --profile dev up api-dev # API em modo dev (hot reload)
docker compose down                     # Parar tudo

# Dashboard (local, sem Docker)
poetry run streamlit run scripts/dashboard.py  # Monitoramento
```

---

## Estrutura de Arquivos Importantes

```
config/project.yaml        ← Mude TUDO aqui primeiro (features, modelo, paths)
src/config.py              ← Paths e constantes carregados do YAML
scripts/train_pipeline.py  ← Pipeline de treinamento padrão
scripts/run_pipeline.py    ← Pipeline de treino + salva métricas em JSON
scripts/optimize_model.py  ← Otimização multi-modelo (LightGBM, XGBoost, CatBoost)
src/feature_engineering.py ← Customize feature engineering
src/evaluate.py            ← Mude métricas (regressão/multiclasse)
src/utils.py               ← Normalização de chaves de métricas
tests/conftest.py          ← Adapte fixtures com seus dados
```

---

## Erros Comuns

### "Feature X not found"
→ Verifique nomes de colunas em `numeric_features` e `categorical_features`

### "Target dimension mismatch"
→ Confira se `TARGET` está correto no `config.py`

### "Module not found"
→ Execute sempre do diretório raiz: `cd /path/to/projeto`

### Testes falhando
→ Adapte fixtures em `tests/conftest.py` com dados do seu domínio

---

## Customizações Rápidas

### Mudar Algoritmo

```python
# config/project.yaml — mude o model_type:
model:
  model_type: "xgboost"  # ou "lightgbm", "catboost"
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.05
```

Ou via código:

```python
# scripts/train_pipeline.py
from src.config import ModelConfig
from src.enums import ModelType

config = ModelConfig(model_type=ModelType.XGBOOST)
```

### Adicionar Transformação

```python
# src/preprocessing.py - no build_preprocessor()

from sklearn.preprocessing import PolynomialFeatures

# Adicione ao pipeline numérico
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2)),  # ← NOVO
    ('scaler', StandardScaler())
])
```

### Dados Desbalanceados

```yaml
# config/project.yaml

model:
  scale_pos_weight: 5  # ← Ajuste o peso da classe positiva
```

Ou use SMOTE via `imbalanced-learn` (já incluído como dependência).

---

## Dependências Extras

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

## Validação Antes do Push

```bash
# 1. Testes
bash scripts/test_ci_locally.sh

# 2. Cobertura (mínimo 70%)
poetry run pytest --cov=src --cov=api --cov-report=term

# 3. Formatação
poetry run ruff check src/ tests/ api/

# 4. Git
git add .
git commit -m "feat: adapta framework para [seu projeto]"
git push
```

---

## Mais Informações

- Tutorial completo: [DOCUMENTATION.md](DOCUMENTATION.md)
- README principal: [README.md](README.md)

---

**Tempo estimado de adaptação: 15-30 minutos**
