import os

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")

# Configuração de dados
TARGET = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Configuração MLflow
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
MLFLOW_EXPERIMENT = "mlflow_test_experiments"
MODEL_NAME = "xgboost_classifier"
MODEL_PARAMS = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    }


DATA_PATH_RAW = "data/raw/dataset.csv"
DATA_PATH_PROCESSED = "data/processed/dataset_clean.csv"

TARGET_COL = "target"
ID_COL = "id"

NUMERIC_COLS = None        # se None → detecta automaticamente
CATEGORICAL_COLS = None   # se None → detecta automaticamente

N_SPLITS = 5
PRIMARY_METRIC = "roc_auc"