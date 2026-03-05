import pandas as pd
from src.config import *
from src.preprocessing import split_data
from src.features import build_preprocessor
from src.train import train_model, train_with_mlflow
from src.evaluate import evaluate

df = pd.read_csv("data/processed/data.csv")

num_features = df.select_dtypes("number").columns.drop(TARGET)
cat_features = df.select_dtypes("object").columns

X_train, X_test, y_train, y_test = split_data(
    df, TARGET, TEST_SIZE, RANDOM_STATE
)

params = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss"
}

preprocessor = build_preprocessor(num_features, cat_features)
pipeline = train_model(preprocessor, params)

pipeline = train_with_mlflow(
    X_train, y_train,
    pipeline,
    params,
    MLFLOW_EXPERIMENT
)

evaluate(pipeline, X_test, y_test)
