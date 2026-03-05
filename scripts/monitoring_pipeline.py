import json
import pandas as pd
from src.monitoring import detect_drift

train_df = pd.read_csv("data/processed/train_reference.csv")
new_df = pd.read_csv("data/monitoring/new_data.csv")

features = train_df.columns

drift = detect_drift(train_df, new_df, features)

with open("reports/drift.json", "w") as f:
    json.dump(drift, f, indent=4)
