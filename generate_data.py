"""
Script para gerar dados de exemplo para teste.
Cria um dataset sintético e salva em data/processed/data.csv
"""

import os
import pandas as pd
import numpy as np

# Definir seed para reprodutibilidade
np.random.seed(42)

# Criar diretórios se não existirem
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Gerar dados sintéticos
n_samples = 500

data = {
    "numeric_0": np.random.randn(n_samples),
    "numeric_1": np.random.randn(n_samples),
    "numeric_2": np.random.randn(n_samples),
    "numeric_3": np.random.randn(n_samples),
    "numeric_4": np.random.randn(n_samples),
    "categorical_0": np.random.choice(["A", "B", "C"], n_samples),
    "categorical_1": np.random.choice(["X", "Y"], n_samples),
    "categorical_2": np.random.choice(["cat1", "cat2", "cat3"], n_samples),
    "target": np.random.choice([0, 1], n_samples),
}

df = pd.DataFrame(data)

# Salvar dados
df.to_csv("data/processed/data.csv", index=False)
print(f"✓ Dataset criado: {df.shape[0]} amostras, {df.shape[1]} colunas")
print(f"✓ Salvo em: data/processed/data.csv")
print(f"\nDistribuição do target:\n{df['target'].value_counts()}")
