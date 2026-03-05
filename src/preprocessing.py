import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import setup_logger

logger = setup_logger(__name__)

def split_data(df, target, test_size, random_state):
    logger.info(f"Iniciando split dos dados", extra={
        'total_samples': len(df),
        'test_size': test_size
    })
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() == 2 else None
    )
    
    logger.info("Split concluído", extra={
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'n_features': X_train.shape[1]
    })

    return X_train, X_test, y_train, y_test
