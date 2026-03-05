from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.logger import setup_logger

logger = setup_logger(__name__)

def build_preprocessor(numeric_features, categorical_features):
    """
    Constrói preprocessador com imputation automática.
    
    Processa features numéricas e categóricas separadamente:
    - Numéricas: Imputation (mediana) → Normalização (StandardScaler)
    - Categóricas: Imputation (missing) → One-Hot Encoding
    
    Args:
        numeric_features: Lista de nomes de features numéricas
        categorical_features: Lista de nomes de features categóricas
    
    Returns:
        ColumnTransformer configurado e pronto para usar
    """
    logger.info("Construindo preprocessador", extra={
        'numeric_features': len(numeric_features),
        'categorical_features': len(categorical_features)
    })
    
    # Pipeline para features numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Mediana é robusta a outliers
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para features categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinar tudo
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Descartar colunas não listadas
    )
    
    logger.info("Preprocessador construído com sucesso")
    
    return preprocessor

