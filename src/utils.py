"""
Utilidades para análise exploratória de dados (EDA).
"""

import pandas as pd
import numpy as np


def calculate_iv_categorical(df: pd.DataFrame, feature_col: str, target_col: str) -> float:
    """
    Calcula o Information Value (IV) de uma variável categórica em relação ao target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados
    feature_col : str
        Nome da coluna da feature categórica
    target_col : str
        Nome da coluna target (binária)
    
    Returns:
    --------
    float
        Valor do Information Value
    """
    # Criar tabela de contingência
    grouped = df.groupby(feature_col)[target_col].agg(['sum', 'count'])
    grouped.columns = ['event', 'total']
    grouped['non_event'] = grouped['total'] - grouped['event']
    
    # Calcular distribuições
    grouped['event_rate'] = grouped['event'] / grouped['event'].sum()
    grouped['non_event_rate'] = grouped['non_event'] / grouped['non_event'].sum()
    
    # Evitar divisão por zero
    grouped['event_rate'] = grouped['event_rate'].replace(0, 0.0001)
    grouped['non_event_rate'] = grouped['non_event_rate'].replace(0, 0.0001)
    
    # Calcular WoE e IV
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
    grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
    
    iv_total = grouped['iv'].sum()
    
    return iv_total


def calculate_iv_numeric(df: pd.DataFrame, feature_col: str, target_col: str, bins: int = 10) -> float:
    """
    Calcula o Information Value (IV) de uma variável numérica em relação ao target.
    Discretiza a variável em bins antes do cálculo.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados
    feature_col : str
        Nome da coluna da feature numérica
    target_col : str
        Nome da coluna target (binária)
    bins : int
        Número de bins para discretização
    
    Returns:
    --------
    float
        Valor do Information Value
    """
    # Criar cópia para não modificar original
    df_temp = df[[feature_col, target_col]].copy()
    
    # Discretizar variável numérica
    df_temp['binned'] = pd.qcut(df_temp[feature_col], q=bins, duplicates='drop')
    
    # Calcular IV usando a função categórica
    return calculate_iv_categorical(df_temp, 'binned', target_col)
