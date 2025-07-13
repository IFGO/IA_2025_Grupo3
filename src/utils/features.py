import logging
import numpy as np
import pandas as pd

from typing import List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features técnicas e baseadas em tempo a partir de dados OHLCV.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'open', 'high', 'low', 'close', 'volume'.

    Returns:
        pd.DataFrame: DataFrame enriquecido com novas features.
    """
    logger.info("Iniciando engenharia de features...")
    
    # Médias Móveis
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_21'] = df['close'].rolling(window=21).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # RSI (Índice de Força Relativa)
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bandas de Bollinger
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = sma_20 + (std_20 * 2)
    df['bollinger_lower'] = sma_20 - (std_20 * 2)
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / sma_20

    # Retornos e Volatilidade
    df['return'] = df['close'].pct_change()
    for lag in range(1, 8):
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
    df['volatility_21'] = df['return'].rolling(window=21).std()

    # Features de Tempo
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Alvo: prever o preço de fechamento do próximo dia
    df['target'] = df['close'].shift(-1)

    # Remover NaNs gerados pelas janelas móveis e pelo shift do alvo
    df.dropna(inplace=True)
    
    logger.info(f"Engenharia de features concluída. {len(df.columns)} colunas totais.")
    return df