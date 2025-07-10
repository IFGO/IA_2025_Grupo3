import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Load and preprocess crypto price data.

    Args:
        file_path (str): Path to CSV.
        window_size (int): Number of past timesteps to use.

    Returns:
        Tuple of input features, targets, and fitted scaler.
    """
    try:
        df = pd.read_csv(file_path, skiprows=1)
        df = df[::-1]  # Oldest first
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)

        scaler = MinMaxScaler()
        df['close_scaled'] = scaler.fit_transform(df[['close']])
        df.rename(columns={'Volume BTC': 'Volume_BTC', 'Volume USDC': 'Volume_USDC'}, inplace=True)

        # remove unnecessary columns
        X = df.drop(columns=['date', 'symbol', 'close', 'close_scaled']).values
        y = df['close_scaled'].values

        return np.array(X), np.array(y), scaler
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise
