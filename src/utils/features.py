import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Classe para engenharia de features para dados de criptomoedas.
    """
    
    def __init__(self):
        self.scalers = {}
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features técnicas básicas.
        
        Args:
            df: DataFrame com colunas 'close', 'volume', 'high', 'low', 'open'
            
        Returns:
            DataFrame com features técnicas adicionadas
        """
        df_features = df.copy()
        
        # Média móvel 7 dias
        df_features['ma_7'] = df_features['close'].rolling(window=7).mean()
        
        # Desvio padrão 7 dias
        df_features['std_7'] = df_features['close'].rolling(window=7).std()
        
        # Variação percentual
        df_features['pct_change'] = df_features['close'].pct_change()
        df_features['pct_change_7d'] = df_features['close'].pct_change(periods=7)
        
        # Volume normalizado (Z-score)
        if 'volume' in df_features.columns:
            df_features['volume_zscore'] = (
                df_features['volume'] - df_features['volume'].rolling(window=30).mean()
            ) / df_features['volume'].rolling(window=30).std()
        
        # RSI (Relative Strength Index)
        df_features['rsi'] = self._calculate_rsi(df_features['close'])
        
        # Bollinger Bands
        df_features['bb_upper'], df_features['bb_lower'] = self._calculate_bollinger_bands(
            df_features['close'], window=20
        )
        
        # MACD
        df_features['macd'], df_features['macd_signal'] = self._calculate_macd(df_features['close'])
        
        # Volatilidade
        df_features['volatility'] = df_features['close'].rolling(window=10).std()
        
        return df_features
    
    def add_external_data(self, df: pd.DataFrame, external_sources: List[str] = None) -> pd.DataFrame:
        """
        Adiciona dados externos como USD/BRL, S&P500, etc.
        
        Args:
            df: DataFrame principal
            external_sources: Lista de símbolos externos a incluir
            
        Returns:
            DataFrame com dados externos adicionados
        """
        if external_sources is None:
            external_sources = ['USDBRL=X', '^GSPC', '^IXIC']  # USD/BRL, S&P500, NASDAQ
        
        df_with_external = df.copy()
        
        # Garantir que temos uma coluna de data
        if 'date' not in df_with_external.columns and df_with_external.index.name != 'date':
            logger.warning("Coluna 'date' não encontrada. Usando índice como data.")
            df_with_external.reset_index(inplace=True)
            if 'timestamp' in df_with_external.columns:
                df_with_external['date'] = pd.to_datetime(df_with_external['timestamp'])
        
        for symbol in external_sources:
            try:
                external_data = self._fetch_external_data(symbol, df_with_external)
                if external_data is not None:
                    df_with_external = pd.merge(
                        df_with_external, external_data, 
                        on='date', how='left', suffixes=('', f'_{symbol.replace("^", "").replace("=X", "")}')
                    )
                    logger.info(f"Dados externos adicionados para {symbol}")
            except Exception as e:
                logger.warning(f"Erro ao buscar dados externos para {symbol}: {e}")
        
        return df_with_external
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str] = None) -> pd.DataFrame:
        """
        Normaliza features usando MinMaxScaler.
        
        Args:
            df: DataFrame com features
            feature_columns: Lista de colunas para normalizar
            
        Returns:
            DataFrame com features normalizadas
        """
        df_normalized = df.copy()
        
        if feature_columns is None:
            # Selecionar colunas numéricas automaticamente
            feature_columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
            # Remover colunas que não devem ser normalizadas
            exclude_cols = ['date', 'timestamp']
            feature_columns = [col for col in feature_columns if col not in exclude_cols]
        
        for col in feature_columns:
            if col in df_normalized.columns:
                scaler = MinMaxScaler()
                df_normalized[f'{col}_scaled'] = scaler.fit_transform(df_normalized[[col]])
                self.scalers[col] = scaler
        
        return df_normalized
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'close', lags: List[int] = None) -> pd.DataFrame:
        """
        Cria features de lag (valores passados).
        
        Args:
            df: DataFrame principal
            target_col: Coluna para criar lags
            lags: Lista de lags a criar
            
        Returns:
            DataFrame com features de lag
        """
        if lags is None:
            lags = [1, 2, 3, 5, 7, 14]
        
        df_with_lags = df.copy()
        
        for lag in lags:
            df_with_lags[f'{target_col}_lag_{lag}'] = df_with_lags[target_col].shift(lag)
        
        return df_with_lags
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcula o RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> tuple:
        """Calcula as Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calcula o MACD."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _fetch_external_data(self, symbol: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Busca dados externos usando yfinance.
        
        Args:
            symbol: Símbolo do Yahoo Finance
            df: DataFrame principal para determinar o período
            
        Returns:
            DataFrame com dados externos ou None se houver erro
        """
        try:
            # Determinar período baseado no DataFrame principal
            if 'date' in df.columns:
                start_date = df['date'].min()
                end_date = df['date'].max()
            else:
                # Usar últimos 2 anos como padrão
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
            
            # Buscar dados
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"Nenhum dado encontrado para {symbol}")
                return None
            
            # Preparar dados
            external_df = pd.DataFrame({
                'date': data.index.date,
                f'{symbol.replace("^", "").replace("=X", "")}_close': data['Close'].values,
                f'{symbol.replace("^", "").replace("=X", "")}_volume': data['Volume'].values,
                f'{symbol.replace("^", "").replace("=X", "")}_pct_change': data['Close'].pct_change().values
            })
            
            external_df['date'] = pd.to_datetime(external_df['date'])
            
            return external_df
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados para {symbol}: {e}")
            return None
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'close') -> Dict[str, float]:
        """
        Calcula importância das features usando correlação.
        
        Args:
            df: DataFrame com features
            target_col: Coluna alvo
            
        Returns:
            Dicionário com importância das features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        
        return correlations.drop(target_col).to_dict()

def create_features_pipeline(df: pd.DataFrame, 
                           include_external: bool = True,
                           external_sources: List[str] = None,
                           normalize: bool = True) -> pd.DataFrame:
    """
    Pipeline completo para criação de features.
    
    Args:
        df: DataFrame original
        include_external: Se deve incluir dados externos
        external_sources: Lista de fontes externas
        normalize: Se deve normalizar features
        
    Returns:
        DataFrame com todas as features criadas
    """
    engineer = FeatureEngineer()
    
    # Criar features técnicas
    df_features = engineer.create_technical_features(df)
    
    # Adicionar dados externos
    if include_external:
        df_features = engineer.add_external_data(df_features, external_sources)
    
    # Criar features de lag
    df_features = engineer.create_lag_features(df_features)
    
    # Normalizar features
    if normalize:
        df_features = engineer.normalize_features(df_features)
    
    # Remover linhas com NaN
    df_features = df_features.dropna()
    
    logger.info(f"Features criadas: {df_features.shape[1]} colunas, {df_features.shape[0]} linhas")
    
    return df_features