import pytest
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.features import FeatureEngineer, create_features_pipeline

class TestFeatureEngineer(unittest.TestCase):
    
    def setUp(self):
        """Configurar dados de teste."""
        self.engineer = FeatureEngineer()
        
        # Criar dados de teste
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Dados simulados de crypto
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        self.test_df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
            'high': [p * 1.05 for p in prices],
            'low': [p * 0.95 for p in prices],
            'open': [p * 1.01 for p in prices]
        })
    
    def test_create_technical_features(self):
        """Testar criação de features técnicas."""
        df_features = self.engineer.create_technical_features(self.test_df)
        
        # Verificar se as features foram criadas
        expected_features = [
            'ma_7', 'std_7', 'pct_change', 'pct_change_7d',
            'volume_zscore', 'rsi', 'bb_upper', 'bb_lower',
            'macd', 'macd_signal', 'volatility'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df_features.columns, f"Feature {feature} não encontrada")
        
        # Verificar se a média móvel está correta
        self.assertAlmostEqual(
            df_features['ma_7'].iloc[10],
            self.test_df['close'].iloc[4:11].mean(),
            places=2
        )
        
        # Verificar se o RSI está no intervalo correto
        rsi_values = df_features['rsi'].dropna()
        self.assertTrue(all(0 <= rsi <= 100 for rsi in rsi_values))
    
    def test_create_lag_features(self):
        """Testar criação de features de lag."""
        lags = [1, 2, 3, 5]
        df_with_lags = self.engineer.create_lag_features(self.test_df, 'close', lags)
        
        # Verificar se as features de lag foram criadas
        for lag in lags:
            lag_col = f'close_lag_{lag}'
            self.assertIn(lag_col, df_with_lags.columns)
            
            # Verificar se os valores de lag estão corretos
            self.assertEqual(
                df_with_lags[lag_col].iloc[lag],
                self.test_df['close'].iloc[0]
            )
    
    def test_normalize_features(self):
        """Testar normalização de features."""
        # Adicionar algumas features numéricas
        df_with_features = self.test_df.copy()
        df_with_features['ma_7'] = df_with_features['close'].rolling(window=7).mean()
        
        # Normalizar
        feature_cols = ['close', 'volume', 'ma_7']
        df_normalized = self.engineer.normalize_features(df_with_features, feature_cols)
        
        # Verificar se as features normalizadas foram criadas
        for col in feature_cols:
            scaled_col = f'{col}_scaled'
            self.assertIn(scaled_col, df_normalized.columns)
            
            # Verificar se os valores estão normalizados (entre 0 e 1)
            scaled_values = df_normalized[scaled_col].dropna()
            self.assertTrue(all(0 <= val <= 1 for val in scaled_values))
    
    @patch('utils.features.yf.Ticker')
    def test_fetch_external_data(self, mock_ticker):
        """Testar busca de dados externos."""
        # Mock do yfinance
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range(start='2023-01-01', periods=5, freq='D'))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Testar busca de dados
        external_data = self.engineer._fetch_external_data('USDBRL=X', self.test_df)
        
        # Verificar se os dados foram retornados
        self.assertIsNotNone(external_data)
        self.assertIn('date', external_data.columns)
        self.assertIn('USDBRL_close', external_data.columns)
        self.assertIn('USDBRL_volume', external_data.columns)
        self.assertIn('USDBRL_pct_change', external_data.columns)
    
    @patch('utils.features.yf.Ticker')
    def test_add_external_data(self, mock_ticker):
        """Testar adição de dados externos."""
        # Mock do yfinance
        mock_data = pd.DataFrame({
            'Close': [5.5, 5.6, 5.7, 5.8, 5.9] * 20,
            'Volume': [1000, 1100, 1200, 1300, 1400] * 20
        }, index=pd.date_range(start='2023-01-01', periods=100, freq='D'))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Testar adição de dados externos
        df_with_external = self.engineer.add_external_data(
            self.test_df, 
            external_sources=['USDBRL=X']
        )
        
        # Verificar se as colunas externas foram adicionadas
        self.assertIn('USDBRL_close', df_with_external.columns)
        self.assertIn('USDBRL_volume', df_with_external.columns)
    
    def test_get_feature_importance(self):
        """Testar cálculo de importância das features."""
        # Criar algumas features correlacionadas
        df_with_features = self.test_df.copy()
        df_with_features['ma_7'] = df_with_features['close'].rolling(window=7).mean()
        df_with_features['correlated_feature'] = df_with_features['close'] * 0.8 + np.random.normal(0, 1000, len(df_with_features))
        
        # Calcular importância
        importance = self.engineer.get_feature_importance(df_with_features, 'close')
        
        # Verificar se as importâncias foram calculadas
        self.assertIsInstance(importance, dict)
        self.assertIn('ma_7', importance)
        self.assertIn('correlated_feature', importance)
        
        # Verificar se os valores estão entre 0 e 1
        for feature, imp in importance.items():
            self.assertTrue(0 <= imp <= 1)
    
    def test_calculate_rsi(self):
        """Testar cálculo do RSI."""
        prices = pd.Series([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03, 46.83, 47.69, 46.49, 46.26])
        rsi = self.engineer._calculate_rsi(prices, window=14)
        
        # Verificar se o RSI está no intervalo correto
        rsi_values = rsi.dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_values))
    
    def test_calculate_bollinger_bands(self):
        """Testar cálculo das Bollinger Bands."""
        prices = pd.Series(self.test_df['close'])
        upper, lower = self.engineer._calculate_bollinger_bands(prices, window=20)
        
        # Verificar se as bandas superiores são maiores que as inferiores
        valid_data = ~(upper.isna() | lower.isna())
        self.assertTrue(all(upper[valid_data] > lower[valid_data]))
    
    def test_calculate_macd(self):
        """Testar cálculo do MACD."""
        prices = pd.Series(self.test_df['close'])
        macd, signal = self.engineer._calculate_macd(prices)
        
        # Verificar se os valores não são todos NaN
        self.assertFalse(macd.isna().all())
        self.assertFalse(signal.isna().all())
        
        # Verificar se o sinal é uma média móvel do MACD
        self.assertEqual(len(macd), len(signal))

class TestFeaturesPipeline(unittest.TestCase):
    
    def setUp(self):
        """Configurar dados de teste."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        prices = np.random.normal(50000, 2000, 50)
        
        self.test_df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 50),
            'high': prices * 1.05,
            'low': prices * 0.95,
            'open': prices * 1.01
        })
    
    @patch('utils.features.yf.Ticker')
    def test_create_features_pipeline(self, mock_ticker):
        """Testar pipeline completo de criação de features."""
        # Mock do yfinance - usar numpy para garantir 50 valores
        mock_data = pd.DataFrame({
            'Close': np.random.uniform(5.5, 5.8, 50),
            'Volume': np.random.randint(1000, 1500, 50)
        }, index=pd.date_range(start='2023-01-01', periods=50, freq='D'))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Executar pipeline
        df_features = create_features_pipeline(
            self.test_df,
            include_external=True,
            external_sources=['USDBRL=X'],
            normalize=True
        )
        
        # Verificar se o pipeline funcionou
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertGreater(len(df_features.columns), len(self.test_df.columns))
        
        # Verificar se não há valores NaN (foram removidos)
        self.assertEqual(df_features.isna().sum().sum(), 0)
    
    def test_create_features_pipeline_no_external(self):
        """Testar pipeline sem dados externos."""
        df_features = create_features_pipeline(
            self.test_df,
            include_external=False,
            normalize=True
        )
        
        # Verificar se o pipeline funcionou
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertGreater(len(df_features.columns), len(self.test_df.columns))
        
        # Verificar se features técnicas foram criadas
        technical_features = ['ma_7', 'std_7', 'pct_change', 'rsi']
        for feature in technical_features:
            feature_found = any(feature in col for col in df_features.columns)
            self.assertTrue(feature_found, f"Feature {feature} não encontrada")

if __name__ == '__main__':
    unittest.main()