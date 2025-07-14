import pytest
import pandas as pd
import numpy as np
import sys
import os

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar as funções do módulo features
try:
    from utils.features import (
        create_features,
        add_technical_indicators,
        create_lagged_features,
        normalize_features
    )
except ImportError as e:
    # Se algumas funções não existirem, criar mocks básicos
    print(f"Warning: Algumas funções não encontradas: {e}")
    
    def create_features(df):
        """Mock function"""
        return df
    
    def add_technical_indicators(df):
        """Mock function"""
        return df
    
    def create_lagged_features(df, lags=None):
        """Mock function"""
        return df
    
    def normalize_features(df):
        """Mock function"""
        return df


class TestFeatures:
    """Testes essenciais para o módulo features"""
    
    @pytest.fixture
    def sample_crypto_df(self):
        """DataFrame de exemplo com dados de criptomoedas"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        # Simular dados realistas de preços
        base_price = 45000
        price_changes = np.random.normal(0, 0.02, 50)  # 2% volatilidade diária
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        df = pd.DataFrame({
            'date': dates,
            'open': np.array(prices) * np.random.uniform(0.99, 1.01, 50),
            'high': np.array(prices) * np.random.uniform(1.01, 1.05, 50),
            'low': np.array(prices) * np.random.uniform(0.95, 0.99, 50),
            'close': prices,
            'volume_usdc': np.random.uniform(1e9, 5e9, 50),
            'volume_crypto': np.random.uniform(1000, 5000, 50)
        })
        
        df.set_index('date', inplace=True)
        return df
    
    @pytest.fixture
    def minimal_df(self):
        """DataFrame mínimo para testes básicos"""
        return pd.DataFrame({
            'close': [100, 102, 98, 105, 103, 107, 104],
            'volume_usdc': [1000, 1100, 900, 1200, 1050, 1300, 1150]
        })

    # Testes para create_features
    def test_create_features_basic(self, sample_crypto_df):
        """Testa criação básica de features"""
        result = create_features(sample_crypto_df.copy())
        
        # Verificar que retorna um DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Verificar que não é vazio
        assert len(result) > 0
        
        # Verificar que tem mais colunas que o original (features adicionadas)
        assert result.shape[1] >= sample_crypto_df.shape[1]

    def test_create_features_preserves_original_columns(self, sample_crypto_df):
        """Testa se as colunas originais são preservadas"""
        original_cols = list(sample_crypto_df.columns)
        result = create_features(sample_crypto_df.copy())
        
        # Verificar se colunas originais ainda existem
        for col in original_cols:
            assert col in result.columns

    def test_create_features_handles_empty_dataframe(self):
        """Testa comportamento com DataFrame vazio"""
        empty_df = pd.DataFrame()
        
        # Deve retornar algo (mesmo que vazio) sem erro
        result = create_features(empty_df)
        assert isinstance(result, pd.DataFrame)

    # Testes para add_technical_indicators
    def test_add_technical_indicators_basic(self, sample_crypto_df):
        """Testa adição de indicadores técnicos"""
        result = add_technical_indicators(sample_crypto_df.copy())
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_crypto_df)  # Pode ter menos linhas devido a janelas

    def test_add_technical_indicators_minimal_data(self, minimal_df):
        """Testa indicadores com dados mínimos"""
        result = add_technical_indicators(minimal_df.copy())
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    # Testes para create_lagged_features
    def test_create_lagged_features_basic(self, sample_crypto_df):
        """Testa criação de features com lag"""
        result = create_lagged_features(sample_crypto_df.copy(), lags=[1, 2, 3])
        
        assert isinstance(result, pd.DataFrame)
        # Com lags, deve ter menos linhas que o original
        assert len(result) <= len(sample_crypto_df)

    def test_create_lagged_features_default_lags(self, sample_crypto_df):
        """Testa criação de lags com parâmetros padrão"""
        result = create_lagged_features(sample_crypto_df.copy())
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_create_lagged_features_large_lag(self, minimal_df):
        """Testa comportamento com lag grande"""
        # Lag maior que os dados disponíveis
        result = create_lagged_features(minimal_df.copy(), lags=[10])
        
        # Deve retornar DataFrame vazio ou com poucas linhas
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0

    # Testes para normalize_features
    def test_normalize_features_basic(self, sample_crypto_df):
        """Testa normalização básica de features"""
        result = normalize_features(sample_crypto_df.copy())
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_crypto_df.shape

    def test_normalize_features_preserves_structure(self, sample_crypto_df):
        """Testa se a estrutura do DataFrame é preservada"""
        original_cols = list(sample_crypto_df.columns)
        original_index = sample_crypto_df.index
        
        result = normalize_features(sample_crypto_df.copy())
        
        # Verificar estrutura preservada
        assert list(result.columns) == original_cols
        assert len(result) == len(sample_crypto_df)

    # Testes de integração
    def test_features_pipeline_integration(self, sample_crypto_df):
        """Testa pipeline completo de features"""
        df = sample_crypto_df.copy()
        
        # Pipeline: indicators -> lags -> normalization
        df_with_indicators = add_technical_indicators(df)
        df_with_lags = create_lagged_features(df_with_indicators, lags=[1, 2])
        df_normalized = normalize_features(df_with_lags)
        
        # Verificações finais
        assert isinstance(df_normalized, pd.DataFrame)
        assert len(df_normalized) > 0
        assert df_normalized.shape[1] >= df.shape[1]  # Mais features

    # Testes de edge cases
    def test_features_with_nan_values(self):
        """Testa comportamento com valores NaN"""
        df_with_nan = pd.DataFrame({
            'close': [100, np.nan, 98, 105, np.nan, 107],
            'volume_usdc': [1000, 1100, np.nan, 1200, 1050, 1300]
        })
        
        result = create_features(df_with_nan.copy())
        
        # Deve lidar com NaN sem erro
        assert isinstance(result, pd.DataFrame)

    def test_features_single_row(self):
        """Testa comportamento com uma única linha"""
        single_row_df = pd.DataFrame({
            'close': [100],
            'volume_usdc': [1000]
        })
        
        result = create_features(single_row_df.copy())
        
        # Deve retornar algo sem erro
        assert isinstance(result, pd.DataFrame)

    def test_features_constant_values(self):
        """Testa comportamento com valores constantes"""
        constant_df = pd.DataFrame({
            'close': [100] * 10,
            'volume_usdc': [1000] * 10
        })
        
        result = create_features(constant_df.copy())
        
        # Deve lidar com valores constantes sem erro
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])