import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import MinMaxScaler

# Importar as funções do data_loader
import sys
sys.path.insert(0, 'src')
from utils.data_loader import prepare_data, read_crypto_data, download_crypto_data


class TestDataLoader:
    """Testes essenciais para o módulo data_loader"""

    @pytest.fixture
    def sample_crypto_df(self):
        """DataFrame de exemplo para testes"""
        return pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Symbol': ['BTC', 'BTC', 'BTC'],
            'Open': [100.0, 105.0, 110.0],
            'High': [110.0, 115.0, 120.0],
            'Low': [90.0, 95.0, 100.0],
            'Close': [105.0, 110.0, 115.0],
            'Volume USDC': [1000, 1100, 1200],
            'Volume BTC': [10, 11, 12]
        })

    def create_test_csv(self, format_type="standard"):
        """Cria arquivo CSV temporário para testes"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Linha inválida que será ignorada pelo skiprows=1
            f.write("https://www.CryptoDataDownload.com\n")
            
            if format_type == "poloniex":
                f.write("unix,date,symbol,open,high,low,close,volume\n")
                f.write("1640995200,2022-01-01,BTC,100.0,110.0,90.0,105.0,1000\n")
                f.write("1641081600,2022-01-02,BTC,105.0,115.0,95.0,110.0,1100\n")
                f.write("1641168000,2022-01-03,BTC,110.0,120.0,100.0,115.0,1200\n")
                f.write("1641254400,2022-01-04,BTC,115.0,125.0,105.0,120.0,1300\n")
                f.write("1641340800,2022-01-05,BTC,120.0,130.0,110.0,125.0,1400\n")
                f.write("1641427200,2022-01-06,BTC,125.0,135.0,115.0,130.0,1500\n")
            else:
                f.write("Date,Symbol,Open,High,Low,Close,Volume USDC,Volume BTC\n")
                f.write("2023-01-01,BTC,100.0,110.0,90.0,105.0,1000,10\n")
                f.write("2023-01-02,BTC,105.0,115.0,95.0,110.0,1100,11\n")
                f.write("2023-01-03,BTC,110.0,120.0,100.0,115.0,1200,12\n")
            
            return f.name

    # Testes para prepare_data
    def test_prepare_data_basic(self, sample_crypto_df):
        """Testa preparação básica de dados"""
        result = prepare_data('BTC', sample_crypto_df.copy())
        
        # Verificar colunas renomeadas
        expected_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume_usdc', 'volume_crypto']
        assert list(result.columns) == expected_cols
        
        # Verificar se está indexado por data
        assert result.index.name == 'date'
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_prepare_data_columns_renamed(self, sample_crypto_df):
        """Testa se as colunas são renomeadas corretamente"""
        result = prepare_data('BTC', sample_crypto_df.copy())
        
        assert 'close' in result.columns
        assert 'open' in result.columns
        assert 'volume_usdc' in result.columns
        assert 'volume_crypto' in result.columns

    # Testes para read_crypto_data
    def test_read_crypto_data_success(self):
        """Testa leitura bem-sucedida de arquivo CSV"""
        temp_file = self.create_test_csv()
        
        try:
            result = read_crypto_data('BTC', temp_file)
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'close' in result.columns
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_crypto_data_file_not_found(self):
        """Testa comportamento com arquivo inexistente"""
        with pytest.raises(Exception):
            read_crypto_data('BTC', 'arquivo_inexistente.csv')

    # Testes para download_crypto_data
    @patch('utils.data_loader.requests')
    def test_download_crypto_data_success(self, mock_requests):
        """Testa download bem-sucedido"""
        # Mock da resposta HTTP
        mock_response = MagicMock()
        mock_response.text = "header\ndata1\ndata2"
        mock_response.raise_for_status.return_value = None
        mock_requests.get.return_value = mock_response
        
        # Mock do arquivo já existente
        with patch('os.path.exists', return_value=True), \
             patch('utils.data_loader.pd.read_csv') as mock_read_csv, \
             patch('utils.data_loader.prepare_data') as mock_prepare:
            
            mock_read_csv.return_value = pd.DataFrame({'col': [1, 2, 3]})
            mock_prepare.return_value = pd.DataFrame({'processed': [1, 2, 3]})
            
            result = download_crypto_data('BTC')
            
            assert result is not None
            mock_prepare.assert_called_once()

    @pytest.mark.skip
    @patch('utils.data_loader.requests')
    @patch('utils.data_loader.os.path.exists')
    def test_download_crypto_data_http_error(self, mock_exists, mock_requests):
        """Testa erro HTTP no download"""
        # Forçar que o arquivo não existe para que tente fazer download
        mock_exists.return_value = False
        
        # Mock do erro HTTP
        mock_requests.get.side_effect = Exception("HTTP Error")
        
        result = download_crypto_data('BTC')
        assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])