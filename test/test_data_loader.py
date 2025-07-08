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
from utils.data_loader import load_data

class TestDataLoader:
    """Testes para o módulo data_loader"""

    def create_test_csv(self, data=None, has_header=True, reverse_order=False):
        """
        Cria um arquivo CSV temporário para testes.
        
        Args:
            data (list): Lista de dados para o CSV
            has_header (bool): Se deve incluir cabeçalho
            reverse_order (bool): Se deve inverter a ordem dos dados
        
        Returns:
            str: Caminho para o arquivo temporário
        """
        if data is None:
            data = [
                ['2023-01-01', 100.0, 110.0, 90.0, 105.0, 1000],
                ['2023-01-02', 105.0, 115.0, 95.0, 110.0, 1100],
                ['2023-01-03', 110.0, 120.0, 100.0, 115.0, 1200],
                ['2023-01-04', 115.0, 125.0, 105.0, 120.0, 1300],
                ['2023-01-05', 120.0, 130.0, 110.0, 125.0, 1400],
                ['2023-01-06', 125.0, 135.0, 115.0, 130.0, 1500],
                ['2023-01-07', 130.0, 140.0, 120.0, 135.0, 1600],
                ['2023-01-08', 135.0, 145.0, 125.0, 140.0, 1700],
                ['2023-01-09', 140.0, 150.0, 130.0, 145.0, 1800],
                ['2023-01-10', 145.0, 155.0, 135.0, 150.0, 1900],
                ['2023-01-11', 150.0, 160.0, 140.0, 155.0, 2000],
                ['2023-01-12', 155.0, 165.0, 145.0, 160.0, 2100],
                ['2023-01-13', 160.0, 170.0, 150.0, 165.0, 2200],
                ['2023-01-14', 165.0, 175.0, 155.0, 170.0, 2300],
                ['2023-01-15', 170.0, 180.0, 160.0, 175.0, 2400],
            ]
        
        if reverse_order:
            data = data[::-1]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Escrever uma linha inválida primeiro (que será ignorada pelo skiprows=1)
            f.write("https://www.CryptoDataDownload.com\n")
            
            # Escrever cabeçalho
            if has_header:
                f.write("date,open,high,low,close,volume\n")
            
            # Escrever dados
            for row in data:
                f.write(','.join(map(str, row)) + '\n')
            
            return f.name

    def test_load_data_basic(self):
        """Testa carregamento básico de dados"""
        temp_file = self.create_test_csv()
        
        # Debug: imprimir o conteúdo do arquivo
        with open(temp_file, 'r') as f:
            print("Conteúdo do CSV:")
            print(f.read())
        
        # Debug: ver como o pandas lê o CSV com skiprows=1
        import pandas as pd
        df_debug = pd.read_csv(temp_file, skiprows=1)
        print("Colunas do DataFrame (após skiprows=1):")
        print(df_debug.columns.tolist())
        print("Primeiras linhas:")
        print(df_debug.head())
        
        try:
            X, y, scaler = load_data(temp_file, window_size=5)
            
            # Verificar shapes
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == 5  # window_size
            assert len(X.shape) == 2
            assert len(y.shape) == 1
            
            # Verificar que temos dados suficientes
            expected_samples = 15 - 5  # 15 linhas - window_size
            assert X.shape[0] == expected_samples
            
            # Verificar se o scaler foi criado
            assert isinstance(scaler, MinMaxScaler)
            
            # Verificar se os dados estão normalizados (entre 0 e 1)
            assert np.all(y >= 0) and np.all(y <= 1)
            
            print(f"✓ Teste básico passou: X.shape={X.shape}, y.shape={y.shape}")
            
        finally:
            # Limpar arquivo temporário
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_data_different_window_sizes(self):
        """Testa diferentes tamanhos de janela"""
        temp_file = self.create_test_csv()
        
        try:
            for window_size in [3, 5, 7, 10]:
                X, y, scaler = load_data(temp_file, window_size=window_size)
                
                # Verificar shapes
                assert X.shape[1] == window_size
                expected_samples = 15 - window_size
                assert X.shape[0] == expected_samples
                
                print(f"✓ Window size {window_size}: X.shape={X.shape}, y.shape={y.shape}")
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_data_reversed_order(self):
        """Testa dados em ordem reversa (mais recente primeiro)"""
        temp_file = self.create_test_csv(reverse_order=True)
        
        try:
            X, y, scaler = load_data(temp_file, window_size=5)
            
            # Verificar que funcionou mesmo com ordem reversa
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == 5
            
            print(f"✓ Teste ordem reversa passou: X.shape={X.shape}, y.shape={y.shape}")
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_data_with_missing_values(self):
        """Testa dados com valores ausentes"""
        # Criar dados com alguns valores NaN
        data_with_nan = [
            ['2023-01-01', 100.0, 110.0, 90.0, 105.0, 1000],
            ['2023-01-02', 105.0, 115.0, 95.0, 'NaN', 1100],  # close como NaN
            ['2023-01-03', 110.0, 120.0, 100.0, 115.0, 1200],
            ['2023-01-04', 115.0, 125.0, 105.0, 120.0, 1300],
            ['2023-01-05', 120.0, 130.0, 110.0, 125.0, 1400],
            ['2023-01-06', 125.0, 135.0, 115.0, 130.0, 1500],
            ['2023-01-07', 130.0, 140.0, 120.0, 135.0, 1600],
            ['2023-01-08', 135.0, 145.0, 125.0, 140.0, 1700],
            ['2023-01-09', 140.0, 150.0, 130.0, 145.0, 1800],
            ['2023-01-10', 145.0, 155.0, 135.0, 150.0, 1900],
        ]
        
        temp_file = self.create_test_csv(data=data_with_nan)
        
        try:
            X, y, scaler = load_data(temp_file, window_size=5)
            
            # Verificar que os dados NaN foram removidos
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == 5
            
            # Verificar que não há NaN nos dados finais
            assert not np.any(np.isnan(X))
            assert not np.any(np.isnan(y))
            
            print(f"✓ Teste dados com NaN passou: X.shape={X.shape}, y.shape={y.shape}")
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_data_insufficient_data(self):
        """Testa comportamento com dados insuficientes"""
        # Criar dados com menos linhas que o window_size
        small_data = [
            ['2023-01-01', 100.0, 110.0, 90.0, 105.0, 1000],
            ['2023-01-02', 105.0, 115.0, 95.0, 110.0, 1100],
            ['2023-01-03', 110.0, 120.0, 100.0, 115.0, 1200],
        ]
        
        temp_file = self.create_test_csv(data=small_data)
        
        try:
            X, y, scaler = load_data(temp_file, window_size=5)
            
            # Com apenas 3 linhas de dados e window_size=5, não deve haver amostras
            assert X.shape[0] == 0
            assert y.shape[0] == 0
            
            print(f"✓ Teste dados insuficientes passou: X.shape={X.shape}, y.shape={y.shape}")
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_data_real_crypto_format(self):
        """Testa com formato real de dados de crypto (similar ao Poloniex)"""
        crypto_data = [
            ['1751414400000', '2025-07-02 00:00:00', 'BTC/USDC', 105308.51, 109514.08, 100000.01, 108751, 14249.8, 0.133857, 6085.84, 0.056621, 106, 106519.8],
            ['1751328000000', '2025-07-01 00:00:00', 'BTC/USDC', 106854.27, 107017.41, 105401, 105401, 3947.72, 0.036926, 3786.33, 0.035401, 27, 106909.81],
            ['1751241600000', '2025-06-30 00:00:00', 'BTC/USDC', 108280, 108565, 106252.31, 106451, 5778.44, 0.054253, 4173.64, 0.039249, 37, 106511.21],
            ['1751155200000', '2025-06-29 00:00:00', 'BTC/USDC', 107399, 108280, 107001.8, 107499, 1712.14, 0.01586, 1635.4, 0.015143, 62, 107954.26],
            ['1751068800000', '2025-06-28 00:00:00', 'BTC/USDC', 107248.99, 107399, 106506.07, 107399, 1997.39, 0.018699, 152.66, 0.001423, 20, 106818.41],
            ['1750982400000', '2025-06-27 00:00:00', 'BTC/USDC', 106516.37, 107498.99, 102280, 106501.01, 3907.02, 0.036874, 764.08, 0.007213, 34, 105968.84],
            ['1750896000000', '2025-06-26 00:00:00', 'BTC/USDC', 107609.84, 108249.52, 101936.73, 106751.86, 14958.44, 0.141258, 3897.54, 0.036395, 83, 105933.32],
            ['1750809600000', '2025-06-25 00:00:00', 'BTC/USDC', 106249.99, 108249.92, 104801, 107500, 54692.02, 0.516458, 46474.16, 0.439343, 111, 105902.22],
            ['1750723200000', '2025-06-24 00:00:00', 'BTC/USDC', 105737.98, 106000, 104023.43, 105749, 1797.09, 0.017062, 445.38, 0.004213, 36, 105329.48],
            ['1750636800000', '2025-06-23 00:00:00', 'BTC/USDC', 99501.19, 105750, 99501.19, 105737.99, 6174.56, 0.060375, 5321.05, 0.05187, 80, 102293.37],
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Primeira linha inválida (será ignorada)
            f.write("https://www.CryptoDataDownload.com\n")
            
            # Cabeçalho real do Poloniex
            f.write("unix,date,symbol,open,high,low,close,Volume BTC,Volume USDC,buyTakerAmount,buyTakerQuantity,tradeCount,weightedAverage\n")
            
            # Dados
            for row in crypto_data:
                f.write(','.join(map(str, row)) + '\n')
            
            temp_file = f.name
        
        try:
            X, y, scaler = load_data(temp_file, window_size=5)
            
            # Verificar que funcionou com dados reais
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == 5
            assert X.shape[0] == 10 - 5  # 10 linhas - window_size
            
            print(f"✓ Teste formato crypto real passou: X.shape={X.shape}, y.shape={y.shape}")
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
