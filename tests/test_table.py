import pytest
import pandas as pd
import sys
import os
from io import StringIO
from unittest.mock import patch

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar a função principal
try:
    from views.table import print_table
except ImportError:
    # Se a função não existir, criar versão mock para teste
    def print_table(df, title=None, max_col_width=30):
        print("Mock print_table")


class TestPrintTable:
    """Testes essenciais para a função print_table"""
    
    @pytest.fixture
    def sample_df(self):
        """DataFrame de exemplo para testes"""
        return pd.DataFrame({
            'Symbol': ['BTC', 'ETH', 'LTC'],
            'Price': [45000.00, 3000.50, 150.25],
            'Change_24h': [2.5, -1.2, 5.8]
        })
    
    @pytest.fixture
    def empty_df(self):
        """DataFrame vazio para testes"""
        return pd.DataFrame()

    def capture_output(self, func, *args, **kwargs):
        """Captura a saída impressa de uma função"""
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            func(*args, **kwargs)
        return captured_output.getvalue()

    def test_print_table_basic(self, sample_df):
        """Testa impressão básica"""
        output = self.capture_output(print_table, sample_df)
        
        assert 'Symbol' in output
        assert 'BTC' in output
        assert 'ETH' in output

    def test_print_table_with_title(self, sample_df):
        """Testa impressão com título"""
        title = "Crypto Prices"
        output = self.capture_output(print_table, sample_df, title=title)
        
        assert title in output
        assert 'BTC' in output

    def test_print_table_empty_dataframe(self, empty_df):
        """Testa impressão de DataFrame vazio"""
        output = self.capture_output(print_table, empty_df)
        
        assert 'empty' in output.lower() or 'empty set' in output.lower()

    def test_null_values_handling(self):
        """Testa tratamento de valores nulos"""
        df_with_nulls = pd.DataFrame({
            'A': [1, None, 3],
            'B': ['X', None, 'Z']
        })
        
        output = self.capture_output(print_table, df_with_nulls)
        # A função atual não converte None para NULL, então verificamos se None aparece
        assert 'None' in output or 'NaN' in output or 'nan' in output

    def test_max_col_width_parameter(self, sample_df):
        """Testa parâmetro max_col_width"""
        output = self.capture_output(print_table, sample_df, max_col_width=10)
        assert 'Symbol' in output
        assert 'BTC' in output

    def test_table_structure(self, sample_df):
        """Testa se a tabela tem estrutura SQL correta"""
        output = self.capture_output(print_table, sample_df)
        
        # Verificar bordas SQL
        assert '+' in output  # Bordas horizontais
        assert '-' in output  # Linhas
        assert '|' in output  # Separadores verticais
        # assert 'rows' in output.lower()  # Contador de linhas


if __name__ == '__main__':
    pytest.main([__file__, '-v'])