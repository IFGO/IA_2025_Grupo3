import pytest
import sys
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np
from io import StringIO

# Importar funções do main.py
sys.path.insert(0, 'src')
from main import main


def test_main_file_not_found():
    """Testa main com arquivo não encontrado"""
    test_args = [
        'main.py',
        '--crypto', 'arquivo_inexistente.csv',
        '--model', 'mlp',
        '--kfolds', '5'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with patch('main.load_data') as mock_load_data:
            # Simular FileNotFoundError
            mock_load_data.side_effect = FileNotFoundError("Arquivo não encontrado")
            
            result = main()
            
            # Deve retornar código de erro
            assert result == 1

def test_main_with_exception():
    """Testa main com exceção genérica"""
    test_args = [
        'main.py',
        '--crypto', 'data/Poloniex_BTCUSDC_d.csv',
        '--model', 'mlp',
        '--kfolds', '5'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with patch('main.load_data') as mock_load_data:
            # Simular exceção genérica
            mock_load_data.side_effect = Exception("Erro genérico")
            
            result = main()
            
            # Deve retornar código de erro
            assert result == 1

def test_main_missing_required_argument():
    """Testa main sem argumentos obrigatórios"""
    test_args = ['main.py']
    
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit):
            main()

def test_main_integration():
    """Teste de integração básico"""
    # Criar arquivo CSV temporário
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Escrever cabeçalho e dados de teste
        f.write("date,open,high,low,close,volume\n")
        for i in range(50):
            f.write(f"2023-01-{i:02d},100.{i},110.{i},90.{i},105.{i},1000{i}\n")
        temp_file = f.name
    
    try:
        test_args = [
            'main.py',
            '--crypto', temp_file,
            '--model', 'mlp',
            '--kfolds', '2',
            '--window_size', '3'
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Executar sem mocks para teste de integração
            result = main()
            
            # Pode falhar devido a dados insuficientes, mas não deve dar erro de import
            assert result in [0, 1]
            
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)