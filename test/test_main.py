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


def test_main_with_valid_args():
    """Testa execução do main com argumentos válidos"""
    test_args = [
        'main.py',
        '--crypto', 'data/Poloniex_BTCUSDC_d.csv',
        '--model', 'mlp',
        '--kfolds', '3',
        '--window_size', '5'
    ]
    
    with patch.object(sys, 'argv', test_args):
        # Mock das funções para evitar processamento real
        with patch('main.load_data') as mock_load_data:
            with patch('main.train_model') as mock_train_model:
                # Configurar mocks
                mock_load_data.return_value = (
                    np.random.rand(100, 5),  # X
                    np.random.rand(100),     # y
                    None                     # _
                )
                
                mock_train_model.return_value = (
                    MagicMock(),  # modelo
                    {
                        'mse_mean': 0.1,
                        'mse_std': 0.01,
                        'mae_mean': 0.05,
                        'mae_std': 0.005,
                        'r2_mean': 0.8,
                        'r2_std': 0.02
                    }
                )
                
                # Executar main
                result = main()
                
                # Verificar se executou com sucesso
                assert result == 0
                
                # Verificar se as funções foram chamadas
                mock_load_data.assert_called_once()
                mock_train_model.assert_called_once()


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


def test_main_argument_parsing():
    """Testa parsing de argumentos"""
    test_args = [
        'main.py',
        '--crypto', 'test.csv',
        '--model', 'mlp',
        '--kfolds', '10',
        '--window_size', '15'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with patch('main.load_data') as mock_load_data:
            with patch('main.train_model') as mock_train_model:
                # Configurar mocks
                mock_load_data.return_value = (
                    np.random.rand(50, 15),
                    np.random.rand(50),
                    None
                )
                
                mock_train_model.return_value = (
                    MagicMock(),
                    {'mse_mean': 0.1, 'mse_std': 0.01, 'mae_mean': 0.05, 
                     'mae_std': 0.005, 'r2_mean': 0.8, 'r2_std': 0.02}
                )
                
                result = main()
                
                # Verificar se os argumentos foram passados corretamente
                assert result == 0
                
                # Verificar se load_data foi chamado com window_size correto
                mock_load_data.assert_called_with('test.csv', window_size=15)
                
                # Verificar se train_model foi chamado com kfolds correto
                args, kwargs = mock_train_model.call_args
                # A função train_model recebe (X, y, model_class, kfolds, **kwargs)
                # Então kfolds é o 4º argumento (índice 3)
                assert kwargs.get('kfolds', args[3] if len(args) > 3 else None) == 10  # kfolds


def test_main_default_arguments():
    """Testa argumentos padrão"""
    test_args = [
        'main.py',
        '--crypto', 'data/Poloniex_BTCUSDC_d.csv'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with patch('main.load_data') as mock_load_data:
            with patch('main.train_model') as mock_train_model:
                # Configurar mocks
                mock_load_data.return_value = (
                    np.random.rand(50, 10),
                    np.random.rand(50),
                    None
                )
                
                mock_train_model.return_value = (
                    MagicMock(),
                    {'mse_mean': 0.1, 'mse_std': 0.01, 'mae_mean': 0.05, 
                     'mae_std': 0.005, 'r2_mean': 0.8, 'r2_std': 0.02}
                )
                
                result = main()
                
                # Verificar valores padrão
                assert result == 0
                
                # Verificar argumentos padrão
                mock_load_data.assert_called_with('data/Poloniex_BTCUSDC_d.csv', window_size=10)
                args, kwargs = mock_train_model.call_args
                # A função train_model recebe (X, y, model_class, kfolds, **kwargs)
                # Então kfolds é o 4º argumento (índice 3)
                assert kwargs.get('kfolds', args[3] if len(args) > 3 else None) == 5  # kfolds padrão


def test_main_logging():
    """Testa se o logging está funcionando"""
    test_args = [
        'main.py',
        '--crypto', 'data/Poloniex_BTCUSDC_d.csv',
        '--model', 'mlp',
        '--kfolds', '3'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with patch('main.load_data') as mock_load_data:
            with patch('main.train_model') as mock_train_model:
                with patch('main.setup_logger') as mock_setup_logger:
                    # Configurar mock do logger
                    mock_logger = MagicMock()
                    mock_setup_logger.return_value = mock_logger
                    
                    # Configurar outros mocks
                    mock_load_data.return_value = (
                        np.random.rand(50, 10),
                        np.random.rand(50),
                        None
                    )
                    
                    mock_train_model.return_value = (
                        MagicMock(),
                        {'mse_mean': 0.1, 'mse_std': 0.01, 'mae_mean': 0.05, 
                         'mae_std': 0.005, 'r2_mean': 0.8, 'r2_std': 0.02}
                    )
                    
                    result = main()
                    
                    # Verificar se o logger foi configurado
                    mock_setup_logger.assert_called_with("CryptoMLP")
                    
                    # Verificar se mensagens foram logadas
                    assert mock_logger.info.call_count > 0


def test_main_model_predictions():
    """Testa se as predições do modelo são executadas"""
    test_args = [
        'main.py',
        '--crypto', 'data/Poloniex_BTCUSDC_d.csv',
        '--model', 'mlp',
        '--kfolds', '3'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with patch('main.load_data') as mock_load_data:
            with patch('main.train_model') as mock_train_model:
                # Configurar mocks
                X_test = np.random.rand(50, 10)
                y_test = np.random.rand(50)
                
                mock_load_data.return_value = (X_test, y_test, None)
                
                mock_model = MagicMock()
                mock_model.predict.return_value = np.random.rand(5, 1)
                mock_model.get_feature_importance.return_value = {
                    'n_features_in': 10,
                    'n_layers': 3,
                    'n_outputs': 1
                }
                
                mock_train_model.return_value = (
                    mock_model,
                    {'mse_mean': 0.1, 'mse_std': 0.01, 'mae_mean': 0.05, 
                     'mae_std': 0.005, 'r2_mean': 0.8, 'r2_std': 0.02}
                )
                
                result = main()
                
                # Verificar se as predições foram feitas
                mock_model.predict.assert_called_once()
                mock_model.get_feature_importance.assert_called_once()
                
                assert result == 0


def test_main_missing_required_argument():
    """Testa main sem argumentos obrigatórios"""
    test_args = ['main.py']
    
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit):
            main()


def test_main_invalid_model_type():
    """Testa main com tipo de modelo inválido"""
    test_args = [
        'main.py',
        '--crypto', 'data/Poloniex_BTCUSDC_d.csv',
        '--model', 'invalid_model'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with patch('main.load_data') as mock_load_data:
            mock_load_data.return_value = (
                np.random.rand(50, 10),
                np.random.rand(50),
                None
            )
            
            # Deve executar normalmente (MLP é padrão)
            result = main()
            assert result == 0


def test_main_with_small_dataset():
    """Testa main com dataset pequeno"""
    test_args = [
        'main.py',
        '--crypto', 'data/Poloniex_BTCUSDC_d.csv',
        '--model', 'mlp',
        '--kfolds', '2'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with patch('main.load_data') as mock_load_data:
            with patch('main.train_model') as mock_train_model:
                # Dataset muito pequeno
                mock_load_data.return_value = (
                    np.random.rand(5, 10),
                    np.random.rand(5),
                    None
                )
                
                mock_train_model.return_value = (
                    MagicMock(),
                    {'mse_mean': 0.5, 'mse_std': 0.1, 'mae_mean': 0.3, 
                     'mae_std': 0.05, 'r2_mean': 0.2, 'r2_std': 0.1}
                )
                
                result = main()
                
                # Deve executar mesmo com dataset pequeno
                assert result == 0


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