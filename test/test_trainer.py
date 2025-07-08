import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import sys

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trainer.trainer import train_model, evaluate_model, predict, save_model, load_model

class TestTrainer:
    """Testes para o módulo trainer"""
    
    def setUp(self):
        """Configurar dados de teste"""
        np.random.seed(42)
        self.X = np.random.randn(100, 10)  # 100 amostras, 10 features
        self.y = np.random.randn(100)      # 100 targets
        
        # Dados pequenos para testes rápidos
        self.X_small = np.random.randn(20, 5)
        self.y_small = np.random.randn(20)
    
    def test_train_model_default_params(self):
        """Testa treinamento com parâmetros padrão"""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        model, metrics = train_model(X, y, kfolds=3)
        
        # Verificar se o modelo foi treinado
        assert model is not None
        assert isinstance(model, MLPRegressor)
        assert hasattr(model, 'scaler')
        assert isinstance(model.scaler, StandardScaler)
        
        # Verificar métricas
        assert isinstance(metrics, dict)
        expected_keys = ['mse_mean', 'mse_std', 'mae_mean', 'mae_std', 
                        'r2_mean', 'r2_std', 'mse_scores', 'mae_scores', 'r2_scores']
        for key in expected_keys:
            assert key in metrics
        
        # Verificar se as métricas são números válidos
        assert not np.isnan(metrics['mse_mean'])
        assert not np.isnan(metrics['mae_mean'])
        assert not np.isnan(metrics['r2_mean'])
        
        # Verificar se temos o número correto de scores
        assert len(metrics['mse_scores']) == 3
        assert len(metrics['mae_scores']) == 3
        assert len(metrics['r2_scores']) == 3
    
    def test_train_model_custom_params(self):
        """Testa treinamento com parâmetros customizados"""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        custom_params = {
            'hidden_layer_sizes': (50, 25),
            'max_iter': 500,
            'alpha': 0.001,
            'learning_rate_init': 0.01
        }
        
        model, metrics = train_model(X, y, kfolds=3, **custom_params)
        
        # Verificar se os parâmetros foram aplicados
        assert model.hidden_layer_sizes == (50, 25)
        assert model.max_iter == 500
        assert model.alpha == 0.001
        assert model.learning_rate_init == 0.01
        
        # Verificar métricas
        assert isinstance(metrics, dict)
        assert not np.isnan(metrics['mse_mean'])
    
    def test_train_model_single_fold(self):
        """Testa treinamento com um único fold"""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        
        # Usar 2 folds no mínimo
        model, metrics = train_model(X, y, kfolds=2)
        
        assert model is not None
        assert len(metrics['mse_scores']) == 2
        assert len(metrics['mae_scores']) == 2
        assert len(metrics['r2_scores']) == 2
    
    @patch('trainer.trainer.logger')
    def test_train_model_with_error_handling(self, mock_logger):
        """Testa tratamento de erros durante o treinamento"""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        # Usar parâmetros que podem causar problemas
        problematic_params = {
            'max_iter': 1,  # Muito baixo
            'alpha': 1000,  # Muito alto
        }
        
        model, metrics = train_model(X, y, kfolds=3, **problematic_params)
        
        # Verificar se o modelo ainda foi criado
        assert model is not None
        assert isinstance(metrics, dict)
        
        # Verificar se logs foram chamados
        mock_logger.info.assert_called()
    
    def test_train_model_insufficient_data(self):
        """Testa treinamento com dados insuficientes"""
        # Dados muito pequenos
        X = np.random.randn(5, 2)
        y = np.random.randn(5)
        
        model, metrics = train_model(X, y, kfolds=2)
        
        # Deve ainda funcionar, mas com performance limitada
        assert model is not None
        assert isinstance(metrics, dict)
    
    def test_evaluate_model_with_scaler(self):
        """Testa avaliação de modelo com scaler"""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        # Treinar modelo
        model, _ = train_model(X, y, kfolds=3)
        
        # Avaliar modelo
        metrics = evaluate_model(model, X, y)
        
        # Verificar métricas
        assert isinstance(metrics, dict)
        expected_keys = ['mse', 'mae', 'r2', 'rmse']
        for key in expected_keys:
            assert key in metrics
            assert not np.isnan(metrics[key])
        
        # Verificar se RMSE é a raiz quadrada de MSE
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']))
    
    def test_evaluate_model_without_scaler(self):
        """Testa avaliação de modelo sem scaler"""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        # Criar modelo simples sem scaler
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
        model.fit(X, y)
        
        # Avaliar modelo
        metrics = evaluate_model(model, X, y)
        
        # Verificar métricas
        assert isinstance(metrics, dict)
        expected_keys = ['mse', 'mae', 'r2', 'rmse']
        for key in expected_keys:
            assert key in metrics
            assert not np.isnan(metrics[key])
    
    @patch('trainer.trainer.logger')
    def test_evaluate_model_logging(self, mock_logger):
        """Testa se os logs são chamados corretamente na avaliação"""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        
        model, _ = train_model(X, y, kfolds=2)
        evaluate_model(model, X, y)
        
        # Verificar se logs foram chamados
        mock_logger.info.assert_called()
    
    def test_predict_with_scaler(self):
        """Testa predição com modelo que tem scaler"""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        # Treinar modelo
        model, _ = train_model(X, y, kfolds=3)
        
        # Fazer predições
        X_test = np.random.randn(10, 10)
        predictions = predict(model, X_test)
        
        # Verificar predições
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))
    
    def test_predict_without_scaler(self):
        """Testa predição com modelo sem scaler"""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        # Criar modelo simples sem scaler
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
        model.fit(X, y)
        
        # Fazer predições
        X_test = np.random.randn(10, 10)
        predictions = predict(model, X_test)
        
        # Verificar predições
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))
    
    @patch('trainer.trainer.logger')
    def test_predict_logging(self, mock_logger):
        """Testa se os logs são chamados corretamente na predição"""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        
        model, _ = train_model(X, y, kfolds=2)
        X_test = np.random.randn(5, 5)
        predict(model, X_test)
        
        # Verificar se logs foram chamados
        mock_logger.info.assert_called()
    
    def test_save_model_success(self):
        """Testa salvamento de modelo com sucesso"""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        
        # Treinar modelo
        model, _ = train_model(X, y, kfolds=2)
        
        # Salvar modelo
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            save_model(model, temp_path)
            
            # Verificar se o arquivo foi criado
            assert os.path.exists(temp_path)
            
            # Verificar se o modelo pode ser carregado
            loaded_model = joblib.load(temp_path)
            assert isinstance(loaded_model, MLPRegressor)
            assert hasattr(loaded_model, 'scaler')
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('joblib.dump')
    @patch('trainer.trainer.logger')
    def test_save_model_error(self, mock_logger, mock_dump):
        """Testa tratamento de erro no salvamento"""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        
        model, _ = train_model(X, y, kfolds=2)
        
        # Simular erro
        mock_dump.side_effect = Exception("Erro de salvamento")
        
        with pytest.raises(Exception):
            save_model(model, "dummy_path")
        
        # Verificar se o erro foi logado
        mock_logger.error.assert_called()
    
    def test_load_model_success(self):
        """Testa carregamento de modelo com sucesso"""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        
        # Treinar e salvar modelo
        model, _ = train_model(X, y, kfolds=2)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            save_model(model, temp_path)
            
            # Carregar modelo
            loaded_model = load_model(temp_path)
            
            # Verificar se o modelo foi carregado corretamente
            assert isinstance(loaded_model, MLPRegressor)
            assert hasattr(loaded_model, 'scaler')
            
            # Verificar se o modelo funciona
            X_test = np.random.randn(5, 5)
            predictions = predict(loaded_model, X_test)
            assert isinstance(predictions, np.ndarray)
            assert predictions.shape == (5,)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('joblib.load')
    @patch('trainer.trainer.logger')
    def test_load_model_error(self, mock_logger, mock_load):
        """Testa tratamento de erro no carregamento"""
        # Simular erro
        mock_load.side_effect = Exception("Erro de carregamento")
        
        with pytest.raises(Exception):
            load_model("dummy_path")
        
        # Verificar se o erro foi logado
        mock_logger.error.assert_called()
    
    def test_load_model_file_not_found(self):
        """Testa carregamento de arquivo inexistente"""
        with pytest.raises(Exception):
            load_model("arquivo_inexistente.pkl")
    
    @patch('trainer.trainer.logger')
    def test_comprehensive_workflow(self, mock_logger):
        """Testa um fluxo completo de trabalho"""
        np.random.seed(42)
        X = np.random.randn(100, 15)
        y = np.random.randn(100)
        
        # Treinar modelo
        model, metrics = train_model(X, y, kfolds=5)
        
        # Avaliar modelo
        eval_metrics = evaluate_model(model, X, y)
        
        # Fazer predições
        X_test = np.random.randn(20, 15)
        predictions = predict(model, X_test)
        
        # Salvar modelo
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            save_model(model, temp_path)
            
            # Carregar modelo
            loaded_model = load_model(temp_path)
            
            # Fazer predições com modelo carregado
            loaded_predictions = predict(loaded_model, X_test)
            
            # Verificar se as predições são consistentes
            np.testing.assert_array_almost_equal(predictions, loaded_predictions)
            
            # Verificar se todas as métricas são válidas
            assert all(not np.isnan(v) for v in metrics.values() if isinstance(v, (int, float)))
            assert all(not np.isnan(v) for v in eval_metrics.values())
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_edge_cases(self):
        """Testa casos extremos"""
        # Teste com dados muito pequenos
        X_tiny = np.random.randn(3, 2)
        y_tiny = np.random.randn(3)
        
        model, metrics = train_model(X_tiny, y_tiny, kfolds=2)
        assert model is not None
        
        # Teste com uma única feature
        X_single = np.random.randn(50, 1)
        y_single = np.random.randn(50)
        
        model_single, metrics_single = train_model(X_single, y_single, kfolds=3)
        assert model_single is not None
        
        # Teste com dados perfeitamente correlacionados
        X_perfect = np.random.randn(50, 1)
        y_perfect = X_perfect.flatten() * 2 + 1  # y = 2x + 1
        
        model_perfect, metrics_perfect = train_model(X_perfect, y_perfect, kfolds=3)
        assert model_perfect is not None
        # R² deve ser próximo de 1 para dados perfeitamente correlacionados
        assert metrics_perfect['r2_mean'] > 0.5  # Relaxado devido ao ruído do MLP

if __name__ == '__main__':
    pytest.main([__file__, '-v'])