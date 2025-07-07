import pytest
import numpy as np
from models.model import MLP
from sklearn.neural_network import MLPRegressor

def test_model_shape():
    """Testa se o modelo retorna o shape correto"""
    model = MLP(input_size=10)
    x = np.random.randn(4, 10)
    y = model(x)
    assert y.shape == (4, 1)

def test_data_loader():
    """Testa o carregamento de dados"""
    from utils.data_loader import load_data
    X, y, _ = load_data("data/Poloniex_BTCUSDC_d.csv", window_size=5)
    assert len(X) == len(y)
    assert X.shape[1] == 5

def test_train_model():
    """Testa o treinamento do modelo"""
    from trainer.trainer import train_model
    from models.model import MLP

    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    model, metrics = train_model(X, y, MLP, kfolds=3)
    assert isinstance(metrics, dict)
    assert 'mse_mean' in metrics
    assert isinstance(metrics['mse_mean'], float)

def test_model_initialization():
    """Testa a inicialização do modelo"""
    model = MLP(input_size=5)
    assert hasattr(model, 'model')
    assert isinstance(model.model, MLPRegressor)
    assert hasattr(model, 'scaler')
    assert hasattr(model, 'is_fitted')
    assert model.is_fitted == False

def test_model_different_input_sizes():
    """Testa diferentes tamanhos de entrada"""
    for input_size in [1, 5, 20, 100]:
        model = MLP(input_size=input_size)
        x = np.random.randn(2, input_size)
        y = model(x)
        assert y.shape == (2, 1)

def test_model_single_sample():
    """Testa predição com uma única amostra"""
    model = MLP(input_size=10)
    x = np.random.randn(1, 10)
    y = model(x)
    assert y.shape == (1, 1)

def test_model_output_type():
    """Testa o tipo de saída do modelo"""
    model = MLP(input_size=10)
    x = np.random.randn(4, 10)
    y = model(x)
    assert isinstance(y, np.ndarray)

def test_model_fit_predict():
    """Testa o ciclo completo de fit e predict"""
    model = MLP(input_size=10, max_iter=100)
    X = np.random.randn(50, 10)
    y = np.random.randn(50)
    
    # Treinar o modelo
    model.fit(X, y)
    assert model.is_fitted == True
    
    # Fazer predições
    predictions = model.predict(X)
    assert predictions.shape == (50, 1)
    assert isinstance(predictions, np.ndarray)

def test_model_predict_without_fit():
    """Testa predição sem treinar o modelo"""
    model = MLP(input_size=10)
    x = np.random.randn(4, 10)
    y = model(x)
    assert y.shape == (4, 1)
    assert np.all(y == 0)  # Deve retornar zeros

def test_model_parameters():
    """Testa os parâmetros do modelo"""
    model = MLP(input_size=10, hidden_layer_sizes=(50, 25), 
                activation='relu', solver='adam', alpha=0.001)
    
    params = model.get_params()
    assert 'hidden_layer_sizes' in params
    assert 'activation' in params
    assert 'solver' in params
    assert 'alpha' in params

def test_model_set_params():
    """Testa a definição de parâmetros"""
    model = MLP(input_size=10)
    
    # Definir novos parâmetros
    model.set_params(hidden_layer_sizes=(30, 15), alpha=0.01)
    
    params = model.get_params()
    assert params['hidden_layer_sizes'] == (30, 15)
    assert params['alpha'] == 0.01

def test_model_score():
    """Testa o score do modelo"""
    model = MLP(input_size=5, max_iter=100)
    X = np.random.randn(50, 5)
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)  # Relação linear com ruído
    
    # Treinar o modelo
    model.fit(X, y)
    
    # Calcular score
    score = model.score(X, y)
    assert isinstance(score, float)
    assert -1 <= score <= 1  # R² está entre -1 e 1

def test_model_zero_input():
    """Testa entrada com zeros"""
    model = MLP(input_size=10)
    x = np.zeros((4, 10))
    y = model(x)
    assert y.shape == (4, 1)
    assert not np.isnan(y).any()

def test_model_large_batch():
    """Testa com batch grande"""
    model = MLP(input_size=10)
    x = np.random.randn(1000, 10)
    y = model(x)
    assert y.shape == (1000, 1)

def test_model_call_method():
    """Testa o método __call__"""
    model = MLP(input_size=10)
    x = np.random.randn(4, 10)
    y1 = model(x)
    y2 = model.predict(x)
    assert np.array_equal(y1, y2)

def test_model_forward_method():
    """Testa o método forward (compatibilidade)"""
    model = MLP(input_size=10)
    x = np.random.randn(4, 10)
    y1 = model.forward(x)
    y2 = model.predict(x)
    assert np.array_equal(y1, y2)

def test_model_feature_importance():
    """Testa informações do modelo"""
    model = MLP(input_size=5, max_iter=100)
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    
    # Treinar o modelo
    model.fit(X, y)
    
    # Obter informações
    info = model.get_feature_importance()
    assert isinstance(info, dict)
    assert 'n_features_in' in info
    assert 'n_layers' in info
    assert 'n_outputs' in info

def test_model_with_different_activations():
    """Testa diferentes funções de ativação"""
    activations = ['relu', 'tanh', 'logistic']
    
    for activation in activations:
        model = MLP(input_size=5, activation=activation)
        x = np.random.randn(10, 5)
        y = model(x)
        assert y.shape == (10, 1)

def test_model_with_different_solvers():
    """Testa diferentes solvers"""
    solvers = ['adam', 'sgd', 'lbfgs']
    
    for solver in solvers:
        model = MLP(input_size=5, solver=solver, max_iter=50)
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        
        # Treinar o modelo
        model.fit(X, y)
        
        # Fazer predições
        predictions = model.predict(X)
        assert predictions.shape == (30, 1)

def test_model_random_state():
    """Testa reprodutibilidade com random_state"""
    X = np.random.randn(50, 10)
    y = np.random.randn(50)
    
    model1 = MLP(input_size=10, random_state=42, max_iter=100)
    model2 = MLP(input_size=10, random_state=42, max_iter=100)
    
    model1.fit(X, y)
    model2.fit(X, y)
    
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    
    # Deve ser muito similares (pode haver pequenas diferenças devido ao solver)
    assert np.allclose(pred1, pred2, rtol=1e-2)

def test_model_regularization():
    """Testa regularização"""
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    # Modelo sem regularização
    model1 = MLP(input_size=10, alpha=0.0, max_iter=100)
    model1.fit(X, y)
    
    # Modelo com regularização
    model2 = MLP(input_size=10, alpha=0.1, max_iter=100)
    model2.fit(X, y)
    
    # Ambos devem treinar sem erro
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    
    assert pred1.shape == pred2.shape == (100, 1)

def test_model_edge_cases():
    """Testa casos extremos"""
    # Muito poucos dados
    model = MLP(input_size=3, max_iter=50)
    X = np.random.randn(5, 3)
    y = np.random.randn(5)
    
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (5, 1)
    
    # Entrada única
    single_pred = model.predict(np.random.randn(1, 3))
    assert single_pred.shape == (1, 1)
