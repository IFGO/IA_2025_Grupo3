import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)

class MLP:
    """
    Wrapper para MLPRegressor do scikit-learn com funcionalidades adicionais.
    """
    
    def __init__(self, input_size=None, hidden_layer_sizes=(100, 50), 
                 activation='relu', solver='adam', alpha=0.0001, 
                 learning_rate_init=0.001, max_iter=1000, random_state=42,
                 **kwargs):
        """
        Inicializa o modelo MLP.
        
        Args:
            input_size (int): Tamanho da entrada (não usado no MLPRegressor)
            hidden_layer_sizes (tuple): Tamanhos das camadas ocultas
            activation (str): Função de ativação
            solver (str): Algoritmo de otimização
            alpha (float): Parâmetro de regularização L2
            learning_rate_init (float): Taxa de aprendizado inicial
            max_iter (int): Número máximo de iterações
            random_state (int): Semente aleatória
            **kwargs: Parâmetros adicionais para MLPRegressor
        """
        self.input_size = input_size
        
        # Filtrar kwargs conflitantes para evitar duplicação de parâmetros
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['hidden_layer_sizes', 'activation', 'solver', 
                                     'alpha', 'learning_rate_init', 'max_iter', 
                                     'random_state', 'verbose']}
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            verbose=False,
            **filtered_kwargs
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Treina o modelo.
        
        Args:
            X (array-like): Features de entrada
            y (array-like): Valores target
            
        Returns:
            self: Retorna a instância do modelo
        """
        logger.info(f"Treinando modelo MLP com {X.shape[0]} amostras")
        
        # Normalizar dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Treinar modelo
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info("Modelo treinado com sucesso")
        return self
        
    def predict(self, X):
        """
        Faz predições.
        
        Args:
            X (array-like): Features de entrada
            
        Returns:
            array: Predições
        """
        X = np.array(X)
        
        # Se o modelo não foi treinado, retornar zeros da forma correta
        if not self.is_fitted:
            if len(X.shape) == 2:
                return np.zeros((X.shape[0], 1))
            else:
                return np.zeros((1, 1))
            
        # Normalizar dados
        X_scaled = self.scaler.transform(X)
        
        # Fazer predições
        predictions = self.model.predict(X_scaled)
        
        # Garantir que as predições tenham a forma correta
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
        
    def forward(self, X):
        """
        Compatibilidade com interface de redes neurais.
        
        Args:
            X (array-like): Features de entrada
            
        Returns:
            array: Predições
        """
        return self.predict(X)
        
    def __call__(self, X):
        """
        Permite chamar o modelo diretamente.
        
        Args:
            X (array-like): Features de entrada
            
        Returns:
            array: Predições
        """
        return self.predict(X)
        
    def score(self, X, y):
        """
        Calcula o score R² do modelo.
        
        Args:
            X (array-like): Features de entrada
            y (array-like): Valores target
            
        Returns:
            float: Score R²
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
            
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
        
    def get_feature_importance(self):
        """
        Retorna informações sobre o modelo treinado.
        
        Returns:
            dict: Informações do modelo
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
            
        info = {
            'n_features_in': self.model.n_features_in_,
            'n_layers': self.model.n_layers_,
            'n_outputs': self.model.n_outputs_,
            'loss': self.model.loss_,
            'n_iter': self.model.n_iter_
        }
        
        return info
        
    def get_params(self):
        """
        Retorna os parâmetros do modelo.
        
        Returns:
            dict: Parâmetros do modelo
        """
        return self.model.get_params()
        
    def eval(self):
        """Compatibilidade com interface de redes neurais - modo de avaliação"""
        return self
        
    def train(self, mode=True):
        """Compatibilidade com interface de redes neurais - modo de treino"""
        return self
        
    def parameters(self):
        """Compatibilidade com interface de redes neurais - retorna parâmetros"""
        if self.is_fitted:
            # Retornar uma lista com os coeficientes do modelo
            params = []
            for layer_coef in self.model.coefs_:
                params.append(layer_coef)
            for layer_bias in self.model.intercepts_:
                params.append(layer_bias)
            return params
        return []

    def cross_validation(self, X, y, cv=5):
        """
        Executa validação cruzada.
        
        Args:
            X (array-like): Features de entrada
            y (array-like): Valores target
            cv (int): Número de folds
            
        Returns:
            tuple: (mean_score, std_score)
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.base import BaseEstimator, RegressorMixin
        
        # Criar wrapper simples
        class SimpleWrapper(BaseEstimator, RegressorMixin):
            def __init__(self):
                # Usar configurações básicas do MLPRegressor
                self.sklearn_model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    random_state=42,
                    verbose=False
                )
                self.scaler = StandardScaler()
                
            def fit(self, X, y):
                X_scaled = self.scaler.fit_transform(X)
                self.sklearn_model.fit(X_scaled, y)
                return self
                
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.sklearn_model.predict(X_scaled)
                
            def get_params(self, deep=True):
                return self.sklearn_model.get_params(deep)
                
            def set_params(self, **params):
                self.sklearn_model.set_params(**params)
                return self
        
        # Criar wrapper
        wrapper = SimpleWrapper()
        
        # Executar validação cruzada
        scores = cross_val_score(wrapper, X, y, cv=cv, scoring='r2')
        
        return scores.mean(), scores.std()

    def set_params(self, **params):
        """
        Define parâmetros do modelo.
        
        Args:
            **params: Parâmetros a serem definidos
        """
        self.model.set_params(**params)
        self.is_fitted = False  # Reset fitting status

# Função auxiliar para criar modelo MLP
def create_mlp_model(input_size=None, **kwargs):
    """
    Cria um modelo MLP com parâmetros personalizados.
    
    Args:
        input_size (int): Tamanho da entrada
        **kwargs: Parâmetros para MLPRegressor
        
    Returns:
        MLP: Instância do modelo MLP
    """
    return MLP(input_size=input_size, **kwargs)
