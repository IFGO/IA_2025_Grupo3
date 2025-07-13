import logging
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

logger = logging.getLogger(__name__)

def create_pipeline(model_type: str, poly_degree: int) -> Pipeline:
    logger.info(f"Modelo selecionado {model_type}")

    if model_type not in ['mlp', 'linear', 'poly']:
        raise ValueError(f"Modelo desconhecido: {model_type}. Use 'mlp', 'linear' ou 'poly'.")
     
    if model_type == 'linear':
        return Pipeline([('scaler', StandardScaler()), ('linear', LinearRegression())])
    
    elif model_type == 'poly':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('linear', LinearRegression())
        ])  
    
    else:
        model =  MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=2000,              # Aumentar para 2000
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,   # Aumentar validação
                n_iter_no_change=50,        # Mais paciência para convergir
                alpha=0.001,                # Regularização mais forte
                learning_rate_init=0.01,    # Taxa de aprendizado maior
                solver='adam',
                tol=1e-4                    # Tolerância para convergência
            )  

        return Pipeline([('scaler', StandardScaler()), ('mlp', model)])
