import logging
from models.model_mlp import MLP
from models.model_poly import Poly
from models.model_linear import Linear
from sklearn.pipeline import Pipeline

from utils.logger import setup_logger
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
# logger = setup_logger("CryptoMLP")

def create_pipeline(model_type: str, featured_data, poly_degree: int) -> Pipeline:
     logger.info(f"Modelo selecionado {model_type}")
     if model_type not in ['mlp', 'linear', 'poly']:
        raise ValueError(f"Modelo desconhecido: {model_type}. Use 'mlp', 'linear' ou 'poly'.")
     
     match model_type:
            case 'mlp':            
                # mlp = MLP(input_size=featured_data.shape[1] - 2)  # -2 para remover 'target' e 'symbol'
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
                )  # -2 para remover 'target' e 'symbol'

                return Pipeline([('scaler', StandardScaler()), ('mlp', model)])
                # return mlp.create_pipeline()
            case 'linear':
                linear = Linear()
                return linear.create_pipeline()
            case 'poly':
                poly = Poly(input_size=featured_data.shape[1] - 2, degree=poly_degree)
                return poly.create_pipeline()
