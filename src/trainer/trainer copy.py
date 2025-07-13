from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging

logger = logging.getLogger(__name__)

def train_model(X: np.ndarray, y: np.ndarray, model_class=MLPRegressor, kfolds: int = 5, **model_kwargs):
    """
    Treina um modelo MLPRegressor usando TimeSeriesSplit para validação cronológica.
    
    Args:
        X (np.ndarray): Features de entrada (ordenadas cronologicamente)
        y (np.ndarray): Valores target (ordenados cronologicamente)
        model_class (class): Classe do modelo (MLPRegressor)
        kfolds (int): Número de splits para validação temporal
        **model_kwargs: Argumentos adicionais para o modelo
    
    Returns:
        tuple: (modelo treinado, métricas)
    """
    logger.info(f"Iniciando treinamento com {kfolds} splits temporais")
    
    # Configurações padrão para MLPRegressor
    default_params = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate_init': 0.001,
        'max_iter': 1000,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10
    }
    
    # Atualizar com parâmetros fornecidos
    default_params.update(model_kwargs)
    
    # Inicializar listas para métricas
    mse_scores = []
    mae_scores = []
    r2_scores = []
    
    # Configurar TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=kfolds)
    
    logger.info(f"Formato dos dados: X={X.shape}, y={y.shape}")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"Processando split temporal {fold + 1}/{kfolds}")
        logger.info(f"  Treino: índices {train_idx[0]} a {train_idx[-1]} ({len(train_idx)} amostras)")
        logger.info(f"  Validação: índices {val_idx[0]} a {val_idx[-1]} ({len(val_idx)} amostras)")
        
        # Dividir dados respeitando a ordem temporal
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Criar e treinar modelo
        model = model_class(**default_params)
        
        try:
            model.fit(X_train_scaled, y_train)
            
            # Fazer predições
            y_pred = model.predict(X_val_scaled)
            
            # Calcular métricas
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            
            logger.info(f"Split {fold + 1} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            
        except Exception as e:
            logger.error(f"Erro no split {fold + 1}: {str(e)}")
            mse_scores.append(np.nan)
            mae_scores.append(np.nan)
            r2_scores.append(np.nan)
    
    # Calcular métricas médias
    avg_mse = np.nanmean(mse_scores)
    avg_mae = np.nanmean(mae_scores)
    avg_r2 = np.nanmean(r2_scores)
    
    std_mse = np.nanstd(mse_scores)
    std_mae = np.nanstd(mae_scores)
    std_r2 = np.nanstd(r2_scores)
    
    logger.info(f"Métricas médias - MSE: {avg_mse:.6f}±{std_mse:.6f}, "
                f"MAE: {avg_mae:.6f}±{std_mae:.6f}, R²: {avg_r2:.6f}±{std_r2:.6f}")
    
    # Treinar modelo final com todos os dados
    logger.info("Treinando modelo final com todos os dados")
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    
    final_model = model_class(**default_params)
    final_model.fit(X_scaled, y)
    final_model.scaler = final_scaler
    
    # Compilar métricas
    metrics = {
        'mse_mean': avg_mse,
        'mse_std': std_mse,
        'mae_mean': avg_mae,
        'mae_std': std_mae,
        'r2_mean': avg_r2,
        'r2_std': std_r2,
        'mse_scores': mse_scores,
        'mae_scores': mae_scores,
        'r2_scores': r2_scores
    }
    
    logger.info("Treinamento concluído com sucesso")
    
    return final_model, metrics

# def train_model_with_walk_forward(X, y, model_class=MLPRegressor, initial_train_size=100, test_size=10):
#     """
#     Treina modelo usando validação walk-forward.
    
#     Args:
#         X: Features
#         y: Target
#         initial_train_size: Tamanho inicial do conjunto de treino
#         test_size: Tamanho do conjunto de teste
        
#     Returns:
#         tuple: (model, metrics)
#     """
    
#     mse_scores = []
#     mae_scores = []
#     r2_scores = []
    
#     # Walk-forward validation
#     for i in range(initial_train_size, len(X) - test_size, test_size):
#         # Dividir dados
#         X_train = X[:i]
#         y_train = y[:i]
#         X_test = X[i:i+test_size]
#         y_test = y[i:i+test_size]
        
#         # Treinar modelo
#         model = MLPRegressor(
#             hidden_layer_sizes=(100, 50),
#             max_iter=500,
#             random_state=42
#         )
#         model.fit(X_train, y_train)
        
#         # Predizer
#         y_pred = model.predict(X_test)
        
#         # Calcular métricas
#         mse_scores.append(mean_squared_error(y_test, y_pred))
#         mae_scores.append(mean_absolute_error(y_test, y_pred))
#         r2_scores.append(r2_score(y_test, y_pred))
    
#     # Treinar modelo final com todos os dados
#     final_model = model_class(
#         hidden_layer_sizes=(100, 50),
#         max_iter=500,
#         random_state=42
#     )
    
#     final_scaler = StandardScaler()
#     X_scaled = final_scaler.fit_transform(X)
    

#     final_model.fit(X_scaled, y)
#     final_model.scaler = final_scaler
    
#     # Retornar métricas
#     metrics = {
#         'mse_scores': mse_scores,
#         'mae_scores': mae_scores,
#         'r2_scores': r2_scores,
#         'mse_mean': np.mean(mse_scores),
#         'mae_mean': np.mean(mae_scores),
#         'r2_mean': np.mean(r2_scores),
#         'mse_std': np.std(mse_scores),
#         'mae_std': np.std(mae_scores),
#         'r2_std': np.std(r2_scores)
#     }
       
#     logger.info("Treinamento concluído com sucesso")
#     return final_model, metrics 

def evaluate_model(model, X, y):
    """
    Avalia um modelo treinado.
    
    Args:
        model: Modelo treinado
        X (np.ndarray): Features de entrada
        y (np.ndarray): Valores target
    
    Returns:
        dict: Métricas de avaliação
    """
    logger.info("Avaliando modelo")
    
    # Normalizar dados se o modelo tem scaler
    if hasattr(model, 'scaler'):
        X_scaled = model.scaler.transform(X)
    else:
        X_scaled = X
    
    # Fazer predições
    y_pred = model.predict(X_scaled)
    
    # Calcular métricas
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }
    
    logger.info(f"Métricas de avaliação - MSE: {mse:.6f}, MAE: {mae:.6f}, "
                f"R²: {r2:.6f}, RMSE: {rmse:.6f}")
    
    return metrics

def predict(model, X):
    """
    Faz predições com um modelo treinado.
    
    Args:
        model: Modelo treinado
        X (np.ndarray): Features de entrada
    
    Returns:
        np.ndarray: Predições
    """
    logger.info(f"Fazendo predições para {X.shape[0]} amostras")
    
    # Normalizar dados se o modelo tem scaler
    if hasattr(model, 'scaler'):
        X_scaled = model.scaler.transform(X)
    else:
        X_scaled = X
    
    predictions = model.predict(X_scaled)
    
    logger.info("Predições concluídas")
    
    return predictions

def save_model(model, filepath):
    """
    Salva um modelo treinado.
    
    Args:
        model: Modelo treinado
        filepath (str): Caminho para salvar o modelo
    """
    import joblib
    
    logger.info(f"Salvando modelo em: {filepath}")
    
    try:
        joblib.dump(model, filepath)
        logger.info("Modelo salvo com sucesso")
    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {str(e)}")
        raise

def load_model(filepath):
    """
    Carrega um modelo salvo.
    
    Args:
        filepath (str): Caminho do modelo salvo
    
    Returns:
        Modelo carregado
    """
    import joblib
    
    logger.info(f"Carregando modelo de: {filepath}")
    
    try:
        model = joblib.load(filepath)
        logger.info("Modelo carregado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise
