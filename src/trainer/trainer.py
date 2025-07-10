from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

def train_model(X: np.ndarray, y: np.ndarray, model_class=MLPRegressor, kfolds: int = 5, **model_kwargs) -> Tuple[object, dict]:
    """
    Treina um modelo MLPRegressor usando validação cruzada k-fold.
    
    Args:
        X (np.ndarray): Features de entrada
        y (np.ndarray): Valores target
        model_class (class): Classe do modelo (MLPRegressor)
        kfolds (int): Número de folds para validação cruzada
        **model_kwargs: Argumentos adicionais para o modelo
    
    Returns:
        tuple: (modelo treinado, métricas)
    """
    logger.info(f"Iniciando treinamento com {kfolds} folds")
    
    # Configurações padrão para MLPRegressor
    default_params = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'constant', #todo add a entrance option for model type {‘constant’, ‘invscaling’, ‘adaptive’}
        'learning_rate_init': 0.001,
        'max_iter': 1000,
        'shuffle': True,
        'random_state': 42,
        'tol': 1e-4,
        'verbose': True,
        'warm_start': False,
        'momentum': 0.9,
        'nesterovs_momentum': True,
        'early_stopping': False,
        'validation_fraction': 0.1,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-8
    }
    
    # Atualizar com parâmetros fornecidos
    default_params.update(model_kwargs)
    
    # Inicializar listas para métricas
    mse_scores = []
    mae_scores = []
    r2_scores = []
    
    # Configurar validação cruzada
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    
    # Modelo final que será treinado em todos os dados
    final_model = None
    
    logger.info(f"Formato dos dados: X={X.shape}, y={y.shape}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"Processando fold {fold + 1}/{kfolds}")
        
        # Dividir dados
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
            
            logger.info(f"Fold {fold + 1} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            
        except Exception as e:
            logger.error(f"Erro no fold {fold + 1}: {str(e)}")
            # Adicionar valores NaN para manter consistência
            mse_scores.append(np.nan)
            mae_scores.append(np.nan)
            r2_scores.append(np.nan)
    
    # Calcular métricas médias (ignorando NaN)
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
    
    # Normalizar todos os dados
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    
    # Treinar modelo final
    final_model = model_class(**default_params)
    final_model.fit(X_scaled, y)
    
    # Adicionar scaler ao modelo para uso futuro
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