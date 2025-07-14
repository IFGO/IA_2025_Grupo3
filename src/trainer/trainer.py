import logging
import numpy as np
import pandas as pd

from scipy import stats
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


logger = logging.getLogger(__name__)

def format_value(    
    symbol: str,
    value: float,
    size: int = 6    
) -> str:
    """
    Formata o valor para exibição com o símbolo e tamanho especificados.

    Args:
        symbol (str): Símbolo da metrica.
        value (float): Valor a ser formatado.
        size (int): Tamanho total do campo, incluindo o símbolo.

    Returns:
        str: Valor formatado com o símbolo.
    """
    return f"{symbol}: {value:>{size - len(symbol)}.2f}"
    

def train_and_evaluate_model(
    df: pd.DataFrame,
    model_pipeline: Pipeline,
    n_splits: int
) -> Tuple[pd.DataFrame, float, float]:
    """
    Treina e avalia um modelo usando TimeSeriesSplit para validação cruzada.

    Args:
        df (pd.DataFrame): DataFrame com features e a coluna 'target'.
        model_pipeline (Pipeline): O pipeline do modelo a ser treinado.
        n_splits (int): Número de folds para TimeSeriesSplit.

    Returns:
        Tuple[pd.DataFrame, float, float]: DataFrame com previsões, RMSE médio e Correlação média.
    """
    X = df.drop(columns=['target', 'symbol'])
    y = df['target']

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    predictions_df = pd.DataFrame(index=df.index)
    predictions_df['actual'] = y
    predictions_df['predicted'] = np.nan

    rmses, corrs, mse_scores, mae_scores, r2_scores  = [], [], [], [], []        

    logger.info(f"Iniciando validação cruzada com {n_splits} folds...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_test)
        
        predictions_df.loc[y_test.index, 'predicted'] = preds
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        corr, _ = stats.pearsonr(y_test, preds)
        rmses.append(rmse)
        corrs.append(corr)
        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        logger.info(f"  - Fold {fold+1}/{n_splits} | {format_value('RMSE', rmse, 11)} | {format_value('MSE', mse, 13 )}" 
                    + f" | {format_value('MAE', mae, 10 )} | {format_value('R²', r2, 6)} | Correlação: {corr:.2f}")
        

    avg_rmse = np.mean(rmses)
    avg_corr = np.mean(corrs)
    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    
    # Desvio padrão para MSE, MAE e R²
    std_rmse = np.nanstd(rmses)
    std_corr = np.nanstd(corrs)
    std_mse = np.nanstd(mse_scores)
    std_mae = np.nanstd(mae_scores)
    std_r2 = np.nanstd(r2_scores)
    
    logger.info(f"Validação cruzada concluída.  , ")
    logger.info(f"  - RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}")
    logger.info(f"  - MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    logger.info(f"  - MAE: {avg_mae:.6f} ± {std_mae:.6f}")
    logger.info(f"  - R²: {avg_r2:.6f}±{std_r2:.6f}")
    logger.info(f"  - Correlação: {avg_corr:.6f} ± {std_corr:.6f}")
    
    return predictions_df.dropna(), avg_rmse, avg_corr

def run_backtest(
    predictions_df: pd.DataFrame,
    initial_capital: float
) -> pd.DataFrame:
    """
    Executa um backtest de uma estratégia de negociação baseada nas previsões.

    Args:
        predictions_df (pd.DataFrame): DataFrame com preços reais e previstos.
        initial_capital (float): O capital inicial para o investimento.

    Returns:
        pd.DataFrame: DataFrame com os resultados do backtest.
    """
    logger.info("Executando backtest da estratégia de investimento...")
    
    backtest_df = pd.DataFrame(index=predictions_df.index)
    backtest_df['actual_close'] = predictions_df['actual']
    
    # O sinal é gerado com base na previsão para o dia seguinte vs o fechamento de hoje.
    # O preço de fechamento de hoje é o 'actual' do dia anterior.
    backtest_df['signal'] = np.where(predictions_df['predicted'] > predictions_df['actual'].shift(1), 1, 0)

    # Retorno do mercado (Buy and Hold)
    backtest_df['market_return'] = backtest_df['actual_close'].pct_change()

    # Retorno da estratégia (sinal de ontem * retorno de hoje)
    backtest_df['strategy_return'] = backtest_df['signal'].shift(1) * backtest_df['market_return']
    backtest_df.fillna(0, inplace=True)

    # Saldo acumulado
    backtest_df['strategy_balance'] = initial_capital * (1 + backtest_df['strategy_return']).cumprod()
    backtest_df['buy_hold_balance'] = initial_capital * (1 + backtest_df['market_return']).cumprod()
    
    logger.info("Backtest concluído.")
    return backtest_df
