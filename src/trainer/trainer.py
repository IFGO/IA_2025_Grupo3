import logging
import numpy as np
import pandas as pd

from scipy import stats
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


logger = logging.getLogger(__name__)

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

    rmses, corrs = [], []

    logger.info(f"Iniciando validação cruzada com {n_splits} folds...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_test)
        
        predictions_df.loc[y_test.index, 'predicted'] = preds
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        corr, _ = stats.pearsonr(y_test, preds)
        rmses.append(rmse)
        corrs.append(corr)
        logger.info(f"  - Fold {fold+1}/{n_splits} | RMSE: {rmse:.2f} | Correlação: {corr:.2f}")

    avg_rmse = np.mean(rmses)
    avg_corr = np.mean(corrs)
    logger.info(f"Validação cruzada concluída. RMSE Médio: {avg_rmse:.2f}, Correlação Média: {avg_corr:.2f}")
    
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
