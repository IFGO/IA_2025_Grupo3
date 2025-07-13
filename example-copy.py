# -*- coding: utf-8 -*-
"""
Trabalho Final - Módulo I - Especialização em IA Aplicada

Este notebook implementa um pipeline completo para análise e previsão de preços de criptomoedas.
Ele está estruturado para ser executado no Google Colab, mas simula uma arquitetura de
projeto modular e robusta, cobrindo todos os requisitos do trabalho.

Versão modificada para aceitar um arquivo CSV estático.
"""

# @title 1. Instalação de Dependências e Importações
# -----------------------------------------------------------------------------
# Instalação de pacotes necessários.
# -----------------------------------------------------------------------------
# !pip install -q pandas scikit-learn matplotlib scipy statsmodels

import logging
import pandas as pd
import numpy as np
from io import StringIO
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# @title 2. Configuração do Ambiente e Simulação da CLI
# -----------------------------------------------------------------------------
# Configuração do logging para fornecer feedback claro durante a execução.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Simulação da Interface de Linha de Comando (CLI).
# Os parâmetros agora são mais genéricos, pois os detalhes virão do CSV.
# -----------------------------------------------------------------------------
class Args:
    # --- PARÂMETROS CONFIGURÁVEIS ---
    model: str = 'mlp'   # Opções: 'mlp', 'linear', 'poly'
    poly_degree: int = 2 # Grau para o modelo polinomial (usado se model='poly')
    kfolds: int = 5
    investment: float = 1000.0

args = Args()

logger.info(f"CONFIGURAÇÃO INICIALIZADA:")
logger.info(f"  - Modelo Selecionado: {args.model}")
if args.model == 'poly':
    logger.info(f"  - Grau Polinomial: {args.poly_degree}")
logger.info(f"  - Folds para TimeSeriesSplit: {args.kfolds}")


# @title 3. Módulo `data_loader`: Aquisição e Preparação de Dados
# -----------------------------------------------------------------------------
# Este módulo é responsável por carregar os dados históricos
# de criptomoedas a partir de um arquivo CSV local.
# -----------------------------------------------------------------------------
def load_crypto_data_from_path(file_path: str) -> Optional[pd.DataFrame]:
    """
    Carrega dados históricos diários de um arquivo CSV.

    Args:
        file_path (str): O caminho para o arquivo CSV.

    Returns:
        Optional[pd.DataFrame]: DataFrame com dados históricos ou None se falhar.
    """
    logger.info(f"Tentando carregar dados de {file_path}")
    
    try:
        # Assume que o formato é o mesmo do cryptodatadownload, com uma linha de cabeçalho a ser pulada.
        df = pd.read_csv(file_path, skiprows=1)
        
        # Mapeamento flexível de colunas para lidar com variações
        column_mapping = {
            'Date': 'date', 'Symbol': 'symbol', 'Open': 'open', 'High': 'high', 
            'Low': 'low', 'Close': 'close', 'Volume USD': 'volume',
            'Volume': 'volume', # Adiciona mapeamento para 'Volume' sem 'USD'
            'Volume AAVE': 'volume_aave', 'Volume BTC': 'volume_btc',
            'buyTakerAmount': 'buy_taker_amount', 'buyTakerQuantity': 'buy_taker_quantity',
            'tradeCount': 'trade_count', 'weightedAverage': 'weighted_average'
        }
        
        # Renomeia apenas as colunas que existem no DataFrame
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Verifica se as colunas essenciais existem
        required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Colunas essenciais ausentes no CSV: {missing}")

        df = df[required_cols]
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)
        df.set_index('date', inplace=True)
        
        logger.info(f"Dados de {file_path} carregados e processados com sucesso.")
        return df

    except FileNotFoundError:
        logger.error(f"Erro: Arquivo não encontrado em '{file_path}'.")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao processar {file_path}: {e}")
        return None

# @title 4. Módulo `features`: Engenharia de Features
# -----------------------------------------------------------------------------
# Este módulo transforma dados brutos em variáveis informativas (features)
# para alimentar os modelos de machine learning.
# -----------------------------------------------------------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features técnicas e baseadas em tempo a partir de dados OHLCV.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'open', 'high', 'low', 'close', 'volume'.

    Returns:
        pd.DataFrame: DataFrame enriquecido com novas features.
    """
    logger.info("Iniciando engenharia de features...")
    
    # Médias Móveis
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_21'] = df['close'].rolling(window=21).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # RSI (Índice de Força Relativa)
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bandas de Bollinger
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = sma_20 + (std_20 * 2)
    df['bollinger_lower'] = sma_20 - (std_20 * 2)
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / sma_20

    # Retornos e Volatilidade
    df['return'] = df['close'].pct_change()
    for lag in range(1, 8):
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
    df['volatility_21'] = df['return'].rolling(window=21).std()

    # Features de Tempo
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Alvo: prever o preço de fechamento do próximo dia
    df['target'] = df['close'].shift(-1)

    # Remover NaNs gerados pelas janelas móveis e pelo shift do alvo
    df.dropna(inplace=True)
    
    logger.info(f"Engenharia de features concluída. {len(df.columns)} colunas totais.")
    return df

# @title 5. Módulo `models`: Criação e Treinamento de Modelos
# -----------------------------------------------------------------------------
# Este módulo contém a lógica para criar, treinar e avaliar os modelos
# de machine learning (MLP, Linear, Polinomial).
# -----------------------------------------------------------------------------
def create_model_pipeline(model_type: str, poly_degree: int = 2) -> Pipeline:
    """
    Cria um pipeline do scikit-learn para o modelo especificado.
    Todos os pipelines incluem escalonamento de dados como primeiro passo.

    Args:
        model_type (str): Tipo do modelo ('mlp', 'linear', 'poly').
        poly_degree (int): Grau para a regressão polinomial.

    Returns:
        Pipeline: O pipeline do modelo não treinado.
    """
    if model_type == 'mlp':
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10
        )
        return Pipeline([('scaler', StandardScaler()), ('mlp', model)])
    
    elif model_type == 'linear':
        return Pipeline([('scaler', StandardScaler()), ('linear', LinearRegression())])

    elif model_type == 'poly':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
    else:
        raise ValueError("Tipo de modelo desconhecido. Use 'mlp', 'linear' ou 'poly'.")


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


# @title 6. Módulo `analysis`: Análise de Resultados e Testes Estatísticos
# -----------------------------------------------------------------------------
# Este módulo contém funções para o backtesting da estratégia de investimento
# e para a realização dos testes de hipótese (Teste-t, ANOVA).
# -----------------------------------------------------------------------------
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


def perform_hypothesis_test(
    daily_returns: pd.Series,
    threshold_pct: float = 0.0,
    significance_level: float = 0.05
) -> Dict:
    """
    Realiza um teste t de uma amostra para os retornos diários.

    Args:
        daily_returns (pd.Series): Série de retornos diários da estratégia.
        threshold_pct (float): Limiar de retorno a ser testado.
        significance_level (float): Nível de significância alfa.

    Returns:
        Dict: Dicionário com os resultados do teste.
    """
    logger.info(f"Realizando Teste-t para H_a: retorno médio > {threshold_pct*100:.2f}%")
    t_statistic, p_value = stats.ttest_1samp(
        a=daily_returns.dropna(),
        popmean=threshold_pct / 100, # Converte % para decimal
        alternative='greater'
    )
    reject_null = p_value < significance_level
    
    return {
        "t_statistic": t_statistic,
        "p_value": p_value,
        "reject_null": reject_null,
        "conclusion": "Há evidência estatística de que o retorno médio é superior ao limiar." if reject_null else "Não há evidência estatística de que o retorno médio é superior ao limiar."
    }

def perform_anova_analysis(all_returns_df: pd.DataFrame) -> Dict:
    """
    Realiza ANOVA e teste post-hoc de Tukey para comparar retornos entre criptos.

    Args:
        all_returns_df (pd.DataFrame): DataFrame com colunas 'return' e 'crypto_symbol'.

    Returns:
        Dict: Dicionário com os resultados da ANOVA e do teste de Tukey.
    """
    if all_returns_df['crypto_symbol'].nunique() < 2:
        logger.warning("ANOVA requer pelo menos dois grupos. Análise pulada.")
        return {
            "anova_p_value": None,
            "tukey_summary": "Não aplicável (apenas um grupo encontrado).",
            "tukey_df": None
        }

    logger.info("Realizando Análise de Variância (ANOVA)...")
    
    # Modelo ANOVA
    model = ols('return ~ C(crypto_symbol)', data=all_returns_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_p_value = anova_table['PR(>F)'][0]
    
    results = {"anova_p_value": anova_p_value}
    
    if anova_p_value < 0.05:
        logger.info("ANOVA significativa. Realizando teste post-hoc de Tukey...")
        tukey_results = pairwise_tukeyhsd(
            endog=all_returns_df['return'],
            groups=all_returns_df['crypto_symbol'],
            alpha=0.05
        )
        results["tukey_summary"] = str(tukey_results)
        results["tukey_df"] = pd.DataFrame(
            data=tukey_results._results_table.data[1:],
            columns=tukey_results._results_table.data[0]
        )
    else:
        logger.info("ANOVA não significativa. Não há evidência de diferença entre os retornos médios.")
        results["tukey_summary"] = "Não aplicável (ANOVA não significativa)."
        results["tukey_df"] = None
        
    return results

# @title 7. Execução do Pipeline Principal
# -----------------------------------------------------------------------------
# Esta seção orquestra a execução de todo o pipeline, desde o carregamento
# dos dados até a análise final.
# -----------------------------------------------------------------------------

# --- PASSO 0: Upload do Arquivo CSV ---
logger.info("Por favor, carregue o arquivo CSV com os dados históricos.")
uploaded = "./data/Bitfinex_BTCUSD_d.csv" # files.upload() # Simula o upload de um arquivo CSV

# Assume que apenas um arquivo foi carregado
if not uploaded:
    logger.critical("Nenhum arquivo foi carregado. Encerrando.")
else:
    file_name = "Bitfinex_BTCUSD_d.csv"  # Simula o nome do arquivo carregado

    # --- PASSO 1: Carregar e Processar Dados do Arquivo ---
    main_data = load_crypto_data_from_path(file_name)

    if main_data is not None:
        # Extrai o símbolo principal do CSV para usar nos relatórios
        # Pega o primeiro símbolo encontrado no arquivo
        main_crypto_symbol = main_data['symbol'].iloc[0]
        logger.info(f"Símbolo principal identificado no arquivo: {main_crypto_symbol}")

        # Filtra dados para apenas o símbolo principal para modelagem
        main_data_filtered = main_data[main_data['symbol'] == main_crypto_symbol]
        featured_data = create_features(main_data_filtered.copy())

        # --- PASSO 2: Treinar o Modelo Selecionado ---
        if args.model == 'poly':
            pipeline = create_model_pipeline(args.model, poly_degree=args.poly_degree)
        else:
            pipeline = create_model_pipeline(args.model)

        predictions_df, avg_rmse, avg_corr = train_and_evaluate_model(
            featured_data, pipeline, n_splits=args.kfolds
        )

        # --- PASSO 3: Executar Backtest ---
        backtest_results = run_backtest(predictions_df, initial_capital=args.investment)
        
        final_strategy_balance = backtest_results['strategy_balance'].iloc[-1]
        final_buy_hold_balance = backtest_results['buy_hold_balance'].iloc[-1]
        
        print(backtest_results['strategy_return'])
        # --- PASSO 4: Executar Teste de Hipótese para o Modelo Principal ---
        hypothesis_test_results = perform_hypothesis_test(backtest_results['strategy_return'])

        # --- PASSO 5: Preparar dados para Análise Comparativa (ANOVA) ---
        # Usa o DataFrame completo carregado (pode conter múltiplos símbolos)
        all_returns_df = main_data.copy()
        all_returns_df['return'] = all_returns_df.groupby('symbol')['close'].pct_change()
        all_returns_df.rename(columns={'symbol': 'crypto_symbol'}, inplace=True)
        all_returns_df.dropna(inplace=True)
        
        # --- PASSO 6: Executar ANOVA ---
        anova_results = perform_anova_analysis(all_returns_df[['return', 'crypto_symbol']])

        # --- PASSO 7: Exibir Resultados ---
        logger.info("\n" + "="*50 + "\nRESULTADOS FINAIS\n" + "="*50)

        # Tabela de Desempenho
        summary_data = {
            'Métrica': ['Criptomoeda', 'Modelo', 'RMSE Médio', 'Correlação Média', 'Saldo Final (Estratégia)', 'Saldo Final (Buy & Hold)'],
            'Valor': [
                main_crypto_symbol,
                f"{args.model.upper()}{f' (G={args.poly_degree})' if args.model == 'poly' else ''}",
                f"${avg_rmse:,.2f}",
                f"{avg_corr:.3f}",
                f"${final_strategy_balance:,.2f}",
                f"${final_buy_hold_balance:,.2f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        print("\n--- Tabela de Desempenho do Modelo ---\n")
        print(summary_df.to_string(index=False))

        # Resultados do Teste de Hipótese
        print(f"\n--- Resultados do Teste de Hipótese (Teste-t) para {main_crypto_symbol} ---\n")
        print(f"Estatística t: {hypothesis_test_results['t_statistic']:.4f}")
        print(f"P-valor: {hypothesis_test_results['p_value']:.4f}")
        print(f"Conclusão: {hypothesis_test_results['conclusion']}")

        # Resultados da ANOVA
        print("\n--- Resultados da Análise de Variância (ANOVA) ---\n")
        if anova_results['anova_p_value'] is not None:
            print(f"P-valor do teste ANOVA: {anova_results['anova_p_value']:.4f}")
            if anova_results['tukey_df'] is not None:
                print("\nResultado do Teste Post-Hoc de Tukey:")
                print(anova_results['tukey_df'].to_string(index=False))
            else:
                print(f"\n{anova_results['tukey_summary']}")
        else:
            print(f"\n{anova_results['tukey_summary']}")


        # --- PASSO 8: Gerar Gráficos ---
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), dpi=150)
        
        # Gráfico 1: Evolução do Lucro
        axes[0].plot(backtest_results.index, backtest_results['strategy_balance'], label=f'Estratégia ({args.model.upper()})', color='royalblue')
        axes[0].plot(backtest_results.index, backtest_results['buy_hold_balance'], label='Comprar e Manter (Buy & Hold)', color='darkorange', linestyle='--')
        axes[0].set_title(f'Evolução do Investimento de ${args.investment:,.2f} - {main_crypto_symbol}', fontsize=16)
        axes[0].set_ylabel('Saldo (USD)')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)

        # Gráfico 2: Diagrama de Dispersão
        sns.regplot(x='actual', y='predicted', data=predictions_df, ax=axes[1],
                    scatter_kws={'alpha':0.3, 'color': 'royalblue'},
                    line_kws={'color':'red', 'linestyle':'--'})
        axes[1].set_title(f'Diagrama de Dispersão: Previsto vs. Real ({main_crypto_symbol})', fontsize=16)
        axes[1].set_xlabel('Preço Real (USD)')
        axes[1].set_ylabel('Preço Previsto (USD)')
        
        plt.tight_layout()
        plt.show()

    else:
        logger.critical("Não foi possível executar o pipeline pois os dados do arquivo não puderam ser carregados.")
