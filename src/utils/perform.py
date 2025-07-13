import logging
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Optional, Tuple, Dict, List
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# logger = setup_logger("CryptoMLP")
logger = logging.getLogger(__name__)

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
    logger.info("Realizando Análise de Variância (ANOVA)...")
    
    # Modelo ANOVA
    all_returns_df = all_returns_df.dropna().copy()
    all_returns_df = all_returns_df[~np.isinf(all_returns_df['return'])]
    all_returns_df = all_returns_df.rename(columns={'return': 'daily_return'})

    model = ols('daily_return ~ C(crypto_symbol)', data=all_returns_df).fit()
    # model = ols('return ~ C(crypto_symbol)', data=all_returns_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_p_value = anova_table['PR(>F)'].iloc[0]
    
    results = {"anova_p_value": anova_p_value}
    
    if anova_p_value < 0.05:
        logger.info("ANOVA significativa. Realizando teste post-hoc de Tukey...")
        tukey_results = pairwise_tukeyhsd(
            endog=all_returns_df['daily_return'],
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
