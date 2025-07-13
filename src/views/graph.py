import time
import datetime
import logging
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_graph(args, backtest_results, predictions_df):
    # --- PASSO 8: Gerar Gráficos ---
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), dpi=150)
    
    # Gráfico 1: Evolução do Lucro
    axes[0].plot(backtest_results.index, backtest_results['strategy_balance'], label=f'Estratégia ({args.model.upper()})', color='royalblue')
    axes[0].plot(backtest_results.index, backtest_results['buy_hold_balance'], label='Comprar e Manter (Buy & Hold)', color='darkorange', linestyle='--')
    axes[0].set_title(f'Evolução do Investimento de ${args.investment:,.2f} - {args.crypto}', fontsize=16)
    axes[0].set_ylabel('Saldo (USD)')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # Gráfico 2: Diagrama de Dispersão
    sns.regplot(x='actual', y='predicted', data=predictions_df, ax=axes[1],
                scatter_kws={'alpha':0.3, 'color': 'royalblue'},
                line_kws={'color':'red', 'linestyle':'--'})
    axes[1].set_title(f'Diagrama de Dispersão: Previsto vs. Real ({args.crypto})', fontsize=16)
    axes[1].set_xlabel('Preço Real (USD)')
    axes[1].set_ylabel('Preço Previsto (USD)')
    
    plt.tight_layout()
    # plt.show()
    # plt.savefig('../figures/crypto_analysis_results.png', dpi=150, bbox_inches='tight')
    file_name = f'./figures/crypto_analysis_{args.model}_results_{datetime.datetime.now().strftime("%y-%m-%d.%f")}.png'
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    logger.info(f"Gráficos salvos em {file_name}")
    plt.close()
