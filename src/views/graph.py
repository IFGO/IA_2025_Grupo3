import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_graph(args, backtest_results, predictions_df):
    try:    
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), dpi=150)
        
        poly_subtitles = ''
        if args.model == 'poly':
            poly_subtitles = f"- Polinomial de {args.poly_degree}º"

        axes[0].plot(backtest_results.index, backtest_results['strategy_balance'], label=f'Estratégia ({args.model.upper()})', color='royalblue')
        axes[0].plot(backtest_results.index, backtest_results['buy_hold_balance'], label='Comprar e Manter (Buy & Hold)', color='darkorange', linestyle='--')
        axes[0].set_title(f'Evolução do Investimento de ${args.investment:,.2f} - {args.crypto} - Para modelo {args.model} {poly_subtitles}', fontsize=16)
        axes[0].set_ylabel('Saldo (USD)')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.regplot(x='actual', y='predicted', data=predictions_df, ax=axes[1],
                    scatter_kws={'alpha':0.3, 'color': 'royalblue'},
                    line_kws={'color':'red', 'linestyle':'--'})
        axes[1].set_title(f'Diagrama de Dispersão: Previsto vs. Real ({args.crypto}) {poly_subtitles}', fontsize=16)
        axes[1].set_xlabel('Preço Real (USD)')
        axes[1].set_ylabel('Preço Previsto (USD)')
        
        plt.tight_layout()
        if args.interative_graph:        
            logger.info("Exibindo gráfico no modo interativo")
            plt.show()
        else:            
            file_name = f'./figures/crypto_analysis_{args.model}_results_{datetime.datetime.now().strftime("%y-%m-%d.%f")}.png'
            plt.savefig(file_name, dpi=150, bbox_inches='tight')
            logger.info(f"Gráficos salvos em {file_name}")
            plt.close()
    except Exception as e:
        logger.exception(f"Não foi possível criar o gráfico erro: {str(e)}")

def generate_multgraph(args, data):
    try:
        backtest_results_mlp = data['mlp']['backtest']
        backtest_results_linear = data['linear']['backtest']
        backtest_results_poly = data['poly']['backtest']

        predictions_df_mlp = data['mlp']['predict']
        predictions_df_linear = data['linear']['predict']
        predictions_df_poly = data['poly']['predict']
        
        sns.set(style="whitegrid")
        
        # Criar figura
        fig = plt.figure(figsize=(18, 12), dpi=150)
        
        poly_subtitles = ''
        if hasattr(args, 'poly_degree'):
            poly_subtitles = f"- Polinomial de {args.poly_degree}º"

        # ============= GRÁFICO PRINCIPAL: Linha 0, ocupando 3 colunas =============
        ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=3, fig=fig)
        
        # Plotar evolução dos investimentos
        ax_main.plot(backtest_results_mlp.index, backtest_results_mlp['buy_hold_balance'], 
                    label='Comprar e Manter (Buy & Hold)', color='darkorange', linestyle='--', linewidth=2)
        ax_main.plot(backtest_results_mlp.index, backtest_results_mlp['strategy_balance'], 
                    label='Estratégia (MLP)', color='royalblue', linewidth=2)
        ax_main.plot(backtest_results_linear.index, backtest_results_linear['strategy_balance'], 
                    label='Estratégia (Linear)', color='magenta', linewidth=2)
        ax_main.plot(backtest_results_poly.index, backtest_results_poly['strategy_balance'], 
                    label=f'Estratégia (Poly {getattr(args, "poly_degree", "N")}º)', color='lime', linewidth=2)
        
        ax_main.set_title(f'Evolução do Investimento de ${args.investment:,.2f} - {args.crypto}', fontsize=18)
        ax_main.set_ylabel('Saldo (USD)', fontsize=14)
        ax_main.legend(fontsize=12)
        ax_main.tick_params(axis='x', rotation=45)
        ax_main.grid(True, alpha=0.3)
        
        # ============= GRÁFICOS DE DISPERSÃO: Linha 1, 3 colunas separadas =============
        
        # MLP - Posição (1,0)
        ax1 = plt.subplot2grid((2, 3), (1, 0), fig=fig)
        sns.regplot(x='actual', y='predicted', data=predictions_df_mlp, ax=ax1,
                    scatter_kws={'alpha': 0.6, 'color': 'royalblue'},
                    line_kws={'color': 'red', 'linestyle': '--'})
        ax1.set_title(f'MLP - Previsto vs Real ({args.crypto})', fontsize=14)
        ax1.set_xlabel('Preço Real (USD)')
        ax1.set_ylabel('Preço Previsto (USD)')
        
        # Linear - Posição (1,1)
        ax2 = plt.subplot2grid((2, 3), (1, 1), fig=fig)
        sns.regplot(x='actual', y='predicted', data=predictions_df_linear, ax=ax2,
                    scatter_kws={'alpha': 0.6, 'color': 'magenta'},
                    line_kws={'color': 'red', 'linestyle': '--'})
        ax2.set_title(f'Linear - Previsto vs Real ({args.crypto})', fontsize=14)
        ax2.set_xlabel('Preço Real (USD)')
        ax2.set_ylabel('Preço Previsto (USD)')
        
        # Polinomial - Posição (1,2)
        ax3 = plt.subplot2grid((2, 3), (1, 2), fig=fig)
        sns.regplot(x='actual', y='predicted', data=predictions_df_poly, ax=ax3,
                    scatter_kws={'alpha': 0.6, 'color': 'lime'},
                    line_kws={'color': 'red', 'linestyle': '--'})
        ax3.set_title(f'Poly - Previsto vs Real ({args.crypto}) {poly_subtitles}', fontsize=14)
        ax3.set_xlabel('Preço Real (USD)')
        ax3.set_ylabel('Preço Previsto (USD)')
        
        plt.tight_layout(pad=3.0)
        
        # Salvar ou exibir
        if getattr(args, 'interative_graph', False):        
            logger.info("Exibindo gráfico no modo interativo")
            plt.show()
        else:            
            file_name = f'./figures/crypto_analysis_comparison_{args.crypto}_{datetime.datetime.now().strftime("%y-%m-%d_%H.%M.%S")}.png'
            plt.savefig(file_name, dpi=150, bbox_inches='tight')
            logger.info(f"Gráfico comparativo salvo em {file_name}")
            plt.close()
            
    except Exception as e:
        logger.exception(f"Não foi possível criar o gráfico erro: {str(e)}")