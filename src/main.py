import argparse
import pandas as pd
from models.model import create_pipeline
from utils.perform import perform_hypothesis_test, perform_anova_analysis
from trainer.trainer import train_and_evaluate_model, run_backtest
from utils.features import create_features
from utils.logger import setup_logger
from views.graph import generate_graph, generate_multgraph
from typing import Optional
from utils.data_loader import load_data, download_crypto_data, read_crypto_data
from views.table import print_table

logger = setup_logger("CryptoMLP")

def args_parser() -> None:
    """
    Classe de configuração de parâmetros para experimentos de previsão de preços de criptomoedas.

    Esta classe centraliza todos os parâmetros ajustáveis do projeto, permitindo fácil customização
    dos experimentos via alteração dos atributos. Os parâmetros controlam desde a escolha da criptomoeda,
    modelo de regressão, janela temporal, até aspectos de investimento e análise comparativa.

    Atributos:
        crypto (str): Símbolo da criptomoeda a ser analisada (ex: 'BTC', 'ETH', 'LTC', 'XRP').
        crypto_file (str): Caminho para o arquivo CSV contendo os dados históricos da criptomoeda.
        model (str): Tipo de modelo de regressão a ser utilizado. Opções: 'mlp' (MLPRegressor), 
            'linear' (Regressão Linear), 'poly' (Regressão Polinomial).
        poly_degree (int): Grau do polinômio para o modelo polinomial (usado apenas se model='poly').
        kfolds (int): Número de splits para validação cruzada temporal (TimeSeriesSplit).
        investment (float): Valor inicial de investimento simulado para análise de retorno.
        window_size (int): Tamanho da janela temporal (número de dias) usada para criação das features.
        crypto_list_for_analysis (List[str]): Lista de símbolos de criptomoedas para análise estatística
            comparativa entre diferentes ativos.

    Exemplo de uso:
        args = Args()
        args.crypto = 'ETH'
        args.model = 'poly'
        args.poly_degree = 3
    """

    logger.info("Inicializando parser de argumentos")
    parser = argparse.ArgumentParser(description="Crypto Price Predictor")
    parser.add_argument('--dwn-not-data-set', type=bool, default=False,
                        help='Se True, baixa o dataset mais recente do cryptodatadownload.com')
    parser.add_argument('--investment', type=float, default=1000.0, help='Valor inicial do investimento em USD')
    parser.add_argument('--poly_degree', type=int, default=2, help='Grau do polinômio para o modelo polinomial (usado se model=poly)')
    parser.add_argument('--show_anova', type=bool, default=False,
                        help='Se True, executa análise ANOVA para comparação entre criptomoedas')
    parser.add_argument('--crypto_list_for_analysis', nargs='+', default=['BTC', 'ETH', 'LTC', 'XRP', 'DOGE'],
                        help='Lista de criptomoedas para análise estatística comparativa')
    parser.add_argument('--crypto', type=str, required=True, default='BTC', help='Deve ser informada a sigal da criptomoeda ex: BTC')
    parser.add_argument('--model', type=str, default="mlp", help='Tipo de modelo: MLPRegressor')
    parser.add_argument('--kfolds', type=int, default=5, help='Número de K-Folds para validação')
    parser.add_argument('--window_size', type=int, default=7, help='Tamanho da janela temporal')
    parser.add_argument('--interative_graph', type=bool, default=False, help='Caso deseje usar o grafico de forma interativa, porém não irá salvar o gráfico.')
    parser.add_argument('--analyse-cryptos', type=bool, default=False, help='Caso esse parametro seja verdadeiro, irá executar uma análise de 10 criptomoeadas predefinidas.')

    args = parser.parse_args()
    logger.info("{args} argumentos carregados com sucesso")
    return args

def main():
    """Função principal do pipeline de treinamento."""
    args = args_parser()

    try:
        main_data = load_data(args.crypto, args.dwn_not_data_set)
        
        # featured_data = create_features(main_data.copy())
        if isinstance(main_data, tuple):            
            featured_data = main_data[0]  # Usar as features já processadas
        else:
            featured_data = create_features(main_data.copy())

        # --- PASSO 2: Treinar o Modelo Selecionado ---
        pipeline = create_pipeline(args.model, args.poly_degree)
        logger.info("Pipeline criado com sucesso.")
    
        predictions_df, avg_rmse, avg_corr = train_and_evaluate_model(
            featured_data, pipeline, n_splits=args.kfolds
        )

        # --- PASSO 3: Executar Backtest ---
        backtest_results = run_backtest(predictions_df, initial_capital=args.investment)
        
        final_strategy_balance = backtest_results['strategy_balance'].iloc[-1]
        final_buy_hold_balance = backtest_results['buy_hold_balance'].iloc[-1]
        
        # --- PASSO 4: Executar Teste de Hipótese para o Modelo Principal ---
        hypothesis_test_results = perform_hypothesis_test(backtest_results['strategy_return'])

        # --- PASSO 5: Coletar dados para Análise Comparativa (ANOVA) ---
        if args.show_anova:
            all_returns_list = []
            for crypto in args.crypto_list_for_analysis:
                temp_data = download_crypto_data(crypto)
                if temp_data is not None:
                    temp_data['return'] = temp_data['close'].pct_change()
                    temp_data['crypto_symbol'] = crypto
                    all_returns_list.append(temp_data[['return', 'crypto_symbol']].dropna())
            
            all_returns_df = pd.concat(all_returns_list)
            
            # --- PASSO 6: Executar ANOVA ---
            anova_results = perform_anova_analysis(all_returns_df)    

        # # --- PASSO 7: Exibir Resultados ---
        
        # Tabela de Desempenho
        summary_data = {
            'Métrica': ['Modelo', 'RMSE Médio', 'Correlação Média', 'Saldo Final (Estratégia)', 'Saldo Final (Buy & Hold)'],
            'Valor': [
            f"{args.model.upper()}{f' (G={args.poly_degree})' if args.model == 'poly' else ''}",
            f"${avg_rmse:,.2f}",
            f"{avg_corr:.3f}",
            f"${final_strategy_balance:,.2f}",
            f"${final_buy_hold_balance:,.2f}"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        print(f"\n--- Tabela de Desempenho do Modelo - {args.crypto} ---\n")
        print_table(summary_df)

        # Resultados do Teste de Hipótese
        print(f"\n--- Resultados do Teste de Hipótese (Teste-t) - {args.crypto} ---\n")        
        hypo_data = {
            'Métrica': ['Estatística t', 'P-valor'],
            'Valor': [
            f"{hypothesis_test_results['t_statistic']:.4f}",
            f"{hypothesis_test_results['p_value']:.4f}",
            ]
        }
        hypo_df = pd.DataFrame(hypo_data)
      
        print_table(hypo_df)
        print(f"Conclusão: {hypothesis_test_results['conclusion']}\n")


        # Resultados da ANOVA
        if args.show_anova:
            print("\n--- Resultados da Análise de Variância (ANOVA) ---\n")
            print(f"P-valor do teste ANOVA: {anova_results['anova_p_value']:.4f}")
            if anova_results['tukey_df'] is not None:
                print("\nResultado do Teste Post-Hoc de Tukey:")
                print(anova_results['tukey_df'].to_string(index=False))
            else:
                print("\nTeste de Tukey não foi realizado (ANOVA não significativa).")

        logger.info("Criango gráficos...")
        generate_multgraph(args, backtest_results, predictions_df)     

    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {args.crypto}")
        return 1
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        logger.exception("Detalhes do erro:")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
