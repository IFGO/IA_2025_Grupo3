import argparse
from utils.data_loader import load_data
from models.model import MLP
from trainer.trainer import train_model #, train_model_with_walk_forward
from utils.logger import setup_logger
from views.graph import generate_graph

def main():
    """Função principal do pipeline de treinamento."""
    logger = setup_logger("CryptoMLP")

    parser = argparse.ArgumentParser(description="Crypto Price Predictor")
    parser.add_argument('--investment', type=float, default=1000.0, help='Valor inicial do investimento em USD')
    parser.add_argument('--poly_degree', type=int, default=2, help='Grau do polinômio para o modelo polinomial (usado se model=poly)')
    parser.add_argument('--crypto_list_for_analysis', nargs='+', default=['BTC', 'ETH', 'LTC', 'XRP', 'DOGE'],
                        help='Lista de criptomoedas para análise estatística comparativa')
    parser.add_argument('--crypto', type=str, required=True, help='Caminho para CSV da criptomoeda')
    parser.add_argument('--crypto_file', type=str, default='data/Poloniex_BTC_d.csv',
                        help='Caminho para o arquivo CSV da criptomoeda (padrão: data/Poloniex_BTC_d.csv)')
    parser.add_argument('--model', type=str, default="mlp", help='Tipo de modelo: MLPRegressor')
    parser.add_argument('--kfolds', type=int, default=5, help='Número de K-Folds para validação')
    parser.add_argument('--window_size', type=int, default=7, help='Tamanho da janela temporal')

    args = parser.parse_args()

    try:
        # logger.info("Iniciando pipeline...")
        # logger.info(f"Arquivo de dados: {args.crypto}")
        # logger.info(f"Modelo: {args.model}")
        # logger.info(f"K-Folds: {args.kfolds}")

        # # Carregar dados
        # logger.info("Carregando dados...")
        # X, y, _ = load_data(args.crypto)
        # logger.info(f"Dados carregados: X={X.shape}, y={y.shape}")

        # # Definir classe do modelo
        # model_class = MLP

        # # Treinar modelo
        # logger.info("Iniciando treinamento...")
        # model, metrics = train_model(X, y, model_class, args.kfolds)

        # # Extrair MSE das métricas
        # mse = metrics['mse_mean']
        # mae = metrics['mae_mean']
        # r2 = metrics['r2_mean']

        # logger.info(f"Modelo treinado com sucesso!")
        # logger.info(f"MSE: {mse:.6f} ± {metrics['mse_std']:.6f}")
        # logger.info(f"MAE: {mae:.6f} ± {metrics['mae_std']:.6f}")
        # logger.info(f"R²: {r2:.6f} ± {metrics['r2_std']:.6f}")

        # # Fazer predições de teste
        # logger.info("Fazendo predições de teste...")
        # test_predictions = model.predict(X[:5])
        # logger.info(f"Primeiras 5 predições: {test_predictions.flat[:5]}")
        # logger.info(f"Valores reais ao lado: {y[:5]}")
        # # logger.info(f"Diferenças previstas: {diff(test_predictions, y[:5]).flat[:5]}")

        # # Informações do modelo
        # model_info = model.get_feature_importance()
        # logger.info(f"Informações do modelo: {model_info}")

        featured_data = create_features(main_data.copy())

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
        
        # --- PASSO 4: Executar Teste de Hipótese para o Modelo Principal ---
        hypothesis_test_results = perform_hypothesis_test(backtest_results['strategy_return'])

        # --- PASSO 5: Coletar dados para Análise Comparativa (ANOVA) ---
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

        # --- PASSO 7: Exibir Resultados ---
        logger.info("\n" + "="*50 + "\nRESULTADOS FINAIS\n" + "="*50)

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
        print("\n--- Tabela de Desempenho do Modelo ---\n")
        print(summary_df.to_string(index=False))

        # Resultados do Teste de Hipótese
        print("\n--- Resultados do Teste de Hipótese (Teste-t) ---\n")
        print(f"Estatística t: {hypothesis_test_results['t_statistic']:.4f}")
        print(f"P-valor: {hypothesis_test_results['p_value']:.4f}")
        print(f"Conclusão: {hypothesis_test_results['conclusion']}")

        # Resultados da ANOVA
        print("\n--- Resultados da Análise de Variância (ANOVA) ---\n")
        print(f"P-valor do teste ANOVA: {anova_results['anova_p_value']:.4f}")
        if anova_results['tukey_df'] is not None:
            print("\nResultado do Teste Post-Hoc de Tukey:")
            print(anova_results['tukey_df'].to_string(index=False))
        else:
            print("\nTeste de Tukey não foi realizado (ANOVA não significativa).")


        generate_graph(args, backtest_results, predictions_df)    

        

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
