import argparse
from utils.data_loader import load_data
from models.model import MLP
from trainer.trainer import train_model
from utils.logger import setup_logger

def main():
    """Função principal do pipeline de treinamento."""
    logger = setup_logger("CryptoMLP")

    parser = argparse.ArgumentParser(description="Crypto Price Predictor")
    parser.add_argument('--crypto', type=str, required=True, help='Caminho para CSV da criptomoeda')
    parser.add_argument('--model', type=str, default="mlp", help='Tipo de modelo: mlp')
    parser.add_argument('--kfolds', type=int, default=5, help='Número de K-Folds para validação')
    parser.add_argument('--window_size', type=int, default=10, help='Tamanho da janela temporal')

    args = parser.parse_args()

    try:
        logger.info("Iniciando pipeline...")
        logger.info(f"Arquivo de dados: {args.crypto}")
        logger.info(f"Modelo: {args.model}")
        logger.info(f"K-Folds: {args.kfolds}")

        # Carregar dados
        logger.info("Carregando dados...")
        X, y, _ = load_data(args.crypto, window_size=args.window_size)
        logger.info(f"Dados carregados: X={X.shape}, y={y.shape}")

        # Definir classe do modelo
        model_class = MLP

        # Treinar modelo
        logger.info("Iniciando treinamento...")
        model, metrics = train_model(X, y, model_class, args.kfolds)

        # Extrair MSE das métricas
        mse = metrics['mse_mean']
        mae = metrics['mae_mean']
        r2 = metrics['r2_mean']

        logger.info(f"Modelo treinado com sucesso!")
        logger.info(f"MSE: {mse:.6f} ± {metrics['mse_std']:.6f}")
        logger.info(f"MAE: {mae:.6f} ± {metrics['mae_std']:.6f}")
        logger.info(f"R²: {r2:.6f} ± {metrics['r2_std']:.6f}")

        # Fazer predições de teste
        logger.info("Fazendo predições de teste...")
        test_predictions = model.predict(X[:5])
        logger.info(f"Primeiras 5 predições: {test_predictions}")
        logger.info(f"Valores reais: {y[:5]}")

        # Informações do modelo
        model_info = model.get_feature_importance()
        logger.info(f"Informações do modelo: {model_info}")

        logger.info("Pipeline finalizado com sucesso!")

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
