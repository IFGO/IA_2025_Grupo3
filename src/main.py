import argparse
from utils.data_loader import load_data
from models.model import MLP
from trainer.trainer import train_model
from utils.logger import setup_logger

def main():
    logger = setup_logger("CryptoMLP")

    parser = argparse.ArgumentParser(description="Crypto Price Predictor")
    parser.add_argument('--crypto', type=str, required=True, help='Caminho para CSV da criptomoeda')
    parser.add_argument('--model', type=str, default="mlp", help='Tipo de modelo: mlp')
    parser.add_argument('--kfolds', type=int, default=5, help='Número de K-Folds para validação')

    args = parser.parse_args()

    logger.info("Iniciando pipeline...")

    X, y, _ = load_data(args.crypto)
    model_class = MLP
    model, mse = train_model(X, y, model_class, args.kfolds)

    logger.info("Pipeline finalizado com sucesso!")

if __name__ == "__main__":
    main()
