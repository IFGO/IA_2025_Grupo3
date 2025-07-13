import pandas as pd
import requests
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def prepare_data(crypto_symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados de criptomoedas para análise.

    Args:
        df (pd.DataFrame): DataFrame com dados brutos.

    Returns:
        pd.DataFrame: DataFrame preparado com colunas renomeadas e indexado por data.
    """
    df.rename(columns={
        'Date': 'date', 'Symbol': 'symbol', 'Open': 'open', 'High': 'high', 
        'Low': 'low', 'Close': 'close', 'Volume USDC': 'volume_usdc', f'Volume {crypto_symbol}': 'volume_crypto'
    }, inplace=True)
    
    # Selecionar e reordenar colunas de interesse
    df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume_usdc', 'volume_crypto']]
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df.sort_values(by='date', inplace=True)
    df.set_index('date', inplace=True)
    
    return df

def download_crypto_data(crypto_symbol: str) -> Optional[pd.DataFrame]:
    """
    Baixa dados históricos diários de criptomoedas do cryptodatadownload.com.
    Args:
        crypto_symbol (str): O símbolo da criptomoeda (ex: 'BTC').

    Returns:
        Optional[pd.DataFrame]: DataFrame com dados históricos ou None se falhar.
    """
    base_url = "https://www.cryptodatadownload.com/cdd/"
    filename = f"Poloniex_{crypto_symbol}USDC_d.csv"
    url = f"{base_url}{filename}"
    local_path = f"./data/{filename}"
    
    logger.info(f"Tentando baixar dados para {crypto_symbol} de {url}")
    
    try:
        # Criar diretório data se não existir
       
        os.makedirs("./data", exist_ok=True)
        
        # Verificar se arquivo já existe localmente
        if not os.path.exists(local_path):
            logger.info(f"Arquivo local não encontrado. Baixando de {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            # Salvar arquivo localmente
            with open(local_path, 'w') as f:
                f.write(response.text)
            logger.info(f"Arquivo salvo em {local_path}")
        else:
            logger.info(f"Usando arquivo local existente: {local_path}")
        
        # Ler arquivo local
        df = pd.read_csv(local_path, skiprows=1)

        # unix,date,symbol,open,high,low,close,Volume USD,Volume BTC

        df = prepare_data(crypto_symbol=crypto_symbol, df=df.copy())
    
        logger.info(f"Dados para {crypto_symbol} processados com sucesso.")
        
        # remove unnecessary columns
#         X = df.drop(columns=['date', 'symbol', 'close', 'close_scaled']).values
#         y = df['close_scaled'].values

#         return np.array(X), np.array(y), scaler        
        return df

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"Erro HTTP ao baixar {crypto_symbol}: {http_err}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao processar {crypto_symbol}: {e}")
        return None

def read_crypto_data(crypto_symbol: str, file_path: str) -> Optional[pd.DataFrame]:
    """
    Lê dados de criptomoedas de um arquivo CSV.

    Args:
        file_path (str): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame com os dados lidos.
    """
    try: 
        os.makedirs("./data", exist_ok=True)
        
        if not os.path.exists(file_path):
           raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        else:
            logger.info(f"Usando arquivo local existente: {file_path}")
        
        # Ler arquivo local
        df = pd.read_csv(file_path, skiprows=1)    
        df = prepare_data(crypto_symbol=crypto_symbol, df=df.copy())        

        return df 
    except Exception as e:
        logger.error(f"Erro ao ler dados de {file_path}: {e}")
        raise
