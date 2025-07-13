class Args:
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
    crypto: str = 'BTC'  # Ex: 'BTC', 'ETH', 'LTC', 'XRP'
    crypto_file: str = 'data/Poloniex_BTC_d.csv'  # Caminho para o arquivo CSV da criptomoeda
    model: str = 'mlp'   # Opções: 'mlp', 'linear', 'poly'
    poly_degree: int = 2 # Grau para o modelo polinomial (usado se model='poly')
    kfolds: int = 5
    investment: float = 1000.0
    window_size: int = 7  # Tamanho da janela temporal para features
    # Lista de criptos para análise estatística comparativa
    crypto_list_for_analysis: List[str] = ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']