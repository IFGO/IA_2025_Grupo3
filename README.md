# 📈 Crypto Price Predictor

> Trabalho Final Módulo I (IFG) - Especialização em Inteligência Artificial Aplicada  
> Professores: Dr. Eduardo Noronha, Me. Otávio Calaça, Dr. Eder Brito  
> Alunos: Fabio Paula, Raony Nogueira, Rafael Fideles, Marcelo Carvalho

Este projeto desenvolve um sistema completo de previsão de preços de criptomoedas utilizando redes neurais e modelos de regressão. O objetivo é analisar dados históricos, construir modelos preditivos, realizar comparações estatísticas e avaliar a rentabilidade de estratégias de investimento automatizadas.

## 🚀 Funcionalidades

- Download e carregamento de datasets históricos do [CryptoDataDownload](https://www.cryptodatadownload.com/data/poloniex/)
- Treinamento de modelos de regressão como:
  - MLPRegressor (rede neural)
  - Regressão Linear
  - Regressão Polinomial (graus 2 a 10)
- Validação cruzada K-Fold
- Cálculo e simulação de lucros com reinvestimento
- Análises estatísticas completas com ANOVA, testes de hipótese e gráficos
- Comparação entre criptomoedas e entre modelos
- CLI configurável com `argparse`
- Testes automatizados com `pytest` + cobertura com `pytest-cov`
- Gráficos salvos em alta resolução (`figures/`, mínimo 150dpi)

![Hint example](hint.gif)

## 📁 Estrutura de Diretórios

```.
data-crypto-ai/
├── data                              # Arquivos CSV de criptomoedas
│   └── Poloniex_BTCUSD_d.csv
├── figures                           # Gráficos salvos em 150 dpi
│   ├── closing_price_boxplots.png
│   ├── closing_price_histograms.png
│   ├── crypto_analysis_comparison_BTC_25-07-14_04.31.31.png
│   ├── daily_volatility_BTC.png
│   └── historical_closing_prices.png
├── pytest.ini
├── README.md                         # Este arquivo com descrição do projeto
├── .gitignore                        # Arquivos e pastas ignorados pelo Git
├── requirements.txt                  # Dependências do projeto
├── setup.py
├── src
│   ├── main.py                       # Script principal com interface CLI
│   ├── models                        # Modelos treinados e serializados
│   │   └── model.py
│   ├── trainer                       # Treinamento e avaliação de modelos
│   │   └── trainer.py
│   ├── utils                         # Funções utilitárias e métricas
│   │   ├── data_loader.py            # Módulo de carregamento e download de dados
│   │   ├── features.py               # Extração e engenharia de features
│   │   ├── logger.py
│   │   ├── perform.py
│   │   └── statistical.py
│   └── views
│       ├── graph.py
│       └── table.py
├── tables                            # Tabelas de resultados e estatísticas
│   ├── Coef.Variation.csv
│   ├── Dispersion measures for - BTC.csv
│   └── Summary statistics for - BTC.csv
└── tests                             # Testes automatizados
    ├── test_data_loader.py
    ├── test_features.py
    ├── test_main.py
    ├── test_model.py
    ├── test_table.py
    └── test_trainer.py
```

## 📁 Estrutura de Pastas

O projeto organiza os resultados e os dados em três pastas principais, com o objetivo de manter os arquivos bem separados e acessíveis conforme o tipo de informação:

### 📊 `figures/`
Esta pasta contém todos os **gráficos gerados** durante as análises estatísticas e modelagens.  
- Os gráficos representam visualmente os desempenhos dos modelos aplicados, retornos das criptomoedas, comparações estatísticas, entre outros.
- Os arquivos são salvos em formatos como `.png` ou `.html` (no caso de gráficos interativos).
- Cada gráfico é nomeado com base na moeda e no tipo de análise realizada.
- Cada gráfico possui uma resolução base de 150 dpi

### 📈 `data/`
Esta pasta armazena os **datasets brutos e processados**, utilizados durante as análises e modelagens.
- Os arquivos estão em formato `.csv`, com preços históricos de criptomoedas obtidos do [cryptodatadownload.com](https://www.cryptodatadownload.com).
- Inclui dados de múltiplas criptomoedas, podendo conter informações como data, preço de fechamento, volume, entre outros.

### 📋 `tables/`
Nesta pasta ficam os **resultados numéricos das análises**, salvos em arquivos `.csv`.
- Cada arquivo representa uma **análise estatística específica** (ex: teste t, ANOVA, métricas de validação dos modelos).
- Os arquivos são segregados por tipo de análise e, quando aplicável, por criptomoeda.
- Facilitam a inspeção, comparação e reuso dos resultados para relatórios ou apresentações.

---

## ⚙️ Parâmetros CLI

python main.py [--param valor] ...

| Parâmetro                   | Tipo   | Padrão                                      | Descrição                                                                 |
|----------------------------|--------|---------------------------------------------|---------------------------------------------------------------------------|
| --dwn-not-data-set         | bool   | False                                       | Se True, baixa o dataset mais recente do [cryptodatadownload.com](https://www.cryptodatadownload.com) |
| --investment               | float  | 1000.0                                      | Valor inicial do investimento em USD                                     |
| --poly_degree              | int    | 2                                           | Grau do polinômio (se model=poly)                                        |
| --show_anova               | bool   | False                                       | Se True, executa análise ANOVA entre criptomoedas                        |
| --crypto_list_for_analysis| list   | ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']        | Criptomoedas para análise estatística comparativa                        |
| --crypto                   | str    | BTC (obrigatório)                           | Sigla da criptomoeda para análise (ex: BTC)                              |
| --model                    | str    | none                                        | Tipo de modelo: MLPRegressor, poly, linear, etc.                         |
| --kfolds                   | int    | 5                                           | Número de Folds para validação cruzada                                   |
| --window_size              | int    | 7                                           | Tamanho da janela temporal                                               |
| --statistical              | bool   | False                                       | Se True, analisa 10 moedas no diretório `data` gerando gráficos e CSVs   |
| --interative_graph         | bool   | False                                       | Se True, exibe gráfico interativo (não será salvo)                        |
| --analyse-cryptos          | bool   | False                                       | Se True, executa análise de 10 criptomoedas predefinidas                 |


## ▶️ Como Executar

Caso deseje executar o projeto localmente, siga os passos fornecidos.

### Requisitos

- **Python**: versão **3.8 até 3.13.3**
- **pip** instalado
- **git** para clonar o repositório

### 1. Clonar o repositório

```bash
git clone https://github.com/Fabioaugustmp/data-crypto-ai
cd data-crypto-ai
```

### 2. Criar e ativar um ambiente virtual (recomendado)

**Para sistemas Unix/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Para sistemas Windows:**
```bash
python -m venv venv
.
env\Scripts ctivate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Executar um exemplo de previsão
 - 4.1 Modelo MLP

```bash
python main.py --crypto BTC --model mlp --investment 1000 --kfolds 5
```

  - 4.2 Todos os Modelos

```bash
python main.py --crypto BTC --investment 1000 --kfolds 5
```

### 5. Rodar com análise estatística e ANOVA

```bash
python main.py --show_anova True --crypto_list_for_analysis BTC ETH LTC XRP DOGE
```
## 🧪 Executar Testes

pytest --cov=.

## 📊 Modelos Suportados

- mlp: Rede Neural Multicamadas com MLPRegressor
- linear: Regressão Linear
- poly: Regressão Polinomial (grau ajustável com --poly_degree)

## 📈 Métricas e Análises Geradas

- Medidas descritivas e de dispersão
- Gráficos de linha, boxplots e histogramas
- Simulação de lucro com reinvestimento diário
- Comparação entre modelos via:
  - Diagrama de dispersão
  - Correlação
  - Equação do regressor
  - Erro padrão
  - Gráfico de lucro acumulado
- Testes de hipótese (nível de significância 5%)
- ANOVA + testes post hoc entre moedas e agrupamentos

## 🧹 Boas Práticas Aplicadas

- black, ruff, flake8 para linting e formatação
- Modularização (data_load.py, models.py, etc.)
- logging e type hints aplicados
- Gráficos salvos automaticamente em figures/
- Testes automatizados com pytest-cov

## 🧠 Sobre o Projeto

Este projeto foi desenvolvido como trabalho final do Módulo I da pós-graduação em Inteligência Artificial Aplicada, e demonstra o uso de IA em aplicações financeiras, combinando Data Science, Machine Learning e Estatística.

## 📧 Créditos

> Fabio Paula -  <a href="mailto:fabioaugustomarquespaula@gmail.com">fabioaugustomarquespaula@gmail.com</a>

> Raony Nascimento - <a href="mailto:nascimento.raony@gmail.com">nascimento.raony@gmail.com</a>

> Rafael Fideles - <a href="mailto:rafaelfideles@live.com@gmail.com">rafaelfideles@live.com@gmail.com</a>

> Marcelo Carvalho - <a href="mailto:mcarvalho.eng@gmail.com@gmail.com">mcarvalho.eng@gmail.com@gmail.com</a>
