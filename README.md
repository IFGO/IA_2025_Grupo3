# 📈 Crypto Price Predictor

> Trabalho Final Módulo I (IFG) - Especialização em Inteligência Artificial Aplicada  
> Professores: Dr. Eduardo Noronha, Me. Otávio Calaça, Dr. Eder Brito  
> Alunos: Fabio Paula, Raony Nogueira, Rafael Fideles, Marcelo 

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

## 📁 Estrutura de Diretórios

```.
data-crypto-ai/
│
├── data/                       # Arquivos CSV de criptomoedas
├── figures/                    # Gráficos salvos em 150 dpi
├── models/                     # Modelos treinados e serializados
├── tests/                      # Testes automatizados (pytest)
│
├── main.py                     # Script principal com interface CLI
├── data_load.py                # Módulo de carregamento e download de dados
├── features.py                 # Extração e engenharia de features
├── models.py                   # Treinamento e avaliação de modelos
├── analysis.py                 # Análises estatísticas e visuais
├── utils.py                    # Funções auxiliares e métricas
│
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

## ⚙️ Parâmetros CLI

python main.py [--param valor] ...

| Parâmetro | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| --dwn-not-data-set | bool | False | Se True, baixa o dataset mais recente |
| --investment | float | 1000.0 | Valor inicial do investimento em USD |
| --poly_degree | int | 2 | Grau do polinômio (se model=poly) |
| --show_anova | bool | False | Executa análise ANOVA entre criptomoedas |
| --crypto_list_for_analysis | list | ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE'] | Criptomoedas para análise estatística |
| --crypto | str | obrigatório | Nome da criptomoeda (ex: BTC) |
| --model | str | mlp | Tipo de modelo (mlp, poly, linear) |
| --kfolds | int | 5 | Número de Folds na validação cruzada |
| --window_size | int | 7 | Janela temporal de features |

## ▶️ Como Executar

Caso deseje executar o projeto localmente, siga os passos fornecidos.

### Requisitos

- **Python**: versão **3.8 até 3.13.3**
- **pip** instalado
- **git** para clonar o repositório

### 1. Clonar o repositório

```bash
git clone https://github.com/Fabioaugustmp/data-crypto-ai -b feat/brainstorm
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

```bash
python main.py --crypto BTC --crypto_file data/Poloniex_BTC_d.csv --model mlp --investment 1000 --kfolds 5
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

<p>Desenvolvido por: Fabio Paula, Raony Nogueira, Rafael Fideles, Marcelo<p>