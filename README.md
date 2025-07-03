# 📈 Crypto Price Prediction using MLP

Este projeto usa uma rede neural do tipo **MLP (Multi-Layer Perceptron)** para prever o preço de fechamento de criptomoedas com base em dados históricos.

## 🚀 Funcionalidades

- Pré-processamento de dados históricos de criptomoedas (Coinbase)
- Treinamento com validação cruzada (K-Fold)
- Arquitetura modular em Python
- Interface via linha de comando (CLI)
- Testes automatizados com `pytest` e relatório de cobertura

---

## 📂 Estrutura do Projeto

```.
├── data
│   └── Poloniex_BTCUSDC_d.csv
├── pytest.ini
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   └── model.py
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils
│       ├── data_loader.py
│       ├── __init__.py
│       └── logger.py
└── test
    ├── __init__.py
    └── test_model.py
```

---

## 📊 Exemplo de Execução

```bash
python src/main.py --crypto data/Poloniex_BTCUSDC_d.csv --model mlp --kfolds 5
```

Parâmetros:

- `--crypto`: Caminho para o CSV baixado do site da [CryptoDataDownload](https://www.cryptodatadownload.com/)
- `--model`: Tipo de modelo (`mlp` por padrão)
- `--kfolds`: Número de divisões para validação cruzada (padrão = 5)

---

## ✅ Executar Testes

```bash
pytest --cov=./ --cov-report=term-missing
```

Gera um relatório de cobertura de testes indicando quais partes do código foram exercitadas.

---

## 📥 Download de Dados

1. Acesse: [CryptoDataDownload - Coinbase](https://www.cryptodatadownload.com/)
2. Baixe um arquivo como: `Coinbase_BTCUSD_1h.csv`
3. Coloque-o na pasta `data/` do projeto

---

## 🧪 Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contribuições

Contribuições são bem-vindas! Abra um *pull request* ou crie uma *issue*.

---

## 🛠️ Tecnologias Usadas

- Python 3.8+
- PyTorch
- NumPy, Pandas
- Scikit-learn
- pytest, pytest-cov

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
