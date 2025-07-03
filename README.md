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

```
crypto_mlp/
├── data/                         # Coloque aqui seu arquivo CSV
├── models/
│   └── model.py                 # Arquitetura da MLP
├── trainer/
│   └── trainer.py               # Treinamento e validação
├── utils/
│   ├── data_loader.py           # Carregamento e pré-processamento
│   └── logger.py                # Configuração de logging
├── teste/
│   └── test_model.py            # Testes automatizados
├── main.py                      # Script principal com argparse
├── requirements.txt
├── .coveragerc
└── README.md
```

---

## 📊 Exemplo de Execução

```bash
python main.py --crypto data/coinbase_BTCUSD_1h.csv --model mlp --kfolds 5
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
