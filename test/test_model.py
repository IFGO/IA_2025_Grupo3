import pytest
import numpy as np
from models.model import MLP
import torch

def test_model_shape():
    model = MLP(input_size=10)
    x = torch.randn(4, 10)
    y = model(x)
    assert y.shape == (4, 1)

def test_data_loader():
    from utils.data_loader import load_data
    X, y, _ = load_data("data/Poloniex_BTCUSDC_d.csv", window_size=5)
    assert len(X) == len(y)
    assert X.shape[1] == 5

def test_train_model():
    from trainer.trainer import train_model
    from models.model import MLP

    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    model, mse = train_model(X, y, MLP, kfolds=3)
    assert isinstance(mse, float)
