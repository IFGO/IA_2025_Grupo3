import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

def train_model(X: np.ndarray, y: np.ndarray, model_class, kfolds: int = 5) -> Tuple[nn.Module, float]:
    """
    Train MLP model with k-fold validation.

    Args:
        X: Input features.
        y: Targets.
        model_class: Model class.
        kfolds: Number of folds.

    Returns:
        Best model and average MSE.
    """
    try:
        kf = KFold(n_splits=kfolds, shuffle=True)
        mse_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Fold {fold + 1}/{kfolds}")
            model = model_class(X.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            X_train = torch.tensor(X[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y[train_idx], dtype=torch.float32).view(-1, 1)
            X_val = torch.tensor(X[val_idx], dtype=torch.float32)
            y_val = torch.tensor(y[val_idx], dtype=torch.float32).view(-1, 1)

            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                mse = criterion(val_pred, y_val).item()
                mse_scores.append(mse)

        avg_mse = np.mean(mse_scores)
        logger.info(f"Média MSE: {avg_mse:.4f}")
        return model, avg_mse

    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        raise
