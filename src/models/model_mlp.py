from sklearn.neural_network import MLPRegressor

class MLP:
    def __init__(self):
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=2000,              # Aumentar para 2000
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,   # Aumentar validação
            n_iter_no_change=50,        # Mais paciência para convergir
            alpha=0.001,                # Regularização mais forte
            learning_rate_init=0.01,    # Taxa de aprendizado maior
            solver='adam',
            tol=1e-4                    # Tolerância para convergência
        )

        return model
        
