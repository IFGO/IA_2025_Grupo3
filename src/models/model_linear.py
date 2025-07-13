from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class Linear:
    def __init__(self):
        pass

    def create_pipeline() -> Pipeline:
        return Pipeline([('scaler', StandardScaler()), ('linear', LinearRegression())])