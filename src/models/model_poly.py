from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class Poly:
    def __init__(self, poly_degree):        
       model =  PolynomialFeatures(degree=poly_degree, include_bias=False)
        
    def create_pipeline() -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', model),
            ('linear', LinearRegression())
        ])    

    