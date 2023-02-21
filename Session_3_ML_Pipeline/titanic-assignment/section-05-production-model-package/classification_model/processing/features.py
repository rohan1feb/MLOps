from sklearn.base import BaseEstimator, TransformerMixin

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y = None):
        return self
        
    def transform(self, X):
        temp = X.copy()
        for var in self.variables:
            temp[var] = temp[var].str[0]
        return temp
