import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class ScalingDealer(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 scaler_name="robust", 
                 **kwargs
                 ):

        self.scaler_name = scaler_name

    def fit(self, X, **kwargs):

        scalers = {
            "robust": RobustScaler,
            "minmax": MinMaxScaler,
            "standard": StandardScaler
        }

        self.scaler_ = scalers[self.scaler_name]().fit(X)
        return self

    def transform(self, X, **kwargs):
        
        X = X.copy()

        X_scaled = self.scaler_.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
