import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class ScalingDealer(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 scaler_name="robust", 
                 **kwargs
                 ):

        self.scaler_name = scaler_name.lower()
        self.kwargs = kwargs

    def fit(self, X):

        scalers = {
            "robust": RobustScaler,
            "minmax": MinMaxScaler,
            "standard": StandardScaler
        }

        if self.scaler_name not in scalers:
            raise ValueError(
                f"Invalid scaler_name '{self.scaler_name}'. "
            )

        self.scaler_ = scalers[self.scaler_name](**self.kwargs).fit(X)
        return self

    def transform(self, X):
        
        X = X.copy()

        return self.scaler_.transform(X)
