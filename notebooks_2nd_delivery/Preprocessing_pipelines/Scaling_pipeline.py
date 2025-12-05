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

        self.cols_to_scale_ = ["car_age", "mileage", "tax", "mpg", "engineSize", "previousOwners"]

        self.scaler_ = scalers[self.scaler_name]().fit(X[self.cols_to_scale_])
        return self

    def transform(self, X, **kwargs):
        
        X = X.copy()

        X_cols = X[self.cols_to_scale_]
        X_encoded = X[[col for col in X.columns if col not in self.cols_to_scale_]]

        X_scaled = self.scaler_.transform(X_cols)

        X_scaled = pd.DataFrame(X_scaled, columns= X_cols.columns, index= X_cols.index)
        
        return pd.concat([X_scaled, X_encoded], axis=1)
