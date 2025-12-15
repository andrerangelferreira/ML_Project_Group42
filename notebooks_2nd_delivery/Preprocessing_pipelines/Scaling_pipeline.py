import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

"""
The ScalingDealer class is responsible for scaling the dataset after all
previous preprocessing transformations have been applied and before the
data is passed to the model.

The class includes a parameter, "scaler_name", which allows the user to
choose the scaling method to apply. The available options are
StandardScaler, MinMaxScaler, and RobustScaler.

Additionally, the class handles cases where One-Hot Encoding is used in
the encoding class. In this scenario, the encoded categorical columns are
not scaled. However, when other encoding methods are applied (e.g., target encoding),
the resulting encoded features are treated as numerical variables and
are therefore scaled before being passed to the model.

"""

class ScalingDealer(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 scaler_name="robust", 
                 one_hot = False,
                 **kwargs
                 ):

        self.scaler_name = scaler_name
        self.one_hot = one_hot

    def fit(self, X, **kwargs):

        scalers = {
            "robust": RobustScaler,
            "minmax": MinMaxScaler,
            "standard": StandardScaler
        }

        if self.one_hot == True:
            
            self.cols_to_scale_ = ["car_age", "mileage", "tax", "mpg", "engineSize", "previousOwners"]

            self.scaler_ = scalers[self.scaler_name]().fit(X[self.cols_to_scale_])

        else:

            self.scaler_ = scalers[self.scaler_name]().fit(X)

        return self

    def transform(self, X, **kwargs):
        
        X = X.copy()

        if self.one_hot == True:

            X_cols = X[self.cols_to_scale_]
            X_encoded = X[[col for col in X.columns if col not in self.cols_to_scale_]]

            X_scaled = self.scaler_.transform(X_cols)

            X_scaled = pd.DataFrame(X_scaled, columns= X_cols.columns, index= X_cols.index)
            
            return pd.concat([X_scaled, X_encoded], axis=1)
        
        else:

            X_scaled = self.scaler_.transform(X)

            X_scaled = pd.DataFrame(X_scaled, columns= X.columns, index= X.index)
            
            return X_scaled