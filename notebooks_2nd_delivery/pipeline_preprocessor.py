import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor_Pipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 outlier_removal= OutliersDealer(),
                 imputer= MissingValuesDealer(),
                 encoder= EncodingDealer(),
                 scaler= ScalingDealer()
                ):
        
        self.outlier_removal = outlier_removal 
        self.imputer = imputer
        self.encoder = encoder
        self.scaler = scaler

    def fit(self, X_train, **kwargs):
       
        #Calculating the outlier methods
        X_train = self.outlier_removal.fit(X_train, **kwargs)

        #Calculating the metrics for missing values imputing
        X_train = self.imputer.fit(X_train, **kwargs)

        X_train = self.encoder.fit(X_train, **kwargs)

        X_train = self.scaler.fit(X_train, **kwargs)

    
    def transform(self, X, **kwargs):

        #Treating the outliers
        X = self.outlier_removal.transform(X, **kwargs) 

        #Imputing Missing values
        X = self.imputer.transform(X, **kwargs) 

        X = self.encoder.transform(X, **kwargs)

        X = self.scaler.transform(X, **kwargs)




class missing_categorical(BaseEstimator, TransformerMixin):

    def __init__(self, brand_col="Brand", numeric_cols=None, n_neighbors=5):
        self.brand_col = brand_col
        self.numeric_cols = numeric_cols
        self.n_neighbors = n_neighbors

        # Internal storage for scalers & imputers by brand
        self.scalers_ = {}
        self.imputers_ = {}

    def fit(self, X, y=None):
        X = X.copy()

        for brand, df_brand in X.groupby(self.brand_col):
            scaler = StandardScaler()
            imputer = KNNImputer(n_neighbors=self.n_neighbors)

            # Fit scaler + imputer on numeric cols for this brand
            scaled = scaler.fit_transform(df_brand[self.numeric_cols])
            imputer.fit(scaled)

            # Store per-brand models
            self.scalers_[brand] = scaler
            self.imputers_[brand] = imputer

        return self

    def transform(self, X):
        X = X.copy()
        result_list = []

        for brand, df_brand in X.groupby(self.brand_col):
            df_temp = df_brand.copy()

            # If brand unseen in training → fallback to global scaler
            if brand not in self.scalers_:
                # Just return imputed but unscaled (safe fallback)
                imputer = KNNImputer(n_neighbors=self.n_neighbors)
                df_temp[self.numeric_cols] = imputer.fit_transform(df_temp[self.numeric_cols])
                result_list.append(df_temp)
                continue

            scaler = self.scalers_[brand]
            imputer = self.imputers_[brand]

            # Scale → Impute → Reverse scale
            scaled = scaler.transform(df_temp[self.numeric_cols])
            imputed_scaled = imputer.transform(scaled)
            df_temp[self.numeric_cols] = scaler.inverse_transform(imputed_scaled)

            result_list.append(df_temp)

        # Concatenate and preserve original row order
        X_clean = pd.concat(result_list).loc[X.index]

        return X_clean

        



            

        
        
