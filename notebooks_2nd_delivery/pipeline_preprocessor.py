import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor_Pipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 outlier_method = "IQR",
                 mod_outliers_cols = [],
                 sev_outliers_cols = [],

                 
                 ):
        
        self.outlier_method = outlier_method 
        self.mod_outliers_cols = mod_outliers_cols
        self.sev_outliers_cols = sev_outliers_cols

    def fit(self, X_train, **kwargs):


        # ------- OUTLIERS ------- #

        # Interquartil Range Method
        if self.outlier_method == "IQR":

            self.q1_ = X_train.quantile(0.25)
            self.q3_ = X_train.quantile(0.75)
            self.iqr_ = self.q3_ - self.q1_

        elif self.outlier_method == "Z-score":
            uhi

        return X_train
    

    def transform(self, X, **kwargs):

        X = X.copy()
            
        if self.outlier_method == "IQR":

            #Capping moderate outliers
            if len(self.mod_outliers_cols) > 0:
                for col in self.mod_outliers_cols:

                    X[col] = np.clip(X[col], self.q1_[col] - 1.5 * self.iqr_[col] , self.q3_[col] + 1.5 * self.iqr_[col])

            #Capping severate outliers        
            if len(self.sev_outliers_cols) > 0:    
                for col in self.sev_outliers_cols:

                    X[col] = np.clip(X[col], self.q1_[col] - 3 * self.iqr_[col] , self.q3_[col] + 3 * self.iqr_[col])


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

        



            

        
        
