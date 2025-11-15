import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

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


    
        



            

        
        
