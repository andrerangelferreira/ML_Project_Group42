import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

class Preprocessor_Pipeline(BaseEstimator):
    
    def __init__(self, 
                 outlier_method = "IQR",
                 mod_outliers_cols = [],
                 sev_outliers_cols = [],

                 
                 ):
        
        self.outlier_method = outlier_method 
        self.mod_outliers_cols = mod_outliers_cols
        self.sev_outliers_cols = sev_outliers_cols

    def fit(self, X_train, **kwargs):

        #Removing outliers
        X_train = self.outliers_removal(X_train)

        #Imputing Missing Values
        X_train = self.missing_values_imputation(X_train)

        #Encoding the categorical columns
        X_train = self.encoding(X_train)

        #Scaling 
        X_train = self.scaling(X_train)

        return X_train
    
    def outliers_removal(self, X_train):

        # Interquartil Range Method
        if self.outlier_method == "IQR":

            #Capping moderate outliers
                if len(self.mod_outliers_cols) > 0:
                    for col in self.mod_outliers_cols:
                        
                        q_1 = X_train[col].quantile(0.25)
                        q_3 = X_train[col].quantile(0.75)
                        IQR = q_3 - q_1
                        X_train[col] = np.clip(X_train[col], q_1 - 1.5 * IQR , q_3 + 1.5 * IQR)

                #Capping severate outliers        
                if len(self.sev_outliers_cols) > 0:    
                    for col in self.sev_outliers_cols:

                        q_1 = X_train[col].quantile(0.25)
                        q_3 = X_train[col].quantile(0.75)
                        IQR = q_3 - q_1
                        X_train[col] = np.clip(X_train[col], q_1 - 3 * IQR , q_3 + 3 * IQR)

        



            

        
        
