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
        X_train = self.outliers_removal(X_train, **kwargs)

        #Imputing Missing Values
        X_train = self.missing_values_imputation(X_train, **kwargs)

        #Encoding the categorical columns
        X_train = self.encoding(X_train, **kwargs)

        #Scaling 
        X_train = self.scaling(X_train, **kwargs)

        return X_train
    
    def outliers_removal(self, X_train, X_val):

        # Interquartil Range Method
        if self.outlier_method == "IQR":
                
            #Probably we'll have to do the capping of outliers in train and validation data all inside thsi function
            # and follow the same reasoning for Missing values, encoding and scaling

            #Capping moderate outliers
                if len(self.mod_outliers_cols) > 0:
                    for col in self.mod_outliers_cols:

                        self.__dict__q1 = {}
                        self.__dict__q3 = {}

                        self.__dict__q1[col] = X_train[col].quantile(0.25)
                        self.__dict__q3[col] = X_train[col].quantile(0.75)
                        IQR = self.__dict__q3[col] - self.__dict__q1[col]
                        X_train[col] = np.clip(X_train[col], self.__dict__q1[col] - 1.5 * IQR , self.__dict__q3[col] + 1.5 * IQR)
                        #X_val[col] = np.clip(X_val[col], q_1 - 1.5 * IQR , q_3 + 1.5 * IQR)

                #Capping severate outliers        
                if len(self.sev_outliers_cols) > 0:    
                    for col in self.sev_outliers_cols:

                        self.__dict__q1[col] = X_train[col].quantile(0.25)
                        self.__dict__q3[col] = X_train[col].quantile(0.75)
                        IQR = self.__dict__q3[col] - self.__dict__q1[col]
                        X_train[col] = np.clip(X_train[col], self.__dict__q1[col] - 3 * IQR , self.__dict__q3[col] + 3 * IQR)

        



            

        
        
