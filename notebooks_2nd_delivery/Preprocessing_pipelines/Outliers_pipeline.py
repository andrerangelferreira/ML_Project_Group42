import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutliersDealer(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 outlier_method = "z-score", # the default method
                 mod_outliers_cols = [],
                 sev_outliers_cols = [],
                 threshold=3, # Pick 2 or 3 as the threshold value of "z"
                 z_columns = [],
                 contamination_IF=0.05, 
                 random_state=42,
                 n_neighbors=20, 
                 contamination_LOF= 0.05,
                 model_columns = [], #columns selected for IF model or LOF model
                 **kwargs
                 ):
        
        self.outlier_method = outlier_method

        # IQR method parameters
        self.mod_outliers_cols = mod_outliers_cols
        self.sev_outliers_cols = sev_outliers_cols

        # Z-Score method parameters
        self.threshold = threshold
        self.z_columns = z_columns

        # Isolation Forest method parameters
        self.contamination_IF = contamination_IF
        self.random_state = random_state
        self.model_columns = model_columns

        # Local Outliers Factor method parameters
        self.n_neighbors = n_neighbors
        self.contamination_LOF = contamination_LOF
        self.model_columns = model_columns


    def fit(self, X_train, **kwargs):

        # Interquartil Range Method
        if self.outlier_method == "IQR":

            self.q1_ = {}
            self.q3_ = {}
            self.iqr_ = {}

            for col in set(self.mod_outliers_cols + self.sev_outliers_cols):
                q1 = X_train[col].quantile(0.25)
                q3 = X_train[col].quantile(0.75)
                iqr = q3 - q1

                self.q1_[col] = q1
                self.q3_[col] = q3
                self.iqr_[col] = iqr
        
        elif self.outlier_method == "z-score":

            self.means_ = {}
            self.stds_ = {}

            for col in self.z_columns:
                self.means_[col] = X_train[col].mean()
                self.stds_[col] = X_train[col].std()

        elif self.outlier_method == "Isolation_Forest":

            self.model = IsolationForest(
            contamination=self.contamination_IF,
            random_state=self.random_state
            )
            self.model.fit(X_train[self.model_columns])

        elif self.outlier_method == "LOF":
            self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination_LOF,
            novelty=True  # required for transform()
            )
            self.model.fit(X_train[self.model_columns])

        return self
    
    def transform(self, X, y, **kwargs):


        X = X.copy()
        y = y.copy()
            
        if self.outlier_method == "IQR":

            #Capping moderate outliers
            if len(self.mod_outliers_cols) > 0:
                for col in self.mod_outliers_cols:

                    X[col] = np.clip(X[col], 
                                     self.q1_[col] - 1.5 * self.iqr_[col], 
                                     self.q3_[col] + 1.5 * self.iqr_[col])

            #Capping severate outliers        
            if len(self.sev_outliers_cols) > 0:    
                for col in self.sev_outliers_cols:

                    X[col] = np.clip(X[col], 
                                     self.q1_[col] - 3 * self.iqr_[col],
                                     self.q3_[col] + 3 * self.iqr_[col])
            
            return X
                
        
        elif self.outlier_method == "z-score":

            for col in self.z_columns:

                X[col] = np.clip(X[col],
                                self.means_[col] - self.threshold * self.stds_[col],
                                self.means_[col] + self.threshold * self.stds_[col]
                            )
            return X
        
        elif self.outlier_method in ["Isolation_Forest", "LOF"]:
            preds = self.model.predict(X[self.model_columns])  # +1 = normal, -1 = outlier
            mask = preds == 1 #Storing the indexes from rows that are considered non outliers
            
            return X[mask], y[mask]
