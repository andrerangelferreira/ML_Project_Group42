import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutliersDealer(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 outlier_method = "Z_Score", # the default method
                 mod_outliers_cols = [],
                 sev_outliers_cols = [],
                 threshold=3, # Pick 2 or 3 as the threshold value of "z"
                 contamination_IF=0.05, 
                 random_state=42,
                 n_neighbors=20, 
                 contamination_LOF=0.05,
                 **kwargs
                 ):
        
        self.outlier_method = outlier_method

        # IQR method parameters
        self.mod_outliers_cols = mod_outliers_cols
        self.sev_outliers_cols = sev_outliers_cols

        # Z-Score method parameters
        self.threshold = threshold

        # Isolation Forest method parameters
        self.contamination_IF = contamination_IF
        self.random_state = random_state

        # Local Outliers Factor method parameters
        self.n_neighbors = n_neighbors
        self.contamination_LOF = contamination_LOF


    def fit(self, X_train, **kwargs):

        # Interquartil Range Method
        if self.outlier_method == "IQR":

            self.q1_ = X_train.quantile(0.25)
            self.q3_ = X_train.quantile(0.75)
            self.iqr_ = self.q3_ - self.q1_

        
        elif self.outlier_method == "Z_Score":
            self.means_ = X_train.mean()
            self.stds_ = X_train.std()

        elif self.outlier_method == "Isolation_Forest":

            self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
            )
            self.model.fit(X_train)

        elif self.outlier_method == "LOF":
            self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination_LOF,
            novelty=True  # required for transform()
            )
            self.model.fit(X_train)

        return self
    
    def transform(self, X, y, **kwargs):


        X = X.copy()
        y = y.copy()
            
        if self.outlier_method == "IQR":

            #Capping moderate outliers
            if len(self.mod_outliers_cols) > 0:
                for col in self.mod_outliers_cols:

                    X[col] = np.clip(X[col], self.q1_[col] - 1.5 * self.iqr_[col] , self.q3_[col] + 1.5 * self.iqr_[col])

            #Capping severate outliers        
            if len(self.sev_outliers_cols) > 0:    
                for col in self.sev_outliers_cols:

                    X[col] = np.clip(X[col], self.q1_[col] - 3 * self.iqr_[col] , self.q3_[col] + 3 * self.iqr_[col])
            
            return X, y
                
        
        elif self.outlier_method == "Z-score":

            z = (X - self.means_) / self.stds_
            mask = (np.abs(z) < self.threshold).all(axis=1)
            return X[mask], y[mask]
        
        elif self.outlier_method in ["Isolation_Forest", "LOF"]:
            preds = self.model.predict(X)  # +1 = normal, -1 = outlier
            mask = preds == 1
            
            return X[mask], y[mask]
