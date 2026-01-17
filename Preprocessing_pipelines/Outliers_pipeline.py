import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutliersDealer(BaseEstimator, TransformerMixin):
    """
    Custom outlier handling transformer compatible with scikit-learn pipelines.

    This class supports multiple outlier detection and treatment strategies:
        - IQR-based capping (moderate and severe outliers)
        - Z-score capping
        - Log-transform + Z-score
        - Isolation Forest (model-based)
        - Local Outlier Factor (LOF)

    Depending on the selected method, outliers are either capped,
    transformed, or adjusted using unsupervised anomaly detection models.
    Outliers are never removed; instead, values are capped to preserve
    dataset size and alignment.

    Parameters
    ----------
    outlier_method : str, default="z_score"
        Outlier handling strategy to apply.
        Options: {"IQR", "z_score", "log", "Isolation_Forest", "LOF"}.

    mod_outliers_cols : list of str, default=[]
        Columns where moderate outliers are capped using the IQR rule
        (1.5 × IQR).

    sev_outliers_cols : list of str, default=[]
        Columns where severe outliers are capped using a stricter IQR
        rule (3 × IQR).

    threshold : int or float, default=3
        Z-score threshold used to cap values when using z-score-based
        methods.

    z_columns : list of str, default=[]
        Columns to which Z-score capping is applied.

    log_columns : list of str, default=[]
        Columns that are log-transformed before optional Z-score capping.

    log_z_columns : list of str, default=[]
        Columns that receive Z-score capping after log transformation.

    contamination_IF : float, default=0.05
        Proportion of expected outliers for Isolation Forest.

    random_state : int, default=42
        Random seed used for model-based outlier detection.

    n_neighbors : int, default=20
        Number of neighbors used in Local Outlier Factor.

    contamination_LOF : float, default=0.05
        Proportion of expected outliers for LOF.

    model_columns : list of str, default=[]
        Subset of numerical columns used by model-based methods
        (Isolation Forest and LOF).

    Notes
    -----
    - Model-based methods cap detected outliers instead of removing them.
    - Designed to work seamlessly inside custom preprocessing pipelines.

    The transformer is designed to be used inside preprocessing pipelines
    without breaking DataFrame structure.
    """

    def __init__(self, 
                 outlier_method = "z_score", # Outlier handling strategy
                 mod_outliers_cols = [],     # Columns with moderate outliers (IQR)
                 sev_outliers_cols = [],     # Columns with severe outliers (IQR)
                 threshold=3, # Pick 2 or 3 as the threshold value of "z"
                 z_columns = [],             # Columns for Z-score method
                 log_z_columns = [],         # Columns for log + Z-score
                 log_columns = [],           # Columns for log-transform
                 contamination_IF=0.05,      # Isolation Forest contamination
                 random_state=42,
                 n_neighbors=20,             # LOF n_neighbors
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

        # Log-transform parameters
        self.log_columns = log_columns
        self.log_z_columns = log_z_columns

        # Isolation Forest method parameters
        self.contamination_IF = contamination_IF
        self.random_state = random_state
        self.model_columns = model_columns

        # Local Outliers Factor method parameters
        self.n_neighbors = n_neighbors
        self.contamination_LOF = contamination_LOF
        self.model_columns = model_columns


    def fit(self, X_train, y = None,  **kwargs):
        """
        Fit the selected outlier handling strategy.

        Statistical methods (IQR, Z-score, log-based) compute summary statistics.
        Model-based methods (Isolation Forest, LOF) fit an unsupervised anomaly
        detection model on the training data.

        Parameters
        ----------
        X_train : pandas.DataFrame

        y : array-like, optional

        Returns
        -------
        self : OutliersDealer
            Fitted transformer instance.
        """

        # ---------- IQR METHOD ----------
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
        
        # ---------- Z-SCORE METHOD ----------
        elif self.outlier_method == "z_score":

            self.means_ = {}
            self.stds_ = {}

            for col in self.z_columns:
                self.means_[col] = X_train[col].mean()
                self.stds_[col] = X_train[col].std()
        
        # ---------- LOG + Z-SCORE METHOD ----------
        elif self.outlier_method == "log":

            # Store offsets (only needed if min = 0)
            self.log_offsets_ = {}

            for col in self.log_columns:
                min_val = X_train[col].min()

                # If the column has non-positive values, shift it
                if min_val <= 0:
                    self.log_offsets_[col] = 1 - min_val
                else:
                    self.log_offsets_[col] = 0

            self.means_ = {}
            self.stds_ = {}

            for col in self.log_z_columns:
                self.means_[col] = X_train[col].mean()
                self.stds_[col] = X_train[col].std()

        # ---------- ISOLATION FOREST ----------
        elif self.outlier_method == "Isolation_Forest":

            self.model_ = IsolationForest(
            contamination=self.contamination_IF,
            random_state=self.random_state
            )
            self.model_.fit(X_train[self.model_columns])

            # Identify outliers (-1) and cap them instead of removing
            preds = self.model_.predict(X_train[self.model_columns])  # +1 = normal, -1 = outlier
            # Find outlier indices
            outlier_mask = (preds == -1)
            
            # Cap outliers to the third quartile (75th percentile)
            for col in self.model_columns:
                q3 = X_train[col].quantile(0.75)
                X_train.loc[outlier_mask, col] = q3
            
            # Store transformed training data
            self.X_ = X_train

        # ---------- LOCAL OUTLIER FACTOR ----------
        elif self.outlier_method == "LOF":

            self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination_LOF, 
            novelty= True
            )
            self.model_.fit(X_train[self.model_columns])

            preds = self.model_.predict(X_train[self.model_columns])  # +1 = normal, -1 = outlier
            # Find outlier indices
            outlier_mask = (preds == -1)
            
            # Cap outliers to the third quartile (75th percentile)
            for col in self.model_columns:
                q3 = X_train[col].quantile(0.75)
                X_train.loc[outlier_mask, col] = q3
            
            # Store the transformed training data so that the same adjustment
            # is reused at transform time, ensuring consistency and preventing
            # data leakage
            self.X_ = X_train

        return self
    
    def transform(self, X, y = None, **kwargs):
        """
        Apply the fitted outlier treatment strategy to new data.

        For model-based methods (Isolation Forest, LOF), the transformation
        learned during training is reused to ensure consistency and avoid
        data leakage.

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature matrix to be transformed.

        y : array-like, optional
            Target variable (not modified; included for pipeline compatibility).

        Returns
        -------
        X_transformed : pandas.DataFrame
            Transformed feature matrix with outliers capped or adjusted
            according to the selected strategy.
        """


        X = X.copy()
        y = y.copy() if y != None else None
            
        # ---------- IQR CAPPING ----------
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
                
        # ---------- Z-SCORE CAPPING ----------
        elif self.outlier_method == "z_score":

            for col in self.z_columns:

                X[col] = np.clip(X[col],
                                self.means_[col] - self.threshold * self.stds_[col],
                                self.means_[col] + self.threshold * self.stds_[col]
                            )
            return X
        
        # ---------- LOG + Z-SCORE ----------
        elif self.outlier_method == "log":

            for col in self.log_columns:
                offset = self.log_offsets_.get(col, 0)
                X[col] = np.log(X[col] + offset)

            for col in self.log_z_columns:

                X[col] = np.clip(X[col],
                                self.means_[col] - self.threshold * self.stds_[col],
                                self.means_[col] + self.threshold * self.stds_[col]
                            )

            return X
        
        # ---------- MODEL-BASED METHODS ----------
        elif self.outlier_method in ["Isolation_Forest", "LOF"]:
    
           return self.X_