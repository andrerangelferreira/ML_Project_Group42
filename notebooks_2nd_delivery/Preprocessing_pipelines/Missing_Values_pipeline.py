import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


class MissingValuesDealer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        imputation_method="simple",  # "simple", "knn", "iterative"
        simple_strategy="mean",      # for simple imputer: mean/median/most_frequent/constant
        fill_value=None,             # used if strategy="constant"
        knn_neighbors=5,
        random_state=42,
        **kwargs
    ):
        self.imputation_method = imputation_method

        # Simple Imputer params
        self.simple_strategy = simple_strategy
        self.fill_value = fill_value

        # KNN Imputer params
        self.knn_neighbors = knn_neighbors

        # Iterative imputer
        self.random_state = random_state


    def fit(self, X_train, **kwargs):

        if self.imputation_method == "simple":
            self.imputer = SimpleImputer(
                strategy=self.simple_strategy,
                fill_value=self.fill_value
            )
            self.imputer.fit(X_train)

        elif self.imputation_method == "knn":
            self.imputer = KNNImputer(
                n_neighbors=self.knn_neighbors
            )
            self.imputer.fit(X_train)

        elif self.imputation_method == "knn_brandwise":

            self.metric_features_ = X_train.select_dtypes(include=np.number)

            self.scalers_ = {}   # scaler per brand
            self.imputers_ = {}  # knn imputer per brand

            for brand, df_brand in X_train.groupby("Brand"):

                scaler = StandardScaler()
                imputer = KNNImputer(n_neighbors=self.knn_neighbors)

                # Fit scaler
                scaled = scaler.fit_transform(df_brand[self.metric_features])

                # Fit imputer on scaled data
                imputer.fit(scaled)

                # Store both
                self.scalers_[brand] = scaler
                self.imputers_[brand] = imputer

        elif self.imputation_method == "iterative":
            self.imputer = IterativeImputer(
                random_state=self.random_state
            )
            self.imputer.fit(X_train)
        return self


    def transform(self, X, y, **kwargs):

        X = X.copy()

        # Simple / KNN / Iterative Imputation
        if self.imputation_method in ["simple", "knn", "iterative"]:
            X_imputed = self.imputer.transform(X)
            X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            return X_imputed, y
        
        # --- BRAND-WISE KNN IMPUTATION ---
        elif self.imputation_method == "knn_brandwise":

            imputed_list = []

            for brand, df_brand in X.groupby("Brand"):

                df_temp = df_brand.copy()

                scaler = self.scalers_[brand]
                imputer = self.imputers_[brand]

                # Scale, impute, inverse scale
                scaled = scaler.transform(df_temp[self.metric_features_])
                imputed_scaled = imputer.transform(scaled)
                df_temp[self.metric_features_] = scaler.inverse_transform(imputed_scaled)

                imputed_list.append(df_temp)

            # Reassemble dataset in original order
            X_imputed = pd.concat(imputed_list, axis=0)
            X_imputed = X_imputed.loc[X.index]

            return X_imputed, y

        else:
            raise ValueError(f"Unknown imputation method: {self.imputation_method}")
