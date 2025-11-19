import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    RFE
)

# Linear Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Tree-Based Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# SVM
from sklearn.svm import SVR

# KNN
from sklearn.neighbors import KNeighborsRegressor



class FeatureSelectionDealer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        selection_method="variance",   # Method options: "variance", "kbest", "model", "rfe"
        threshold=0.0,                  # for variance threshold
        k=10,                           # for SelectKBest    
        model_name= "rf",               # model acronym
        n_features_to_select=None,      # for RFE
        random_state=42,
        **kwargs
    ):
        
        self.selection_method = selection_method

        # Variance threshold parameters
        self.threshold = threshold

        # SelectKBest parameters
        self.k = k

        # Model-based selection parameters
    
        self.model_name = model_name
        self.random_state = random_state

        # RFE parameters
        self.n_features_to_select = n_features_to_select


    def get_model(self):

        # ---- LINEAR MODELS ----
        if self.model_name == "linear":
            return LinearRegression()

        elif self.model_name == "ridge":
            return Ridge()

        elif self.model_name == "lasso":
            return Lasso()

        elif self.model_name == "elasticnet":
            return ElasticNet()


        # ---- TREE-BASED MODELS ----
        elif self.model_name == "rf":   # Random Forest
            return RandomForestRegressor(random_state=self.random_state)

        elif self.model_name == "et":   # Extra Trees
            return ExtraTreesRegressor(random_state=self.random_state)

        elif self.model_name == "dt":   # Decision Tree
            return DecisionTreeRegressor(random_state=self.random_state)


        # ---- GRADIENT BOOSTING ----
        elif self.model_name == "gboost":
            return GradientBoostingRegressor(random_state=self.random_state)

        elif self.model_name == "adaboost":
            return AdaBoostRegressor(random_state=self.random_state)

        elif self.model_name == "hgb":
            return HistGradientBoostingRegressor(random_state=self.random_state)


        # ---- SUPPORT VECTOR MACHINE ----
        elif self.model_name == "svm":
            return SVR()


        # ---- K-NEAREST NEIGHBORS ----
        elif self.model_name == "knn":
            return KNeighborsRegressor()


        # ---- ERROR ----
        else:
            raise ValueError(f"Unknown regression model '{self.model_name}'")


    def fit(self, X, y=None, **kwargs):

        X = X.copy()

        # 1) Variance Threshold
        if self.selection_method == "variance":
            self.selector = VarianceThreshold(threshold=self.threshold)
            self.selector.fit(X)

        # 2) SelectKBest
        elif self.selection_method == "kbest":

            self.selector = SelectKBest(score_func=f_regression, k=self.k)
            self.selector.fit(X, y)

        # 3) Model-based feature selection
        elif self.selection_method == "model":
            model = self.get_model()
            model.fit(X, y)
            importances = np.abs(model.feature_importances_ 
                                  if hasattr(model, "feature_importances_") 
                                  else model.coef_.ravel())

            # store mask of selected features (top k)
            k = self.k if self.k < X.shape[1] else X.shape[1]
            top_k_idx = np.argsort(importances)[::-1][:k]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_k_idx] = True

            self.support_mask_ = mask

        # 4) RFE
        elif self.selection_method == "rfe":
            model = self.get_model()
            self.selector = RFE(
                estimator=model,
                n_features_to_select=self.n_features_to_select
            )
            self.selector.fit(X, y)

        else:
            raise ValueError(f"Unknown method: {self.selection_method}")

        return self


    def transform(self, X, y=None, **kwargs):

        X = X.copy()

        if self.selection_method in ["variance", "kbest", "rfe"]:
            X_sel = self.selector.transform(X)
            # Ensure dataframe structure is preserved
            cols = np.array(X.columns)[self.selector.get_support()]
            X_sel = pd.DataFrame(X_sel, columns=cols, index=X.index)
            return X_sel, y

        elif self.selection_method == "model":
            cols = np.array(X.columns)[self.support_mask_]
            X_sel = X.loc[:, cols]
            return X_sel, y

        else:
            raise ValueError(f"Unknown method: {self.selection_method}")
