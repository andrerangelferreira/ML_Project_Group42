import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


class FeatureSelectionDealer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        selection_method="variance",   # "variance", "kbest", "model", "rfe"
        threshold=0.0,                  # for variance threshold
        k=10,                           # for SelectKBest
        score_func="f_classif",         # or "f_regression"
        model_type="classifier",        # "classifier" or "regressor"
        model_name="rf",                # "rf" or "linear"
        n_features_to_select=None,      # for RFE
        random_state=42,
        **kwargs
    ):
        self.selection_method = selection_method

        # Variance threshold parameters
        self.threshold = threshold

        # SelectKBest parameters
        self.k = k
        self.score_func = score_func

        # Model-based selection parameters
        self.model_type = model_type
        self.model_name = model_name
        self.random_state = random_state

        # RFE parameters
        self.n_features_to_select = n_features_to_select


    def _get_score_func(self):
        if self.score_func == "f_classif":
            return f_classif
        elif self.score_func == "f_regression":
            return f_regression
        else:
            raise ValueError(f"Unknown score func: {self.score_func}")


    def _get_model(self):
        if self.model_type == "classifier":
            if self.model_name == "rf":
                return RandomForestClassifier(random_state=self.random_state)
            elif self.model_name == "linear":
                return LogisticRegression(max_iter=500)
        else:
            if self.model_name == "rf":
                return RandomForestRegressor(random_state=self.random_state)
            elif self.model_name == "linear":
                return LinearRegression()

        raise ValueError(f"Unknown model: {self.model_name} ({self.model_type})")


    def fit(self, X, y=None, **kwargs):

        X = X.copy()

        # 1) Variance Threshold
        if self.selection_method == "variance":
            self.selector = VarianceThreshold(threshold=self.threshold)
            self.selector.fit(X)

        # 2) SelectKBest
        elif self.selection_method == "kbest":
            score_func = self._get_score_func()
            self.selector = SelectKBest(score_func=score_func, k=self.k)
            self.selector.fit(X, y)

        # 3) Model-based feature selection
        elif self.selection_method == "model":
            model = self._get_model()
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
            self.selector = None  # not using scikit-learn selector here

        # 4) RFE
        elif self.selection_method == "rfe":
            model = self._get_model()
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
