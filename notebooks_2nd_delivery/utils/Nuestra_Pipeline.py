import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

class NuestraPipeline(RegressorMixin, BaseEstimator):

    """
    Custom regressor that integrates preprocessing, feature extraction,
    optional outlier removal, scaling, and a final regression model.
    
    Intended as the regression counterpart of HermeticClassifier.
    
    Parameters
    ----------
    preprocessor : object
        pipeline that include classes with all the steps of preprocessing

    model : object
        Regressor to fit (e.g., RandomForestRegressor).

    """
    
    def __init__(
        self, 
        preprocessor,
        model, 
        **kwargs
    ):
        self.preprocessor = preprocessor
        self.model = model

    def fit(self, X, y, **kwargs):
        """Fits the complete hermetic regression pipeline."""

        output = self.preprocessor.fit_transform(X, y, **kwargs)

        # Handle preprocessors that return only X or (X, y)
        if isinstance(output, tuple):
            X_clean, y_clean = output
        else:
            X_clean = output
            y_clean = y

        # Clone for sklearn compatibility
        self.model_ = clone(self.model)
        self.model_.fit(X_clean, y_clean)

        # Store for inspection
        self.X_ = X_clean
        self.y_ = y_clean

        return self

    def predict(self, X, **kwargs):
        """Predicts regression output given raw input data."""

        check_is_fitted(self, "model_")

        X_clean = self.preprocessor.transform(X, **kwargs)

        y_preds = self.model_.predict(X_clean)

        return y_preds
