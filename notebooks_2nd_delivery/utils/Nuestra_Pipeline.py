import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_sample_weight

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
        imputer,
        outlier_remover,
        encoder,
        scaler,
        selector,
        model, 
        q = 10,
        **kwargs
    ):

        self.imputer = imputer
        self.outlier_remover = outlier_remover
        self.encoder = encoder
        self.scaler = scaler
        self.selector = selector
        self.model = model
        self.q = q

        for parameter, value in kwargs.items():  #these lines are used to store the hyperparameter values that come from the searches
            setattr(self, parameter, value)

    def fit(self, X, y, **kwargs):
        """Fits the complete hermetic regression pipeline."""

        X = self.imputer.fit_transform(X, **kwargs)

        output = self.outlier_remover.fit_transform(X, y, **kwargs)

        # Handle preprocessors that return only X or (X, y)
        if isinstance(output, tuple):
            X, y_clean = output
        else:
            X = output
            y_clean = y

        X = self.encoder.fit_transform(X, y_clean, **kwargs)

        X = self.scaler.fit_transform(X, **kwargs)

        X_clean = self.selector.fit_transform(X, y_clean, **kwargs)

        # Create weights inversely proportional to price frequency
        price_bins = pd.qcut(y_clean, q = self.q, labels=False)  # Divide into deciles
        sample_weights = compute_sample_weight('balanced', price_bins)

        # Clone for sklearn compatibility
        self.model_ = clone(self.model)
        self.model_.fit(X_clean, y_clean, sample_weight = sample_weights)
        #self.model_.fit(X_clean, y_clean)

        # Store for inspection
        self.X_ = X_clean
        self.y_ = y

        return self

    def predict(self, X, **kwargs):
        """Predicts regression output given raw input data."""

        X = self.imputer.transform(X, **kwargs)

        X = self.encoder.transform(X, **kwargs)

        X = self.scaler.transform(X, **kwargs)

        X_clean = self.selector.transform(X, **kwargs)

        check_is_fitted(self, "model_")

        y_preds = self.model_.predict(X_clean)

        return y_preds
