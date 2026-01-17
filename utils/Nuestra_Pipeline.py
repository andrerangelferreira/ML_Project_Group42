import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_sample_weight

class NuestraPipeline(RegressorMixin, BaseEstimator):

    """
    Custom regressor that integrates preprocessing, feature extraction,
    optional outlier removal, scaling, and a final regression model.
    
    Parameters
    ----------
    imputer : object
        Transformer responsible for handling missing values.

    outlier_remover : object
        Optional transformer that detects and removes outliers. May return
        either transformed X or a tuple (X_clean, y_clean).

    encoder : object
        Transformer that encodes categorical features into numerical form.

    scaler : object
        Transformer used to scale numerical features when required.

    selector : object
        Feature selection transformer applied after preprocessing.

    model : object
        Final regression estimator (e.g., RandomForestRegressor).

    q : int, default=10
        Number of quantile bins used to compute balanced sample weights
        over the target variable.

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

    def fit(self, X, y, **kwargs):
        """
        Fit the complete regression pipeline on the training data.

        This method sequentially applies all preprocessing steps, performs
        optional outlier removal, computes sample weights to mitigate target
        imbalance, and fits the final regression model.

        Parameters
        ----------
        X : array-like
            Raw input features.

        y : array-like
            Target variable (car price).

        Returns
        -------
        self : object
            Fitted pipeline instance.
        """

        # Step 1 — Handle missing values
        X = self.imputer.fit_transform(X, **kwargs)

        # Step 2 — Outlier removal (may modify X and y)
        output = self.outlier_remover.fit_transform(X, y, **kwargs)

        # Handle preprocessors that return only X or (X, y)
        if isinstance(output, tuple):
            X, y_clean = output
        else:
            X = output
            y_clean = y

        # Step 3 — Encode categorical variables
        X = self.encoder.fit_transform(X, y_clean, **kwargs)

        # Step 4 — Scale numerical features
        X = self.scaler.fit_transform(X, **kwargs)

        # Step 5 — Feature selection
        X_clean = self.selector.fit_transform(X, y_clean, **kwargs)

        # Create weights inversely proportional to price frequency
        price_bins = pd.qcut(y_clean, q = self.q, labels=False)  # Divide into deciles
        sample_weights = compute_sample_weight('balanced', price_bins)

        # Clone for sklearn compatibility
        self.model_ = clone(self.model)
        
        try:
            self.model_.fit(X_clean, y_clean, sample_weight=sample_weights)
        except TypeError:
            # Fallback for models that do not support sample weights
            self.model_.fit(X_clean, y_clean)

        # Store for inspection
        self.X_ = X_clean
        self.y_ = y

        return self

    def predict(self, X, **kwargs):
        """
        Generate price predictions for new, unseen data.

        Applies the same preprocessing steps learned during training and
        outputs predictions from the fitted regression model.

        Parameters
        ----------
        X : array-like
            Raw input features for which predictions are required.

        Returns
        -------
        y_pred : array-like
            Predicted car prices.
        """

        # Apply preprocessing steps in the same order as during training
        X = self.imputer.transform(X, **kwargs)

        X = self.encoder.transform(X, **kwargs)

        X = self.scaler.transform(X, **kwargs)

        X_clean = self.selector.transform(X, **kwargs)

        # Ensure that the pipeline has been fitted before generating predictions
        check_is_fitted(self, "model_")

        # Generate price predictions using the fitted regression model
        y_preds = self.model_.predict(X_clean)

        return y_preds
