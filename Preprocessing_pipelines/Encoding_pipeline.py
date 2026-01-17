import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from category_encoders import CountEncoder, TargetEncoder


from sklearn.base import BaseEstimator, TransformerMixin

class EncodingDealer(BaseEstimator, TransformerMixin):
    """
    Flexible categorical encoding transformer supporting multiple strategies.

    This transformer provides a unified interface for handling categorical
    variables using one of three encoding methods:
        - One-Hot Encoding
        - Target Encoding
        - Frequency (Count) Encoding

    Parameters
    ----------
    method : str, default="onehot"
        Encoding strategy to apply. Supported options:
        {"onehot", "target", "freq"}.

    cols : list or None, default=None
        List of categorical columns to encode. If None, categorical columns
        are automatically inferred based on data types.

    handle_unknown : str, default="ignore"
        Strategy to handle unseen categories during transformation
        (used only for One-Hot Encoding).

    min_freq : int, default=0
        Minimum frequency threshold for categories (reserved for extensions).

    Notes
    -----
    - Target encoding requires the target variable `y` during fitting.
    - All encodings replace the original categorical columns with their
      encoded numerical representations.
    
    """
    def __init__(
        self,
        method="onehot",         # "onehot", "target", "freq"
        cols=None,
        handle_unknown="ignore",
        min_freq=0,
        **kwargs
    ):
        self.method = method
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.min_freq = min_freq

        # learned attributes (post-fit)
        self.cols_ = None
        self.categories_ = {}
        self.target_means_ = {}
        self.freqs_ = {}
        self.brand_categories_ = None
        self.model_encoders_ = {}

    def fit(self, X, y=None):
        """
        Fit the selected encoding strategy on the training data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features containing categorical variables.

        y : array-like, optional
            Target variable, required for target encoding.

        Returns
        -------
        self : object
            Fitted encoder instance.
        """

        # determine categorical columns to operate on
        if self.cols is None:
            self.cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            # if user passed cols, keep those (but only existing ones)
            self.cols_ = [c for c in list(self.cols) if c in X.columns]

        # ONE-HOT
        if self.method == "onehot":
            # initialize the encoder
            self.ohe_ = OneHotEncoder(
                handle_unknown=self.handle_unknown,
                sparse_output=False
            )

            # fit only on categorical columns
            self.ohe_.fit(X[self.cols_])

        # TARGET
        elif self.method == "target":
            if y is None:
                raise ValueError("Target variable 'y' must be provided for target encoding.")

            # Create encoder for selected columns
            self.target_encoder_ = TargetEncoder(cols=self.cols_)

            # Fit encoder (X and y must be aligned)
            self.target_encoder_.fit(X[self.cols_], y)

        # FREQUENCY
        elif self.method == "freq":
            # Create a count encoder for selected columns
            self.freq_encoder_ = CountEncoder(cols=self.cols_)

            # Fit encoder (X and y must be aligned)
            self.freq_encoder_.fit(X[self.cols_], y)

        return self

    def transform(self, X):
        """
        Transform categorical variables using the fitted encoding strategy.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features to be transformed.

        Returns
        -------
        X_transformed : pandas.DataFrame
            DataFrame with encoded categorical features and original
            non-categorical features preserved.
        """
        
        X = X.copy()

        # ONE-HOT
        if self.method == "onehot":
            # transform categorical columns using fitted encoder
            ohe_array = self.ohe_.transform(X[self.cols_])

            # assemble encoded features into DataFrame
            ohe_df = pd.DataFrame(ohe_array, columns=self.ohe_.get_feature_names_out(), index=X.index)

            # drop original categorical columns
            X = X.drop(columns=self.cols_)

            # concatenate encoded columns
            X = pd.concat([X, ohe_df], axis=1)
            
        # TARGET
        elif self.method == "target":
            # transform categorical columns using fitted encoder
            target_array = self.target_encoder_.transform(X[self.cols_])

            # assemble encoded features into DataFrame
            target_df = pd.DataFrame(target_array, columns=self.target_encoder_.get_feature_names_out(), index=X.index)

            # drop original categorical columns
            X = X.drop(columns=self.cols_)

            # concatenate encoded columns
            X = pd.concat([X, target_df], axis=1)


        # FREQUENCY
        elif self.method == "freq":
            # transform categorical columns using fitted encoder
            freq_array = self.freq_encoder_.transform(X[self.cols_])

            # assemble encoded features into DataFrame
            freq_df = pd.DataFrame(freq_array, columns=self.freq_encoder_.get_feature_names_out(), index=X.index)

            # drop original categorical columns
            X = X.drop(columns=self.cols_)

            # concatenate encoded columns
            X = pd.concat([X, freq_df], axis=1)

        return X
