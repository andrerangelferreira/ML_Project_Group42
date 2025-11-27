import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from category_encoders import CountEncoder, TargetEncoder


from sklearn.base import BaseEstimator, TransformerMixin

class EncodingDealer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method="onehot",         # "onehot", "target", "freq", "hybrid"
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
            self.target_encoder_.fit(X, y)

        # FREQUENCY
        elif self.method == "freq":
            # Create a count encoder for selected columns
            self.freq_encoder_ = CountEncoder(cols=self.cols_)

            # Fit it (does not use y)
            self.freq_encoder_.fit(X)

        # HYBRID
        elif self.method == "hybrid":
            if "Brand" not in X.columns or "model" not in X.columns:
                raise ValueError("Hybrid encoding requires 'Brand' and 'model' columns.")
            self.brand_categories_ = X["Brand"].astype("object").dropna().unique().tolist()
            for brand in self.brand_categories_:
                models = X.loc[X["Brand"] == brand, "model"].astype("category").cat.categories.tolist()
                mapping = {m: i+1 for i, m in enumerate(models)}
                self.model_encoders_[brand] = mapping

        return self

    def transform(self, X):
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
            X = self.target_encoder_.transform(X)

        # FREQUENCY
        elif self.method == "freq":
            X = self.freq_encoder_.transform(X)

        # HYBRID
        elif self.method == "hybrid":
            # create Brand one-hot columns (or zero if Brand missing)
            for brand in (self.brand_categories_ or []):
                X[f"Brand_{brand}"] = (X.get("Brand", pd.Series(index=X.index)) == brand).astype(int)

            # then replace 1s with model code where applicable
            for brand in (self.brand_categories_ or []):
                mask = X.get("Brand", pd.Series(index=X.index)) == brand
                mapping = self.model_encoders_.get(brand, {})
                if "model" in X.columns:
                    mapped = X.loc[mask, "model"].map(mapping).fillna(0).astype(int)
                    X.loc[mask, f"Brand_{brand}"] = mapped
                else:
                    X.loc[mask, f"Brand_{brand}"] = 0

            # drop original columns if exist
            for c in ["Brand", "model"]:
                if c in X.columns:
                    X = X.drop(columns=[c])

        return X
