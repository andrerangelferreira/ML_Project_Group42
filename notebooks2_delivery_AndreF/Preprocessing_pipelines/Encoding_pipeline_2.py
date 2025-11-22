import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# custom transformer of categorical encoding from sklearn
# make the class compatible with sklearn: BaseEstimator, TransformerMixin
class EncodingDealer(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        method="onehot",         # methods: "onehot", "target", "freq", "hybrid"
        cols=None,               # list of columns to encode
        handle_unknown="ignore", # how to handle unknown categories
        min_freq=0,              # for rare category handling
        **kwargs                # additional parameters    
    ):
        
        self.method = method
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.min_freq = min_freq
        
        # attributes created during fit
        self.categories_ = None
        self.target_means_ = None
        self.freqs_ = None

        # HYBRID attributes
        self.brand_categories_ = None
        self.model_encoders_ = {}   # one LabelEncoder per brand


    def fit(self, X, y=None):
        """
        Learn the required statistics for each encoding method:
        - unique categories (One-Hot)
        - target means per category (Target Encoding)
        - category frequencies (Frequency Encoding)
        - brand categories and model encoders (Hybrid Encoding)
        """
        X = X.copy()

        # auto-detect columns except in hybrid
        if self.method in ["onehot", "target", "freq"] and self.cols is None:
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # ------------------------------ ONE-HOT ------------------------------
        if self.method == "onehot":
            self.categories_ = {
                col: X[col].unique().tolist() 
                for col in self.cols
            }

        # ------------------------------ TARGET ------------------------------
        elif self.method == "target":
            if y is None:
                raise ValueError("Target variable 'y' must be provided for target encoding.")
            self.target_means_ = {
                col: X.groupby(col)[y.name].mean().to_dict()
                for col in self.cols
            }

        # ------------------------------ FREQUENCY ------------------------------
        elif self.method == "freq":
            self.freqs_ = {
                col: X[col].value_counts(normalize=True).to_dict()
                for col in self.cols
            }

        # ------------------------------ HYBRID ------------------------------
        elif self.method == "hybrid":

            # Ensure Brand and model exist
            if "Brand" not in X.columns or "model" not in X.columns:
                raise ValueError("Hybrid encoding requires 'Brand' and 'model' columns.")

            # 1. Learn all Brand categories
            self.brand_categories_ = X["Brand"].unique().tolist()

            # 2. For each brand, learn a label encoder for the models of that brand
            for brand in self.brand_categories_:
                models = X.loc[X["Brand"] == brand, "model"].astype("category").cat.categories.tolist()
                # create mapping starting at 1 (never 0)
                mapping = {m: i+1 for i, m in enumerate(models)}
                self.model_encoders_[brand] = mapping

        return self
    
    # In fit() -> learns categories, averages, frequencies.
    # In transform() -> use this to convert text to numbers


    def transform(self, X):
        """
        Apply the encoding learned during fit().
        """
        X = X.copy() # avoid modifying original data

        # ------------------------------ ONE-HOT ------------------------------
        if self.method == "onehot":
            for col in self.cols:
                for category in self.categories_[col]:
                    X[f"{col}_{category}"] = (X[col] == category).astype(int)
            X = X.drop(columns=self.cols)

        # During fit() self.categories_[col] = list of categories that existed in training
        # For each selected categorical column (self.cols):
        # For each category learned in training:
        # Creates new column
        # Fills with 1 if matches category, else 0
        # Finally, drops original categorical columns

        # ------------------------------ TARGET ------------------------------
        elif self.method == "target":
            for col in self.cols:
                mapping = self.target_means_[col]
                X[col] = X[col].map(mapping)
                # handle unseen categories
                if self.handle_unknown == "ignore":
                    X[col] = X[col].fillna(X[col].mean())

        # ------------------------------ FREQUENCY ------------------------------
        elif self.method == "freq":
            for col in self.cols:
                mapping = self.freqs_[col]
                X[col] = X[col].map(mapping)
                X[col] = X[col].fillna(0)  # unseen categories -> frequency 0

        # ------------------------------ HYBRID ------------------------------
        elif self.method == "hybrid":

            # 1. One-hot brand columns
            for brand in self.brand_categories_:
                X[f"Brand_{brand}"] = (X["Brand"] == brand).astype(int)

            # 2. Replace the "1" in Brand columns with model codes
            for brand in self.brand_categories_:
                mask = X["Brand"] == brand
                mapping = self.model_encoders_[brand]

                # Replace only where Brand == brand
                X.loc[mask, f"Brand_{brand}"] = (
                    X.loc[mask, "model"]
                    .map(mapping)
                    .fillna(0)   # unknown model → 0
                    .astype(int)
                )

            # 3. Drop original columns
            X = X.drop(columns=["Brand", "model"])

        return X
