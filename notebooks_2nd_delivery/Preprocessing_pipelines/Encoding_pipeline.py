import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# custom transformer of categorical encoding from sklearn
# make the class compatible with sklearn: BaseEstimator, TransformerMixin
class EncodingDealer(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        method="onehot",         # methods: "onehot", "target", "freq"
        cols=None,               # list of columns to encode
        handle_unknown="ignore", # how to handle unknown categories
        min_freq=0,              # for rare category handling
        **kwargs                 # additional parameters
    ):
        
        self.method = method
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.min_freq = min_freq
        
        # atributes to be created on fit
        self.categories_ = None
        self.target_means_ = None


    def fit(self, X, y=None):
        """
        Learn the required statistics for each encoding method:
        - unique categories (One-Hot)
        - target means per category (Target Encoding)
        - category frequencies (Frequency Encoding)
        """
        X = X.copy() # avoid modifying original data
        
        if self.cols is None:
            # automatically detect categorical columns
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # learn categories
        if self.method == "onehot":
            self.categories_ = {
                col: X[col].unique().tolist() 
                for col in self.cols
            }
        
        # For each category, calculate the target average.
        elif self.method == "target":
            if y is None:
                raise ValueError("Target variable 'y' must be provided for target encoding.")
            self.target_means_ = {
                col: X.groupby(col)[y.name].mean().to_dict() for col in self.cols
            }
        
        # For each category, calculate its relative frequency.
        elif self.method == "freq":
            self.freqs_ = {
                col: X[col].value_counts(normalize=True).to_dict() for col in self.cols
            }
        
        # This first part builds an encoding transformer that is fully compatible with sklearn.
        return self
        
    
    # In fit() -> learns categories, averages, frequencies.
    # In transform() -> use this to convert text to numbers

    # Applies what was learned in fit() to the new data.
    def transform(self, X):
        """
        Apply the encoding learned during fit()
        """
        X = X.copy() # avoid modifying original data

        # One-Hot Encoding (manual implementation)
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

        # Target Encoding
        elif self.method == "target":
            for col in self.cols:
                mapping = self.target_means_[col]
                X[col] = X[col].map(mapping)
                # handle unseen categories
                if self.handle_unknown == "ignore":
                    X[col] = X[col].fillna(X[col].mean())

        # Frequency Encoding
        elif self.method == "freq":
            for col in self.cols:
                mapping = self.freqs_[col]
                X[col] = X[col].map(mapping)
                X[col] = X[col].fillna(0)

        # Return the transformed dataframe.
        # This causes the pipeline to continue to the next step.
        return X
