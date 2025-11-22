import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from Preprocessing_pipelines.Outliers_pipeline import OutliersDealer
from Preprocessing_pipelines.Missing_Values_pipeline import MissingValuesDealer
from Preprocessing_pipelines.Encoding_pipeline import EncodingDealer
from Preprocessing_pipelines.Scaling_pipeline import ScalingDealer
from Preprocessing_pipelines.Feature_Selection_pipeline import FeatureSelectionDealer


class Preprocessor_Pipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 outlier_removal= OutliersDealer(),
                 imputer= MissingValuesDealer(),
                 encoder= EncodingDealer(),
                 scaler= ScalingDealer(), 
                 selector = FeatureSelectionDealer()
                ):
        
        self.outlier_removal = outlier_removal 
        self.imputer = imputer
        self.encoder = encoder
        self.scaler = scaler
        self.selector = selector

    def fit(self, X_train, y = None, **kwargs):
       
        #Calculating the outlier methods
        self.outlier_removal.fit(X_train, **kwargs)

        #Calculating the metrics for missing values imputing
        self.imputer.fit(X_train, **kwargs)

        self.encoder.fit(X_train, **kwargs)

        self.scaler.fit(X_train, **kwargs)

        self.selector.fit(X_train, **kwargs)

        return self
    
    def transform(self, X, y= None, **kwargs):

        #Treating the outliers
        output = self.outlier_removal.transform(X, y, **kwargs) 

        if isinstance(output, tuple):
            X, y = output
        else:
            X = output
            
        #Imputing Missing values
        X = self.imputer.transform(X, **kwargs) 

        X = self.encoder.transform(X, **kwargs)

        X = self.scaler.transform(X, **kwargs)

        X = self.selector.transform(X, **kwargs)

        if isinstance(output, tuple):
            return X, y
        return X
