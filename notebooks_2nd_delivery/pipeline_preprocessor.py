import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from Preprocessing_pipelines.Outliers_pipeline import OutliersDealer
from Preprocessing_pipelines.Missing_Values_pipeline import MissingValuesDealer



class Preprocessor_Pipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 outlier_removal= OutliersDealer(),
                 imputer= MissingValuesDealer(),
                 encoder= Encoder(),
                 scaler= StandardScalerCustom()
                ):
        
        self.outlier_removal = outlier_removal 
        self.imputer = imputer
        self.encoder = encoder
        self.scaler = scaler

    def fit(self, X_train, **kwargs):
       
        #Calculating the outlier methods
        X_train = self.outlier_removal.fit(X_train, **kwargs)

        #Calculating the metrics for missing values imputing
        X_train = self.imputer.fit(X_train, **kwargs)

    
    def transform(self, X, **kwargs):

        #Treating the outliers
        X = self.outlier_removal.transform(X, **kwargs) 

        #Imputing Missing values
        X = self.imputer.transform(X, **kwargs) 

        



    
        



            

        
        
