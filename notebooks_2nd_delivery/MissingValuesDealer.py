import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class MissingValuesDealer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        imputation_method="simple",  # "simple", "knn", "iterative"
        simple_strategy_num="mean",      # for simple imputer with numerical
        strategy_cat="most_frequent", #  imputer for categorical
        fill_value=None,             # used if strategy="constant"
        knn_neighbors=5,
        random_state=42,
        knn_scaling_method= "standard",
        **kwargs
    ):
        self.imputation_method = imputation_method

        # Simple Imputer params
        self.simple_strategy_num = simple_strategy_num  #numerical variables
        self.strategy_cat = strategy_cat  #categorical variables
        self.fill_value = fill_value

        # KNN Imputer params
        self.knn_neighbors = knn_neighbors
        self.knn_scaling_method = knn_scaling_method

        # Iterative imputer
        self.random_state = random_state




    def fit(self, X_train, **kwargs):

        #----- FITTING WITH SIMPLE IMPUTER -----

        if self.imputation_method == "simple":

            #imputer for numerical
            self.imputer_num = SimpleImputer(
                strategy=self.simple_strategy_num,
                fill_value=self.fill_value
            )
            self.imputer_num.fit(X_train.select_dtypes(include=np.number))

            #imputer for categorical
            self.imputer_cat = SimpleImputer(
                strategy=self.strategy_cat,
                fill_value=self.fill_value
            )
            self.imputer_cat.fit(X_train.select_dtypes(exclude=np.number))

        #----- FITTING WITH KNN -----

        elif self.imputation_method == "knn":
            self.metric_features = X_train.select_dtypes(include=np.number).columns
  
            #scaler for knn

            if self.knn_scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.knn_scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.knn_scaling_method == "robust":
                self.scaler = RobustScaler()

            #fit scaler      
            self.scaler.fit(X_train[self.metric_features])
            #Transform the data set to discover knn imputer 
            scaled = self.scaler.transform(X_train[self.metric_features])


            #imputer for numerical
            self.imputer_num = KNNImputer(
                n_neighbors=self.knn_neighbors
            )
            self.imputer_num.fit(scaled)

            #imputer for categorical
            self.imputer_cat = SimpleImputer(
                strategy=self.strategy_cat,
                fill_value=self.fill_value
            )
            self.imputer_cat.fit(X_train.select_dtypes(exclude=np.number))


         #----- FITTING WITH KNN_BRANDWISE -----

        elif self.imputation_method == "knn_brandwise":

            #Imputers and scalers for numerical imputation
            self.metric_features = X_train.select_dtypes(include=np.number).columns

            self.scalers_ = {}   # scaler per brand
            self.imputers_ = {}  # knn imputer per brand

            for brand, df_brand in X_train.groupby("Brand"):

                if self.knn_scaling_method == "standard":
                    scaler = StandardScaler()
                elif self.knn_scaling_method == "minmax":
                    scaler = MinMaxScaler()
                elif self.knn_scaling_method == "robust":
                    scaler = RobustScaler()
                
                imputer = KNNImputer(n_neighbors=self.knn_neighbors)

                # Fit scaler
                scaled = scaler.fit_transform(df_brand[self.metric_features])

                # Fit imputer on scaled data
                imputer.fit(scaled)

                # Store both
                self.scalers_[brand] = scaler
                self.imputers_[brand] = imputer
            
            #imputer for categorical
            self.imputer_cat = SimpleImputer(
                strategy=self.strategy_cat,
                fill_value=self.fill_value
            )
            self.imputer_cat.fit(X_train.select_dtypes(exclude=np.number))

        # ----- FITTING WITH ITERATIVE IMPUTER -----       

        elif self.imputation_method == "iterative":

            #imputer for numerical
            self.imputer_num = IterativeImputer(
                random_state=self.random_state
            )
            self.imputer_num.fit(X_train.select_dtypes(include=np.number))

            #imputer for categorical
            self.imputer_cat = SimpleImputer(
                strategy=self.strategy_cat,
                fill_value=self.fill_value
            )
            self.imputer_cat.fit(X_train.select_dtypes(exclude=np.number))

        return self


    def transform(self, X, y, **kwargs):

        X = X.copy()

        # Simple / Iterative Imputation
        if self.imputation_method in ["simple", "iterative"]:
            
            # Split columns
            num_cols = X.select_dtypes(include=np.number).columns
            cat_cols = X.select_dtypes(exclude=np.number).columns

            #impute
            X_num_imputed = self.imputer_num.transform(X.select_dtypes(include=np.number))
            X_cat_imputed = self.imputer_cat.transform(X.select_dtypes(exclude=np.number))

            # Convert back to DataFrames
            df_num = pd.DataFrame(X_num_imputed, columns=num_cols, index=X.index)
            df_cat = pd.DataFrame(X_cat_imputed, columns=cat_cols, index=X.index)

            # Combine
            X_imputed = pd.concat([df_num, df_cat], axis=1)
            X_imputed = X_imputed[X.columns]

            #Correcting previousOwners that should only have integers 
            X_imputed["previousOwners"] = X_imputed["previousOwners"].round()

            return X_imputed, y
        
        #----- KNN IMPUTATION -----

        elif self.imputation_method == "knn":

            # Split columns
            num_cols = X.select_dtypes(include=np.number).columns
            cat_cols = X.select_dtypes(exclude=np.number).columns

            #Scale numeric values
            scaled = self.scaler.transform(X[num_cols])
            #impute scaled values
            imputed_scaled = self.imputer_num.transform(scaled)

            #inverse scale
            X_num_imputed = self.scaler.inverse_transform(imputed_scaled) 

            #Categorical values imputation
            X_cat_imputed = self.imputer_cat.transform(X[cat_cols])

            # Convert back to DataFrames
            df_num = pd.DataFrame(X_num_imputed, columns=num_cols, index=X.index)
            df_cat = pd.DataFrame(X_cat_imputed, columns=cat_cols, index=X.index)

             # Combine
            X_imputed = pd.concat([df_num, df_cat], axis=1)
            X_imputed = X_imputed[X.columns]

            #Correcting previousOwners that should only have integers 
            X_imputed["previousOwners"] = X_imputed["previousOwners"].round()

            return X_imputed, y


        
        # ----- BRAND-WISE KNN IMPUTATION -----
        elif self.imputation_method == "knn_brandwise":

            # Split columns
            num_cols = X.select_dtypes(include=np.number).columns
            cat_cols = X.select_dtypes(exclude=np.number).columns


            #Impute Numerical
            imputed_list = []

            for brand, df_brand in X.groupby("Brand"):

                df_temp = df_brand.copy()

                scaler = self.scalers_[brand]
                imputer = self.imputers_[brand]

                # Scale, impute, inverse scale
                scaled = scaler.transform(df_temp[self.metric_features])
                imputed_scaled = imputer.transform(scaled)
                df_temp[self.metric_features] = scaler.inverse_transform(imputed_scaled)

                imputed_list.append(df_temp[self.metric_features])

            # Reassemble dataset in original order
            X_num_imputed = pd.concat(imputed_list, axis=0)
            X_num_imputed = X_num_imputed.loc[X.index]

            #Impute Categorical 
            X_cat_imputed = self.imputer_cat.transform(X.select_dtypes(exclude=np.number))
            df_cat = pd.DataFrame(X_cat_imputed, columns=cat_cols, index=X.index)

            # Combine
            X_imputed = pd.concat([ X_num_imputed, df_cat], axis=1)
            X_imputed = X_imputed[X.columns]

            #Correcting previousOwners that should only have integers 
            X_imputed["previousOwners"] = X_imputed["previousOwners"].round()

            return X_imputed, y

        else:
            raise ValueError(f"Unknown imputation method: {self.imputation_method}")
        

# Um possivel error no knn_branwise poderá acontecer caso não haja rows suficientes para uma brand especifica 


