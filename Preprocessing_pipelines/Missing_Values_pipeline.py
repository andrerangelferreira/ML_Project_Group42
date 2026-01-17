import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

"""
MissingValuesDealer is a custom class designed to be used within our custom
data processing pipeline, specifically to handle and impute missing values
in the dataset.This class provides multiple strategies for missing value imputation. 
Some of these strategies rely on existing imputation methods, such as
SimpleImputer and IterativeImputer, while others are custom approaches
developed for this project, namely knn_brandwise and knn_modelwise.

The class implements two main methods: fit and transform. During the fit
stage, the selected imputer (and scalers, in the case of KNN-based methods)
is fitted and stored. During the transform stage, the stored imputers are
applied to the input data to fill all missing values.

Imputation methods implemented:

- SimpleImputer:
  Imputes missing values using a constant strategy such as the mean.

- KNN Imputer:
  Uses a KNN-based approach to fill missing values. Since KNN imputation is
  sensitive to feature scaling, the data is scaled before imputation. The
  scaling method can be controlled via the knn_scaling_method parameter, and
  the number of neighbors via the knn_neighbors parameter.

- Iterative Imputer:
  Fills missing values using a multivariate approach by iteratively predicting
  each feature with missing data based on the other features in the dataset.

- knn_brandwise:
  A custom imputation method that applies KNN imputation separately within
  each brand group, under the assumption that vehicles from the same brand
  are more similar. During fitting, a separate scaler and imputer are stored
  for each brand. This method also uses the knn_scaling_method and
  knn_neighbors parameters.

- knn_modelwise:
  A more granular version of knn_brandwise that applies KNN imputation within
  each model group. Due to the large number of distinct models, some models
  may have too few samples to apply KNN reliably. These are treated as rare
  models.

  For models with fewer samples than a predefined threshold, missing values
  are imputed using the median of that model. This threshold is controlled by
  the min_model_size_for_knn parameter.

  In cases where a model contains only missing values for a given feature or
  when unseen models appear during transformation, a global KNN imputer is
  used as a fallback strategy.


  All the categorical values where filled with the SimpleImputer with the most_frequent value.
"""



class MissingValuesDealer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        imputation_method="simple",    # "simple", "knn", "iterative", "knn_brandwise", "knn_modelwise"
        simple_strategy_num="mean",    # for simple imputer with numerical
        strategy_cat="most_frequent",  #  imputer for categorical
        fill_value=None,               # used if strategy="constant"
        knn_neighbors=5,           
        random_state=42,
        knn_scaling_method= "standard",
        min_model_size_for_knn=15,     #min_model_size_for_knn can never be lower than knn_neighbors
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

        #knn-modelwise
        self.min_model_size_for_knn = min_model_size_for_knn

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

            X_train_ = X_train.copy()

            # First we have to impute the brand, therefore knn_brandwise will leave missing values in rows where brand is missing
            self.imputer_brand = SimpleImputer(strategy="most_frequent")
            self.imputer_brand.fit(X_train_[["Brand"]])


            # Replace missing values before groupby
            brand_imputed = self.imputer_brand.transform(X_train_[["Brand"]])      
            X_train_["Brand"] = pd.Series(brand_imputed.flatten(), index=X_train_.index)          

            self.metric_features = X_train.select_dtypes(include=np.number).columns
   
            # train brand-specific scalers and imputers

            self.scalers_ = {}   # scaler per brand
            self.imputers_ = {}  # knn imputer per brand

            for brand, df_brand in X_train_.groupby("Brand"):

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

        # ----- FITTING WITH KNN MODELWISE IMPUTER -----

        elif self.imputation_method == "knn_modelwise":

            X_train_ = X_train.copy()

            # Impute model first
            self.imputer_model = SimpleImputer(strategy="most_frequent")
            X_train_["model"] = self.imputer_model.fit_transform(X_train_[["model"]]).ravel() #fit_transform returns a 2D array, so we have to ravel it to have a 1D array, therefore we would get an error

            # Identify numeric features
            self.metric_features = X_train_.select_dtypes(include=np.number).columns

            # Count the number of cars per model
            model_counts = X_train_["model"].value_counts()

            # Store rare models
            self.rare_models_ = model_counts[model_counts < self.min_model_size_for_knn].index.tolist()

            # Prepare dictionaries
            self.scalers_ = {}
            self.imputers_ = {}
            self.model_medians_ = {}

            #GLOBAL fallback scaler and imputer for unseen models 
            if self.knn_scaling_method == "standard":
                self.global_scaler_ = StandardScaler()
            elif self.knn_scaling_method == "minmax":
                self.global_scaler_ = MinMaxScaler()
            elif self.knn_scaling_method == "robust":
                self.global_scaler_ = RobustScaler()

            scaled_global = self.global_scaler_.fit_transform(X_train_[self.metric_features])
            self.global_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors).fit(scaled_global)

            #Fit model-specific imputers
            for model, df_model in X_train_.groupby("model"):

                # if Model is rare we store medians of the model
                if model in self.rare_models_:
                    median_values = df_model[self.metric_features].median()
                    # keep NaN if all values missing; we'll handle it in transform
                    self.model_medians_[model] = median_values
                    continue   #we use continue so that the code doesn't have to verify the other condition and skip to the next model

                # - Model as enough cars we fit scaler and knn
                if self.knn_scaling_method == "standard":
                    scaler = StandardScaler()
                elif self.knn_scaling_method == "minmax":
                    scaler = MinMaxScaler()
                elif self.knn_scaling_method == "robust":
                    scaler = RobustScaler()

                scaled = scaler.fit_transform(df_model[self.metric_features])
                imputer = KNNImputer(n_neighbors=self.knn_neighbors).fit(scaled)

                self.scalers_[model] = scaler
                self.imputers_[model] = imputer

            # Fit categorical imputer
            self.imputer_cat = SimpleImputer(
                strategy=self.strategy_cat,
                fill_value=self.fill_value
            )
            self.imputer_cat.fit(X_train_.select_dtypes(exclude=np.number))



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


    def transform(self, X, y = None, **kwargs):

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

            return X_imputed
        
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

            return X_imputed


        # ----- BRAND-WISE KNN IMPUTATION -----
        elif self.imputation_method == "knn_brandwise":

            #Impute brand first
            brand_imputed = self.imputer_brand.transform(X[["Brand"]])  
            X["Brand"] = brand_imputed.ravel()

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
            X_cat_imputed = self.imputer_cat.transform(X[cat_cols])
            df_cat = pd.DataFrame(X_cat_imputed, columns=cat_cols, index=X.index)

            # Combine
            X_imputed = pd.concat([ X_num_imputed, df_cat], axis=1)
            X_imputed = X_imputed[X.columns]

            #Correcting previousOwners that should only have integers 
            X_imputed["previousOwners"] = X_imputed["previousOwners"].round()

            return X_imputed
        

        # ----- MODEL-WISE KNN IMPUTATION -----
        elif self.imputation_method == "knn_modelwise":

            # Impute model first
            X["model"] = self.imputer_model.transform(X[["model"]]).ravel() #we use ravel for the same reason as before

            num_cols = X.select_dtypes(include=np.number).columns
            cat_cols = X.select_dtypes(exclude=np.number).columns

            imputed_list = []

            for model, df_model in X.groupby("model"):

                df_temp = df_model.copy()

                # Use global scaler and global KNN for unseen models in the trainning set
                if model not in self.scalers_ and model not in self.rare_models_:
                    scaled = self.global_scaler_.transform(df_temp[self.metric_features])
                    imputed_scaled = self.global_imputer_.transform(scaled)
                    df_temp[self.metric_features] = self.global_scaler_.inverse_transform(imputed_scaled)
                    imputed_list.append(df_temp[self.metric_features])
                    continue

                # Use the median to impute rare models or global KNN for columns with only missing values 
                if model in self.rare_models_:
                    median_values = self.model_medians_[model] #getting all the medians from that model
                    
                    # Columns where median is NaN (all values missing)
                    cols_missing = median_values[median_values.isna()].index.tolist() 

                    # Fill available medians
                    df_temp[self.metric_features] = df_temp[self.metric_features].fillna(median_values)

                    # Fill completely missing columns with global imputer
                    if cols_missing:
                        #Getting the all df_temp fill with global knn, because we need all columns seen in fit to use it
                        scaled_full = self.global_scaler_.transform(df_temp[self.metric_features])
                        imputed_full = self.global_imputer_.transform(scaled_full)
                        imputed_full = self.global_scaler_.inverse_transform(imputed_full)

                        # Getting it as a dataframe
                        df_imputed_full = pd.DataFrame(
                            imputed_full,
                            columns=self.metric_features,
                            index=df_temp.index
                        )

                        # Only replacing the columns of df_temp where there were only missing values
                        df_temp[cols_missing] = df_imputed_full[cols_missing]

                    #Appending the resulting imputed dataframe to imputed_list    
                    imputed_list.append(df_temp[self.metric_features])
                    continue

                # Use scaler and KNN for models with enough cars
                scaler = self.scalers_[model]
                imputer = self.imputers_[model]

                scaled = scaler.transform(df_temp[self.metric_features])
                imputed_scaled = imputer.transform(scaled)
                df_temp[self.metric_features] = scaler.inverse_transform(imputed_scaled)

                imputed_list.append(df_temp[self.metric_features])

            # Reassemble numeric values
            X_num_imputed = pd.concat(imputed_list, axis=0).loc[X.index]

            # Categorical imputation
            X_cat_imputed = self.imputer_cat.transform(X[cat_cols])
            df_cat = pd.DataFrame(X_cat_imputed, columns=cat_cols, index=X.index)

            # Combine
            X_imputed = pd.concat([X_num_imputed, df_cat], axis=1)
            X_imputed = X_imputed[X.columns]

            # Fix previousOwners by rounding
            X_imputed["previousOwners"] = X_imputed["previousOwners"].round()

            return X_imputed