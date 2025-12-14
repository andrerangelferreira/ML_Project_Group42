# ------ IMPORTS ------


# ------ Standard Library Imports ------
import math
import re
# ------ Data Manipulation ------
import pandas as pd
import numpy as np
import ast

# ------ Visualization ------
import matplotlib.pyplot as plt   # use pyplot instead of pylab
import seaborn as sns

# ------ Machine Learning - Preprocessing ------
from sklearn.preprocessing import (
    StandardScaler, 
    RobustScaler, 
    OneHotEncoder, 
    LabelEncoder
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.base import clone

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv 
# now import normally from model_selection
from sklearn.model_selection import HalvingRandomSearchCV


# ------ Evaluation metrics ------
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    make_scorer
)

# ------ Machine Learning - Algorithms ------
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPClassifier

# ------ Missing Data Imputation ------
from sklearn.impute import KNNImputer

# ------ Statistics & Tests ------
import scipy.stats as stats
from scipy.stats import chi2_contingency

# ------ String Matching / Fuzzy Matching ------
from rapidfuzz import process, fuzz

# ------ Pipeline ------
from sklearn.pipeline import Pipeline

import joblib






# ------ FUNCTIONS CREATED THROUGHOUT THE PROJECT ------

def normalize_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower().strip()  # lowercase + remove spaces
    x = re.sub(r'[^a-z0-9\s\-]', '', x)  # keep alphanumeric & hyphens
    x = re.sub(r'\s+', ' ', x)  # collapse multiple spaces
    return x

def num_per_cat(data, numerical_var, cat_var):
    sns.set()

    # Computing mean income per education level
    CLV_mean = data.groupby(cat_var)[numerical_var].mean().reset_index().sort_values(by=numerical_var, ascending= False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=CLV_mean, x=cat_var, y=numerical_var, hue = cat_var, legend=False)

    plt.title(f"Average {numerical_var} by {cat_var}")
    plt.xlabel(cat_var)
    plt.ylabel(numerical_var)
    plt.xticks()
    plt.tight_layout()
    plt.show()


def boxplotter(data, metric_features, n_rows, n_cols):

    # Plot ALL Numeric Variables' Histograms in one figure

    sns.set(style= "darkgrid", context= "notebook") ## Reset to darkgrid

    # Prepare figure. Create individual axes where each histogram will be placed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10),tight_layout=True)

    # Plot data
    # Iterate across axes objects and associate each histogram (hint: use the ax.hist() instead of plt.hist()):
    for ax, feat in zip(axes.flatten(), metric_features): # Notice the zip() function and flatten() method
        sns.boxplot(x=data[feat], ax=ax)
        
    # Layout
    # Add a centered title to the figure:
    plt.suptitle("Numeric Variables' Box Plots", fontsize=20, y=1.02, fontweight='bold')
    plt.show()

def custom_combiner(feature, category):
    return f"{category}"

def correlation_matrix(data, threshold):
    
    corr = data.corr(method="pearson")
    corr = corr.round(2)

    mask_annot = np.absolute(corr.values) >= threshold

    annot = np.where(mask_annot, corr.values, np.full(corr.shape,"")) 

    fig = plt.figure(figsize=(10, 8))

    # Plotting the heatmap of the correlation matrix
    sns.heatmap(data=corr, 
                annot=annot, # Specifing custom annotation
                fmt='s', # The annotation matrix now has strings, so we need to explicitly say this
                vmin=-1, vmax=1, 
                center=0,
                square=True, # Make each cell square-shaped
                linewidths=.5, # Adding lines between cells
                cmap='PiYG' # Diverging color map
                )

    plt.show()

def TestIndependence(X,y,var,alpha=0.05):        
    dfObserved = pd.crosstab(y,X) 
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    if p<alpha:
        result="{0} is IMPORTANT for Prediction".format(var)
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)
    print(result)

def plot_importance(coef,name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(8,10))
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using " + name + " Model")
    plt.show()

def calculate_regression_metrics(y_true, y_pred):
    
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, rmse


def evaluate_best_model_with_cv(best_model, X, y, model_name, cv=5):
    """
    Performs k-fold cross-validation and plots predictions vs actual values 
    for both train and validation folds.
    
    Parameters:
    -----------
    best_model : sklearn estimator
        The best model from RandomizedSearchCV (best_estimator_)
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    cv : int
        Number of folds for cross-validation (default=5)
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Arrays to store predictions
    train_actuals = []
    train_predictions = []
    val_preds = np.zeros(len(y))
    
    # Perform cross-validation manually to get both train and val predictions
    for train_idx, val_idx in kfold.split(X):
        # Use .iloc for proper DataFrame indexing
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
        
        # Clone and train model on this fold
        fold_model = clone(best_model)
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Get predictions for train fold
        train_fold_preds = fold_model.predict(X_train_fold)
        train_actuals.extend(y_train_fold)
        train_predictions.extend(train_fold_preds)
        
        # Get predictions for validation fold
        val_preds[val_idx] = fold_model.predict(X_val_fold)
    
    # Convert lists to arrays for metrics calculation
    train_actuals = np.array(train_actuals)
    train_predictions = np.array(train_predictions)
    
    # Calculate metrics for aggregated training folds
    train_r2, train_mae, train_rmse = calculate_regression_metrics(train_actuals, train_predictions)

    # Calculate validation metrics
    val_r2, val_mae, val_rmse = calculate_regression_metrics(y, val_preds)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Training folds predictions vs actual
    axes[0].scatter(train_actuals, train_predictions, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].plot([train_actuals.min(), train_actuals.max()], 
                 [train_actuals.min(), train_actuals.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title(f'Training Results of {model_name}\nR²={train_r2:.4f}, MAE={train_mae:.2f}, RMSE={train_rmse:.2f}', 
                      fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation predictions vs actual
    axes[1].scatter(y, val_preds, alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
    y_min, y_max = np.min(y), np.max(y)
    axes[1].plot([y_min, y_max], [y_min, y_max], 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Values', fontsize=12)
    axes[1].set_ylabel('Predicted Values', fontsize=12)
    axes[1].set_title(f'Validation Resultsof {model_name}\nR²={val_r2:.4f}, MAE={val_mae:.2f}, RMSE={val_rmse:.2f}', 
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()