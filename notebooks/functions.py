# ------ Standard Library Imports ------
import math
import re
# ------ Data Manipulation ------
import pandas as pd
import numpy as np

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
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoCV, Ridge

# ------ Machine Learning - Algorithms ------
from sklearn.cluster import DBSCAN

# ------ Missing Data Imputation ------
from sklearn.impute import KNNImputer

# ------ Statistics & Tests ------
import scipy.stats as stats
from scipy.stats import chi2_contingency

# ------ String Matching / Fuzzy Matching ------
from rapidfuzz import process, fuzz


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