import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import re
from rapidfuzz import process, fuzz
import math


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
