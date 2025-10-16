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
