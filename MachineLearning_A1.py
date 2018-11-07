import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from IPython.display import Image
import statsmodels.formula.api as smf
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats

dataset = load_boston()
print(dataset.keys())

houseData = pd.DataFrame(dataset.data)
houseData.columns = [dataset.feature_names]
