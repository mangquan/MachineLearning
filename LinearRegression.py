
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

AdvertisingData=pd.read_csv('Advertising.csv')
X = AdvertisingData[['Radio', 'TV','Newspaper']].values
Y = AdvertisingData.Sales
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, random_state= 0)
linreg = LinearRegression().fit(X_train, Y_train)
print( 'The MSE of fitted data is ' + str(mean_squared_error(Y_test,linreg.predict(X_test))))