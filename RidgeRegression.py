from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np

dataset = load_boston()
X=dataset.data
Y=dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state= 0)

model = Ridge(alpha= 100)
model.fit(X_train, Y_train)


print( 'The R2 score of Ridge with alpha =100 is '+ str( model.score(X_test,Y_test)))
print('The number of non zero coefficients is ' + str(np.where(model.coef_ != 0)[0].shape[0]) )
print(' ')

model2 = Lasso(alpha=100)
model2.fit(X_train, Y_train)

print( 'The R2 score of Lasso with alpha =100 is '+ str( model2.score(X_test,Y_test)))
print('The number of non zero coefficients is ' + str(np.where(model2.coef_ != 0)[0].shape[0]) )
print(' ')

model3 = Lasso(alpha=0.001)
model3.fit(X_train, Y_train)

print( 'The R2 score of Ridge with alpha =0.001 is '+ str( model3.score(X_test,Y_test)))