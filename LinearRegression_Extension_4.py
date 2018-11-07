from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

AutoData=read_csv('Auto_modify.csv')

X_auto_hp=AutoData.horsepower.values.reshape(-1,1) # define features: horsepower
Y_new = AutoData.mpg.values.reshape(-1,1).astype(int)


lab_enc = preprocessing.LabelEncoder()
Y_new = lab_enc.fit_transform(Y_new)

X_train, X_test, Y_train, Y_test = train_test_split(X_auto_hp, Y_new.reshape(-1,), random_state=0)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, Y_train)

print('The R2 score of this model without housrsepower^2 term is ' + str(model.score(X_test, Y_test)))

X_auto_hp_2 = np.concatenate((X_auto_hp,np.square(AutoData.horsepower.values.reshape(-1,1))),axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_auto_hp_2, Y_new, random_state=0)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, Y_train)

print('The R2 score of this model with housrsepower^2 term is ' + str(model.score(X_test, Y_test)))
