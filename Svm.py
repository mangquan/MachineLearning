from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import numpy as np

CancerDataset = load_breast_cancer()

model = SVC(gamma=0.2, C=0.1)

x_train, x_test, y_train, y_test = train_test_split(CancerDataset.data, CancerDataset.target, random_state=0)

model.fit(x_train, y_train)

print('The accuracy of svc with gamma = 0.2, c = 0.1 is ',model.score(x_test, y_test))

scaler = MinMaxScaler()

scaler.fit(CancerDataset.data)

x_train, x_test, y_train, y_test = train_test_split(scaler.transform(CancerDataset.data), CancerDataset.target, random_state=0)

model.fit(x_train, y_train)

print('The accuracy of svc with scaled data is ', model.score(x_test, y_test))

C = [0.1, 1, 5]

Gamma = [0.1, 1, 5]

best_score = 0

kfolds = 5

for c in C:
    for g in Gamma:
        scores = cross_val_score(SVC(gamma=g, C=c), X=x_train, y=y_train, cv= kfolds)
        score = np.mean(scores)

        if score > best_score:
            best_c = c
            best_g = g
            best_score = score

SelectedModel = SVC(C=best_c, gamma=best_g).fit(x_train, y_train)

print('Best c is ', best_c)

print('Best gamma is ', best_g)

print('Best score on train set is ', best_score)

print('Best score on test set is ', SelectedModel.score(x_test, y_test))


