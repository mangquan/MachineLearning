from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

DataCancer = load_breast_cancer()
print(DataCancer.keys())
print(DataCancer.DESCR)

X_features = DataCancer.data
Y_targetClass = DataCancer.target

scaler = preprocessing.StandardScaler()

X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_targetClass, random_state=0)

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

LogModel = LogisticRegression(C=1)


LogModel.fit(X_train_scaled, Y_train)

print(LogModel.score(X_test_scaled, Y_test))

#######################################################################################

kfolds = 5

candidateC = [0.01, 0.1, 1, 10, 100]

highScore = 0

bestC = 0

for candidate in candidateC:
    Model = LogisticRegression(C=candidate)

    scores = cross_val_score(Model, X_train_scaled, Y_train, cv=kfolds)

    score = np.mean(scores)

    print('This score of C=' + str(candidate) + ':' + str(score))

    if score > highScore:
        bestC = candidate
        highScore = score

SelectedModel = LogisticRegression(C=bestC).fit(X_train_scaled, Y_train)

test_score = SelectedModel.score(X_test_scaled, Y_test)

print('Final score is', highScore)
print('Best C is', bestC)
print('Test score is ', test_score)



