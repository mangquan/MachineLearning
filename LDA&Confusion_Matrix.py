import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import numpy as np


DigitsData=load_digits()
print(DigitsData.keys())
print(DigitsData.DESCR)
print(DigitsData.data[1])

plt.gray()
plt.matshow(DigitsData.images[1000])
plt.show()

X = DigitsData.data

Y = DigitsData.target

Y_label = Y==9

print('The number of True label is:', sum(Y_label == True))

print('The number of False label is:', sum(Y_label == False))

x_train, x_test, y_train, y_test = train_test_split(X, Y_label, random_state = 0)

LDA_model = LinearDiscriminantAnalysis()

LDA_model.fit(x_train, y_train)

LDA_model.score(x_train, y_train)

y_predict = LDA_model.predict(x_test)

print(confusion_matrix(y_test, y_predict))

print(precision_score(y_test, y_predict))

print(recall_score(y_test, y_predict))



