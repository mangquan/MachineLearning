import statsmodels.formula.api as smf
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

credit = read_csv('Credit2.csv')

# Assign values to discrete variables
credit.Student = credit.Student.map({'No': 0, 'Yes': 1})
credit.Married = credit.Married.map({'No': 0, 'Yes': 1})
credit.Ethnicity = credit.Ethnicity.map({'Caucasian': 0, 'Asian': 1, 'African American': 2})

# Insert the interaction term
credit['Interaction'] = credit.Income * credit.Student
credit['Interaction2'] = credit.Limit * credit.Student

# Create the model
model = smf.ols('Balance ~ Student + Limit + Income + Interaction + Interaction2', credit)

# fit the model
fittingResult = model.fit()

# the result shows very low p-values regarding to these parameters which mean the actual relationships are not additive
print(fittingResult.summary)


# Section 2

X = credit[['Limit', 'Income', 'Student']].values

Y = credit[['Balance']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

model = LinearRegression().fit(X_train, Y_train)

print('The R2 score of this model without interaction term is' + str(model.score(X_test, Y_test)))

# Retrain the model with interaction term

X2 = credit[['Limit', 'Income', 'Student', 'Interaction', 'Interaction2']].values

X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, random_state=0)

model = LinearRegression().fit(X_train, Y_train)

print('The R2 score of this model with interaction term is' + str(model.score(X_test, Y_test)))





