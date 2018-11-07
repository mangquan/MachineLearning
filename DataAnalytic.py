import pandas as pd
import statsmodels.formula.api as smf

#
# (b) perform linear regression of apret on tstsc and salar
#     separately and then of apret on both tstsc and salar.


test = pd.read_csv('Retention.csv')

model1 = smf.ols('apret ~ tstsc + salar ', test)

model2 = smf.ols('apret ~  salar ', test)

model3 = smf.ols('apret ~ tstsc ', test)

result1 = model1.fit()

result2 = model2.fit()

result3 = model3.fit()

print(result1.summary())

print(result2.summary())

print(result3.summary())