from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import  load_iris
from sklearn.tree import  DecisionTreeClassifier
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble import AdaBoostClassifier

irisData = load_iris()
X = irisData.data
y = irisData.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

# model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=0)
#
# model.fit(x_train, y_train)

# dot_data = export_graphviz(model, class_names=irisData.target_names, feature_names=irisData.feature_names, out_file=None)

# graph = graphviz.Source(dot_data)
#
# graph.render(view=True)

# print(model.score(x_test, y_test))

modelRamdonforest = RandomForestClassifier(n_estimators=4, max_depth=2, max_features=2, random_state=0)

modelRamdonforest.fit(x_train, y_train)

print(modelRamdonforest.score(x_test, y_test))