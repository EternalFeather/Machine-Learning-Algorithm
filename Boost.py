from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 1, random_state = 0)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))

predictions = model.predict(x_test)
print(predictions)
print(accuracy_score(y_test, predictions))