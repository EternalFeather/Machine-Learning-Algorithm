from sklearn import linear_model
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
print(linear.score(x_train, y_train))

print('Coefficient: %s\nIntercept: %s' % (linear.coef_, linear.intercept_))

predictions = linear.predict(x_test)
print(predictions.astype(int), "\n", y_test)
print(accuracy_score(y_test, predictions.astype(int)))

