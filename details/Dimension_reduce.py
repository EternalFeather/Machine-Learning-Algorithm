from sklearn import decomposition
from sklearn import tree
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

pca = decomposition.PCA(n_components = 3)
train_reduce = pca.fit_transform(x_train)
test_reduce = pca.transform(x_test)
model = tree.DecisionTreeClassifier()
model.fit(train_reduce, y_train)
print(model.score(train_reduce, y_train))

predictions = model.predict(test_reduce)
print(predictions)
print(accuracy_score(y_test, predictions))