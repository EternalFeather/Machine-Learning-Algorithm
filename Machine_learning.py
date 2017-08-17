from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser(description = 'Machine_learning.py')
parser.add_argument("-mode", required = True, help = "Choose mode of machine learning modules.")
opt = parser.parse_args()

# Linear regression
from sklearn.linear_model import LinearRegression
# Logical regression
from sklearn.linear_model import LogisticRegression
# Support vector machine
from sklearn.svm import SVC
# Decision tree
from sklearn.tree import DecisionTreeClassifier
# Random forest
from sklearn.ensemble import RandomForestClassifier
# K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
# Naive bayesian
from sklearn.naive_bayes import GaussianNB
# K-means
from sklearn.cluster import KMeans
# Dimension reduce(PCA)
from sklearn.decomposition import PCA
# Boost
from sklearn.ensemble import GradientBoostingClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

if opt.mode == 'LinearRegression':
	linear = LinearRegression()
	linear.fit(x_train, y_train)
	print("MSG : Training accuracy = ", linear.score(x_train, y_train))
	prediction = linear.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction.astype(int)))
elif opt.mode == 'LogisticRegression':
	logistic = LogisticRegression()
	logistic.fit(x_train, y_train)
	print("MSG : Training accuracy = ", logistic.score(x_train, y_train))
	prediction = logistic.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
elif opt.mode == 'SVM':
	svm = SVC()
	svm.fit(x_train, y_train)
	print("MSG : Training accuracy = ", svm.score(x_train, y_train))
	prediction = svm.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
elif opt.mode == 'DecisionTree':
	decisiontree = DecisionTreeClassifier()
	decisiontree.fit(x_train, y_train)
	print("MSG : Training accuracy = ", decisiontree.score(x_train, y_train))
	prediction = decisiontree.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
elif opt.mode == 'RandomForest':
	randomforest = RandomForestClassifier()
	randomforest.fit(x_train, y_train)
	print("MSG : Training accuracy = ", randomforest.score(x_train, y_train))
	prediction = randomforest.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
elif opt.mode == 'KNN':
	knn = KNeighborsClassifier()
	knn.fit(x_train, y_train)
	print("MSG : Training accuracy = ", knn.score(x_train, y_train))
	prediction = knn.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
elif opt.mode == 'NaiveBayes':
	naivebayes = GaussianNB()
	naivebayes.fit(x_train, y_train)
	print("MSG : Training accuracy = ", naivebayes.score(x_train, y_train))
	prediction = naivebayes.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
elif opt.mode == 'K-means':
	kmeans = KMeans(n_clusters = 3, random_state = 0)
	kmeans.fit(x_train, y_train)
	prediction = kmeans.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
elif opt.mode == 'Dimension_reduce':
	pca = PCA(n_components = 3)
	decisiontree = DecisionTreeClassifier()
	train_reduce = pca.fit_transform(x_train)
	test_reduce = pca.transform(x_test)
	decisiontree.fit(train_reduce, y_train)
	print("MSG : Training accuracy = ", decisiontree.score(train_reduce, y_train))
	prediction = decisiontree.predict(test_reduce)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
elif opt.mode == 'Boost':
	boost = GradientBoostingClassifier()
	boost.fit(x_train, y_train)
	print("MSG : Training accuracy = ", boost.score(x_train, y_train))
	prediction = boost.predict(x_test)
	print("MSG : Testing accuracy = ", accuracy_score(y_test, prediction))
else:
	print("MSG : No such module.")
	exit()








