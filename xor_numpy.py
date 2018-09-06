from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import numpy as np
np.random.seed(1)


x = np.array([[0, 0], 
			  [0, 1],
			  [1, 0],
			  [1, 1]])

y = np.array([[0],
			  [1],
			  [1],
			  [0]])

def linear_regression():
	linear = LinearRegression()
	linear.fit(x, y)
	print("MSG : Training accuracy = ", linear.score(x, y), "\n")
	print("Pred : \n")
	print(linear.predict(x).astype(int))
	print("Truth : \n")
	print(y)


def sigmoid(x, derive=False):
	if derive:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))


def feedforward_neural_networks(lr):
	W_0 = 2 * np.random.random((2, 5)) - 1
	W_1 = 2 * np.random.random((5, 1)) - 1

	for i in range(60000):

		l_0 = x
		l_1 = sigmoid(np.dot(l_0, W_0))
		l_2 = sigmoid(np.dot(l_1, W_1))

		l_2_error = y - l_2

		if i % 10000 == 0:
			print('MSG : Mean Square Error is ', str(np.mean((l_2_error) ** 2)))

		l_2_delta = l_2_error * sigmoid(l_2, derive=True)
		l_1_error = l_2_delta.dot(W_1.T)
		l_1_delta = l_1_error * sigmoid(l_1, derive=True)

		W_1 += l_1.T.dot(lr * l_2_delta)
		W_0 += l_0.T.dot(lr * l_1_delta)

	print('MSG : Output after training is :')
	print(l_2)


if __name__ == '__main__':
	# linear_regression()
	feedforward_neural_networks(0.1)