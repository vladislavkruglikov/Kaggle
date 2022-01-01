import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            pred = self.predict(X)
            loss = pred - y

            dw = X.T @ loss
            db = loss.mean(axis=0)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return X @ self.weights + self.bias
