import numpy as np


from DecisionTreeRegressor import DecisionTreeRegressor


def bootstrap_sample(X, y, size):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=int(n_samples * size), replace=True)
    return(X[idxs], y[idxs])


class RandomForestRegressor:
    def __init__(self, min_samples_split=2, max_depth=100, n_estimators=5, bootstrap=0.9, max_features=1):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.models = []
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_features = max_features

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_sample, y_sample = bootstrap_sample(X, y, size=self.bootstrap)
            model = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        n_samples, n_features = X.shape
        res = np.zeros(n_samples)
        for model in self.models:
            res += model.predict(X)
        return res / self.n_estimators
