import numpy as np

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(self,
                 n_estimators=100,
                 max_depth=1,
                 max_features=1,
                 learning_rate=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.lr = learning_rate
        self.models = []

    def fit(self, X, y):
        model_pred_sum = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = y - model_pred_sum
            model = DecisionTreeRegressor(max_depth=self.max_depth,
                                          max_features=self.max_features)
            model.fit(X, residuals)
            model_pred_sum += self.lr * model.predict(X)
            self.models.append(model)

    def predict(self, X):
        k = np.zeros(len(X))
        for model in self.models:
            k += self.lr * model.predict(X)

        return k