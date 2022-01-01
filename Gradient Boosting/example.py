from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from GradientBoostingRegressor import GradientBoostingRegressor


data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

model = GradientBoostingRegressor(n_estimators=10, max_depth=1)
model.fit(X_train, y_train)
MSE = mean_squared_error(y_test, model.predict(X_test))
print('MSE: {}'.format(MSE))