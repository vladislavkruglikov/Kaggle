import pandas as pd


from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_iris
from DecisionTreeClassifier import DecisionTreeClassifier
from DecisionTreeRegressor import DecisionTreeRegressor
from RandomForestRegressor import RandomForestRegressor

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('DecisionTreeRegressor MSE: {}'.format(mean_squared_error(y_test, y_pred)))


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train);
print('DecisionTreeClassifier accuracy: {}'.format(accuracy_score(y_test, clf.predict(X_test))))


data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

model = RandomForestRegressor(n_estimators=3,
                                    bootstrap=0.8,
                                    max_depth=10,
                                    min_samples_split=3,
                                    max_features=1)
model.fit(X_train, y_train)
MSE = mean_squared_error(y_test, model.predict(X_test))
print('RandomForestRegressor MSE: {}'.format(MSE))