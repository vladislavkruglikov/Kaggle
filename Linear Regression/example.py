from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from LinearRegression import LinearRegression

X, y = datasets.load_diabetes(return_X_y=True)

# We will be using only one feature with index 2
X = X[:, [2]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression(n_iters=10000)

model.fit(X_train, y_train)

MSE_train = mean_squared_error(y_train, model.predict(X_train))
MSE_test = mean_squared_error(y_test, model.predict(X_test))

print('Train set MSE: {}'.format(MSE_train))
print('Test set MSE: {}'.format(MSE_test))
