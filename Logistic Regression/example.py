from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

print('Accuracy score for train data: {}'.format(accuracy_score(y_train, model.predict(X_train))))
print('Accuracy score for train data: {}'.format(accuracy_score(y_test, model.predict(X_test))))
