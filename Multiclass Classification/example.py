from sklearn import datasets
from MulticlassClassification import MulticlassClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)

label_encoding = LabelEncoder()
y = label_encoding.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf = MulticlassClassification()

clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))
