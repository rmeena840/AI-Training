from sklearn.datasets import load_digits
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()

X = pd.DataFrame(digits.data)
Y = digits.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.4)
model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
predict = model.predict()
print("Accuracy:", acc)
print("Predict:", predict)
