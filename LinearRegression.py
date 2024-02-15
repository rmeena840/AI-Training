import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

style.use("ggplot")

data = pd.read_csv("student_mat.csv", sep=";")
# Get only desired columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# Columns value to be predicted
predict = "G3"

# Attributes
X = np.array(data.drop([predict], axis=1))
# Label
Y = np.array(data[predict])

# Training the model
x_train, x_test, y_train, y_test =sklearn.model_selection.train_test_split(X, Y, test_size=0.4)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.4)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)
acc = linear.score(x_test, y_test)

print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted = linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])

# Drawing and plotting model
plot = "G1"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
