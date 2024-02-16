import sklearn
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_csv("insurance_data.csv")
data = data[["age", "bought_insurance"]]

predict = "bought_insurance"

X = np.array(data.drop([predict], axis=1))
# Label
Y = np.array(data[predict])

# Training the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

model = linear_model.LogisticRegression()

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

plt.scatter(X, Y, marker="+", color="red")
plt.show()
