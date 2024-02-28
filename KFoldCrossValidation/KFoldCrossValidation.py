from sklearn.datasets import load_digits
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

digits = load_digits()

X = pd.DataFrame(digits.data)
Y = digits.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(x_train, y_train)
lr.score(x_test, y_test)

svm = SVC(gamma='auto')
svm.fit(x_train, y_train)
svm.score(x_test, y_test)

rf = RandomForestClassifier(n_estimators=40)
rf.fit(x_train, y_train)
rf.score(x_test, y_test)

kf = KFold(n_splits=3)


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data, digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

print("#### calculating cross validation function without library method ####")
print("Logistics: ", scores_logistic)
print("SVM: ", scores_svm)
print("Logistics: ", scores_rf)
print("#############")

print("#### calculating cross validation score with library method ####")
logistics_cross_validation_score = cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), digits.data, digits.target, cv=3)
svm_cross_validation_score = cross_val_score(SVC(gamma='auto'), digits.data, digits.target, cv=3)
random_forrest_cross_validation_score = cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target, cv=3)
print("Logistics: ", logistics_cross_validation_score)
print("SVM: ", svm_cross_validation_score)
print("Logistics: ", random_forrest_cross_validation_score)
print("#############")
