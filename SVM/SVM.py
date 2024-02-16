import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

classes = ['malignant' 'benign']

clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("SVM accuracy: ",acc)

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(x_train, y_train)
knn_acc = knn.score(x_test, y_test)
print("KNN Acc: ", knn_acc)
