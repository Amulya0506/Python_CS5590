from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics

# Loading digits data set available in scikit-learn
iris = datasets.load_digits()
data = iris.data
labels = iris.target
# splitting train and test data for linear kernel
X_train, X_test,y_train,y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
# splitting train and test data for  rbf kernel
X_train_rbf, X_test_rbf, y_train_rbf, y_test_rbf = train_test_split(data, labels, test_size=0.2, random_state=1)
# define  linear kernel model
linear_model = SVC(kernel='linear')
# define rbf kernel
rbf_model = SVC(kernel='rbf')
# training data in linear kernel model
linear_model.fit(X_train, y_train)
# predict the test data using linear kernel model
linear_prediction = linear_model.predict(X_test)
# print(linear_prediction)
# accuracy score for linear kernel
print("linear kernel Accuracy score is", metrics.accuracy_score(y_test, linear_prediction) * 100)
# training data in RBF kernel model
rbf_model.fit(X_train_rbf, y_train_rbf)
# predict the test data using rbc kernel
rbf_prediction = rbf_model.predict(X_test_rbf)
# print(rbf_prediction)
# calc accuracy for RBF kernel
print("RBF kernel accuracy score is", metrics.accuracy_score(y_test_rbf, rbf_prediction) * 100)
