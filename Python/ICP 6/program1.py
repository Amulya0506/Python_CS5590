from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plot
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Loading Iris datasets
iris = datasets.load_iris()

# Fitting Naive Bayes model to the data
model = GaussianNB()
model.fit(iris.data, iris.target)
# Making Prediction based on X, Y
expected = iris.target
predicted = model.predict(iris.data)

# Matlib Plot
plot.plot(expected, predicted)
print(expected)
print(predicted)
plot.show()
# Cross Validation compare the predicted and expected values
print(metrics.classification_report(expected, predicted))
# Matrix which is used to find the accuracy of classification
print(metrics.confusion_matrix(expected, predicted))
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)
# Training the Model on Training Set
model.fit(X_train, Y_train)
# Training the Model on Testing Set
Y_predicted = model.predict(X_test)
# Evaluating the Model based on Testing Part
print("Gaussian Model Accuracy is ", metrics.accuracy_score(Y_test, Y_predicted) * 100)