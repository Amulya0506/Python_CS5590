from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plot
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Loading iris data set available in scikit-learn
iris = datasets.load_iris()
data = iris.data
labels = iris.target
target_names = iris.target_names
print("Classes: ",target_names)

# splitting the data set for training and testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)

# Applying Linear Discriminant Analysis
lda_clf = LinearDiscriminantAnalysis(n_components=2)
X_r = lda_clf.fit(data, labels).transform(data)
# Making Prediction on complete data
expected = labels
lda_predict = lda_clf.predict(data)
# print(expected)
# print(lda_predict)
print("Linear Discriminant Model Accuracy: ", metrics.accuracy_score(labels, lda_predict) * 100)

# Applying Linear Discriminant Analysis on partitioned data
lda_clf.fit(X_train, y_train).transform(data)
# Making Predictions on test data
lda_y_predict = lda_clf.predict(X_test)
# print(y_test)
# print(lda_y_predict)
print("Linear Discriminant Model Accuracy on partitioned data: ", metrics.accuracy_score(y_test, lda_y_predict) * 100)

# Applying Logistic Regression
logit_clf = LogisticRegression()
logit_clf.fit(X_train,y_train)
logit_y_predict = logit_clf.predict(X_test)
# print(y_test)
# print(logit_y_predict)
print("Logistic Regrssion Model Accuracy on partitioned data: ", metrics.accuracy_score(y_test, logit_y_predict) * 100)

# plotting the classification
plot.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plot.scatter(X_r[labels == i, 0], X_r[labels== i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plot.legend(loc='best', shadow=False, scatterpoints=1)
plot.title('LDA of IRIS dataset')
plot.show()

