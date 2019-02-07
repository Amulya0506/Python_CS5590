from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Loading digits data set available in scikit-learn
iris = datasets.load_iris()
X = iris.data
y = iris.target
# considering K values
k_range = [1,25,50]
scores=[]
for i in k_range:
    neigh = KNeighborsClassifier(n_neighbors=i)
    # splitting the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=143)
    # training the model
    neigh.fit(X_train,y_train)
    # predicting the model on test data
    y_predict = neigh.predict(X_test)
    # calculating accuracy
    print("Accuracy Score for K = %d:  %f"%(i ,(metrics.accuracy_score(y_test, y_predict)*100)))
    scores.append(metrics.accuracy_score(y_test, y_predict))
# import Matplotlib (scientific plotting library)


# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()




