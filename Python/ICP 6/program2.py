import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# importing the csv data
data_set = pd.read_csv("sample_stocks.csv")
data = data_set.iloc[:,:].values

# plotting the loaded data
plt.scatter(data[:,0],data[:,1], label='True Position')
plt.show()
k_means = KMeans(n_clusters=4, random_state=0).fit(data)
print(k_means.labels_)
print(k_means.cluster_centers_)

# plotting the labels
plt.scatter(data[:,0],data[:,1], c=k_means.labels_, cmap='rainbow')
# plotting the centroids
plt.scatter(k_means.cluster_centers_[:,0] ,k_means.cluster_centers_[:,1], marker = 'x')
plt.show()

# To plot the elbow graph
# Elbow Method - Technique to determine the K Location of a bend (knee) in the plot is generally considered as an indicator of the appropriate number of clusters.
ran = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in ran]
score = [kmeans[i].fit(data).score(data) for i in range(len(kmeans))]
plt.plot(ran,score)
plt.show()



