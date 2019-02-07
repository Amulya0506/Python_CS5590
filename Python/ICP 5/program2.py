import numpy as np
import matplotlib.pyplot as plt
import random

def create_cluster(X, centroid_pts):
    cluster = {}
  # To calculate normalized distance
    for x in X:
        value = min([(i[0],np.linalg.norm(x - centroid_pts[i[0]]))for i in enumerate(centroid_pts)], key=lambda s:s[1])[0]
        try:
            cluster[value].append(x)
        except:
            cluster[value] = [x]
    return cluster


def calculate_new_center(cluster):
    keys =sorted(cluster.keys())
    # Calculating the mean for new centroid points
    newmu = np.array([(np.mean(cluster[k],axis = 0))for k in keys])
    return newmu

# To compare the new and old centroids
def matched(new_centroids, old_centroids):
    return (set([tuple(a)for a in new_centroids]) == set([tuple(a)for a in old_centroids]))

def Apply_Kmeans(X, K, N):
    # selecting random centroids from dataset and by number of clusters.
    old_centroids = np.random.randint(N, size = K)
    old_centroid_pts = np.array([X[i]for i in old_centroids])
    print("old :",old_centroids)
    print(old_centroid_pts)
    cluster_info = create_cluster(X, old_centroid_pts)
    print("Initial cluster information:")
    print(cluster_info)
    new_centroids = np.random.randint(N, size=K)
    new_centroid_pts = np.array([X[i] for i in new_centroids])
    print("new :", new_centroid_pts)
    itr = 0
    print("Graph after selecting initial clusters with initial centroids:")
    plot_cluster(old_centroid_pts,cluster_info,itr)
    # comparing new centroid and old centroid
    while not matched(new_centroid_pts, old_centroid_pts):
        itr = itr + 1
        old_centroid_pts = new_centroid_pts
        cluster_info = create_cluster(X,new_centroid_pts)
        plot_cluster(new_centroid_pts, cluster_info,itr)
        new_centroid_pts = calculate_new_center(cluster_info)
    print("Results after final iteration:")
    # Plotting centroids points
    plot_cluster(new_centroid_pts, cluster_info, itr)
    return

# To plot the clusters in different colors
def plot_cluster(mu,cluster, itr):
    color = 10 * ['r.','g.','k.','c.','b.','m.']
    print('Iteration number : ',itr)
    for l in cluster.keys():
        for m in range(len(cluster[l])):
            plt.plot(cluster[l][m][0], cluster[l][m][1], color[l], markersize=10)
    plt.scatter(mu[:,0],mu[:,1],marker = 'x', s = 150, linewidths = 5, zorder = 10)
    plt.show()

def init_graph(N, p1, p2):
    X = np.array([(random.choice(p1),random.choice(p2))for i in range(N)])
    return X


def Simulate_Clusters():
    print(".........Starting Cluster Simulation.........")
    N = int(input("Enter the number of points: "))
    K = int(input("Enter the number of Clusters: "))
    # Creating Data sets of height and weight of persons
    height = np.array([20.22, 30, 10, 25, 34, 48, 18, 29, 32, 39, 42,22,34,22,34,33,44])
    weight = np.array([10, 12, 14, 20, 16, 18,22,45,67,56,22,33])
    X = init_graph(N, height, weight)
    print(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    Apply_Kmeans(X, K, N)

if __name__ == '__main__':
    Simulate_Clusters()
