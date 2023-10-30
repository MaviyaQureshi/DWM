"""   Implement Clustering Algorithms using Python    """


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create a sample dataset (replace this with your own data)
data = np.array([[1, 2], [5, 6], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Initialize the K-Means clustering algorithm with the desired number of clusters
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)

# Fit the model to the data
kmeans.fit(data)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the cluster centers
centers = kmeans.cluster_centers_

# Visualize the clusters
colors = ["g.", "r.", "c.", "y."]
for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=10)

for center in centers:
    plt.scatter(center[0], center[1], marker="x", s=200, linewidths=3, color="k")

plt.show()
