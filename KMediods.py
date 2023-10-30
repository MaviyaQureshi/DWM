import numpy as np
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Create a sample dataset (replace this with your own data)
data = np.array([[1, 2], [5, 6], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Initialize the K-Medoids clustering algorithm with the desired number of clusters
num_clusters = 2
kmedoids = KMedoids(n_clusters=num_clusters)

# Fit the model to the data
kmedoids.fit(data)

# Get the cluster medoids and labels
medoids = data[kmedoids.medoid_indices_]
labels = kmedoids.labels_

# Visualize the clusters
colors = ["g.", "r.", "c.", "y."]
for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=10)

for medoid in medoids:
    plt.scatter(medoid[0], medoid[1], marker="x", s=200, linewidths=3, color="k")

plt.show()
