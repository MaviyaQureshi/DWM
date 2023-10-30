from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[1, 2], [4, 7], [9, 5], [12, 2], [5, 1], [3, 4]])

num_cluster = 2
mediod = KMedoids(n_clusters=num_cluster)

mediod.fit(data)

labels = mediod.labels_

centers = data[mediod.medoid_indices_]

colors = ["r.", "g.", "y.", "c."]

for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=10)

for center in centers:
    plt.scatter(center[0], center[1], marker="x", s=100, linewidths=2, color="k")


plt.show()
