from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(0)
X = np.random.randn(100, 2)

# Define the number of clusters (k)
k = 3

# Initialize the KMeans object
kmeans = KMeans(n_clusters=k)

# Fit the data
kmeans.fit(X)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
