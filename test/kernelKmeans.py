from sklearn.datasets import make_moons
from sklearn_extra.cluster import KernelKMeans
import matplotlib.pyplot as plt

# Generate synthetic data (moons)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Kernel K-means clustering
kernel_kmeans = KernelKMeans(n_clusters=2, kernel="rbf", gamma=1)
kernel_kmeans.fit(X)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kernel_kmeans.labels_, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kernel_kmeans.cluster_centers_[:, 0], kernel_kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Cluster Centers')
plt.title('Kernel K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
