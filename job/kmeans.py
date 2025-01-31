import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    """
    Initialize k centroids by randomly selecting k points from the dataset.
    """
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    """
    Assign each point in X to the nearest centroid.
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """
    Calculate the new centroids as the mean of all points assigned to each cluster.
    """
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    Apply K-means algorithm to find k clusters in the dataset X.
    """
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            print(f"Converged after {i+1} iterations")
            break
        centroids = new_centroids
    return centroids, labels

# Example usage
np.random.seed(42)
X = np.random.rand(300, 2)  # Generate some random 2D data
k = 3  # Number of clusters

centroids, labels = kmeans(X, k)

# Plotting the result
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("K-means Clustering")
plt.show()