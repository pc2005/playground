import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iter=100, tolerance=1e-4):
    # randomly select k points as initial centroid
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    
    for i in range(max_iter):
        # calculcate distance to centroids
        distances = np.linalg.norm(X[:, np.newaxis]-centroids, axis=2)
        
        # get labels by distance
        labels = np.argmin(distances, axis=1)
        
        # mean location as new centroid``
        new_centroids = np.array([X[labels==j].mean(axis=0) for j in range(k)])

        # check convergence
        if np.linalg.norm(new_centroids - centroids) < tolerance:
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