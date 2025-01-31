import numpy as np

def dbscan(X, eps, min_samples):
    """
    Perform DBSCAN clustering from scratch using numpy.
    
    Parameters:
    X : ndarray of shape (n_samples, n_features)
        The input data points.
    eps : float
        The maximum distance between two samples for them to be considered as neighbors.
    min_samples : int
        The minimum number of points required to form a dense region (a cluster).
        
    Returns:
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point. Noisy samples are labeled as -1.
    """
    
    # Initialize labels for each point as unclassified (-1)
    n_points = X.shape[0]
    labels = np.full(n_points, -1)
    cluster_id = 0
    
    # Function to calculate neighbors
    def region_query(point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= eps)[0]
    
    # Expand cluster from a given point
    def expand_cluster(point_idx, neighbors):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if labels[neighbor_idx] == -1:  # If it's unclassified, mark it as part of the cluster
                labels[neighbor_idx] = cluster_id
            
            elif labels[neighbor_idx] == -1:  # If it's noise, change it to a border point
                labels[neighbor_idx] = cluster_id
            
            # Get neighbors for the current neighbor and check if it can expand the cluster
            new_neighbors = region_query(neighbor_idx)
            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate((neighbors, new_neighbors))
            
            i += 1
    
    # Main loop through each point
    for point_idx in range(n_points):
        if labels[point_idx] != -1:  # Already processed
            continue
        
        neighbors = region_query(point_idx)
        
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            expand_cluster(point_idx, neighbors)
            cluster_id += 1
    
    return labels

# Example usage
np.random.seed(0)
X = np.random.rand(100, 2) * 10  # Generate some random 2D data
eps = 1.0  # Maximum distance for points to be considered neighbors
min_samples = 5  # Minimum number of points to form a cluster

labels = dbscan(X, eps, min_samples)

# Plotting the result
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("DBSCAN Clustering")
plt.show()