import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class OctreeNode:
    def __init__(self, points, depth, max_depth):
        self.points = points
        self.children = []
        self.depth = depth
        self.max_depth = max_depth
        self.centroid = np.mean(points, axis=0) if points.size else None
        if self.depth < self.max_depth and len(points) > 1:
            self.subdivide()

    def subdivide(self):
        # Find the midpoint along each axis
        midpoints = np.mean(self.points, axis=0)
        for i in range(8):
            # Determine which points belong to this octant
            octant_filter = np.all(
                [
                    (self.points[:, dim] < midpoints[dim]) if (i & (1 << dim)) == 0 else (self.points[:, dim] >= midpoints[dim])
                    for dim in range(3)
                ],
                axis=0
            )
            octant_points = self.points[octant_filter]
            if len(octant_points) > 0:
                self.children.append(OctreeNode(octant_points, self.depth + 1, self.max_depth))

def build_octree(points, max_depth):
    return OctreeNode(points, 0, max_depth)

def collect_centroids(node):
    if not node.children:
        return [node.centroid] if node.centroid is not None else []
    centroids = []
    for child in node.children:
        centroids.extend(collect_centroids(child))
    return centroids

def octree_subsample(points, max_depth):
    octree = build_octree(points, max_depth)
    centroids = collect_centroids(octree)
    return np.array(centroids), octree

# Generate random points
np.random.seed(42)
points = np.random.rand(2000, 3)

# Define maximum depth for the octree
max_depth = 2

# Build octree and get subsampled points
subsampled_points, octree = octree_subsample(points, max_depth)

# Prepare for animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', marker='o')

# Store the points to be animated at each level
levels_points = []

def collect_levels(node, level_points, depth):
    if node.depth == depth:
        level_points.extend(collect_centroids(node))
    else:
        for child in node.children:
            collect_levels(child, level_points, depth)

for depth in range(max_depth + 1):
    level_points = []
    collect_levels(octree, level_points, depth)
    levels_points.append(np.array(level_points))

def update(frame):
    sc._offsets3d = (levels_points[frame][:, 0], levels_points[frame][:, 1], levels_points[frame][:, 2])
    ax.set_title(f'Octree Subsampling at Depth {frame}')
    return sc,

ani = FuncAnimation(fig, update, frames=range(max_depth + 1), blit=False, repeat=True)

plt.show()