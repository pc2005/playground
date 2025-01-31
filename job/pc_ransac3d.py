import numpy as np

def fit_plane(points):
    p1, p2, p3 = points
    v1 = p1 - p2
    v2 = p2 - p3
    normal = np.cross(v1, v2)
    a,b,c = normal
    d = -np.dot(normal, p1)
    return a,b,c,d

def calc_distance(point, plane_parameters):
    a,b,c,d = plane_parameters
    x,y,z = point
    return abs(a*x + b*y + c*z +d)/np.sqrt(a**2 + b**2 + c**2)

def ransac_3d(points, threshold, max_iterations, min_inliers):
    best_plane = None
    best_inliers = []

    for _ in range(max_iterations):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        plane_params = fit_plane(sample)
        inliers = []
        for i, point in enumerate(points):
            if calc_distance(point, plane_params) < threshold:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = plane_params

        if len(best_inliers) > min_inliers:
            break

    return best_plane, best_inliers


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    n_points = 100
    inlier_points = np.random.uniform(-10, 10, (n_points, 3))
    plane_params = [1, -2, 1, -3]
    noise = np.random.normal(0, 0.1, inlier_points.shape)
    inlier_points[:, 2] = (-plane_params[0] * inlier_points[:, 0] - 
                           plane_params[1] * inlier_points[:, 1] - 
                           plane_params[3]) / plane_params[2] + noise[:, 2]

    outliers = np.random.uniform(-10, 10, (int(n_points * 0.2), 3))
    points = np.vstack((inlier_points, outliers))

    # Run RANSAC
    threshold = 0.2
    max_iterations = 1000
    min_inliers = n_points // 2
    best_plane, best_inliers = ransac_3d(points, threshold, max_iterations, min_inliers)
    best_plane /= np.min(np.abs(best_plane))
   
    print("Best plane parameters:", best_plane)
    print("Number of inliers:", len(best_inliers))