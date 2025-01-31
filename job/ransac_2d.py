import numpy as np
import random

def fit_line(points):
    """
    Fits a line through two points in 2D.
    Returns the line parameters (a, b, c) for the equation: ax + by + c = 0.
    """
    (x1, y1), (x2, y2) = points
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c

def calculate_distance(point, line_params):
    """
    Calculates the perpendicular distance from a point to a line.
    """
    a, b, c = line_params
    x, y = point
    return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

def ransac_2d(points, threshold, max_iterations, min_inliers):
    """
    RANSAC algorithm for fitting a line to 2D points.
    """
    best_line = None
    best_inliers = []

    for _ in range(max_iterations):
        # Randomly select 2 points to define a line
        sample = points[np.random.choice(points.shape[0], 2, replace=False)]
        
        # Fit a line to the sample points
        line_params = fit_line(sample)
        
        # Determine inliers
        inliers = []
        for i, point in enumerate(points):
            distance = calculate_distance(point, line_params)
            if distance < threshold:
                inliers.append(i)
        
        # Update the best model if the current one has more inliers
        if len(inliers) > len(best_inliers):
            best_line = line_params
            best_inliers = inliers

        # Early stopping if enough inliers are found
        if len(best_inliers) >= min_inliers:
            break
    
    return best_line, best_inliers

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    n_points = 100
    x = np.random.uniform(0, 10, n_points)
    y = 2 * x + 1 + np.random.normal(0, 0.5, n_points)  # y = 2x + 1 with noise
    inlier_points = np.column_stack((x, y))

    # Add outliers
    outliers = np.random.uniform(0, 10, (int(n_points * 0.2), 2))
    points = np.vstack((inlier_points, outliers))

    # Run RANSAC
    threshold = 0.5
    max_iterations = 1000
    min_inliers = n_points // 2
    best_line, best_inliers = ransac_2d(points, threshold, max_iterations, min_inliers)

    print("Best line parameters (a, b, c):", best_line)
    print("Number of inliers:", len(best_inliers))

    # Visualization
    import matplotlib.pyplot as plt
    plt.scatter(points[:, 0], points[:, 1], color='gray', label='Data points')
    plt.scatter(points[best_inliers, 0], points[best_inliers, 1], color='blue', label='Inliers')
    x_vals = np.linspace(0, 10, 100)
    if best_line:
        a, b, c = best_line
        y_vals = (-a * x_vals - c) / b
        plt.plot(x_vals, y_vals, color='red', label='Best fit line')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RANSAC Line Fitting')
    plt.show()