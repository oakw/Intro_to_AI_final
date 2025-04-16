import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from test_interface import get_score
import random
from collections import defaultdict
import heapq
import warnings
from tqdm import tqdm
import sys
from test_performance import visualize_path

def get_neighbors(point):
    """Get adjacent points (up, right, down, left)"""
    x, y = point
    return [(x+dx, y+dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if -100 <= x+dx <= 100 and -100 <= y+dy <= 100]

def find_best_path(gp, start_point, path_length=10):
    """
    Find best connected path using beam search with GP predictions
    Returns the path and its total predicted score
    """
    beam_width = 20
    beam = [(0, [start_point])]  # (total_score, path)
    visited = {start_point}
    
    while len(beam[0][1]) < path_length:
        new_beam = []
        for total_score, path in beam:
            current = path[-1]
            # Get valid neighbors (not visited and within bounds)
            neighbors = [(x, y) for x, y in get_neighbors(current)
                        if (x, y) not in visited]
            
            if not neighbors:
                continue
                
            # Predict scores for neighbors
            X_pred = np.array(neighbors)
            scores, _ = gp.predict(X_pred, return_std=True)
            
            # Add each neighbor to new candidates
            for neighbor, score in zip(neighbors, scores):
                new_path = path + [neighbor]
                new_score = total_score + score
                new_beam.append((new_score, new_path))
                
        # Keep top beam_width paths
        new_beam.sort(reverse=True)
        beam = new_beam[:beam_width]
        
        # Mark points in best paths as visited
        for _, path in beam:
            visited.add(path[-1])
            
    return beam[0][1]  # Return the best path

def optimize_field(n_iterations=200, path_length=10, layout_id='1'):
    # Initialize data storage
    X_samples = []
    y_samples = []
    
    # Initial random sampling (20 points)
    for _ in range(20):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        z = get_score(x, y, layout_id)
        if z is not None:
            X_samples.append([x, y])
            y_samples.append(z)
    
    # Convert to numpy arrays
    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples)
    
    # Initialize Gaussian Process with carefully chosen kernel bounds
    # Scale the constant kernel bounds based on the variance of observed values
    y_std = np.std(y_samples) if len(y_samples) > 0 else 1.0
    constant_kernel = ConstantKernel(
        constant_value=y_std,
        constant_value_bounds=(y_std/10, y_std*10)
    )
    
    # Set RBF length scale based on the input space
    # Our input space is 200x200 (-100 to 100 in both dimensions)
    # A reasonable length scale would be around 10-20% of the input space
    length_scale = 40.0  # 20% of 200
    rbf_kernel = RBF(
        length_scale=length_scale,
        length_scale_bounds=(length_scale/10, length_scale*10)
    )
    
    kernel = constant_kernel * rbf_kernel
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=0,  # Changed from 5 to 0 to avoid bounds error
        # TODO: ^^^^ yet the above error still occurs from time to time
        random_state=42,
        normalize_y=True  # This helps with numerical stability
    )
    
    # Main optimization loop
    for i in tqdm(range(n_iterations)):
        # Fit GP model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_samples, y_samples)
        
        # Generate candidate points
        candidates = []
        for _ in range(100):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            candidates.append([x, y])
        candidates = np.array(candidates)
        
        # Calculate acquisition function (Upper Confidence Bound)
        mean, std = gp.predict(candidates, return_std=True)
        beta = 2.0  # Exploration-exploitation trade-off parameter
        acquisition = mean + beta * std
        
        # Select best point according to acquisition function
        best_idx = np.argmax(acquisition)
        next_x, next_y = candidates[best_idx]
        
        # Query the actual value
        z = get_score(int(next_x), int(next_y), layout_id)
        if z is not None:
            X_samples = np.vstack((X_samples, [next_x, next_y]))
            y_samples = np.append(y_samples, z)
    
    # Fit final GP model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_samples, y_samples)
    
    # Find best starting point using grid search
    grid_points = []
    for x in range(-100, 101, 5):
        for y in range(-100, 101, 5):
            grid_points.append([x, y])
    grid_points = np.array(grid_points)
    mean_scores, _ = gp.predict(grid_points, return_std=True)
    
    best_start_idx = np.argmax(mean_scores)
    start_point = tuple(map(int, grid_points[best_start_idx]))
    
    # Find best path from the starting point
    best_path = find_best_path(gp, start_point, path_length)
    return best_path

def find_best_locations(layout_id='1'):
    """
    Find the best connected path of 10 positions that yields the highest total z value.
    Returns list of (x, y) tuples representing the path.
    """
    path = optimize_field(n_iterations=200, path_length=10, layout_id=layout_id)
    return path

if __name__ == "__main__":
    layout_id = '1'
    if len(sys.argv) > 1:
        layout_id = sys.argv[1]

    # Test the implementation
    best_path = find_best_locations(layout_id)
    print("Best connected path found:")
    total_z = 0
    for i, (x, y) in enumerate(best_path, 1):
        z = get_score(x, y, layout_id)
        total_z += z
        print(f"{i}. Position (x={x}, y={y}), z={z}")
    print(f"Total sum of z values: {total_z}")
    
    # Print path connections
    print("\nPath connections:")
    for i in range(len(best_path)-1):
        x1, y1 = best_path[i]
        x2, y2 = best_path[i+1]
        dx = x2 - x1
        dy = y2 - y1
        direction = {(0, 1): "right", (0, -1): "left", (1, 0): "down", (-1, 0): "up"}[(dx, dy)]
        print(f"Step {i+1}: Move {direction} from ({x1}, {y1}) to ({x2}, {y2})")
    
    # Visualize the path
    output_file = f'path_visualization_layout_{layout_id}.png'
    visualize_path(best_path, layout_id=layout_id, output_file=output_file)
