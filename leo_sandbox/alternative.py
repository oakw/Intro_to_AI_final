import numpy as np
import time
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Callable


def get_neighbors(point):
    """Get adjacent points (up, right, down, left)"""
    x, y = point
    return [(x+dx, y+dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if -100 <= x+dx <= 100 and -100 <= y+dy <= 100]

class GaussianProcessExploreExploit:
    def __init__(
            self,
            space_bounds = (-100, 100),
            n_iterations: int = 200,
            path_length: int = 10,
            avoid_negative: bool = False,
            predetermined_sample: list[tuple[int, int, int]]|None = None,
        ):
        """
        Initialize the Gaussian Process for exploration and exploitation.
        
        :param space_bounds: Tuple of (min, max) bounds for the search space
        :param n_iterations: Number of iterations for exploration
        :param path_length: Length of the path to be exploited
        :param avoid_negative: Whether to avoid negative regions during path finding
        :param predetermined_sample: List of tuples (x, y, z) for predetermined samples
        """
        self.space_bounds = space_bounds
        self.n_iterations = n_iterations
        self.path_length = path_length
        self.avoid_negative = avoid_negative
        self.predetermined_sample = predetermined_sample if predetermined_sample is not None else []
        
        # Data storage
        self.train_x = []
        self.train_y = []
        self.model = None
        self.best_path = None
        self.grid_data = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_model(self):
        """Return the trained Gaussian Process model"""
        return self.model
    
    def explore(self, query_z: Callable[[float, float], int]):
        """
        Explore the environment using Gaussian Process optimization.
        The function will sample points in the space, query their z values,
        and use them to train the Gaussian Process model.
        
        :param query_z: Function to query the z value at a given (x, y) position
        :return: Trained Gaussian Process model
        """
        print(f"Starting Gaussian Process exploration with {self.n_iterations} iterations...")
        print(f"Using device: {self.device}")
        
        # Define bounds for the search space
        bounds_tensor = torch.tensor([[self.space_bounds[0], self.space_bounds[0]], 
                                     [self.space_bounds[1], self.space_bounds[1]]], 
                                    dtype=torch.float64, device=self.device)
        
        if self.predetermined_sample:
            # Use predetermined samples if provided
            self.train_x = [[x, y] for x, y, _ in self.predetermined_sample]
            self.train_y = [z for _, _, z in self.predetermined_sample]
        else:
            # Initial random sampling within bounds
            # TODO:
            n_initial = 10
            X_initial = torch.rand(n_initial, 2, device=self.device, dtype=torch.float64)
            # Transform from [0, 1] to [min_bound, max_bound]
            X_initial = X_initial * (self.space_bounds[1] - self.space_bounds[0]) + self.space_bounds[0]
            
            print(f"Starting with {n_initial} initial random points...")
            for i in range(n_initial):
                x, y = X_initial[i].tolist()
                z = query_z(x, y)
                
                if z is not None:
                    self.train_x.append([x, y])
                    self.train_y.append(z)
                
                time.sleep(0.1)  # Be nice to the API
            
            # Main Bayesian Optimization loop
            # TODO: 
            remaining_calls = 100 - len(self.train_x)
            
            # Convert to tensors with double precision
            train_x_tensor = torch.tensor(self.train_x, dtype=torch.float64, device=self.device)
            train_y_tensor = torch.tensor(self.train_y, dtype=torch.float64, device=self.device).unsqueeze(-1)
            
            found_optimal = False  # Flag for early stopping if we find value of 1000
            
            for iteration in range(remaining_calls):
                # Check if we have enough good points and should stop early
                if found_optimal and iteration > min(30, remaining_calls // 2):
                    print("Found optimal point, transitioning to exploitation phase")
                    break
                
                # Step 1: Update normalized and standardized data
                train_x_normalized = normalize(train_x_tensor, bounds=bounds_tensor)
                train_y_standardized = standardize(train_y_tensor)
                
                # Step 2: Fit GP model
                self.model = SingleTaskGP(train_x_normalized, train_y_standardized)
                mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                fit_gpytorch_model(mll)
                
                # Step 3: Define acquisition function
                best_value = train_y_standardized.max()
                EI = ExpectedImprovement(
                    self.model, 
                    best_f=best_value,
                    maximize=True,
                )
                
                # Step 4: Optimize acquisition function in normalized space
                bounds_ei = torch.stack([
                    torch.zeros(2, device=self.device, dtype=torch.float64),
                    torch.ones(2, device=self.device, dtype=torch.float64)
                ])
                
                new_x_normalized, _ = optimize_acqf(
                    acq_function=EI,
                    bounds=bounds_ei,
                    q=1,
                    num_restarts=20, 
                    raw_samples=500,
                )
                
                # Convert normalized point back to original space
                new_x = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * new_x_normalized
                new_x_value, new_y_value = new_x[0].tolist()
                
                # Query the value at the new point
                z = query_z(new_x_value, new_y_value)
                if z is not None:
                    # Update training data
                    self.train_x.append([new_x_value, new_y_value])
                    self.train_y.append(z)
                    train_x_tensor = torch.tensor(self.train_x, dtype=torch.float64, device=self.device)
                    train_y_tensor = torch.tensor(self.train_y, dtype=torch.float64, device=self.device).unsqueeze(-1)
                    
                    # Check if we found optimal value (1000)
                    if z == 1000:
                        found_optimal = True
                        print(f"Found optimal value at ({new_x_value}, {new_y_value})!")
                
                time.sleep(0.1)  # Be nice to the API
        
        # Fit final model
        if len(self.train_x) > 0:
            train_x_tensor = torch.tensor(self.train_x, dtype=torch.float64, device=self.device)
            train_y_tensor = torch.tensor(self.train_y, dtype=torch.float64, device=self.device).unsqueeze(-1)
            train_x_normalized = normalize(train_x_tensor, bounds=bounds_tensor)
            train_y_standardized = standardize(train_y_tensor)
            
            try:
                self.model = SingleTaskGP(train_x_normalized, train_y_standardized)
                mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                fit_gpytorch_model(mll)
            except Exception as e:
                print(f"Warning: Could not fit final GP model: {e}")
                # Fallback to previous model if fit fails
        
        # Prepare grid data for path planning
        self._prepare_grid_data()

        print("Run a possible exploit grid search...")
        # Use the exploration moves here to be better prepared for the exploit phase by preparing the best path
        self.best_path = self._exploit_grid_search(query_z)
        
        return self.model
    
    def _prepare_grid_data(self):
        """Prepare grid data for path planning by storing all explored points"""
        X_array = np.array(self.train_x)
        Y_array = np.array(self.train_y)
        
        # Store explored data in grid_data
        for i, (x, y) in enumerate(X_array):
            # Round to nearest integer for grid indexing
            int_x, int_y = int(round(x)), int(round(y))
            self.grid_data[(int_x, int_y)] = Y_array[i]
    
    def _find_best_path(self, start_point):
        """
        Find the best path of length self.path_length starting from a given point
        using depth-first search.
        
        :param start_point: The starting point (x, y) for the path
        :return: The best path as a list of points and its total value
        """
        paths = []
        visited = set()
        
        def dfs(node, path, total_value):
            if len(path) == self.path_length:
                paths.append((path[:], total_value))
                return
            
            visited.add(node)
            
            # Sort neighbors by value to prioritize better moves
            neighbors = []
            for neighbor in get_neighbors(node):
                if neighbor not in visited and neighbor in self.grid_data:
                    neighbors.append((neighbor, self.grid_data[neighbor]))
            
            # Sort neighbors by value (descending)
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Try better neighbors first (pruning low-value paths)
            for neighbor, value in neighbors:
                # Skip very negative neighbors if avoid_negative is True
                if self.avoid_negative and value < -100:
                    continue
                    
                new_value = total_value + value
                path.append(neighbor)
                dfs(neighbor, path, new_value)
                path.pop()
            
            visited.remove(node)
        
        dfs(start_point, [start_point], self.grid_data[start_point])
        
        if not paths:
            return [], 0
        
        # Find the path with the highest total value
        best_path, best_value = max(paths, key=lambda x: x[1])
        return best_path, best_value
    
    def _exploit_grid_search(self, query_z: Callable[[float, float], int]):
        """
        Perform a grid search to find the best connected path.
        
        :param query_z: Function to query the z value at a given (x, y) position
        :return: The best path as a list of points
        """
        # Get the best point from exploration phase
        if len(self.train_x) == 0 or len(self.train_y) == 0:
            print("No exploration data available. Cannot perform exploitation.")
            return []
        
        best_idx = np.argmax(self.train_y)
        best_x, best_y = self.train_x[best_idx]
        best_value = self.train_y[best_idx]
        print(f"Best point found during exploration: ({best_x}, {best_y}) with value: {best_value}")
        
        # Fine-grained exploration around the best point
        grid_size = 10
        lower_x = max(self.space_bounds[0], int(best_x) - grid_size // 2)
        upper_x = min(self.space_bounds[1], int(best_x) + grid_size // 2)
        lower_y = max(self.space_bounds[0], int(best_y) - grid_size // 2)
        upper_y = min(self.space_bounds[1], int(best_y) + grid_size // 2)
        
        print(f"Performing grid search of size {grid_size}x{grid_size} around best point...")
        
        # Top points from exploration for prioritized grid evaluation
        top_points = sorted([(int(round(x)), int(round(y)), z) for (x, y), z in zip(self.train_x, self.train_y)], 
                           key=lambda x: x[2], reverse=True)[:5]
        
        # Exploration around top points
        evaluated_points = set([(int(round(x)), int(round(y))) for x, y in self.train_x])
        
        for center_x, center_y, _ in top_points:
            # Explore a focused grid around each top point
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = center_x + dx, center_y + dy
                    
                    # Stay within global boundaries
                    if not (self.space_bounds[0] <= x <= self.space_bounds[1] and 
                            self.space_bounds[0] <= y <= self.space_bounds[1]):
                        continue
                    
                    # Skip if already evaluated
                    if (x, y) in evaluated_points:
                        continue
                    
                    # Query the value at this point
                    z = query_z(x, y)
                    
                    if z is not None:
                        # Store result in grid_data
                        self.grid_data[(x, y)] = z
                        evaluated_points.add((x, y))
                    
                    time.sleep(0.1)  # Be nice to the API
        
        # Find best path from the top points
        best_overall_path = []
        best_overall_value = float('-inf')
        
        # Try the top points as starting points
        for point, _ in sorted(list(self.grid_data.items()), key=lambda x: x[1], reverse=True)[:10]:
            path, value = self._find_best_path(point)
            if value > best_overall_value and len(path) == self.path_length:
                best_overall_value = value
                best_overall_path = path
        
        # If we didn't find a valid path, use greedy approach
        if not best_overall_path or len(best_overall_path) < self.path_length:
            print("No valid path found with DFS. Using greedy approach...")
            best_grid_idx = np.argmax(list(self.grid_data.values())) if self.grid_data else 0
            start_point = list(self.grid_data.keys())[best_grid_idx] if self.grid_data else (0, 0)
            
            # Greedy path finding
            path = [start_point]
            visited = set([start_point])
            
            while len(path) < self.path_length:
                current = path[-1]
                candidates = []
                
                # Check valid neighbors
                for neighbor in get_neighbors(current):
                    if neighbor not in visited:
                        # If we don't know the value yet, query it
                        if neighbor not in self.grid_data:
                            z = query_z(neighbor[0], neighbor[1])
                            if z is not None:
                                self.grid_data[neighbor] = z
                        
                        if neighbor in self.grid_data:
                            candidates.append((neighbor, self.grid_data[neighbor]))
                
                if candidates:
                    # Sort by value (descending)
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    best_neighbor, _ = candidates[0]
                    path.append(best_neighbor)
                    visited.add(best_neighbor)
                else:
                    # If no valid neighbors, try random direction
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = current[0] + dx, current[1] + dy
                        neighbor = (nx, ny)
                        
                        if (self.space_bounds[0] <= nx <= self.space_bounds[1] and 
                            self.space_bounds[0] <= ny <= self.space_bounds[1] and 
                            neighbor not in visited):
                            
                            # Query new point
                            z = query_z(nx, ny)
                            if z is not None:
                                self.grid_data[neighbor] = z
                                path.append(neighbor)
                                visited.add(neighbor)
                                break
                    else:
                        # If we couldn't add any point, break
                        break
            
            best_overall_path = path
        
        # Ensure path is exactly path_length
        if len(best_overall_path) > self.path_length:
            best_overall_path = best_overall_path[:self.path_length]
        
        # Validate path - ensure all moves are orthogonal and single-step
        for i in range(len(best_overall_path)-1):
            x1, y1 = best_overall_path[i]
            x2, y2 = best_overall_path[i+1]
            dx = x2 - x1
            dy = y2 - y1
            
            # Make sure this is a valid orthogonal move of distance 1
            if not ((dx == 0 and abs(dy) == 1) or (dy == 0 and abs(dx) == 1)):
                print(f"Warning: Invalid move from ({x1}, {y1}) to ({x2}, {y2}). Correcting path.")
                # Insert intermediate point for diagonals
                if dx != 0 and dy != 0:
                    best_overall_path.insert(i+1, (x1, y2))
                    break
                # Insert intermediate point for long moves
                elif abs(dx) > 1:
                    best_overall_path.insert(i+1, (x1 + (1 if dx > 0 else -1), y1))
                    break
                elif abs(dy) > 1:
                    best_overall_path.insert(i+1, (x1, y1 + (1 if dy > 0 else -1)))
                    break
        
        return best_overall_path
    
    def exploit(self, query_z: Callable[[float, float], int]):
        """
        Exploit the environment using the trained Gaussian Process model.
        The function will find the best path in the environment based on the model's predictions.
        
        :param query_z: Function to query the z value at a given (x, y) position
        """

        # Verify and query the path points
        if len(self.best_path) < self.path_length:
            # TODO: what to do now?
            print(f"Warning: Path length is only {len(self.best_path)}, expected {self.path_length}")
        
        # Query all points in the path to make sure we have their true values
        for point in self.best_path:
            x, y = point
            query_z(x, y)
        
        return self.best_path


# import requests
# import sys
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import networkx
# import torch.nn.functional as F
# import csv
# from datetime import datetime
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from martins_sandbox.test_interface import get_score as external_get_score
# from martins_sandbox.test_performance import visualize_path


# # API endpoint - update to use local mock server
# BASE_URL = "http://157.180.73.240:8080/"  # Remote server
# BASE_URL = "http://127.0.0.1:5000/"  # Local mock server
# temp_calls = 100  # Number of calls for exploration phase
# # Set default layout_id to '1'
# DEFAULT_LAYOUT_ID = '7'

# # Flag to determine whether to use the mock server or external API
# USE_MOCK_SERVER = True

# # API call counter
# api_call_counter = 0

# # Function to call the API and get the result
# def call_api(x, y, layout_id=DEFAULT_LAYOUT_ID):
#     global api_call_counter
#     if api_call_counter >= 199:
#         print("WARNING: API call limit reached!")
#         return -1000  # Return penalty value
    
#     api_call_counter += 1
#     url = f"{BASE_URL}{layout_id}/{x}/{y}"  # Note the URL format change to match the Flask route
#     print(f"Calling API with x={x}, y={y}, layout={layout_id}...")
#     try:
#         response = requests.get(url, timeout=5)
#         if response.status_code == 200:
#             try:
#                 # Parse the response as JSON
#                 data = response.json()
#                 print(f"API response: {data}")
#                 return float(data.get('z', -1000))
#             except ValueError:
#                 # If JSON parsing fails, try to parse as text
#                 print(f"Failed to parse JSON: {response.text}")
#                 return -1000
            
#         else:
#             print(f"Error: {response.status_code} - {response.text}")
#             return -1000  # Return a penalty value for errors
#     except Exception as e:
#         print(f"Exception: {e}")
#         return -1000  # Return a penalty value for exceptions


# # Function to get the score (using either the mock server or external API)
# def get_score(x, y, layout_id=DEFAULT_LAYOUT_ID):
#     if USE_MOCK_SERVER:
#         return call_api(x, y, layout_id)
#     else:
#         return external_get_score(x, y, layout_id)

# # Function to find the best path using BoTorch optimization - standalone version
# def find_best_locations(layout_id='1', avoid_negative=False):
#     # Set up device for BoTorch
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Define bounds for the search space
#     bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]], dtype=torch.float64, device=device)
    
#     # Store all evaluated points
#     train_x = []
#     train_y = []
    
#     # Flag to indicate if we found a value of 1000
#     found_optimal = False
#     optimal_x = None
#     optimal_y = None
    
#     # Initial points - random sampling within bounds
#     n_initial = 10
#     X_initial = torch.rand(n_initial, 2, device=device, dtype=torch.float64) * 200 - 100
#     Y_initial = []
    
#     print(f"Starting with {n_initial} initial random points...")
#     for i in range(n_initial):
#         x, y = X_initial[i].tolist()
#         result = call_api(x, y, layout_id)
#         Y_initial.append(result)
        
#         # Store for later use
#         train_x.append([x, y])
#         train_y.append(result)
        
#         # Check if we found the optimal value
#         if result == 1000:
#             found_optimal = True
#             optimal_x, optimal_y = x, y
#             print(f"Found optimal value at ({x}, {y})!")
        
#         time.sleep(0.1)  # Be nice to the API
    
#     # Convert to tensors with double precision
#     train_x_tensor = torch.tensor(train_x, dtype=torch.float64, device=device)
#     train_y_tensor = torch.tensor(train_y, dtype=torch.float64, device=device).unsqueeze(-1)

#     # Normalize inputs to [0,1] range
#     bounds_tensor = torch.tensor([[-100.0, -100.0], [100.0, 100.0]], dtype=torch.float64, device=device)
#     train_x_normalized = normalize(train_x_tensor, bounds=bounds_tensor)

#     # Standardize outputs
#     train_y_standardized = standardize(train_y_tensor)

#     # Main Bayesian Optimization loop
#     remaining_calls = temp_calls - n_initial
    
#     for iteration in range(remaining_calls):
#         if api_call_counter >= 199:
#             print("API call limit reached, stopping optimization")
#             break
            
#         # If we found optimal value and have made enough calls, transition to exploitation
#         if found_optimal and iteration > min(30, remaining_calls // 2):
#             print("Found optimal point, transitioning to exploitation phase")
#             break
        
#         # Step 1: Update normalized and standardized data
#         train_x_normalized = normalize(train_x_tensor, bounds=bounds_tensor)
#         train_y_standardized = standardize(train_y_tensor)
        
#         # Step 2: Fit GP model using normalized and standardized data
#         model = SingleTaskGP(train_x_normalized, train_y_standardized)
#         mll = ExactMarginalLogLikelihood(model.likelihood, model)
#         fit_gpytorch_model(mll)
        
#         # Step 3: Define acquisition function (with standardized best value)
#         best_value = train_y_standardized.max()
#         EI = ExpectedImprovement(
#             model, 
#             best_f=best_value,
#             maximize=True,
#             # beta=0.1  # Add some exploration
#         )
        
#         # Step 4: Optimize acquisition function in normalized space
#         bounds_ei = torch.stack([
#             torch.zeros(2, device=device, dtype=torch.float64),
#             torch.ones(2, device=device, dtype=torch.float64)
#         ])
        
#         # Use a larger number of random restarts and samples
#         new_x_normalized, _ = optimize_acqf(
#             acq_function=EI,
#             bounds=bounds_ei,
#             q=1,
#             num_restarts=20,  # Increased from 10
#             raw_samples=500,  # Increased from 100
#         )
        
#         # Convert normalized point back to original space
#         new_x = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * new_x_normalized
#         new_x_value, new_y_value = new_x[0].tolist()
        
#         # Step 5: Query the API for the next point
#         next_result = call_api(new_x_value, new_y_value, layout_id)
        
#         # Step 6: Update training data (using original scale)
#         train_x.append([new_x_value, new_y_value])
#         train_y.append(next_result)
#         train_x_tensor = torch.tensor(train_x, dtype=torch.float64, device=device)
#         train_y_tensor = torch.tensor(train_y, dtype=torch.float64, device=device).unsqueeze(-1)
        
#         # Check if we found the optimal value
#         if next_result == 1000:
#             found_optimal = True
#             optimal_x, optimal_y = new_x_value, new_y_value
#             print(f"Found optimal value at ({new_x_value}, {new_y_value})!")
        
#         print(f"Iteration {iteration + 1}/{remaining_calls}: x={new_x_value:.2f}, y={new_y_value:.2f}, z={next_result}")
        
#         # Add a small delay
#         time.sleep(0.1)
    
#     # Extract exploration results for analysis
#     X_array = np.array(train_x)
#     Y_array = np.array(train_y)
    
#     # Get the best point from the exploration phase
#     best_idx = np.argmax(Y_array)
#     best_x, best_y = X_array[best_idx, 0], X_array[best_idx, 1]
#     best_value = Y_array[best_idx]
    
#     print(f"Best point found during exploration: ({best_x}, {best_y}) with value: {best_value}")
    
#     # Define grid boundaries for exploitation phase
#     if found_optimal:
#         center_x, center_y = optimal_x, optimal_y
#     else:
#         center_x, center_y = best_x, best_y
    
#     grid_size = 10
#     lower_x = max(-100, int(center_x) - grid_size // 2)
#     upper_x = min(100, int(center_x) + grid_size // 2)
#     lower_y = max(-100, int(center_y) - grid_size // 2)
#     upper_y = min(100, int(center_y) + grid_size // 2)
    
#     # Calculate calls used and remaining
#     calls_used = api_call_counter
#     calls_remaining = 200 - calls_used
    
#     print(f"API calls used so far: {calls_used}, remaining: {calls_remaining}")
#     print(f"Exploring {grid_size}x{grid_size} grid around best point with remaining calls")
    
#     # Store grid values and points
#     grid_points = []
#     grid_values = []
    
#     # First, reuse already evaluated points
#     explored_data = {}
#     for i, (x, y) in enumerate(X_array):
#         # Round to nearest integer for grid indexing
#         int_x, int_y = int(round(x)), int(round(y))
#         explored_data[(int_x, int_y)] = Y_array[i]
        
#         # Include if within grid boundaries
#         if lower_x <= int_x <= upper_x and lower_y <= int_y <= upper_y:
#             grid_points.append((int_x, int_y))
#             grid_values.append(Y_array[i])
    
#     # Set to track points we've already evaluated
#     evaluated_points = set([(int(round(x)), int(round(y))) for x, y in X_array])
    
#     # Sort the top points from exploration for prioritized grid evaluation
#     top_points = sorted([(int(round(x)), int(round(y)), y) for (x, y), y in zip(X_array, Y_array)], 
#                        key=lambda x: x[2], reverse=True)[:5]
    
#     # Exploitation phase: focused search around top points
#     new_calls = 0
#     max_new_calls = min(calls_remaining, 50)  # Stay within API call limit
    
#     for center_x, center_y, _ in top_points:
#         if new_calls >= max_new_calls:
#             break
            
#         # Explore a focused grid around each top point
#         for dx in range(-2, 3):
#             for dy in range(-2, 3):
#                 if new_calls >= max_new_calls:
#                     break
                    
#                 x, y = center_x + dx, center_y + dy
                
#                 # Stay within global boundaries
#                 if not (-100 <= x <= 100 and -100 <= y <= 100):
#                     continue
                    
#                 # Skip if already evaluated
#                 if (x, y) in evaluated_points:
#                     continue
                
#                 # Call API for this new point
#                 value = call_api(x, y, layout_id)
#                 new_calls += 1
                
#                 # Store the result
#                 grid_points.append((x, y))
#                 grid_values.append(value)
#                 evaluated_points.add((x, y))
                
#                 time.sleep(0.1)  # Be nice to the API
    
#     # Find the best integer point from our combined exploration
#     best_grid_idx = np.argmax(grid_values) if grid_values else 0
#     if grid_points:
#         best_grid_x, best_grid_y = grid_points[best_grid_idx]
#         best_grid_value = grid_values[best_grid_idx]
#         print(f"Best integer point: ({best_grid_x}, {best_grid_y}) with value: {best_grid_value}")
#     else:
#         best_grid_x, best_grid_y = int(round(best_x)), int(round(best_y))
#         best_grid_value = best_value
#         print(f"No grid points evaluated, using rounded best point: ({best_grid_x}, {best_grid_y})")
    
#     # Create a 2D grid to store all values for path planning
#     grid_data = {}
#     for (x, y), value in zip(grid_points, grid_values):
#         grid_data[(x, y)] = value
    
#     # Also add any points from the exploration phase
#     for (x, y), value in explored_data.items():
#         if (x, y) not in grid_data:
#             grid_data[(x, y)] = value
    
#     print(f"Grid has {len(grid_data)} points for path planning")
#     print(f"Total API calls made: {api_call_counter}")
    
#     # Create a graph to find the best path
#     G = networkx.DiGraph()
    
#     # Add nodes and edges for all possible moves
#     for (x, y), value in grid_data.items():
#         # Add the node
#         G.add_node((x, y), value=value)
        
#         # Add edges to all possible neighbors (4 directions)
#         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Only orthogonal moves
#             nx, ny = x + dx, y + dy
#             if (nx, ny) in grid_data:
#                 # Add an edge with weight based on the neighbor's value
#                 # Negative weight because we want to find the maximum value path
#                 G.add_edge((x, y), (nx, ny), weight=-grid_data[(nx, ny)])
    
#     # Function to find the best path of length 10 starting from a given point
#     def find_best_path(start_point):
#         paths = []
#         visited = set()
        
#         def dfs(node, path, total_value):
#             if len(path) == 10:
#                 paths.append((path[:], total_value))
#                 return
            
#             visited.add(node)
            
#             # Sort neighbors by value to prioritize better moves
#             neighbors = []
#             for neighbor in G.neighbors(node):
#                 if neighbor not in visited:
#                     neighbors.append((neighbor, grid_data[neighbor]))
            
#             # Sort neighbors by value (descending)
#             neighbors.sort(key=lambda x: x[1], reverse=True)
            
#             # Try better neighbors first (pruning low-value paths)
#             for neighbor, value in neighbors:
#                 # Skip very negative neighbors if avoid_negative is True
#                 if avoid_negative and value < -100:
#                     continue
                    
#                 new_value = total_value + value
#                 path.append(neighbor)
#                 dfs(neighbor, path, new_value)
#                 path.pop()
            
#             visited.remove(node)
        
#         dfs(start_point, [start_point], grid_data[start_point])
        
#         if not paths:
#             return [], 0
        
#         # Find the path with the highest total value
#         best_path, best_value = max(paths, key=lambda x: x[1])
#         return best_path, best_value
    
#     # Try different starting points to find the best overall path
#     best_overall_path = []
#     best_overall_value = float('-inf')
    
#     # Try the top 10 integer points from all our data
#     top_points = sorted(list(grid_data.items()), key=lambda x: x[1], reverse=True)[:10]
    
#     for (point), _ in top_points:
#         path, value = find_best_path(point)
#         if value > best_overall_value and len(path) == 10:
#             best_overall_value = value
#             best_overall_path = path
    
#     # If we didn't find a valid path, use a greedy approach
#     if not best_overall_path or len(best_overall_path) < 10:
#         print("No valid path found or path too short. Using enhanced greedy approach...")
        
#         # First, ensure we have evaluated points in all four directions from the best point
#         start_point = (best_grid_x, best_grid_y)
#         path = [start_point]
#         visited = set([start_point])
        
#         # Check remaining API calls
#         calls_left = 200 - api_call_counter
#         print(f"API calls remaining for enhanced exploration: {calls_left}")
        
#         # More aggressive exploration around best points if we have calls left
#         if calls_left > 10:
#             print(f"Using remaining {min(calls_left - 5, 40)} API calls for enhanced exploration...")
#             # Get top N points to explore around
#             top_exploration_points = sorted(list(grid_data.items()), 
#                                           key=lambda x: x[1], reverse=True)[:5]
            
#             max_exploration_calls = min(calls_left - 5, 40)  # Leave some calls for path completion
#             exploration_calls = 0
            
#             # Create a more comprehensive grid around top points
#             for (cx, cy), _ in top_exploration_points:
#                 if exploration_calls >= max_exploration_calls:
#                     break
                    
#                 # Explore in a spiral pattern around best points
#                 for radius in range(1, 4):
#                     if exploration_calls >= max_exploration_calls:
#                         break
                        
#                     # Explore horizontal and vertical lines at this radius
#                     for dx in range(-radius, radius + 1):
#                         for dy in [-radius, radius]:  # Top and bottom edges
#                             if exploration_calls >= max_exploration_calls:
#                                 break
                                
#                             x, y = cx + dx, cy + dy
#                             point = (x, y)
                            
#                             # Check if valid and not already evaluated
#                             if (-100 <= x <= 100 and -100 <= y <= 100 and 
#                                 point not in grid_data and point not in evaluated_points):
#                                 value = call_api(x, y, layout_id)
#                                 time.sleep(0.1)
#                                 grid_data[point] = value
#                                 exploration_calls += 1
                                
#                                 # Add to graph with connections to neighbors
#                                 G.add_node(point, value=value)
#                                 for ndx, ndy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                                     neighbor = (x + ndx, y + ndy)
#                                     if neighbor in grid_data:
#                                         G.add_edge(point, neighbor, weight=-grid_data[neighbor])
#                                         G.add_edge(neighbor, point, weight=-value)
                    
#                     # Left and right edges
#                     for dy in range(-radius+1, radius):
#                         for dx in [-radius, radius]:  # Left and right edges
#                             if exploration_calls >= max_exploration_calls:
#                                 break
                                
#                             x, y = cx + dx, cy + dy
#                             point = (x, y)
                            
#                             # Check if valid and not already evaluated
#                             if (-100 <= x <= 100 and -100 <= y <= 100 and 
#                                 point not in grid_data and point not in evaluated_points):
#                                 value = call_api(x, y, layout_id)
#                                 time.sleep(0.1)
#                                 grid_data[point] = value
#                                 exploration_calls += 1
                                
#                                 # Add to graph with connections to neighbors
#                                 G.add_node(point, value=value)
#                                 for ndx, ndy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                                     neighbor = (x + ndx, y + ndy)
#                                     if neighbor in grid_data:
#                                         G.add_edge(point, neighbor, weight=-grid_data[neighbor])
#                                         G.add_edge(neighbor, point, weight=-value)
            
#             print(f"Enhanced exploration complete. Used {exploration_calls} additional API calls")
            
#             # Try to find paths using DFS again with the enhanced grid
#             best_overall_path = []
#             best_overall_value = float('-inf')
            
#             # Try the top points from our enhanced grid
#             top_points = sorted(list(grid_data.items()), key=lambda x: x[1], reverse=True)[:15]
            
#             for (point), _ in top_points:
#                 path, value = find_best_path(point)
#                 if value > best_overall_value and len(path) == 10:
#                     best_overall_value = value
#                     best_overall_path = path
            
#             if best_overall_path and len(best_overall_path) == 10:
#                 print("Found valid path after enhanced exploration!")
#                 return best_overall_path
        
#         # Try to add more points to the grid if needed to ensure connectivity
#         for distance in range(1, 6):  # Try exploring further if needed
#             for dx, dy in [(0, distance), (0, -distance), (distance, 0), (-distance, 0)]:
#                 if len(grid_data) >= 30 and api_call_counter >= 180:
#                     break  # Don't make too many extra calls
                
#                 x, y = start_point[0] + dx, start_point[1] + dy
#                 if not (-100 <= x <= 100 and -100 <= y <= 100):
#                     continue
                
#                 point = (x, y)
#                 if point not in grid_data and point not in evaluated_points:
#                     value = call_api(x, y, layout_id)
#                     time.sleep(0.1)
#                     grid_data[point] = value
                    
#                     # Add to graph
#                     G.add_node(point, value=value)
#                     # Connect to any existing neighbors
#                     for ndx, ndy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                         neighbor = (x + ndx, y + ndy)
#                         if neighbor in grid_data:
#                             G.add_edge(point, neighbor, weight=-grid_data[neighbor])
#                             G.add_edge(neighbor, point, weight=-value)
        
#         # Enhanced greedy approach that ensures a path of length 10
#         path = [start_point]
#         visited = set([start_point])
        
#         # Attempt 1: Try to find a connected path using grid_data points
#         print("Attempt 1: Finding connected path in explored grid...")
#         while len(path) < 10:
#             current = path[-1]
#             candidates = []
            
#             # Check all possible neighbors (including further ones if needed)
#             for distance in range(1, 6):
#                 for dx, dy in [(0, distance), (0, -distance), (distance, 0), (-distance, 0)]:
#                     nx, ny = current[0] + dx, current[1] + dy
#                     neighbor = (nx, ny)
                    
#                     if neighbor in grid_data and neighbor not in visited:
#                         candidates.append((neighbor, grid_data[neighbor], distance))
            
#             # If we found candidates, pick the best one (highest value, lowest distance)
#             if candidates:
#                 # Sort by value (descending) then by distance (ascending)
#                 candidates.sort(key=lambda x: (-x[1], x[2]))
#                 best_neighbor, _, _ = candidates[0]
#                 path.append(best_neighbor)
#                 visited.add(best_neighbor)
#             else:
#                 break
        
#         # If we still don't have 10 points, use the non-optimal approach: generate artificial points
#         if len(path) < 10:
#             print(f"Attempt 2: Adding artificial points to complete the path (current length: {len(path)})...")
#             # Make sure we have enough valid moves from current position
#             last_point = path[-1]
            
#             # Continue adding points in cardinal directions until we have 10
#             directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
#             dir_index = 0
            
#             while len(path) < 10:
#                 dx, dy = directions[dir_index]
#                 nx, ny = last_point[0] + dx, last_point[1] + dy
#                 new_point = (nx, ny)
                
#                 # Check if the new point is within bounds
#                 if -100 <= nx <= 100 and -100 <= ny <= 100:
#                     if new_point not in visited:
#                         # If we don't know the value yet, query the API
#                         if new_point not in grid_data and api_call_counter < 198:
#                             value = call_api(nx, ny, layout_id)
#                             time.sleep(0.1)
#                             grid_data[new_point] = value
#                         # If we can't query or already know it, just add it
#                         else:
#                             if new_point not in grid_data:
#                                 grid_data[new_point] = -100  # Placeholder value
                                
#                         path.append(new_point)
#                         visited.add(new_point)
#                         last_point = new_point
                
#                 # Try the next direction
#                 dir_index = (dir_index + 1) % 4
        
#         best_overall_path = path
        
#         # Validate the final path - make sure all moves are orthogonal
#         for i in range(len(best_overall_path)-1):
#             x1, y1 = best_overall_path[i]
#             x2, y2 = best_overall_path[i+1]
#             dx = x2 - x1
#             dy = y2 - y1
            
#             # Make sure this is a valid orthogonal move
#             if not ((dx == 0 and abs(dy) == 1) or (dy == 0 and abs(dx) == 1)):
#                 print(f"Warning: Invalid move from ({x1}, {y1}) to ({x2}, {y2}). Fixing...")
#                 # Fix the path by inserting intermediate points
#                 if dx != 0 and dy != 0:
#                     # Split this move into two orthogonal moves
#                     best_overall_path.insert(i+1, (x1, y2))  # Horizontal move first
#                     break  # Restart the validation
#                 elif abs(dx) > 1:
#                     # Insert intermediate horizontal point
#                     best_overall_path.insert(i+1, (x1 + (1 if dx > 0 else -1), y1))
#                     break  # Restart the validation
#                 elif abs(dy) > 1:
#                     # Insert intermediate vertical point
#                     best_overall_path.insert(i+1, (x1, y1 + (1 if dy > 0 else -1)))
#                     break  # Restart the validation
    
#     # Final check: ensure path length is exactly 10
#     while len(best_overall_path) > 10:
#         best_overall_path.pop()
        
#     return best_overall_path[:10]  # Ensure we return exactly 10 points

# # Updated main with default layout_id
# if __name__ == "__main__":
#     layout_id = DEFAULT_LAYOUT_ID
#     if len(sys.argv) > 1:
#         layout_id = sys.argv[1]
    
#     print(f"Using layout_id: {layout_id}")
#     print(f"Using {'mock server' if USE_MOCK_SERVER else 'external API'}")
    
#     # Test the implementation
#     best_path = find_best_locations(layout_id)
#     print("Best connected path found:")
#     total_z = 0
    
#     # Calculate weighted path score
#     scores = []
#     for i, (x, y) in enumerate(best_path, 1):
#         z = get_score(x, y, layout_id)
#         total_z += z
#         scores.append(z)
#         print(f"{i}. Position (x={x}, y={y}), z={z}")
    
#     # Calculate path quality metrics
#     avg_score = total_z / len(best_path) if best_path else 0
#     min_score = min(scores) if scores else 0
#     score_variance = sum((s - avg_score)**2 for s in scores) / len(scores) if scores else 0
    
#     print(f"Total sum of z values: {total_z}")
#     print(f"Average score per position: {avg_score:.2f}")
#     print(f"Minimum position score: {min_score}")
#     print(f"Score variance: {score_variance:.2f}")
    
#     # Evaluate path quality
#     path_quality = "Excellent"
#     if min_score < -500:
#         path_quality = "Poor"
#     elif min_score < 0:
#         path_quality = "Fair"
#     elif score_variance > 10000:
#         path_quality = "Varied"
        
#     print(f"Path quality assessment: {path_quality}")
    
#     # If path isn't good enough, try to improve it
#     if path_quality in ["Poor", "Fair"] and api_call_counter < 190:
#         print("Attempting to improve path by avoiding negative regions...")
#         # Try to generate an alternative path using the find_best_locations function
#         # with modified parameters to avoid negative regions
#         improved_path = find_best_locations(layout_id, avoid_negative=True)
        
#         # Calculate new path score
#         new_total_z = sum(get_score(x, y, layout_id) for x, y in improved_path)
        
#         # Use the improved path if it's better
#         if new_total_z > total_z:
#             best_path = improved_path
#             total_z = new_total_z
#             print("Found improved path!")
#             print("Updated path:")
#             for i, (x, y) in enumerate(best_path, 1):
#                 z = get_score(x, y, layout_id)
#                 print(f"{i}. Position (x={x}, y={y}), z={z}")
#             print(f"New total sum of z values: {total_z}")
    
#     # Print path connections
#     print("\nPath connections:")
#     for i in range(len(best_path)-1):
#         x1, y1 = best_path[i]
#         x2, y2 = best_path[i+1]
#         dx = x2 - x1
#         dy = y2 - y1
#         direction = {(0, 1): "right", (0, -1): "left", (1, 0): "down", (-1, 0): "up"}.get((dx, dy), "unknown")
#         print(f"Step {i+1}: Move {direction} from ({x1}, {y1}) to ({x2}, {y2})")    
    
#     # Visualize the path
#     output_file = f'path_visualization_layout_{layout_id}.png'
#     visualize_path(best_path, layout_id=layout_id, output_file=output_file)

#     # Write results to CSV
#     csv_filename = 'path_results.csv'
#     file_exists = os.path.isfile(csv_filename)
    
#     # Current timestamp for the record
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     # Prepare data for CSV
#     csv_data = [
#         timestamp,
#         layout_id,
#         total_z,
#         avg_score,
#         min_score,
#         score_variance,
#         path_quality,
#         api_call_counter
#     ]
    
#     # Add each position and its score
#     for i, (x, y) in enumerate(best_path):
#         z = scores[i]
#         csv_data.extend([f"({x},{y})", z])
    
#     # Write to CSV
#     with open(csv_filename, 'a', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
        
#         # Write header if file doesn't exist
#         if not file_exists:
#             header = [
#                 'Timestamp', 'Layout_ID', 'Total_Score', 'Avg_Score', 
#                 'Min_Score', 'Score_Variance', 'Path_Quality', 'API_Calls'
#             ]
#             # Add position headers
#             for i in range(1, 11):
#                 header.extend([f'Position_{i}', f'Score_{i}'])
            
#             csv_writer.writerow(header)
        
#         # Write data row
#         csv_writer.writerow(csv_data)
    
#     print(f"\nResults saved to {csv_filename}")

#     print(f"Total API calls made: {api_call_counter}")
#     if api_call_counter > 200:
#         print(f"WARNING: Exceeded API call limit with {api_call_counter} calls!")