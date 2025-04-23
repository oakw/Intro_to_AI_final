import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Callable
from tqdm import tqdm
import warnings


def get_neighbors(point):
    """Get adjacent points (up, right, down, left and diagonals)"""
    x, y = point
    return [(x+dx, y+dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] #, (1, 1), (-1, -1), (1, -1), (-1, 1)]
            if -100 <= x+dx <= 100 and -100 <= y+dy <= 100]

class GaussianProcessExploreExploit:
    def __init__(
            self,
            space_bounds = (-100, 100),
            initial_random_point_count: int = 10,
            explore_iterations: int = 100,
            path_length: int = 10,
            acqf_optimization_kwargs: dict|None = None,
            avoid_negative: bool = False,
            predetermined_sample: list[tuple[int, int, int]]|None = None,
        ):
        """
        Initialize the Gaussian Process for exploration and exploitation.
        
        :param space_bounds: Tuple of (min, max) bounds for the search space
        :param initial_random_point_count: Number of initial random points for exploration
        :param explore_iterations: Number of iterations for exploration
        :param path_length: Length of the path to be exploited
        :param avoid_negative: Whether to avoid negative regions during path finding
        :param predetermined_sample: List of tuples (x, y, z) for predetermined samples
        """
        self.space_bounds = space_bounds
        self.explore_iterations = explore_iterations
        self.path_length = path_length
        self.avoid_negative = avoid_negative
        self.initial_random_point_count = initial_random_point_count
        self.predetermined_sample = predetermined_sample if predetermined_sample is not None else []
        self.acqf_optimization_kwargs = acqf_optimization_kwargs if acqf_optimization_kwargs is not None else {}
        
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
        print(f"Starting Gaussian Process exploration with {self.explore_iterations} iterations...")
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
            X_initial = torch.rand(self.initial_random_point_count, 2, device=self.device, dtype=torch.float64)
            # Transform from [0, 1] to [min_bound, max_bound]
            X_initial = X_initial * (self.space_bounds[1] - self.space_bounds[0]) + self.space_bounds[0]
            
            print(f"Starting with {self.initial_random_point_count} initial random points...")
            for i in tqdm(range(self.initial_random_point_count), desc="Sampling initial points"):
                x, y = X_initial[i].tolist()
                z = query_z(x, y)
                
                if z is not None:
                    self.train_x.append([x, y])
                    self.train_y.append(z)
                            
            # Main Bayesian Optimization loop
            remaining_calls = self.explore_iterations - self.initial_random_point_count
            
            # Convert to tensors with double precision
            train_x_tensor = torch.tensor(self.train_x, dtype=torch.float64, device=self.device)
            train_y_tensor = torch.tensor(self.train_y, dtype=torch.float64, device=self.device).unsqueeze(-1)
                        
            for _ in tqdm(range(remaining_calls), desc="Exploration", unit="iteration"):
                # Step 1: Update normalized and standardized data
                train_x_normalized = normalize(train_x_tensor, bounds=bounds_tensor)
                train_y_standardized = standardize(train_y_tensor)
                
                # Step 2: Fit GP model
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model = SingleTaskGP(train_x_normalized, train_y_standardized)
                    mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                    fit_gpytorch_model(mll)
                    
                    # Step 3: Define acquisition function
                    best_value = train_y_standardized.max()
                    EI = qExpectedImprovement(
                        self.model, 
                        best_f=best_value,
                    )
                
                # Step 4: Optimize acquisition function in normalized space
                bounds_ei = torch.stack([
                    torch.zeros(2, device=self.device, dtype=torch.float64),
                    torch.ones(2, device=self.device, dtype=torch.float64)
                ])
            
                new_x_normalized, _ = optimize_acqf(
                    acq_function=EI,
                    bounds=bounds_ei,
                    q=10,
                    **self.acqf_optimization_kwargs
                )
                
                new_x = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * new_x_normalized

                new_x_value, new_y_value = new_x[0].tolist()
                for i in range(10):
                    # Convert normalized point back to original space and take if already not explored
                    new_x_value, new_y_value = new_x[i].tolist()
                    new_x_value = round(new_x_value, 2)
                    new_y_value = round(new_y_value, 2)
                    if [new_x_value, new_y_value] not in self.train_x:
                        break
                
                # Query the value at the new point
                z = query_z(new_x_value, new_y_value)
                if z is not None:
                    self.train_x.append([new_x_value, new_y_value])
                    self.train_y.append(z)
                    train_x_tensor = torch.tensor(self.train_x, dtype=torch.float64, device=self.device)
                    train_y_tensor = torch.tensor(self.train_y, dtype=torch.float64, device=self.device).unsqueeze(-1)
                
        
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
        
        # Prepare grid data for path planning
        self._prepare_grid_data()

        print("Run a possible exploit grid search. May take a while...")
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
        
        # Top points from exploration for prioritized grid evaluation
        top_points = sorted([(int(round(x)), int(round(y)), z) for (x, y), z in zip(self.train_x, self.train_y)], 
                           key=lambda x: x[2], reverse=True)[:3]
        
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
                            
        # Find best path from the top points
        best_overall_path = []
        best_overall_value = float('-inf')
        
        # Try the top points as starting points
        # Try all points as potential starting points to find the best overall path
        for point in self.grid_data.keys():
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
                    for dx, dy in get_neighbors((0, 0)):
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
        
        # Validate path - ensure all moves are  single-step
        for i in range(len(best_overall_path)-1):
            x1, y1 = best_overall_path[i]
            x2, y2 = best_overall_path[i+1]
            dx = x2 - x1
            dy = y2 - y1
            
            # Make sure this is a valid move of distance 1
            if not ((dx == 0 and abs(dy) == 1) or (dy == 0 and abs(dx) == 1) or
                    (abs(dx) == 1 and abs(dy) == 1)):
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
        if self.best_path is None:
            print("No path found. Please run explore first.")
            return []

        # Verify and query the path points
        if len(self.best_path) < self.path_length:
            # TODO: what to do now?
            print(f"Warning: Path length is only {len(self.best_path)}, expected {self.path_length}")
        
        # Query all points in the path to make sure we have their true values
        for point in self.best_path:
            x, y = point
            query_z(x, y)
        
        return self.best_path
