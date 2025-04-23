import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
import warnings
from tqdm import tqdm
from typing import Callable


def get_neighbors(point):
    """Get adjacent points (up, right, down, left and diagonals)"""
    x, y = point
    return [(x+dx, y+dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] #, (1, 1), (-1, -1), (1, -1), (-1, 1)]
            if -100 <= x+dx <= 100 and -100 <= y+dy <= 100]

class RandomForestRegressorExploreExploit:
    def __init__(
            self,
            space_bounds = (-100, 100),
            n_iterations: int = 200,
            path_length: int = 10,
            predetermined_sample: list[tuple[int, int, int]]|None = None,
        ):
        """
        Initialize the Random Forest Regressor for exploration and exploitation.
        
        :param space_bounds: Tuple of (min, max) bounds for the search space.
        :param n_iterations: Number of iterations for exploration.
        :param path_length: Length of the path to be exploited
        :param predetermined_sample: List of tuples (x, y, z) for predetermined samples.
            Useful to perform the exploitation with a set of known points.
        """
        self.rf_model = None
        self.space_bounds = space_bounds
        self.n_iterations = n_iterations
        self.path_length = path_length
        self.predetermined_sample = predetermined_sample if predetermined_sample is not None else []

    def get_model(self) -> RandomForestRegressor|None:
        """Accessor for getting the trained Random Forest model"""
        return self.rf_model

    def explore(self, query_z: Callable[[float, float], int]) -> RandomForestRegressor:
        """
        Explore the environment using Random Forest regression.
        The function will sample points in the space, query their z values,
        and use them to train the Random Forest model.
        """
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

        if self.predetermined_sample:
            # Use predetermined samples if provided
            X_samples = np.array([[x, y] for x, y, _ in self.predetermined_sample])
            y_samples = np.array([z for _, _, z in self.predetermined_sample])

        else:
            RANDOM_SAMPLES = 50  
            X_samples = []
            y_samples = []
            
            # Initial random sampling (20 points)
            for _ in tqdm(range(RANDOM_SAMPLES), desc="Exploring with random sampling", unit="sample"):
                x = random.randint(self.space_bounds[0], self.space_bounds[1])
                y = random.randint(self.space_bounds[0], self.space_bounds[1])
                z = query_z(x, y)
                if z is not None:
                    X_samples.append([x, y])
                    y_samples.append(z)
            
            # Convert to numpy arrays
            X_samples = np.array(X_samples)
            y_samples = np.array(y_samples)
            
            for _ in tqdm(range(self.n_iterations - RANDOM_SAMPLES), desc="Exploring with iterated fitting", unit="iteration"):
                # Fit RF model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.rf_model.fit(X_samples, y_samples)
                
                # Generate candidate points
                candidates = []
                for _ in range(100):
                    x = random.randint(self.space_bounds[0], self.space_bounds[1])
                    y = random.randint(self.space_bounds[0], self.space_bounds[1])
                    candidates.append([x, y])
                candidates = np.array(candidates)
                
                # Calculate predictions for candidates
                predictions = self.rf_model.predict(candidates)
                
                # Select best point according to predictions
                best_idx = np.argmax(predictions)
                next_x, next_y = candidates[best_idx]
                
                # Query the actual value
                z = query_z(next_x, next_y)
                if z is not None:
                    X_samples = np.vstack((X_samples, [next_x, next_y]))
                    y_samples = np.append(y_samples, z)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.rf_model.fit(X_samples, y_samples)

        return self.rf_model

    def exploit(self, query_z: Callable[[float, float], int]):
        """
        Exploit the environment using the trained Random Forest model.
        The function will find the best path in the environment based on the model's predictions.
        """
        if not hasattr(self, 'start_point') or self.start_point is None:
            self.start_point = self._exploit_start_point()

        self.best_path = self._find_best_path()
        for point in self.best_path:
            x, y = point
            query_z(x, y)

    def _exploit_start_point(self, n_candidates=5) -> list[tuple]:
        """
        Find multiple promising starting points for exploitation using Random Forest predictions
        Returns a list of starting points as tuples (x, y)
        """
        if self.rf_model is None:
            raise ValueError("Model not trained. Call explore first.")

        grid_points = []
        for x in range(self.space_bounds[0], self.space_bounds[1] + 1):
            for y in range(self.space_bounds[0], self.space_bounds[1] + 1):
                grid_points.append([x, y])
        grid_points = np.array(grid_points)
        mean_scores = self.rf_model.predict(grid_points)
        
        # Get indices of top n_candidates points
        top_indices = np.argsort(mean_scores)[-n_candidates:]
        start_points = [tuple(map(int, grid_points[idx])) for idx in top_indices]
        return start_points

    def _find_best_path(self, n_trials=5):
        """
        Find best connected path using beam search with Random Forest predictions
        Tries multiple starting points and iterations to find the globally best path
        Returns the path as a list of tuples (x, y)
        """
        if self.rf_model is None:
            raise ValueError("Model not trained. Call explore first.")

        start_points = self._exploit_start_point(n_candidates=5)
        best_overall_path = None
        best_overall_score = float('-inf')
        
        for start in start_points:
            for trial in tqdm(range(n_trials), desc=f"Finding paths from {start}", unit="trial"):
                beam_width = 20
                beam = [(0, [start])]  # (total_score, path)
                visited = {start}
                
                while len(beam[0][1]) < self.path_length:
                    new_beam = []
                    for total_score, path in beam:
                        current = path[-1]
                        neighbors = [(x, y) for x, y in get_neighbors(current)
                                   if (x, y) not in visited]
                        
                        if not neighbors:
                            continue
                            
                        X_pred = np.array(neighbors)
                        scores = self.rf_model.predict(X_pred)
                        
                        # Add randomization to avoid getting stuck in local optima
                        noise = np.random.normal(0, 0.1, size=len(scores))
                        scores += noise
                        
                        for neighbor, score in zip(neighbors, scores):
                            new_path = path + [neighbor]
                            new_score = total_score + score
                            new_beam.append((new_score, new_path))
                            
                    new_beam.sort(reverse=True)
                    beam = new_beam[:beam_width]
                    
                    for _, path in beam:
                        visited.add(path[-1])
                
                # Update best overall path if current path is better
                if beam[0][0] > best_overall_score:
                    best_overall_score = beam[0][0]
                    best_overall_path = beam[0][1]
                
        return best_overall_path
