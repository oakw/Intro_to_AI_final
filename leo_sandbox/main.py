import numpy as np
import requests
import time
import sys
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from martins_sandbox.test_interface import get_score as external_get_score
from martins_sandbox.test_performance import visualize_path

# API endpoint - update to use local mock server
# BASE_URL = "http://157.180.73.240:8080/"  # Remote server
BASE_URL = "http://127.0.0.1:5000/"  # Local mock server
temp_calls = 100  # Number of calls for exploration phase
# Set default layout_id to '2'
DEFAULT_LAYOUT_ID = '1'

# Flag to determine whether to use the mock server or external API
USE_MOCK_SERVER = True

# API call counter
api_call_counter = 0

# Function to call the API and get the result
def call_api(x, y, layout_id=DEFAULT_LAYOUT_ID):
    global api_call_counter
    if api_call_counter >= 199:
        print("WARNING: API call limit reached!")
        return -1000  # Return penalty value
    
    api_call_counter += 1
    url = f"{BASE_URL}{layout_id}/{x}/{y}"  # Note the URL format change to match the Flask route
    print(f"Calling API with x={x}, y={y}, layout={layout_id}...")
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            try:
                # Parse the response as JSON
                data = response.json()
                print(f"API response: {data}")
                return float(data.get('z', -1000))
            except ValueError:
                # If JSON parsing fails, try to parse as text
                print(f"Failed to parse JSON: {response.text}")
                return -1000
            
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return -1000  # Return a penalty value for errors
    except Exception as e:
        print(f"Exception: {e}")
        return -1000  # Return a penalty value for exceptions

# Function to get the score (using either the mock server or external API)
def get_score(x, y, layout_id=DEFAULT_LAYOUT_ID):
    if USE_MOCK_SERVER:
        return call_api(x, y, layout_id)
    else:
        return external_get_score(x, y, layout_id)

# Function to find the best path - modified version of your existing code
def find_best_locations(layout_id='1'):
    # Define the search space
    space = [Real(-100, 100, name='x'), Real(-100, 100, name='y')]
    
    # Store all evaluated points
    X = []
    Y = []
    
    # Flag to indicate if we found a value of 1000
    found_optimal = False
    optimal_x = None
    optimal_y = None
    
    # Function to optimize (negative because we want to maximize the result)
    def objective(params):
        nonlocal found_optimal, optimal_x, optimal_y
        x, y = params
        result = call_api(x, y, layout_id)
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)
        
        # Store the point and its value
        X.append((x, y))  # Store point directly during optimization
        Y.append(result)  # Store result directly during optimization
        
        # Check if we found the optimal value (1000)
        if result == 1000:
            found_optimal = True
            optimal_x, optimal_y = x, y
            print(f"Found optimal value 1000 at ({x}, {y})! Stopping random search...")
            
        return -result  # Negate because we want to maximize
    
    # Create a custom callback to check for early stopping
    def callback(res):
        if found_optimal:
            return True  # Stop the optimization
        return False
    
    # Run the exploration phase with up to temp_calls calls, but stop early if we find optimal value
    print(f"Starting exploration phase with up to {temp_calls} API calls...")
    res_gp = gp_minimize(objective, space, n_calls=temp_calls, random_state=42, 
                         verbose=True, callback=callback)
    
    # Extract the results
    X_array = np.array(res_gp.x_iters)
    Y_array = -np.array(res_gp.func_vals)
    
    # If we found an optimal value, explore around it with remaining calls
    calls_used = len(X)
    calls_remaining = 200 - calls_used
    
    if found_optimal:
        print(f"Found optimal value at ({optimal_x}, {optimal_y}). Using remaining {calls_remaining} calls to explore nearby.")
        # Define the grid boundaries - focus tightly around the optimal point
        grid_size = 10  # Explore a reasonable area around the optimal point
        lower_x = max(-100, int(optimal_x) - grid_size // 2)
        upper_x = min(100, int(optimal_x) + grid_size // 2)
        lower_y = max(-100, int(optimal_y) - grid_size // 2)
        upper_y = min(100, int(optimal_y) + grid_size // 2)
    else:
        # We didn't find an optimal value, use the best point we found
        best_idx = np.argmax(Y_array)
        best_x, best_y = X_array[best_idx, 0], X_array[best_idx, 1]
        best_value = Y_array[best_idx]
        print(f"Best point found during exploration: ({best_x}, {best_y}) with value: {best_value}")
        
        # Define the grid boundaries - focus tightly around the best point
        grid_size = 8  # Small enough to stay within our remaining calls
        lower_x = max(-100, int(best_x) - grid_size // 2)
        upper_x = min(100, int(best_x) + grid_size // 2)
        lower_y = max(-100, int(best_y) - grid_size // 2)
        upper_y = min(100, int(best_y) + grid_size // 2)
    
    print(f"API calls used so far: {calls_used}, remaining: {calls_remaining}")
    print(f"Exploring {grid_size}x{grid_size} grid around best point with remaining {calls_remaining} calls")
    
    # Store grid values and points
    grid_points = []
    grid_values = []
    
    # First, reuse the points we've already sampled in the exploration phase
    # Create a dictionary to track evaluated points
    explored_data = {}
    for i, (x, y) in enumerate(X_array):
        # Round to nearest integer for grid indexing
        int_x, int_y = int(round(x)), int(round(y))
        explored_data[(int_x, int_y)] = Y_array[i]
        
        # Only include if within our grid boundaries
        if lower_x <= int_x <= upper_x and lower_y <= int_y <= upper_y:
            grid_points.append((int_x, int_y))
            grid_values.append(Y_array[i])
            
    # Set to track points we've already evaluated
    evaluated_points = set([(int(round(x)), int(round(y))) for x, y in X_array])
            
    # Sort the top points from exploration for prioritized grid evaluation
    top_points = sorted([(int(round(x)), int(round(y)), y) for (x, y), y in zip(X_array, Y_array)], 
                       key=lambda x: x[2], reverse=True)[:5]
    
    # Prioritize exploration around the top points
    new_calls = 0
    max_new_calls = min(calls_remaining, 50)  # Stay within API call limit
    
    for center_x, center_y, _ in top_points:
        if new_calls >= max_new_calls:
            break
            
        # Explore a 5x5 grid around each top point
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if new_calls >= max_new_calls:
                    break
                    
                x, y = center_x + dx, center_y + dy
                
                # Stay within global boundaries
                if not (-100 <= x <= 100 and -100 <= y <= 100):
                    continue
                    
                # Skip if already evaluated
                if (x, y) in evaluated_points:
                    continue
                
                # Call API for this new point
                value = call_api(x, y, layout_id)
                new_calls += 1
                
                # Store the result
                grid_points.append((x, y))
                grid_values.append(value)
                evaluated_points.add((x, y))
                
                time.sleep(0.1)  # Be nice to the API

    # Find the best integer point from our combined exploration
    best_grid_idx = np.argmax(grid_values)
    best_grid_x, best_grid_y = grid_points[best_grid_idx]
    best_grid_value = grid_values[best_grid_idx]
    print(f"Best integer point: ({best_grid_x}, {best_grid_y}) with value: {best_grid_value}")
    
    # Create a 2D grid to store all values for path planning
    grid_data = {}
    for (x, y), value in zip(grid_points, grid_values):
        grid_data[(x, y)] = value
    
    # Also add any points from the exploration phase that we didn't include yet
    for (x, y), value in explored_data.items():
        if (x, y) not in grid_data:
            grid_data[(x, y)] = value
    
    print(f"Grid has {len(grid_data)} points for path planning")
    print(f"Total API calls made: {api_call_counter}")
    
    # Create a graph to find the best path
    G = networkx.DiGraph()
    
    # Add nodes and edges for all possible moves
    for (x, y), value in grid_data.items():
        # Add the node
        G.add_node((x, y), value=value)
        
        # Add edges to all possible neighbors (4 directions)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Only orthogonal moves
            nx, ny = x + dx, y + dy
            if (nx, ny) in grid_data:
                # Add an edge with weight based on the neighbor's value
                # Negative weight because we want to find the maximum value path
                G.add_edge((x, y), (nx, ny), weight=-grid_data[(nx, ny)])
    
    # Function to find the best path of length 10 starting from a given point
    def find_best_path(start_point):
        paths = []
        visited = set()
        
        def dfs(node, path, total_value):
            if len(path) == 10:
                paths.append((path[:], total_value))
                return
            
            visited.add(node)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    new_value = total_value + grid_data[neighbor]
                    path.append(neighbor)
                    dfs(neighbor, path, new_value)
                    path.pop()
            visited.remove(node)
        
        dfs(start_point, [start_point], grid_data[start_point])
        
        if not paths:
            return [], 0
        
        # Find the path with the highest total value
        best_path, best_value = max(paths, key=lambda x: x[1])
        return best_path, best_value
    
    # Try different starting points to find the best overall path
    best_overall_path = []
    best_overall_value = float('-inf')
    
    # Try the top 10 integer points from all our data
    top_points = sorted(list(grid_data.items()), key=lambda x: x[1], reverse=True)[:10]
    
    for (point), _ in top_points:
        path, value = find_best_path(point)
        if value > best_overall_value and len(path) == 10:
            best_overall_value = value
            best_overall_path = path
    
    # If we didn't find a valid path, use a greedy approach
    if not best_overall_path or len(best_overall_path) < 10:
        print("No valid path found or path too short. Using enhanced greedy approach...")
        
        # First, ensure we have evaluated points in all four directions from the best point
        start_point = (best_grid_x, best_grid_y)
        path = [start_point]
        visited = set([start_point])
        
        # Try to add more points to the grid if needed to ensure connectivity
        for distance in range(1, 6):  # Try exploring further if needed
            for dx, dy in [(0, distance), (0, -distance), (distance, 0), (-distance, 0)]:
                if len(grid_data) >= 30 and api_call_counter >= 180:
                    break  # Don't make too many extra calls
                
                x, y = start_point[0] + dx, start_point[1] + dy
                if not (-100 <= x <= 100):
                    continue
                
                point = (x, y)
                if point not in grid_data and point not in evaluated_points:
                    value = call_api(x, y, layout_id)
                    time.sleep(0.1)
                    grid_data[point] = value
                    
                    # Add to graph
                    G.add_node(point, value=value)
                    # Connect to any existing neighbors
                    for ndx, ndy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = (x + ndx, y + ndy)
                        if neighbor in grid_data:
                            G.add_edge(point, neighbor, weight=-grid_data[neighbor])
                            G.add_edge(neighbor, point, weight=-value)
                    
        # Enhanced greedy approach that ensures a path of length 10
        path = [start_point]
        visited = set([start_point])
        
        # Attempt 1: Try to find a connected path using grid_data points
        print("Attempt 1: Finding connected path in explored grid...")
        while len(path) < 10:
            current = path[-1]
            candidates = []
            
            # Check all possible neighbors (including further ones if needed)
            for distance in range(1, 6):
                for dx, dy in [(0, distance), (0, -distance), (distance, 0), (-distance, 0)]:
                    nx, ny = current[0] + dx, current[1] + dy
                    neighbor = (nx, ny)
                    
                    if neighbor in grid_data and neighbor not in visited:
                        candidates.append((neighbor, grid_data[neighbor], distance))
            
            # If we found candidates, pick the best one (highest value, lowest distance)
            if candidates:
                # Sort by value (descending) then by distance (ascending)
                candidates.sort(key=lambda x: (-x[1], x[2]))
                best_neighbor, _, _ = candidates[0]
                path.append(best_neighbor)
                visited.add(best_neighbor)
            else:
                break
        
        # If we still don't have 10 points, use the non-optimal approach: generate artificial points
        if len(path) < 10:
            print(f"Attempt 2: Adding artificial points to complete the path (current length: {len(path)})...")
            # Make sure we have enough valid moves from current position
            last_point = path[-1]
            
            # Continue adding points in cardinal directions until we have 10
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
            dir_index = 0
            
            while len(path) < 10:
                dx, dy = directions[dir_index]
                nx, ny = last_point[0] + dx, last_point[1] + dy
                new_point = (nx, ny)
                
                # Check if the new point is within bounds
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    if new_point not in visited:
                        # If we don't know the value yet, query the API
                        if new_point not in grid_data and api_call_counter < 198:
                            value = call_api(nx, ny, layout_id)
                            time.sleep(0.1)
                            grid_data[new_point] = value
                        # If we can't query or already know it, just add it
                        else:
                            if new_point not in grid_data:
                                grid_data[new_point] = -100  # Placeholder value
                                
                        path.append(new_point)
                        visited.add(new_point)
                        last_point = new_point
                
                # Try the next direction
                dir_index = (dir_index + 1) % 4
        
        best_overall_path = path
        
        # Validate the final path - make sure all moves are orthogonal
        for i in range(len(best_overall_path)-1):
            x1, y1 = best_overall_path[i]
            x2, y2 = best_overall_path[i+1]
            dx = x2 - x1
            dy = y2 - y1
            
            # Make sure this is a valid orthogonal move
            if not ((dx == 0 and abs(dy) == 1) or (dy == 0 and abs(dx) == 1)):
                print(f"Warning: Invalid move from ({x1}, {y1}) to ({x2}, {y2}). Fixing...")
                # Fix the path by inserting intermediate points
                if dx != 0 and dy != 0:
                    # Split this move into two orthogonal moves
                    best_overall_path.insert(i+1, (x1, y2))  # Horizontal move first
                    break  # Restart the validation
                elif abs(dx) > 1:
                    # Insert intermediate horizontal point
                    best_overall_path.insert(i+1, (x1 + (1 if dx > 0 else -1), y1))
                    break  # Restart the validation
                elif abs(dy) > 1:
                    # Insert intermediate vertical point
                    best_overall_path.insert(i+1, (x1, y1 + (1 if dy > 0 else -1)))
                    break  # Restart the validation
    
    # Final check: ensure path length is exactly 10
    while len(best_overall_path) > 10:
        best_overall_path.pop()
        
    return best_overall_path[:10]  # Ensure we return exactly 10 points

# Updated main with default layout_id
if __name__ == "__main__":
    layout_id = DEFAULT_LAYOUT_ID
    if len(sys.argv) > 1:
        layout_id = sys.argv[1]
    
    print(f"Using layout_id: {layout_id}")
    print(f"Using {'mock server' if USE_MOCK_SERVER else 'external API'}")
    
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
        direction = {(0, 1): "right", (0, -1): "left", (1, 0): "down", (-1, 0): "up"}.get((dx, dy), "unknown")
        print(f"Step {i+1}: Move {direction} from ({x1}, {y1}) to ({x2}, {y2})")    
    # Visualize the path
    output_file = f'path_visualization_layout_{layout_id}.png'
    visualize_path(best_path, layout_id=layout_id, output_file=output_file)

    print(f"Total API calls made: {api_call_counter}")
    if api_call_counter > 200:
        print(f"WARNING: Exceeded API call limit with {api_call_counter} calls!")