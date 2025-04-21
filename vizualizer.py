import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_environment(explore_exploit_instance):
    """
    Visualize the environment of explore/exploit instance using matplotlib.
    This function will plot the predicted z values over the grid of points.
    """
    if not hasattr(explore_exploit_instance, 'get_model') or explore_exploit_instance.get_model() is None:
        raise ValueError("Model not trained or instance has no get_model function. Call explore first.")
    if not hasattr(explore_exploit_instance, 'space_bounds') or explore_exploit_instance.space_bounds is None:
        raise ValueError("Space bounds not defined. Set space_bounds in the instance.")

    grid_points = []
    for x in range(explore_exploit_instance.space_bounds[0], explore_exploit_instance.space_bounds[1] + 1):
        for y in range(explore_exploit_instance.space_bounds[0], explore_exploit_instance.space_bounds[1] + 1):
            grid_points.append([x, y])
    grid_points = np.array(grid_points)
    
    model = explore_exploit_instance.get_model()
    
    # Predict values with appropriate model type
    if hasattr(model, 'predict'):
        mean_scores = model.predict(grid_points)
    else:
        mean_scores = predict_gp(model, grid_points, explore_exploit_instance.space_bounds)

    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=mean_scores, cmap='viridis')
    plt.colorbar(label='Predicted z value')

    if hasattr(explore_exploit_instance, 'best_path') and explore_exploit_instance.best_path is not None:
        best_path_x, best_path_y = zip(*explore_exploit_instance.best_path)
        plt.plot(best_path_x, best_path_y, color='red', linewidth=2, label='Best Path')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Predicted Environment Visualization')
    plt.show()

def predict_gp(model, points, bounds):
    """Helper function for GP model prediction with proper scaling"""
    device = next(model.parameters()).device
    x_tensor = torch.tensor(points, dtype=torch.float64, device=device)
    
    # Normalize inputs to [0,1] range
    min_bound, max_bound = bounds
    x_normalized = (x_tensor - min_bound) / (max_bound - min_bound)
    
    # Get predictions
    posterior = model.posterior(x_normalized)
    
    # Scale to match typical range of scores (-1000 to 1000)
    mean = posterior.mean.detach().cpu().numpy().flatten()
    return mean * 1000