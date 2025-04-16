import os
from PIL import Image, ImageDraw
import numpy as np

def visualize_path(path, layout_id='1', output_file='path_visualization.png'):
    """
    Draw the path on the layout image from mock-server
    
    Args:
        path: List of (x, y) tuples representing the path points
        layout_id: ID of the layout to use as background
        output_file: Name of the output file to save the visualization
    """
    # Load the layout image
    layouts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mock-server', 'layouts')
    layout_path = os.path.join(layouts_dir, f"{layout_id}.png")
    
    if not os.path.exists(layout_path):
        raise FileNotFoundError(f"Layout {layout_id} not found")
    
    # Open and convert to RGB for colored visualization
    layout = Image.open(layout_path).convert('RGB')
    draw = ImageDraw.Draw(layout)
    
    # Convert coordinates from [-100, 100] to image coordinates [0, 799]
    def transform_coords(x, y):
        img_x = int((x + 100) * 799 / 200)
        img_y = int((y + 100) * 799 / 200)
        return img_x, img_y
    
    # Draw the path
    point_radius = 2
    line_width = 3
    
    # Draw lines between consecutive points
    for i in range(len(path) - 1):
        start = transform_coords(path[i][0], path[i][1])
        end = transform_coords(path[i+1][0], path[i+1][1])
        draw.line([start, end], fill=(255, 0, 0), width=line_width)
    
    # Draw points and numbers
    for i, (x, y) in enumerate(path):
        img_x, img_y = transform_coords(x, y)
        
        # Draw point
        draw.ellipse(
            [(img_x - point_radius, img_y - point_radius),
             (img_x + point_radius, img_y + point_radius)],
            fill=(0, 255, 0),
            outline=(0, 0, 0)
        )
        
        # Draw point number
        # draw.text(
        #     (img_x + point_radius + 2, img_y - point_radius - 2),
        #     str(i + 1),
        #     fill=(0, 0, 255)
        # )
    
    # Save the visualization
    # layout.show()
    layout.save(output_file)
    print(f"Visualization saved as {output_file}")

if __name__ == "__main__":
    # Example usage
    test_path = [(0, 0), (10, 10), (20, 20), (30, 30)]
    visualize_path(test_path, layout_id='1')