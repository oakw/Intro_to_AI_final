from flask import Flask, jsonify
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

LAYOUTS_DIR = os.path.join(os.path.dirname(__file__), 'layouts')

def load_layout(layout_id):
    """Load layout image file based on ID"""
    layout_path = os.path.join(LAYOUTS_DIR, f"{layout_id}.png")
    
    if not os.path.exists(layout_path):
        return None
        
    return Image.open(layout_path).convert('L')  # Convert to grayscale

def get_score_at_position(layout_image, x, y):
    """
    Get score at position (x, y) in the image
    x, y are in range [-100, 100]
    Returns value from -1000 (black) to 1000 (white)
    """
    # Check if coordinates are within valid range
    if x < -100 or x > 100 or y < -100 or y > 100:
        return -1000
    
    # Map from [-100, 100] to [0, 799] (image coordinates)
    img_x = int((x + 100) * 799 / 200)
    img_y = int((y + 100) * 799 / 200)
    
    # Get pixel value (0 is black, 255 is white)
    # Need to flip the y-coordinate since image origin is at top-left
    # but we want bottom-left to be (-100, -100)
    img_y = int((100 - y) * 799 / 200)  # Flip y-axis
    pixel_value = layout_image.getpixel((img_x, img_y))
    
    # Map from [0, 255] to [-1000, 1000]
    return int(pixel_value * (2000 / 255) - 1000)

@app.route('/<int:layout_id>/<x>/<y>', methods=['GET'])
def get_layout_score(layout_id, x, y):
    """Handle requests for layout scores at specific coordinates"""
    layout = load_layout(layout_id)
    x = float(x)
    y = float(y)
    
    if layout is None:
        return jsonify({"error": f"Layout {layout_id} not found"}), 404
    
    score = get_score_at_position(layout, x, y)
    return jsonify({"z": score})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)