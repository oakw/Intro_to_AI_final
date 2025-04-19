import requests


def get_score(x, y, layout_id = '1'):
    """
    Get score at position (x, y) in the image
    x, y are in range [-100, 100]
    """
    # Make a request to the server
    url = f"http://127.0.0.1:5000/{layout_id}/{x}/{y}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json().get("z")
    else:
        return None