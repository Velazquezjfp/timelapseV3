import os
from flask import Flask, request, jsonify
import base64
import numpy as np
from face_module import blur_heads
from detector import process_image_multi_detector
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve secret key from environment variables
SECRET_KEY = os.environ.get('APP_SECRET_KEY')

@app.route("/")
def init_connect():
    return "Endpoint reachable"

@app.route("/detection", methods=["POST"])
def detect():
    # Validate secret key
    provided_secret_key = request.headers.get('secret-key')
    if not provided_secret_key or provided_secret_key != SECRET_KEY:
        return jsonify({"status": "error", "message": "Unauthorized access"}), 401

    # Get headers
    blur_faces = request.headers.get('blur-faces')
    detect_objects = request.headers.get('detect-objects')
    blur_mode = request.headers.get('blur-mode', 'standard')  # 'standard' or 'fast'

    # Check for active services
    if not blur_faces and not detect_objects:
        return jsonify({"status": "success", "system_message": "Please include headers 'blur_faces' and 'detect_objects' with a true or false value."})

    if blur_faces == "false" and detect_objects == "false":
        return jsonify({"status": "success", "system_message": "Please assign a 'true' value to one or both of the headers: 'blur_faces', 'detect_objects'."})

    # Access JSON data from the request
    request_data = request.get_json()
    base64_image_string = request_data.get('image', '')

    # Decode the base64 image string
    image_data = base64.b64decode(base64_image_string)

    # Convert binary image data to a numpy array
    numpy_image = np.frombuffer(image_data, np.uint8)

    # Process image based on headers
    detection_data, original_size = process_image_multi_detector(numpy_image)

    # Initialize response
    response = {"status": "success"}

    # Handle face/head blurring if required
    if blur_faces == 'true' and 'person' in detection_data and len(detection_data['person']) > 0:
        person_coordinates = [entry['coordinate'] for entry in detection_data['person']]

        if person_coordinates:
            # Use new face_module with MediaPipe Pose-based head detection
            blurred_image_base64 = blur_heads(
                person_coordinates,
                './image.jpg',
                original_size,
                mode=blur_mode
            )

            # Send response based on headers values. Blurred image could be None value.
            if detect_objects == 'true':
                response.update({
                    "coordinates_data": detection_data,
                    "blured_image": blurred_image_base64
                })
            else:
                response["blured_image"] = blurred_image_base64
        else:
            # If there were no people detected, no blurring needed
            if detect_objects == 'true':
                response.update({
                    "coordinates_data": detection_data,
                    "blured_image": None
                })
            else:
                response["blured_image"] = None

    # Add detection data if detect-objects is true
    elif detect_objects == 'true':
        response["coordinates_data"] = detection_data

    return jsonify(response)

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000)
    pass
