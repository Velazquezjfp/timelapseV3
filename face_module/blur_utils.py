"""
Blurring utilities for head/face privacy protection.
Provides Gaussian blur application, fallback blur logic, and base64 encoding.
"""

import cv2
import base64
import numpy as np


def apply_gaussian_blur(image, bbox, kernel_size=(99, 99), sigma=30):
    """
    Apply Gaussian blur to a specific region of the image.

    Args:
        image: OpenCV image (numpy array, modified in place)
        bbox: Bounding box tuple (x, y, w, h)
        kernel_size: Gaussian kernel size (must be odd numbers)
        sigma: Gaussian sigma value

    Returns:
        True if blur was applied, False otherwise
    """
    x, y, w, h = bbox

    # Validate bounding box
    if w <= 0 or h <= 0:
        return False

    # Ensure coordinates are within image bounds
    img_h, img_w = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    if x2 <= x or y2 <= y:
        return False

    try:
        region = image[y:y2, x:x2]
        if region.size == 0:
            return False

        blurred = cv2.GaussianBlur(region, kernel_size, sigma)
        image[y:y2, x:x2] = blurred
        return True
    except Exception as e:
        print(f"Error applying blur: {e}")
        return False


def apply_fallback_blur(image, person_bbox, head_ratio=0.25):
    """
    Apply fallback blur to the top portion of a person bounding box.
    Only applies if the person appears to be standing (height > width * 1.5).

    Args:
        image: OpenCV image (numpy array, modified in place)
        person_bbox: Person bounding box tuple (x, y, w, h)
        head_ratio: Ratio of height to blur (default 25%)

    Returns:
        True if fallback blur was applied, False otherwise
    """
    x, y, w, h = person_bbox

    # Only apply to standing persons (height > width * 1.5)
    if h <= w * 1.5:
        print(f"Skipping fallback blur: not standing (h={h}, w={w})")
        return False

    # Calculate head region (top portion)
    head_height = int(h * head_ratio)
    if head_height < 5:  # Minimum blur height
        return False

    head_bbox = (x, y, w, head_height)

    result = apply_gaussian_blur(image, head_bbox)
    if result:
        print(f"Applied fallback blur to top {head_ratio*100:.0f}% at: x={x}, y={y}, w={w}, h={head_height}")

    return result


def encode_image_base64(image):
    """
    Encode an OpenCV image to base64 JPEG string.

    Args:
        image: OpenCV image (numpy array)

    Returns:
        Base64 encoded string, or None on error
    """
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def calculate_head_bbox_from_landmarks(landmarks, person_bbox, image_shape, padding=0.2, scale=2.0):
    """
    Calculate a head bounding box from pose landmarks.

    Args:
        landmarks: Dictionary with landmark positions {name: (x, y, visibility)}
        person_bbox: Person bounding box (x, y, w, h) for context
        image_shape: Shape of the subframe image (height, width)
        padding: Extra padding around detected landmarks (ratio)
        scale: Scale factor for final box size (2.0 = 100% bigger, doubled)

    Returns:
        Head bounding box (x, y, w, h) relative to the subframe, or None
    """
    head_landmark_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                           'left_eye_inner', 'right_eye_inner', 'left_eye_outer', 'right_eye_outer']

    # Collect valid landmarks
    points = []
    for name in head_landmark_names:
        if name in landmarks:
            x, y, visibility = landmarks[name]
            if visibility > 0.5:  # Only use visible landmarks
                points.append((x, y))

    if len(points) < 2:  # Need at least 2 points to estimate head
        return None

    # Calculate bounding box around landmarks
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add padding
    width = max_x - min_x
    height = max_y - min_y

    # Estimate full head size (landmarks only cover part of head)
    # Head is typically taller than the face landmarks suggest
    estimated_head_height = height * 2.0  # Double the landmark height
    estimated_head_width = max(width * 1.5, estimated_head_height * 0.8)

    # Center the box on the landmarks
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2 - height * 0.3  # Shift up slightly

    # Apply padding and scale factor (scale=2.0 doubles the box size proportionally)
    final_width = estimated_head_width * (1 + padding) * scale
    final_height = estimated_head_height * (1 + padding) * scale

    x = int(center_x - final_width / 2)
    y = int(center_y - final_height / 2)
    w = int(final_width)
    h = int(final_height)

    # Clamp to image bounds
    img_h, img_w = image_shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return None

    return (x, y, w, h)
